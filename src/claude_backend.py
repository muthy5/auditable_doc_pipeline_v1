from __future__ import annotations

import json
import logging
import os
import socket
import time
from typing import Any, Dict

from .exceptions import BackendError
from .llm_interface import LocalLLMBackend
from .retry import RetryConfig
from .token_budget import TokenWindowTracker, estimate_tokens

LOGGER = logging.getLogger(__name__)

# Pricing per million tokens (USD) as of 2026-03.
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00, "cache_write": 1.25, "cache_read": 0.10},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00, "cache_write": 3.75, "cache_read": 0.30},
}
_DEFAULT_PRICING: dict[str, float] = {"input": 3.00, "output": 15.00, "cache_write": 3.75, "cache_read": 0.30}


class CostTracker:
    """Accumulate estimated USD cost across API calls."""

    def __init__(self, budget: float = 0.25) -> None:
        self.budget = budget
        self.total_cost: float = 0.0
        self._lock = __import__("threading").Lock()

    def record(self, model: str, input_tokens: int, output_tokens: int,
               cache_creation_tokens: int = 0, cache_read_tokens: int = 0) -> float:
        pricing = _MODEL_PRICING.get(model, _DEFAULT_PRICING)
        # Standard input tokens exclude cached tokens
        standard_input = max(0, input_tokens - cache_creation_tokens - cache_read_tokens)
        cost = (
            standard_input * pricing["input"] / 1_000_000
            + output_tokens * pricing["output"] / 1_000_000
            + cache_creation_tokens * pricing["cache_write"] / 1_000_000
            + cache_read_tokens * pricing["cache_read"] / 1_000_000
        )
        with self._lock:
            self.total_cost += cost
        LOGGER.info("Cost: pass=$%.4f cumulative=$%.4f budget=$%.2f model=%s", cost, self.total_cost, self.budget, model)
        return cost

    def check_budget(self, pass_name: str) -> None:
        with self._lock:
            current = self.total_cost
        if current >= self.budget:
            raise BudgetExceededError(
                f"Budget of ${self.budget:.2f} exceeded (${current:.4f} spent) before pass '{pass_name}'."
            )


class BudgetExceededError(Exception):
    """Raised when the run cost exceeds the configured budget."""


def _compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


class ClaudeBackendConfig:
    """Configuration for the Claude API backend."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        max_retries: int = 4,
        enable_prompt_caching: bool = True,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.enable_prompt_caching = enable_prompt_caching


class ClaudeAPIBackend(LocalLLMBackend):
    """Backend that calls the Anthropic Claude API."""

    def __init__(self, config: ClaudeBackendConfig, token_tracker: TokenWindowTracker | None = None, cost_tracker: CostTracker | None = None) -> None:
        if not config.api_key:
            raise ValueError(
                "Claude API key must be provided via --claude-api-key or ANTHROPIC_API_KEY environment variable."
            )
        self.config = config
        try:
            import anthropic

            self._client = anthropic.Anthropic(api_key=config.api_key)
        except ImportError:
            raise BackendError(
                "The 'anthropic' package is required for the Claude backend. "
                "Install it with: pip install anthropic"
            )
        self.token_tracker = token_tracker
        self.cost_tracker = cost_tracker

    def generate_json(
        self,
        pass_name: str,
        prompt_text: str,
        payload: Dict[str, Any],
        schema: Dict[str, Any] | None = None,
        model_override: str | None = None,
    ) -> Dict[str, Any]:
        """Generate a JSON response from the Claude API with retry + backoff."""
        effective_model = model_override or self.config.model
        if self.cost_tracker is not None:
            self.cost_tracker.check_budget(pass_name)
        composed_prompt = self._compose_prompt(pass_name=pass_name, prompt_text=prompt_text, payload=payload, schema=schema)
        static_part = ""
        dynamic_part = ""
        if self.config.enable_prompt_caching:
            static_part, dynamic_part = self._split_prompt(
                pass_name=pass_name,
                prompt_text=prompt_text,
                payload=payload,
                schema=schema,
            )

        def _call() -> Dict[str, Any]:
            call_start = time.perf_counter()
            if self.config.enable_prompt_caching:
                response = self._client.messages.create(
                    model=effective_model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    system=[
                        {
                            "type": "text",
                            "text": static_part,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": dynamic_part}],
                )
            else:
                response = self._client.messages.create(
                    model=effective_model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": composed_prompt}],
                )
            call_elapsed = time.perf_counter() - call_start
            LOGGER.info("Claude API call timing: pass=%s model=%s duration_seconds=%.2f", pass_name, effective_model, call_elapsed)
            usage = getattr(response, "usage", None)
            if usage is not None:
                cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", None) or 0
                cache_read_tokens = getattr(usage, "cache_read_input_tokens", None) or 0
                LOGGER.debug(
                    "Claude prompt cache usage: pass=%s cache_creation_input_tokens=%s cache_read_input_tokens=%s",
                    pass_name,
                    cache_creation_tokens,
                    cache_read_tokens,
                )
                if self.cost_tracker is not None:
                    self.cost_tracker.record(
                        model=effective_model,
                        input_tokens=getattr(usage, "input_tokens", 0),
                        output_tokens=getattr(usage, "output_tokens", 0),
                        cache_creation_tokens=cache_creation_tokens,
                        cache_read_tokens=cache_read_tokens,
                    )
            text_blocks: list[str] = []
            for block in response.content:
                block_type = getattr(block, "type", None)
                if block_type is None and isinstance(block, dict):
                    block_type = block.get("type")
                if block_type != "text":
                    continue
                block_text = getattr(block, "text", None)
                if block_text is None and isinstance(block, dict):
                    block_text = block.get("text")
                if isinstance(block_text, str):
                    text_blocks.append(block_text)
            if not text_blocks:
                raise BackendError("No text blocks found in Claude response content.")
            raw_text = "".join(text_blocks)
            parsed = self._extract_json_object(raw_text)
            if not isinstance(parsed, dict):
                raise BackendError("Parsed JSON is not an object.")
            return parsed

        def _status_code(exc: Exception) -> int | None:
            status_code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
            return status_code if isinstance(status_code, int) else None

        def _is_rate_limit_error(exc: Exception) -> bool:
            if _status_code(exc) == 429:
                return True
            text = str(exc).lower()
            return "rate limit" in text

        def _should_retry(exc: Exception) -> bool:
            if isinstance(exc, (json.JSONDecodeError, BackendError, TimeoutError, socket.timeout)):
                return True
            status_code = _status_code(exc)
            if status_code is not None:
                if status_code == 429:
                    return True
                return status_code in {408, 409, 425, 500, 502, 503, 504}
            text = str(exc).lower()
            return any(token in text for token in ["rate limit", "timeout", "temporar", "overloaded", "connection"])

        retry_config = RetryConfig(max_retries=self.config.max_retries, base_delay_s=2.0, max_delay_s=90.0)
        estimated_tokens = estimate_tokens(composed_prompt)

        try:
            attempt = 0
            while True:
                try:
                    if self.token_tracker is not None:
                        self.token_tracker.sleep_if_needed(estimated_tokens)
                    result = _call()
                    if self.token_tracker is not None:
                        self.token_tracker.record_usage(estimated_tokens)
                    return result
                except Exception as exc:  # noqa: BLE001
                    if attempt >= retry_config.max_retries or not _should_retry(exc):
                        raise
                    if _is_rate_limit_error(exc):
                        LOGGER.warning("Rate limit hit on pass '%s'. Waiting 62s for window reset.", pass_name)
                        delay = 62.0
                    else:
                        delay = min(retry_config.base_delay_s * (2**attempt), retry_config.max_delay_s)
                        LOGGER.warning("Claude retry attempt %s for pass '%s': %s", attempt + 1, pass_name, exc)
                    time.sleep(delay)
                    attempt += 1
        except Exception as exc:
            raise BackendError(
                f"Failed to produce valid JSON after {self.config.max_retries + 1} attempts "
                f"for pass '{pass_name}': {exc}"
            ) from exc

    def _summarize_schema_required_fields(self, schema: Dict[str, Any]) -> str:
        """Build a compact schema summary focused on required fields."""
        required = schema.get("required", []) if isinstance(schema, dict) else []
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        optional_count = max(0, len(properties) - len(required)) if isinstance(properties, dict) else 0

        if not required or not isinstance(properties, dict):
            return json.dumps(schema, ensure_ascii=False, indent=2)

        required_lines: list[str] = []
        for field in required:
            field_schema = properties.get(field, {}) if isinstance(field, str) else {}
            field_type = field_schema.get("type", "any") if isinstance(field_schema, dict) else "any"
            field_desc = field_schema.get("description", "") if isinstance(field_schema, dict) else ""
            if isinstance(field_type, list):
                field_type = " | ".join(str(item) for item in field_type)
            line = f"- {field} ({field_type})"
            if field_desc:
                line += f": {field_desc}"
            required_lines.append(line)

        if optional_count >= 5:
            return (
                "Schema summary (required fields only due to many optional fields):\n"
                + "\n".join(required_lines)
                + f"\nOptional fields omitted from prompt: {optional_count}."
            )
        return json.dumps(schema, ensure_ascii=False, indent=2)

    def _compose_prompt(
        self,
        pass_name: str,
        prompt_text: str,
        payload: Dict[str, Any],
        schema: Dict[str, Any] | None = None,
    ) -> str:
        """Build the full prompt for the Claude API."""
        parts = [
            "You are a JSON-only backend for an auditable document analysis pipeline.",
            "Return a single valid JSON object only. No markdown, no code fences, no explanation.",
            f"Pass: {pass_name}",
            "Prompt instructions:",
            prompt_text,
        ]
        if schema is not None:
            parts.extend(
                [
                    "Your output MUST conform to this JSON schema:",
                    self._summarize_schema_required_fields(schema),
                ]
            )
        parts.extend(
            [
                "Input payload JSON:",
                _compact_json(payload),
            ]
        )
        return "\n\n".join(parts)

    def _split_prompt(
        self,
        pass_name: str,
        prompt_text: str,
        payload: Dict[str, Any],
        schema: Dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        static_parts = [
            "You are a JSON-only backend for an auditable document analysis pipeline.",
            "Return a single valid JSON object only. No markdown, no code fences, no explanation.",
            f"Pass: {pass_name}",
            "Prompt instructions:",
            prompt_text,
        ]
        if schema is not None:
            static_parts.extend(
                [
                    "Your output MUST conform to this JSON schema:",
                    self._summarize_schema_required_fields(schema),
                ]
            )
        static_part = "\n\n".join(static_parts)
        dynamic_part = "Input payload JSON:\n" + _compact_json(payload)
        return static_part, dynamic_part

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        """Extract a JSON object from model output."""
        stripped = text.strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        start = stripped.find("{")
        if start == -1:
            raise BackendError("No JSON object found in model output.")

        depth = 0
        in_string = False
        escape = False
        end = -1

        for i in range(start, len(stripped)):
            ch = stripped[i]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end == -1:
            raise BackendError("No complete JSON object found in model output.")

        candidate = stripped[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise BackendError(f"Failed to parse extracted JSON: {exc}") from exc
