from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict

from .exceptions import BackendError
from .llm_interface import LocalLLMBackend

LOGGER = logging.getLogger(__name__)


class ClaudeBackendConfig:
    """Configuration for the Claude API backend."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries


class ClaudeAPIBackend(LocalLLMBackend):
    """Backend that calls the Anthropic Claude API."""

    def __init__(self, config: ClaudeBackendConfig) -> None:
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

    def generate_json(
        self,
        pass_name: str,
        prompt_text: str,
        payload: Dict[str, Any],
        schema: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Generate a JSON response from the Claude API with retry logic."""
        composed_prompt = self._compose_prompt(
            pass_name=pass_name,
            prompt_text=prompt_text,
            payload=payload,
            schema=schema,
        )
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                call_start = time.perf_counter()
                response = self._client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": composed_prompt}],
                )
                call_elapsed = time.perf_counter() - call_start
                LOGGER.info(
                    "Claude API call timing: pass=%s attempt=%d duration_seconds=%.2f",
                    pass_name,
                    attempt + 1,
                    call_elapsed,
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
            except (json.JSONDecodeError, BackendError) as exc:
                LOGGER.warning(
                    "Retryable error on attempt %d/%d for pass '%s': %s: %s",
                    attempt + 1,
                    self.config.max_retries + 1,
                    pass_name,
                    type(exc).__name__,
                    exc,
                )
                last_error = exc
            except Exception as exc:
                LOGGER.warning(
                    "API error on attempt %d/%d for pass '%s': %s: %s",
                    attempt + 1,
                    self.config.max_retries + 1,
                    pass_name,
                    type(exc).__name__,
                    exc,
                )
                last_error = exc

        raise BackendError(
            f"Failed to produce valid JSON after {self.config.max_retries + 1} attempts "
            f"for pass '{pass_name}': {last_error}"
        ) from last_error

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
                json.dumps(payload, ensure_ascii=False, indent=2),
            ]
        )
        return "\n\n".join(parts)

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
