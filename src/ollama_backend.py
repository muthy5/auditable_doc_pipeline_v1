from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .exceptions import BackendError
from .llm_interface import LocalLLMBackend
from .retry import RetryConfig, retry_with_backoff

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OllamaBackendConfig:
    """Configuration for Ollama local backend."""

    base_url: str = "http://127.0.0.1:11434"
    model: str = ""
    timeout_s: float = 120.0
    temperature: float = 0.0
    num_predict: int = 2048
    max_retries: int = 2
    max_context_tokens: int = 4096


class OllamaResponseError(RuntimeError):
    """Error wrapper for Ollama response failures."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        """Initialize response error details."""
        super().__init__(message)
        self.status_code = status_code


class OllamaLocalBackend(LocalLLMBackend):
    """JSON-producing backend powered by a local Ollama server."""

    def __init__(self, config: OllamaBackendConfig) -> None:
        """Initialize backend configuration."""
        if not config.model.strip():
            raise ValueError("Ollama model name must be provided.")
        self.config = config

    @staticmethod
    def _estimate_token_count(text: str) -> int:
        """Estimate tokens from word count using a conservative multiplier."""
        return int(len(text.split()) * 1.3)

    def _is_permanent_error(self, error: Exception) -> bool:
        """Return True when error indicates a non-retryable client failure."""
        if not isinstance(error, OllamaResponseError):
            return False
        code = error.status_code
        return code is not None and 400 <= code < 500 and code not in {408, 429}

    def _retry_config(self) -> RetryConfig:
        """Build retry configuration for request attempts."""
        return RetryConfig(max_retries=self.config.max_retries, base_delay_s=0.25, max_delay_s=2.0)

    def health_check(self) -> None:
        """Verify server connectivity and that the selected model is available."""
        url = self.config.base_url.rstrip("/") + "/api/tags"
        req = urllib.request.Request(url=url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise BackendError(f"Unable to reach Ollama server at {url}: {exc}") from exc
        models = [item.get("name", "") for item in payload.get("models", []) if isinstance(item, dict)]
        model = self.config.model
        accepted_model_names = {model, f"{model}:latest"} if ":" not in model else {model, model.split(":", 1)[0]}
        if not any(candidate in models for candidate in accepted_model_names):
            raise BackendError(f"Ollama model '{self.config.model}' not found. Available: {models}")

    def generate_json(
        self,
        pass_name: str,
        prompt_text: str,
        payload: dict[str, Any],
        schema: dict[str, Any] | None = None,
        model_override: str | None = None,
    ) -> dict[str, Any]:
        """Generate one JSON object for a pipeline pass."""
        composed_prompt = self._compose_prompt(pass_name, prompt_text, payload, schema)

        def _call() -> dict[str, Any]:
            raw_text = self._call_ollama(composed_prompt)
            parsed = self._extract_json_object(raw_text)
            if not isinstance(parsed, dict):
                raise OllamaResponseError("Parsed JSON is not an object.")
            return parsed

        def _should_retry(exc: Exception) -> bool:
            return not self._is_permanent_error(exc)

        try:
            return retry_with_backoff(
                fn=_call,
                should_retry=_should_retry,
                config=self._retry_config(),
                on_retry=lambda attempt, exc: LOGGER.warning("Ollama retry attempt %s due to: %s", attempt, exc),
            )
        except Exception as exc:  # noqa: BLE001
            if self._is_permanent_error(exc):
                raise BackendError(f"Permanent Ollama error: {exc}") from exc
            raise BackendError(f"Failed to produce valid JSON after retries: {exc}") from exc

    def _compose_prompt(
        self,
        pass_name: str,
        prompt_text: str,
        payload: dict[str, Any],
        schema: dict[str, Any] | None = None,
    ) -> str:
        """Compose backend prompt, truncating payload when context is too large."""
        payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
        parts = [
            "You are a JSON-only backend for an auditable pipeline.",
            "Return a single valid JSON object only.",
            "Do not wrap output in markdown or code fences.",
            f"Pass: {pass_name}",
            "Prompt instructions:",
            prompt_text,
        ]
        if schema is not None:
            parts.extend(["Your output MUST conform to this JSON schema:", json.dumps(schema, ensure_ascii=False, indent=2)])

        estimated = self._estimate_token_count(payload_json)
        if estimated > self.config.max_context_tokens:
            LOGGER.warning("Payload exceeds max context tokens (%s > %s); truncating.", estimated, self.config.max_context_tokens)
            payload_json = payload_json[: max(1000, self.config.max_context_tokens * 3)] + "..."

        parts.extend(["Input payload JSON:", payload_json])
        return "\n\n".join(parts)

    def _call_ollama(self, prompt: str) -> str:
        """Send a generation request to Ollama."""
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.config.temperature, "num_predict": self.config.num_predict},
        }
        data = json.dumps(payload).encode("utf-8")
        url = self.config.base_url.rstrip("/") + "/api/generate"
        req = urllib.request.Request(url=url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise OllamaResponseError(f"Ollama HTTP error {exc.code}: {exc.reason}", status_code=exc.code) from exc
        except urllib.error.URLError as exc:
            raise OllamaResponseError(f"Failed to connect to Ollama at {url}: {exc}") from exc
        parsed = json.loads(raw)
        text = parsed.get("response")
        if not isinstance(text, str) or not text.strip():
            raise OllamaResponseError("Ollama response did not contain text in 'response'.")
        return text

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        """Extract the first parseable JSON object from model text."""
        stripped = text.strip()
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        start = stripped.find("{")
        if start == -1:
            raise OllamaResponseError("No JSON object found in model output.")
        depth = 0
        in_string = False
        escape = False
        end = -1
        for i, ch in enumerate(stripped[start:], start=start):
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
            raise OllamaResponseError("No complete JSON object found in model output.")
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise OllamaResponseError(f"Failed to parse extracted JSON object: {exc}") from exc
