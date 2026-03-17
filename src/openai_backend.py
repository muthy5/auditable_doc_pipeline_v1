from __future__ import annotations

import json
import logging
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict

from .exceptions import BackendError
from .llm_interface import LocalLLMBackend
from .retry import RetryConfig, retry_with_backoff

LOGGER = logging.getLogger(__name__)


class OpenAIResponseError(BackendError):
    """Transport/API error with optional HTTP status metadata."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class OpenAIBackendConfig:
    """Configuration for OpenAI-compatible API backends.

    Works with OpenAI, Azure OpenAI, OpenRouter, vLLM, llama.cpp server,
    LM Studio, and any provider exposing an OpenAI-compatible chat completions
    endpoint.
    """

    api_key: str = ""
    model: str = "gpt-4o"
    base_url: str = "https://api.openai.com/v1"
    max_tokens: int = 4096
    temperature: float = 0.0
    max_retries: int = 2

    @classmethod
    def from_env(cls, **overrides: Any) -> "OpenAIBackendConfig":
        """Build config from environment variables with optional overrides."""
        return cls(
            api_key=overrides.get("api_key") or os.environ.get("OPENAI_API_KEY", ""),
            model=overrides.get("model") or os.environ.get("OPENAI_MODEL", "gpt-4o"),
            base_url=overrides.get("base_url") or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            max_tokens=overrides.get("max_tokens", 4096),
            temperature=overrides.get("temperature", 0.0),
            max_retries=overrides.get("max_retries", 2),
        )


class OpenAICompatibleBackend(LocalLLMBackend):
    """Backend that calls any OpenAI-compatible chat completions API."""

    def __init__(self, config: OpenAIBackendConfig) -> None:
        if not config.api_key:
            raise ValueError(
                "OpenAI-compatible API key must be provided via --openai-api-key "
                "or OPENAI_API_KEY environment variable."
            )
        self.config = config

    def generate_json(
        self,
        pass_name: str,
        prompt_text: str,
        payload: Dict[str, Any],
        schema: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Generate a JSON response from an OpenAI-compatible API with retry logic."""
        composed_prompt = self._compose_prompt(pass_name, prompt_text, payload, schema)

        def _call() -> Dict[str, Any]:
            raw_text = self._chat_completion(composed_prompt)
            parsed = self._extract_json_object(raw_text)
            if not isinstance(parsed, dict):
                raise BackendError("Parsed JSON is not an object.")
            return parsed

        def _should_retry(exc: Exception) -> bool:
            if isinstance(exc, OpenAIResponseError):
                if exc.status_code is None:
                    return True
                return exc.status_code in {408, 409, 425, 429, 500, 502, 503, 504}
            return isinstance(exc, (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError, BackendError))

        try:
            return retry_with_backoff(
                fn=_call,
                should_retry=_should_retry,
                config=RetryConfig(max_retries=self.config.max_retries, base_delay_s=0.5, max_delay_s=5.0),
                on_retry=lambda attempt, exc: LOGGER.warning(
                    "OpenAI-compatible retry attempt %s for pass '%s': %s", attempt, pass_name, exc
                ),
            )
        except Exception as exc:
            raise BackendError(
                f"Failed to produce valid JSON after {self.config.max_retries + 1} attempts "
                f"for pass '{pass_name}': {exc}"
            ) from exc

    def _chat_completion(self, prompt: str) -> str:
        """Send a chat completion request and return the assistant message text."""
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        request_body = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a JSON-only backend. Return a single valid JSON object. "
                    "No markdown, no code fences, no explanation.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        data = json.dumps(request_body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        req = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:  # noqa: BLE001
                pass
            raise OpenAIResponseError(f"OpenAI API HTTP {exc.code}: {exc.reason}. {body}", status_code=exc.code) from exc
        except urllib.error.URLError as exc:
            raise OpenAIResponseError(f"Failed to connect to OpenAI-compatible API at {url}: {exc}") from exc

        parsed = json.loads(raw)
        choices = parsed.get("choices", [])
        if not choices:
            raise BackendError("OpenAI API returned no choices.")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise BackendError("OpenAI API returned empty content.")
        return content

    def _compose_prompt(
        self,
        pass_name: str,
        prompt_text: str,
        payload: Dict[str, Any],
        schema: Dict[str, Any] | None = None,
    ) -> str:
        """Build the user prompt for the chat completion."""
        parts = [
            f"Pass: {pass_name}",
            "Prompt instructions:",
            prompt_text,
        ]
        if schema is not None:
            parts.extend(
                [
                    "Your output MUST conform to this JSON schema:",
                    json.dumps(schema, ensure_ascii=False, indent=2),
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
        # Strip markdown code fences if present
        if stripped.startswith("```"):
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1 :]
            if stripped.endswith("```"):
                stripped = stripped[:-3].strip()

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
