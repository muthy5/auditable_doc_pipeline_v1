from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict

try:
    import requests
except ImportError:  # pragma: no cover
    class _RequestsExceptions:
        RequestException = Exception

    class _RequestsModule:
        exceptions = _RequestsExceptions()

    requests = _RequestsModule()

from .exceptions import BackendError
from .llm_interface import LocalLLMBackend


@dataclass(frozen=True)
class OllamaBackendConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = ""
    timeout_s: float = 120.0
    temperature: float = 0.0
    num_predict: int = 2048
    max_retries: int = 2


class OllamaResponseError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OllamaLocalBackend(LocalLLMBackend):
    def __init__(self, config: OllamaBackendConfig) -> None:
        if not config.model.strip():
            raise ValueError("Ollama model name must be provided.")
        self.config = config

    def _is_permanent_error(self, error: OllamaResponseError) -> bool:
        status_code = error.status_code
        return status_code is not None and 400 <= status_code < 500 and status_code not in {408, 429}

    def generate_json(
        self,
        pass_name: str,
        prompt_text: str,
        payload: Dict[str, Any],
        schema: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        composed_prompt = self._compose_prompt(
            pass_name=pass_name, prompt_text=prompt_text, payload=payload, schema=schema
        )
        last_error: Exception | None = None
        retryable_errors = (
            OllamaResponseError,
            json.JSONDecodeError,
            ConnectionError,
            TimeoutError,
            requests.exceptions.RequestException,
        )

        for _attempt in range(self.config.max_retries + 1):
            try:
                raw_text = self._call_ollama(prompt=composed_prompt)
                parsed = self._extract_json_object(raw_text)
                if not isinstance(parsed, dict):
                    raise OllamaResponseError("Parsed JSON is not an object.")
                return parsed
            except OllamaResponseError as exc:
                if self._is_permanent_error(exc):
                    raise BackendError(f"Permanent Ollama error: {exc}") from exc
                last_error = exc
            except retryable_errors as exc:
                last_error = exc

        raise BackendError(f"Failed to produce valid JSON after retries: {last_error}") from last_error

    def _compose_prompt(
        self,
        pass_name: str,
        prompt_text: str,
        payload: Dict[str, Any],
        schema: Dict[str, Any] | None = None,
    ) -> str:
        parts = [
            "You are a JSON-only backend for an auditable pipeline.",
            "Return a single valid JSON object only.",
            "Do not wrap output in markdown or code fences.",
            f"Pass: {pass_name}",
            "Prompt instructions:",
            prompt_text,
        ]
        if schema is not None:
            parts.extend([
                "Your output MUST conform to this JSON schema:",
                json.dumps(schema, ensure_ascii=False, indent=2),
            ])
        parts.extend([
            "Input payload JSON:",
            json.dumps(payload, ensure_ascii=False, indent=2),
        ])
        return "\n\n".join(parts)

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.num_predict,
            },
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

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        stripped = text.strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        start = stripped.find("{")
        if start == -1:
            raise OllamaResponseError("No JSON object found in model output.")

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
            raise OllamaResponseError("No complete JSON object found in model output.")

        candidate = stripped[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise OllamaResponseError(f"Failed to parse extracted JSON object: {exc}") from exc
