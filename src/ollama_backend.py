from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict

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
    pass


class OllamaLocalBackend(LocalLLMBackend):
    def __init__(self, config: OllamaBackendConfig) -> None:
        if not config.model.strip():
            raise ValueError("Ollama model name must be provided.")
        self.config = config

    def generate_json(self, pass_name: str, prompt_text: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        composed_prompt = self._compose_prompt(pass_name=pass_name, prompt_text=prompt_text, payload=payload)
        last_error: Exception | None = None
        for _attempt in range(self.config.max_retries + 1):
            try:
                raw_text = self._call_ollama(prompt=composed_prompt)
                parsed = self._extract_json_object(raw_text)
                if not isinstance(parsed, dict):
                    raise OllamaResponseError("Parsed JSON is not an object.")
                return parsed
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise OllamaResponseError(f"Failed to produce valid JSON after retries: {last_error}")

    def _compose_prompt(self, pass_name: str, prompt_text: str, payload: Dict[str, Any]) -> str:
        return "\n\n".join(
            [
                "You are a JSON-only backend for an auditable pipeline.",
                "Return a single valid JSON object only.",
                "Do not wrap output in markdown or code fences.",
                f"Pass: {pass_name}",
                "Prompt instructions:",
                prompt_text,
                "Input payload JSON:",
                json.dumps(payload, ensure_ascii=False, indent=2),
            ]
        )

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
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise OllamaResponseError("No JSON object found in model output.")

        candidate = stripped[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise OllamaResponseError(f"Failed to parse extracted JSON object: {exc}") from exc
