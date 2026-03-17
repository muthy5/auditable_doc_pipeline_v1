from __future__ import annotations

import importlib.util
import json
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class CapabilityStatus:
    """Status for one runtime capability."""

    available: bool
    message: str


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def check_ollama(base_url: str, model: str, timeout_s: float = 5.0) -> CapabilityStatus:
    """Validate Ollama server reachability and model existence."""
    if not model.strip():
        return CapabilityStatus(False, "Missing Ollama model name.")
    url = base_url.rstrip("/") + "/api/tags"
    req = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return CapabilityStatus(False, f"Ollama server unreachable at {url}: {exc}")
    except Exception as exc:  # noqa: BLE001
        return CapabilityStatus(False, f"Failed to query Ollama server at {url}: {exc}")
    models = [item.get("name", "") for item in payload.get("models", []) if isinstance(item, dict)]
    if model not in models:
        return CapabilityStatus(False, f"Ollama model '{model}' not found. Available: {models}")
    return CapabilityStatus(True, "Ollama server reachable and selected model is installed.")


def run_preflight(
    *,
    backend: str,
    enable_search: bool,
    claude_api_key: str,
    brave_api_key: str,
    ollama_base_url: str,
    ollama_model: str,
    ollama_timeout_s: float = 5.0,
    openai_api_key: str = "",
) -> dict[str, CapabilityStatus]:
    """Collect runtime dependency/backend checks for CLI and Streamlit."""
    statuses: dict[str, CapabilityStatus] = {
        "demo_backend": CapabilityStatus(True, "Demo backend is always available."),
        "claude_backend": CapabilityStatus(_has_module("anthropic"), "'anthropic' package available." if _has_module("anthropic") else "Install 'anthropic' to use Claude backend."),
        "ollama_backend": CapabilityStatus(True, "Ollama not selected."),
        "openai_backend": CapabilityStatus(True, "OpenAI-compatible backend not selected."),
        "pdf_parsing": CapabilityStatus(_has_module("pypdf"), "'pypdf' package available." if _has_module("pypdf") else "Install 'pypdf' to parse PDF files."),
        "docx_parsing": CapabilityStatus(_has_module("docx"), "'python-docx' package available." if _has_module("docx") else "Install 'python-docx' to parse DOCX files."),
        "web_search": CapabilityStatus(True, "Web search disabled."),
    }

    if backend == "claude":
        if not statuses["claude_backend"].available:
            statuses["claude_backend"] = CapabilityStatus(False, "Claude backend unavailable: missing 'anthropic' package.")
        elif not claude_api_key.strip():
            statuses["claude_backend"] = CapabilityStatus(False, "Claude backend unavailable: missing API key.")
        else:
            statuses["claude_backend"] = CapabilityStatus(True, "Claude backend configured.")

    if backend == "ollama":
        statuses["ollama_backend"] = check_ollama(ollama_base_url, ollama_model, timeout_s=ollama_timeout_s)

    if backend == "openai":
        if not openai_api_key.strip():
            statuses["openai_backend"] = CapabilityStatus(False, "OpenAI-compatible backend unavailable: missing API key.")
        else:
            statuses["openai_backend"] = CapabilityStatus(True, "OpenAI-compatible backend configured.")

    if enable_search:
        if brave_api_key.strip():
            statuses["web_search"] = CapabilityStatus(True, "Web search configured.")
        else:
            statuses["web_search"] = CapabilityStatus(False, "Web search enabled but BRAVE API key is missing.")

    return statuses
