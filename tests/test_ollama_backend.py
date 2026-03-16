import json

import pytest
from src.exceptions import BackendError
from src.ollama_backend import requests, OllamaBackendConfig, OllamaLocalBackend, OllamaResponseError


def _backend(max_retries: int = 2) -> OllamaLocalBackend:
    return OllamaLocalBackend(OllamaBackendConfig(model="llama3", max_retries=max_retries))


def test_generate_json_retries_on_transient_failure_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(max_retries=2)
    calls = {"count": 0}

    def fake_call(prompt: str) -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise ConnectionError("temporary")
        return '{"ok": true}'

    monkeypatch.setattr(backend, "_call_ollama", fake_call)

    result = backend.generate_json(pass_name="p", prompt_text="t", payload={})

    assert result == {"ok": True}
    assert calls["count"] == 3


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("conn"),
        TimeoutError("timeout"),
        json.JSONDecodeError("bad", "x", 0),
        requests.exceptions.RequestException("request"),
        OllamaResponseError("retryable"),
    ],
)
def test_generate_json_raises_backend_error_after_retry_exhaustion(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    backend = _backend(max_retries=1)

    def always_fail(prompt: str) -> str:
        raise error

    monkeypatch.setattr(backend, "_call_ollama", always_fail)

    with pytest.raises(BackendError):
        backend.generate_json(pass_name="p", prompt_text="t", payload={})


def test_generate_json_does_not_retry_on_permanent_4xx(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(max_retries=3)
    calls = {"count": 0}

    def permanent(prompt: str) -> str:
        calls["count"] += 1
        raise OllamaResponseError("bad request", status_code=400)

    monkeypatch.setattr(backend, "_call_ollama", permanent)

    with pytest.raises(BackendError):
        backend.generate_json(pass_name="p", prompt_text="t", payload={})

    assert calls["count"] == 1
