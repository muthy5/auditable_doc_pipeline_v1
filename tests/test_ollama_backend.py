from __future__ import annotations

import json

import pytest

from src.exceptions import BackendError
from src.ollama_backend import OllamaBackendConfig, OllamaLocalBackend, OllamaResponseError


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


@pytest.mark.parametrize("error", [ConnectionError("conn"), TimeoutError("timeout"), json.JSONDecodeError("bad", "x", 0), OllamaResponseError("retryable")])
def test_generate_json_raises_backend_error_after_retry_exhaustion(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    backend = _backend(max_retries=1)
    monkeypatch.setattr(backend, "_call_ollama", lambda prompt: (_ for _ in ()).throw(error))
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


def test_health_check_accepts_latest_tag_for_unqualified_model(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = OllamaLocalBackend(OllamaBackendConfig(model="llama3"))

    class FakeResponse:
        def read(self) -> bytes:
            return json.dumps({"models": [{"name": "llama3:latest"}]}).encode("utf-8")

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr("src.ollama_backend.urllib.request.urlopen", lambda *args, **kwargs: FakeResponse())

    backend.health_check()
