from __future__ import annotations

import json
import urllib.error

import pytest

from src.exceptions import BackendError
from src.openai_backend import OpenAIBackendConfig, OpenAICompatibleBackend


def test_openai_backend_raises_value_error_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key must be provided"):
        OpenAICompatibleBackend(OpenAIBackendConfig(api_key=""))


def test_openai_backend_initializes_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = OpenAICompatibleBackend(OpenAIBackendConfig(api_key="test-key"))
    assert backend.config.api_key == "test-key"


def test_extract_json_object_parses_clean_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = OpenAICompatibleBackend(OpenAIBackendConfig(api_key="test-key"))
    result = backend._extract_json_object('{"a": 1, "b": "hello"}')
    assert result == {"a": 1, "b": "hello"}


def test_extract_json_object_strips_code_fences(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = OpenAICompatibleBackend(OpenAIBackendConfig(api_key="test-key"))
    result = backend._extract_json_object('```json\n{"a": 1}\n```')
    assert result == {"a": 1}


def test_extract_json_object_finds_embedded_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = OpenAICompatibleBackend(OpenAIBackendConfig(api_key="test-key"))
    result = backend._extract_json_object('Here is the result: {"x": true} done.')
    assert result == {"x": True}


def test_extract_json_object_raises_on_no_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = OpenAICompatibleBackend(OpenAIBackendConfig(api_key="test-key"))
    with pytest.raises(BackendError, match="No JSON object found"):
        backend._extract_json_object("this has no json at all")


def test_compose_prompt_includes_pass_and_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = OpenAICompatibleBackend(OpenAIBackendConfig(api_key="test-key"))
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    prompt = backend._compose_prompt("test_pass", "Do the thing", {"input": 1}, schema=schema)
    assert "Pass: test_pass" in prompt
    assert "Do the thing" in prompt
    assert '"x"' in prompt
    assert '"input": 1' in prompt


def test_generate_json_calls_api_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response_body = json.dumps({
        "choices": [{"message": {"content": '{"result": "ok"}'}}]
    }).encode("utf-8")

    class FakeResponse:
        def read(self) -> bytes:
            return response_body

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    def fake_urlopen(req: object, timeout: object = None) -> FakeResponse:
        return FakeResponse()

    monkeypatch.setattr("src.openai_backend.urllib.request.urlopen", fake_urlopen)
    backend = OpenAICompatibleBackend(OpenAIBackendConfig(api_key="test-key", max_retries=0))
    result = backend.generate_json(pass_name="p", prompt_text="x", payload={"a": 1})
    assert result == {"result": "ok"}


def test_generate_json_raises_on_empty_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response_body = json.dumps({"choices": []}).encode("utf-8")

    class FakeResponse:
        def read(self) -> bytes:
            return response_body

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    def fake_urlopen(req: object, timeout: object = None) -> FakeResponse:
        return FakeResponse()

    monkeypatch.setattr("src.openai_backend.urllib.request.urlopen", fake_urlopen)
    backend = OpenAICompatibleBackend(OpenAIBackendConfig(api_key="test-key", max_retries=0))
    with pytest.raises(BackendError, match="no choices"):
        backend.generate_json(pass_name="p", prompt_text="x", payload={})


def test_generate_json_raises_on_invalid_json_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response_body = json.dumps({
        "choices": [{"message": {"content": "not json at all"}}]
    }).encode("utf-8")

    class FakeResponse:
        def read(self) -> bytes:
            return response_body

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    def fake_urlopen(req: object, timeout: object = None) -> FakeResponse:
        return FakeResponse()

    monkeypatch.setattr("src.openai_backend.urllib.request.urlopen", fake_urlopen)
    backend = OpenAICompatibleBackend(OpenAIBackendConfig(api_key="test-key", max_retries=0))
    with pytest.raises(BackendError, match="Failed to produce valid JSON"):
        backend.generate_json(pass_name="p", prompt_text="x", payload={})


def test_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
    config = OpenAIBackendConfig.from_env()
    assert config.api_key == "env-key"
    assert config.model == "gpt-4o-mini"
    assert config.base_url == "http://localhost:8080/v1"


def test_config_from_env_with_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    config = OpenAIBackendConfig.from_env(api_key="override-key", model="custom-model")
    assert config.api_key == "override-key"
    assert config.model == "custom-model"


def test_generate_json_retries_on_rate_limit_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls = {"n": 0}

    class FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def read(self) -> bytes:
            return self._payload

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    def fake_urlopen(req: object, timeout: object = None) -> FakeResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError("http://x", 429, "Too Many Requests", hdrs=None, fp=None)
        body = json.dumps({"choices": [{"message": {"content": '{"ok": true}'}}]}).encode("utf-8")
        return FakeResponse(body)

    monkeypatch.setattr("src.openai_backend.urllib.request.urlopen", fake_urlopen)
    backend = OpenAICompatibleBackend(OpenAIBackendConfig(api_key="test-key", max_retries=1))
    parsed = backend.generate_json(pass_name="p", prompt_text="x", payload={})
    assert parsed == {"ok": True}
    assert calls["n"] == 2


def test_openai_backend_allows_missing_api_key_for_local_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = OpenAICompatibleBackend(OpenAIBackendConfig(api_key="", base_url="http://localhost:8000/v1"))
    assert backend.config.base_url == "http://localhost:8000/v1"
