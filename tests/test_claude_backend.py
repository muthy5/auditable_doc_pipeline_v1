from __future__ import annotations

import sys

import pytest

from src.claude_backend import ClaudeAPIBackend, ClaudeBackendConfig, _compact_json
from src.exceptions import BackendError


class _DummyClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key


class _DummyAnthropicModule:
    class Anthropic:
        def __init__(self, api_key: str) -> None:
            self._client = _DummyClient(api_key=api_key)


class _FakeMessagesClient:
    def __init__(self, response: object) -> None:
        self._response = response

    def create(self, **kwargs: object) -> object:
        return self._response


def test_claude_backend_raises_value_error_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError):
        ClaudeAPIBackend(ClaudeBackendConfig(api_key=""))


def test_extract_json_object_parses_json_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "anthropic", _DummyAnthropicModule())
    backend = ClaudeAPIBackend(ClaudeBackendConfig(api_key="test-key"))

    parsed = backend._extract_json_object('prefix {"a": 1, "nested": {"b": true}} suffix')

    assert parsed == {"a": 1, "nested": {"b": True}}


def test_extract_json_object_raises_backend_error_on_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "anthropic", _DummyAnthropicModule())
    backend = ClaudeAPIBackend(ClaudeBackendConfig(api_key="test-key"))

    with pytest.raises(BackendError):
        backend._extract_json_object("this is not json")


def test_generate_json_joins_all_text_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    class _Module:
        class Anthropic:
            def __init__(self, api_key: str) -> None:
                response = type(
                    "Response",
                    (),
                    {"content": [{"type": "text", "text": '{"a":'}, {"type": "text", "text": ' 1}'}, {"type": "tool_use", "name": "x"}]},
                )()
                self.messages = _FakeMessagesClient(response)

    monkeypatch.setitem(sys.modules, "anthropic", _Module())
    backend = ClaudeAPIBackend(ClaudeBackendConfig(api_key="test-key"))

    parsed = backend.generate_json(pass_name="p", prompt_text="x", payload={})

    assert parsed == {"a": 1}


def test_generate_json_raises_when_no_text_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    class _Module:
        class Anthropic:
            def __init__(self, api_key: str) -> None:
                response = type("Response", (), {"content": [{"type": "tool_use", "name": "x"}]})()
                self.messages = _FakeMessagesClient(response)

    monkeypatch.setitem(sys.modules, "anthropic", _Module())
    backend = ClaudeAPIBackend(ClaudeBackendConfig(api_key="test-key", max_retries=0))

    with pytest.raises(BackendError, match="No text blocks found in Claude response content"):
        backend.generate_json(pass_name="p", prompt_text="x", payload={})


def test_compose_prompt_uses_required_field_schema_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "anthropic", _DummyAnthropicModule())
    backend = ClaudeAPIBackend(ClaudeBackendConfig(api_key="test-key"))

    schema = {
        "type": "object",
        "required": ["doc_id", "status"],
        "properties": {
            "doc_id": {"type": "string", "description": "Document identifier"},
            "status": {"type": "string", "description": "Processing status"},
            "opt1": {"type": "string"},
            "opt2": {"type": "string"},
            "opt3": {"type": "string"},
            "opt4": {"type": "string"},
            "opt5": {"type": "string"},
            "opt6": {"type": "string"},
        },
    }

    prompt = backend._compose_prompt("pass", "instructions", {"x": 1}, schema=schema)

    assert "required fields only" in prompt
    assert "- doc_id (string): Document identifier" in prompt
    assert "Optional fields omitted from prompt" in prompt


def test_generate_json_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("time.sleep", lambda _s: None)
    calls = {"n": 0}

    class _Module:
        class Anthropic:
            def __init__(self, api_key: str) -> None:
                self.messages = self

            def create(self, **kwargs: object) -> object:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("rate limit exceeded")
                return type("Response", (), {"content": [{"type": "text", "text": '{"ok": true}'}]})()

    monkeypatch.setitem(sys.modules, "anthropic", _Module())
    backend = ClaudeAPIBackend(ClaudeBackendConfig(api_key="test-key", max_retries=1))

    parsed = backend.generate_json(pass_name="p", prompt_text="x", payload={})

    assert parsed == {"ok": True}
    assert calls["n"] == 2


def test_prompt_caching_uses_system_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    captured: dict[str, object] = {}

    class _Module:
        class Anthropic:
            def __init__(self, api_key: str) -> None:
                self.messages = self

            def create(self, **kwargs: object) -> object:
                captured.update(kwargs)
                usage = type("Usage", (), {"cache_creation_input_tokens": 12, "cache_read_input_tokens": 34})()
                return type("Response", (), {"content": [{"type": "text", "text": '{"ok": true}'}], "usage": usage})()

    monkeypatch.setitem(sys.modules, "anthropic", _Module())
    backend = ClaudeAPIBackend(ClaudeBackendConfig(api_key="test-key", enable_prompt_caching=True))

    parsed = backend.generate_json(pass_name="p", prompt_text="x", payload={"a": 1})

    assert parsed == {"ok": True}
    assert "system" in captured
    assert captured["messages"] == [{"role": "user", "content": "Input payload JSON:\n{\"a\":1}"}]


def test_prompt_caching_disabled_uses_single_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    captured: dict[str, object] = {}

    class _Module:
        class Anthropic:
            def __init__(self, api_key: str) -> None:
                self.messages = self

            def create(self, **kwargs: object) -> object:
                captured.update(kwargs)
                return type("Response", (), {"content": [{"type": "text", "text": '{"ok": true}'}]})()

    monkeypatch.setitem(sys.modules, "anthropic", _Module())
    backend = ClaudeAPIBackend(ClaudeBackendConfig(api_key="test-key", enable_prompt_caching=False))

    parsed = backend.generate_json(pass_name="p", prompt_text="x", payload={"a": 1})

    assert parsed == {"ok": True}
    assert "system" not in captured
    assert isinstance(captured.get("messages"), list)
    assert captured["messages"]


def test_compact_json_no_indent() -> None:
    compact = _compact_json({"a": 1, "b": {"c": 2}})

    assert compact == '{"a":1,"b":{"c":2}}'
    assert "\n" not in compact
