from __future__ import annotations

import sys

import pytest

from src.claude_backend import ClaudeAPIBackend, ClaudeBackendConfig
from src.exceptions import BackendError


class _DummyClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key


class _DummyAnthropicModule:
    class Anthropic:
        def __init__(self, api_key: str) -> None:
            self._client = _DummyClient(api_key=api_key)


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
