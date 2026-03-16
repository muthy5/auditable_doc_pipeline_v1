from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from src.schemas import load_schema
from src.search import BraveSearchClient


class _FakeHTTPResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None


def test_brave_search_client_normalizes_results(monkeypatch: pytest.MonkeyPatch) -> None:
    client = BraveSearchClient(api_key="test-key")

    payload = {
        "web": {
            "results": [
                {"title": "Result A", "url": "https://example.com/a", "description": "Snippet A"},
                {"title": "Result B", "url": "https://example.com/b", "description": "Snippet B"},
            ]
        }
    }

    def fake_urlopen(request, timeout=0):  # noqa: ANN001
        assert request.headers["X-subscription-token"] == "test-key"
        return _FakeHTTPResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    results = client.search("fresh lemonade process", count=2)

    assert results == [
        {"title": "Result A", "url": "https://example.com/a", "snippet": "Snippet A"},
        {"title": "Result B", "url": "https://example.com/b", "snippet": "Snippet B"},
    ]


def test_brave_search_client_retries_and_returns_empty_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    client = BraveSearchClient(api_key="test-key", max_retries=1)

    def failing_urlopen(request, timeout=0):  # noqa: ANN001
        raise urllib.error.URLError("network down")

    monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)
    assert client.search("query") == []


def test_search_queries_schema_accepts_valid_payload() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    schema = load_schema(repo_root / "schemas", "search_queries.schema.json")
    payload = {"queries": ["query one", "query two", "query three"]}
    Draft202012Validator(schema).validate(payload)
