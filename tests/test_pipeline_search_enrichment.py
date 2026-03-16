from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.config import PipelineConfig
from src.exceptions import PipelineError
from src.pipeline import AuditablePipeline


def test_build_web_context_raises_without_brave_api_key() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pipeline = AuditablePipeline(
        repo_root=repo_root,
        backend_name="demo",
        config=PipelineConfig(enable_search=True, brave_api_key=""),
    )

    with pytest.raises(
        PipelineError,
        match="Web search is enabled but no Brave API key was provided. Set --brave-api-key or BRAVE_API_KEY environment variable.",
    ):
        pipeline._build_web_context(normalize={}, document_text="text")


def test_build_web_context_reraises_when_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pipeline = AuditablePipeline(
        repo_root=repo_root,
        backend_name="demo",
        config=PipelineConfig(enable_search=True, brave_api_key="test-key"),
    )

    monkeypatch.setattr(pipeline, "_generate_search_queries", lambda normalize, document_text: ["query"])

    class _BrokenClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def search(self, query: str) -> list[dict[str, str]]:
            raise RuntimeError("boom")

    monkeypatch.setattr("src.pipeline.BraveSearchClient", _BrokenClient)

    with pytest.raises(RuntimeError, match="boom"):
        pipeline._build_web_context(normalize={}, document_text="text", strict=True)


def test_build_web_context_warns_and_returns_empty_in_non_strict(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pipeline = AuditablePipeline(
        repo_root=repo_root,
        backend_name="demo",
        config=PipelineConfig(enable_search=True, brave_api_key="test-key"),
    )

    monkeypatch.setattr(pipeline, "_generate_search_queries", lambda normalize, document_text: ["query"])

    class _BrokenClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def search(self, query: str) -> list[dict[str, str]]:
            raise RuntimeError("boom")

    monkeypatch.setattr("src.pipeline.BraveSearchClient", _BrokenClient)

    with caplog.at_level(logging.WARNING):
        output = pipeline._build_web_context(normalize={}, document_text="text", strict=False)

    assert output == []
    assert "Web search enrichment skipped due to error: boom" in caplog.text

