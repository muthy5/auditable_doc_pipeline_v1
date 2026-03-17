from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from src.config import PipelineConfig
from src.fallback import build_fallback_queries, detect_gaps
from src.pipeline import AuditablePipeline


# ---------------------------------------------------------------------------
# detect_gaps
# ---------------------------------------------------------------------------

def _minimal_merge(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "doc_id": "doc_001",
        "chunks_seen": ["c1"],
        "global_entities": {},
        "global_defined_terms": [],
        "global_undefined_terms": [],
        "all_explicit_facts": [],
        "all_claims": [],
        "all_steps": [],
        "all_dependencies": [],
        "all_inputs_required": [],
        "all_outputs_produced": [],
        "all_missing_information_signals": [],
        "cross_reference_graph": [],
        "term_registry": [],
        "merge_warnings": [],
    }
    base.update(overrides)
    return base


def _minimal_normalize(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "task": {"primary_goal": "build a birdhouse", "domain": "woodworking"},
        "questions_to_answer": [],
    }
    base.update(overrides)
    return base


def test_detect_gaps_no_materials_or_steps() -> None:
    merge = _minimal_merge()
    normalize = _minimal_normalize()
    gaps = detect_gaps(merge, normalize)
    areas = {g["area"] for g in gaps}
    assert "materials" in areas
    assert "steps" in areas


def test_detect_gaps_with_materials_and_steps() -> None:
    merge = _minimal_merge(
        global_entities={"materials": ["wood", "nails"], "equipment": ["hammer"]},
        all_steps=[{"step_id": "s1", "text": "Cut wood"}],
    )
    normalize = _minimal_normalize()
    gaps = detect_gaps(merge, normalize)
    areas = {g["area"] for g in gaps}
    assert "materials" not in areas
    assert "steps" not in areas


def test_detect_gaps_missing_information_signals() -> None:
    merge = _minimal_merge(
        all_missing_information_signals=[{"text": "drying time not specified"}],
        all_steps=[{"step_id": "s1", "text": "paint"}],
        global_entities={"materials": ["paint"]},
    )
    normalize = _minimal_normalize()
    gaps = detect_gaps(merge, normalize)
    assert any(g["area"] == "missing_information" for g in gaps)


def test_detect_gaps_undefined_terms() -> None:
    merge = _minimal_merge(
        global_undefined_terms=["rabbet joint"],
        all_steps=[{"step_id": "s1", "text": "cut"}],
        global_entities={"materials": ["wood"]},
    )
    normalize = _minimal_normalize()
    gaps = detect_gaps(merge, normalize)
    assert any(g["area"] == "undefined_term" and "rabbet joint" in g["description"] for g in gaps)


def test_detect_gaps_unresolved_inputs() -> None:
    merge = _minimal_merge(
        all_inputs_required=["lumber", "screws"],
        all_outputs_produced=["lumber"],
        all_steps=[{"step_id": "s1", "text": "assemble"}],
        global_entities={"materials": ["lumber"]},
    )
    normalize = _minimal_normalize()
    gaps = detect_gaps(merge, normalize)
    assert any(g["area"] == "unresolved_input" and "screws" in g["description"] for g in gaps)


def test_detect_gaps_open_questions() -> None:
    merge = _minimal_merge(
        all_steps=[{"step_id": "s1", "text": "x"}],
        global_entities={"materials": ["x"]},
    )
    normalize = _minimal_normalize(questions_to_answer=["What size screws?"])
    gaps = detect_gaps(merge, normalize)
    assert any(g["area"] == "open_question" for g in gaps)


# ---------------------------------------------------------------------------
# build_fallback_queries
# ---------------------------------------------------------------------------

def test_build_fallback_queries_materials_gap() -> None:
    gaps = [{"area": "materials", "description": "No materials or equipment listed"}]
    normalize = _minimal_normalize()
    queries = build_fallback_queries(gaps, normalize, "build a birdhouse")
    assert len(queries) == 1
    assert "materials" in queries[0].lower()


def test_build_fallback_queries_steps_gap() -> None:
    gaps = [{"area": "steps", "description": "No procedural steps found"}]
    normalize = _minimal_normalize()
    queries = build_fallback_queries(gaps, normalize, "build a birdhouse")
    assert len(queries) == 1
    assert "step" in queries[0].lower()


def test_build_fallback_queries_respects_max() -> None:
    gaps = [{"area": "open_question", "description": f"question {i}"} for i in range(10)]
    normalize = _minimal_normalize()
    queries = build_fallback_queries(gaps, normalize, "goal", max_queries=3)
    assert len(queries) == 3


def test_build_fallback_queries_deduplicates() -> None:
    gaps = [
        {"area": "open_question", "description": "same question"},
        {"area": "open_question", "description": "same question"},
    ]
    normalize = _minimal_normalize()
    queries = build_fallback_queries(gaps, normalize, "goal")
    assert len(queries) == 1


# ---------------------------------------------------------------------------
# Pipeline integration: _build_fallback_context
# ---------------------------------------------------------------------------

def test_build_fallback_context_disabled_by_default() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pipeline = AuditablePipeline(repo_root=repo_root, backend_name="demo")
    result = pipeline._build_fallback_context(
        merge=_minimal_merge(),
        normalize=_minimal_normalize(),
        user_goal="test",
        existing_web_context=[],
    )
    assert result == []


def test_build_fallback_context_warns_without_brave_key(caplog: pytest.LogCaptureFixture) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pipeline = AuditablePipeline(
        repo_root=repo_root,
        backend_name="demo",
        config=PipelineConfig(enable_fallback_search=True, brave_api_key=""),
    )
    with caplog.at_level(logging.WARNING):
        result = pipeline._build_fallback_context(
            merge=_minimal_merge(),
            normalize=_minimal_normalize(),
            user_goal="test",
            existing_web_context=[],
        )
    assert result == []
    assert "no Brave API key" in caplog.text


def test_build_fallback_context_no_gaps_returns_empty(caplog: pytest.LogCaptureFixture) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pipeline = AuditablePipeline(
        repo_root=repo_root,
        backend_name="demo",
        config=PipelineConfig(enable_fallback_search=True, brave_api_key="test-key"),
    )
    merge = _minimal_merge(
        all_steps=[{"step_id": "s1", "text": "do"}],
        global_entities={"materials": ["stuff"]},
    )
    with caplog.at_level(logging.INFO):
        result = pipeline._build_fallback_context(
            merge=merge,
            normalize=_minimal_normalize(),
            user_goal="test",
            existing_web_context=[],
        )
    assert result == []
    assert "No information gaps detected" in caplog.text


def test_build_fallback_context_executes_search(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pipeline = AuditablePipeline(
        repo_root=repo_root,
        backend_name="demo",
        config=PipelineConfig(enable_fallback_search=True, brave_api_key="test-key"),
    )

    search_calls: list[str] = []

    class _FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def search(self, query: str, count: int = 5) -> list[dict[str, str]]:
            search_calls.append(query)
            return [{"title": "Result", "url": "https://example.com", "snippet": "info"}]

    monkeypatch.setattr("src.pipeline.BraveSearchClient", _FakeClient)

    result = pipeline._build_fallback_context(
        merge=_minimal_merge(),  # no materials or steps -> gaps detected
        normalize=_minimal_normalize(),
        user_goal="build a birdhouse",
        existing_web_context=[],
    )
    assert len(result) > 0
    assert all(entry.get("source") == "fallback" for entry in result)
    assert len(search_calls) > 0


def test_build_fallback_context_deduplicates_against_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pipeline = AuditablePipeline(
        repo_root=repo_root,
        backend_name="demo",
        config=PipelineConfig(enable_fallback_search=True, brave_api_key="test-key"),
    )

    # Produce a single gap with one known query
    monkeypatch.setattr("src.pipeline.detect_gaps", lambda merge, normalize: [{"area": "materials", "description": "no materials"}])
    monkeypatch.setattr(
        "src.pipeline.build_fallback_queries",
        lambda gaps, normalize, user_goal: ["woodworking build a birdhouse required materials and equipment list"],
    )

    # Existing context already has the same query
    existing = [{"query": "woodworking build a birdhouse required materials and equipment list", "results": []}]
    result = pipeline._build_fallback_context(
        merge=_minimal_merge(),
        normalize=_minimal_normalize(),
        user_goal="build a birdhouse",
        existing_web_context=existing,
    )
    assert result == []


def test_build_fallback_context_graceful_on_search_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pipeline = AuditablePipeline(
        repo_root=repo_root,
        backend_name="demo",
        config=PipelineConfig(enable_fallback_search=True, brave_api_key="test-key"),
    )

    class _BrokenClient:
        def __init__(self, api_key: str) -> None:
            pass

        def search(self, query: str, count: int = 5) -> list[dict[str, str]]:
            raise RuntimeError("network down")

    monkeypatch.setattr("src.pipeline.BraveSearchClient", _BrokenClient)

    with caplog.at_level(logging.WARNING):
        result = pipeline._build_fallback_context(
            merge=_minimal_merge(),
            normalize=_minimal_normalize(),
            user_goal="test",
            existing_web_context=[],
        )
    assert result == []
    assert "Fallback web search failed" in caplog.text


def test_build_fallback_context_strict_reraises(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pipeline = AuditablePipeline(
        repo_root=repo_root,
        backend_name="demo",
        config=PipelineConfig(enable_fallback_search=True, brave_api_key="test-key"),
    )

    class _BrokenClient:
        def __init__(self, api_key: str) -> None:
            pass

        def search(self, query: str, count: int = 5) -> list[dict[str, str]]:
            raise RuntimeError("boom")

    monkeypatch.setattr("src.pipeline.BraveSearchClient", _BrokenClient)

    with pytest.raises(RuntimeError, match="boom"):
        pipeline._build_fallback_context(
            merge=_minimal_merge(),
            normalize=_minimal_normalize(),
            user_goal="test",
            existing_web_context=[],
            strict=True,
        )
