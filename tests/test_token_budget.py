from __future__ import annotations

from src.token_budget import (
    TokenWindowTracker,
    estimate_tokens,
    strip_debug_keys,
    trim_for_plan,
    trim_for_synthesis,
)


def test_estimate_tokens_basic() -> None:
    assert estimate_tokens("abcd" * 10) == 10


def test_window_tracker_prunes_old_entries(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    now = [1000.0]
    monkeypatch.setattr("src.token_budget.time.time", lambda: now[0])
    tracker = TokenWindowTracker(tokens_per_minute=100)

    tracker.record_usage(40)
    now[0] = 1065.0
    tracker.record_usage(30)

    assert tracker.tokens_used_in_window() == 30


def test_sleep_if_needed_triggers(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    now = [1000.0]
    slept: list[float] = []

    monkeypatch.setattr("src.token_budget.time.time", lambda: now[0])
    monkeypatch.setattr("src.token_budget.time.sleep", lambda s: slept.append(s))

    tracker = TokenWindowTracker(tokens_per_minute=100)
    tracker.record_usage(90)

    sleep_s = tracker.sleep_if_needed(20)

    assert sleep_s == 60.0
    assert slept == [60.0]


def test_sleep_if_needed_skips(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    now = [1000.0]
    slept: list[float] = []

    monkeypatch.setattr("src.token_budget.time.time", lambda: now[0])
    monkeypatch.setattr("src.token_budget.time.sleep", lambda s: slept.append(s))

    tracker = TokenWindowTracker(tokens_per_minute=100)
    tracker.record_usage(20)

    sleep_s = tracker.sleep_if_needed(30)

    assert sleep_s == 0.0
    assert slept == []


def test_trim_for_plan_removes_merge() -> None:
    payload = {
        "task": {"x": 1},
        "merge": {"huge": True},
        "schema_audit": {"blocking_gaps": ["a"]},
        "dependency_audit": {"blocking_dependencies": ["d"]},
        "synthesis": {"final_answer": "ok"},
    }

    trimmed = trim_for_plan(payload)

    assert "merge" not in trimmed


def test_trim_for_plan_removes_web_context() -> None:
    payload = {
        "task": {"x": 1},
        "schema_audit": {"blocking_gaps": []},
        "dependency_audit": {"blocking_dependencies": []},
        "synthesis": {"final_answer": "ok"},
        "web_context": [{"url": "https://example.com"}],
    }

    trimmed = trim_for_plan(payload)

    assert "web_context" not in trimmed


def test_trim_for_synthesis_keeps_blocking_gaps() -> None:
    payload = {
        "task": {"x": 1},
        "merge": {"a": 1},
        "schema_audit": {"blocking_gaps": ["missing"], "other": ["drop"]},
        "dependency_audit": {"blocking_dependencies": [], "ordering_constraints": []},
        "assumption_audit": {"blocking_assumptions": [], "uncertainty_points": []},
        "evidence_audit": {"claim_registry": []},
        "web_context": [],
        "reference_context": [],
    }

    trimmed = trim_for_synthesis(payload)

    assert trimmed["schema_audit"]["blocking_gaps"] == ["missing"]


def test_strip_debug_keys() -> None:
    payload = {"a": 1, "_debug": 2, "nested": {"_meta": 3, "k": [{"_raw": "x", "v": 1}]}}

    cleaned = strip_debug_keys(payload)

    assert cleaned == {"a": 1, "nested": {"k": [{"v": 1}]}}
