from __future__ import annotations

import copy
import json
import time
from typing import Any


def estimate_tokens(text: str) -> int:
    """Estimate model tokens conservatively using 4 chars/token."""
    return len(text) // 4


def estimate_payload_tokens(payload: dict[str, Any]) -> int:
    """Estimate token count for a JSON-serializable payload."""
    serialized = json.dumps(payload, ensure_ascii=False)
    return estimate_tokens(serialized)


def strip_debug_keys(obj: Any) -> Any:
    """Return a cleaned copy with keys starting with '_' removed recursively."""
    if isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for key, value in obj.items():
            if isinstance(key, str) and key.startswith("_"):
                continue
            cleaned[key] = strip_debug_keys(value)
        return cleaned
    if isinstance(obj, list):
        return [strip_debug_keys(item) for item in obj]
    return copy.deepcopy(obj)


class TokenWindowTracker:
    """Track approximate token usage over a 60-second rolling window."""

    def __init__(self, tokens_per_minute: int = 25000) -> None:
        self.tokens_per_minute = tokens_per_minute
        self._entries: list[tuple[float, int]] = []

    def _prune(self) -> None:
        cutoff = time.time() - 60.0
        self._entries = [(ts, count) for ts, count in self._entries if ts >= cutoff]

    def record_usage(self, token_count: int) -> None:
        self._prune()
        self._entries.append((time.time(), token_count))

    def tokens_used_in_window(self) -> int:
        self._prune()
        return sum(count for _, count in self._entries)

    def sleep_if_needed(self, estimated_tokens: int) -> float:
        self._prune()
        if self.tokens_used_in_window() + estimated_tokens <= self.tokens_per_minute:
            return 0.0

        required_tokens = (self.tokens_used_in_window() + estimated_tokens) - self.tokens_per_minute
        released = 0
        now = time.time()
        sleep_seconds = 0.0
        for ts, count in sorted(self._entries, key=lambda item: item[0]):
            released += count
            sleep_seconds = max(0.0, (ts + 60.0) - now)
            if released >= required_tokens:
                break
        if sleep_seconds > 0.0:
            time.sleep(sleep_seconds)
        return sleep_seconds


def _sub(payload: dict[str, Any], key: str, field: str) -> list:
    """Extract a nested list field from a sub-dict of payload."""
    return payload.get(key, {}).get(field, [])


def _trim_merge_for_audit(merge: dict[str, Any]) -> dict[str, Any]:
    """Lighter merge: drop verbose source_refs from facts."""
    trimmed: dict[str, Any] = {}
    for key, value in merge.items():
        if key == "all_explicit_facts" and isinstance(value, list):
            trimmed[key] = [
                {
                    "fact_id": f.get("fact_id", ""),
                    "text": f.get("text", ""),
                    "source_chunk_ids": f.get("source_chunk_ids", []),
                }
                for f in value
            ]
        elif key == "cross_reference_graph" and isinstance(value, list):
            trimmed[key] = [
                {
                    "source_chunk_id": r.get("source_chunk_id", ""),
                    "ref_text": r.get("ref_text", ""),
                }
                for r in value
            ]
        else:
            trimmed[key] = value
    return trimmed


def trim_for_schema_audit(p: dict[str, Any]) -> dict[str, Any]:
    """Trim payload for schema audit pass."""
    return {
        "task": p.get("task"),
        "merge": _trim_merge_for_audit(p.get("merge", {})),
        "chunk_summaries": p.get("chunk_summaries"),
        "document_type": p.get("document_type"),
        "document_type_schema": p.get("document_type_schema"),
        "web_context": p.get("web_context"),
        "reference_context": p.get("reference_context"),
    }


def trim_for_dependency_audit(p: dict[str, Any]) -> dict[str, Any]:
    """Trim payload for dependency audit."""
    return {
        "task": p.get("task"),
        "merge": _trim_merge_for_audit(p.get("merge", {})),
        "web_context": p.get("web_context"),
        "reference_context": p.get("reference_context"),
    }


def trim_for_assumption_audit(p: dict[str, Any]) -> dict[str, Any]:
    """Trim payload for assumption audit."""
    return {
        "task": p.get("task"),
        "merge": _trim_merge_for_audit(p.get("merge", {})),
        "schema_audit": {
            "blocking_gaps": _sub(p, "schema_audit", "blocking_gaps"),
            "nonblocking_gaps": _sub(p, "schema_audit", "nonblocking_gaps"),
        },
        "dependency_audit": {
            "blocking_dependencies": _sub(p, "dependency_audit", "blocking_dependencies"),
            "ordering_constraints": _sub(p, "dependency_audit", "ordering_constraints"),
        },
        "web_context": p.get("web_context"),
        "reference_context": p.get("reference_context"),
    }


def trim_for_evidence_audit(p: dict[str, Any]) -> dict[str, Any]:
    """Trim payload for evidence audit."""
    return {
        "merge": _trim_merge_for_audit(p.get("merge", {})),
        "schema_audit": {
            "blocking_gaps": _sub(p, "schema_audit", "blocking_gaps"),
        },
        "dependency_audit": {
            "blocking_dependencies": _sub(p, "dependency_audit", "blocking_dependencies"),
        },
        "web_context": p.get("web_context"),
        "reference_context": p.get("reference_context"),
    }


def trim_for_synthesis(p: dict[str, Any]) -> dict[str, Any]:
    """Trim pass input for synthesis while preserving key audit outputs."""
    payload = {
        "task": p.get("task"),
        "merge": p.get("merge"),
        "schema_audit": {
            "blocking_gaps": _sub(p, "schema_audit", "blocking_gaps"),
            "nonblocking_gaps": _sub(p, "schema_audit", "nonblocking_gaps"),
        },
        "dependency_audit": {
            "blocking_dependencies": _sub(p, "dependency_audit", "blocking_dependencies"),
            "ordering_constraints": _sub(p, "dependency_audit", "ordering_constraints"),
        },
        "assumption_audit": {
            "blocking_assumptions": _sub(p, "assumption_audit", "blocking_assumptions"),
            "uncertainty_points": _sub(p, "assumption_audit", "uncertainty_points"),
        },
        "evidence_audit": {
            "claim_registry": _sub(p, "evidence_audit", "claim_registry"),
        },
        "web_context": p.get("web_context"),
        "reference_context": p.get("reference_context"),
    }
    return strip_debug_keys(payload)


def trim_for_plan(p: dict[str, Any]) -> dict[str, Any]:
    """Trim pass input for plan generation."""
    payload = {
        "task": p.get("task"),
        "synthesis": p.get("synthesis"),
        "schema_audit": {
            "blocking_gaps": _sub(p, "schema_audit", "blocking_gaps"),
        },
        "dependency_audit": {
            "blocking_dependencies": _sub(p, "dependency_audit", "blocking_dependencies"),
        },
    }
    return strip_debug_keys(payload)
