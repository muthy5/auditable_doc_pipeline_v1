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


def trim_for_synthesis(full_payload: dict[str, Any]) -> dict[str, Any]:
    """Trim pass input for synthesis while preserving key audit outputs."""
    payload = {
        "task": full_payload.get("task"),
        "merge": full_payload.get("merge"),
        "schema_audit": {
            "blocking_gaps": full_payload.get("schema_audit", {}).get("blocking_gaps", []),
            "nonblocking_gaps": full_payload.get("schema_audit", {}).get("nonblocking_gaps", []),
        },
        "dependency_audit": {
            "blocking_dependencies": full_payload.get("dependency_audit", {}).get("blocking_dependencies", []),
            "ordering_constraints": full_payload.get("dependency_audit", {}).get("ordering_constraints", []),
        },
        "assumption_audit": {
            "blocking_assumptions": full_payload.get("assumption_audit", {}).get("blocking_assumptions", []),
            "uncertainty_points": full_payload.get("assumption_audit", {}).get("uncertainty_points", []),
        },
        "evidence_audit": {
            "claim_registry": full_payload.get("evidence_audit", {}).get("claim_registry", []),
        },
        "web_context": full_payload.get("web_context"),
        "reference_context": full_payload.get("reference_context"),
    }
    return strip_debug_keys(payload)


def trim_for_plan(full_payload: dict[str, Any]) -> dict[str, Any]:
    """Trim pass input for plan generation to avoid duplicate large context."""
    payload = {
        "task": full_payload.get("task"),
        "synthesis": full_payload.get("synthesis"),
        "schema_audit": {
            "blocking_gaps": full_payload.get("schema_audit", {}).get("blocking_gaps", []),
        },
        "dependency_audit": {
            "blocking_dependencies": full_payload.get("dependency_audit", {}).get("blocking_dependencies", []),
        },
    }
    return strip_debug_keys(payload)
