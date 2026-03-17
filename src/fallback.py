from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


def detect_gaps(merge_output: dict[str, Any], normalize_output: dict[str, Any]) -> list[dict[str, str]]:
    """Analyze merge output and normalized task to detect information gaps.

    Returns a list of gap descriptions with ``area`` and ``description`` keys.
    Each gap represents a piece of information that the uploaded document did
    not explicitly provide but that downstream passes are likely to need.
    """
    gaps: list[dict[str, str]] = []

    # 1. Missing information signals surfaced during chunk extraction
    for signal in merge_output.get("all_missing_information_signals", []):
        text = signal.get("text", "").strip() if isinstance(signal, dict) else str(signal).strip()
        if text:
            gaps.append({"area": "missing_information", "description": text})

    # 2. Undefined terms — referenced but never defined in the document
    for term in merge_output.get("global_undefined_terms", []):
        gaps.append({"area": "undefined_term", "description": f"Term '{term}' is referenced but not defined"})

    # 3. Required inputs that are never produced by any step
    inputs_required = set(merge_output.get("all_inputs_required", []))
    outputs_produced = set(merge_output.get("all_outputs_produced", []))
    unresolved_inputs = inputs_required - outputs_produced
    for inp in sorted(unresolved_inputs):
        gaps.append({"area": "unresolved_input", "description": f"Required input '{inp}' is not produced by any documented step"})

    # 4. Empty entity buckets that the task likely needs
    entities = merge_output.get("global_entities", {})
    task = normalize_output.get("task", {}) if isinstance(normalize_output, dict) else {}
    domain = task.get("domain", "").lower() if isinstance(task, dict) else ""

    if not entities.get("materials") and not entities.get("equipment"):
        gaps.append({"area": "materials", "description": "No materials or equipment listed in the document"})

    if not merge_output.get("all_steps"):
        gaps.append({"area": "steps", "description": "No procedural steps found in the document"})

    # 5. Questions the normalize pass identified as needing answers
    for question in normalize_output.get("questions_to_answer", []):
        if isinstance(question, str) and question.strip():
            gaps.append({"area": "open_question", "description": question.strip()})

    return gaps


def build_fallback_queries(
    gaps: list[dict[str, str]],
    normalize_output: dict[str, Any],
    user_goal: str,
    max_queries: int = 5,
) -> list[str]:
    """Build targeted search queries from detected gaps.

    Produces up to *max_queries* short, focused web search queries designed to
    retrieve information that can fill the identified gaps.
    """
    task = normalize_output.get("task", {}) if isinstance(normalize_output, dict) else {}
    primary_goal = task.get("primary_goal", "") if isinstance(task, dict) else ""
    domain = task.get("domain", "") if isinstance(task, dict) else ""

    context_prefix = " ".join(filter(None, [domain, primary_goal])).strip()
    if not context_prefix:
        context_prefix = user_goal.strip()

    queries: list[str] = []
    seen: set[str] = set()

    for gap in gaps:
        if len(queries) >= max_queries:
            break
        desc = gap.get("description", "")
        area = gap.get("area", "")

        if area == "materials":
            query = f"{context_prefix} required materials and equipment list"
        elif area == "steps":
            query = f"{context_prefix} step by step procedure"
        elif area == "undefined_term":
            term = desc.split("'")[1] if "'" in desc else desc
            query = f"{term} definition {domain}".strip()
        elif area == "unresolved_input":
            inp = desc.split("'")[1] if "'" in desc else desc
            query = f"how to obtain {inp} {context_prefix}".strip()
        elif area == "open_question":
            query = f"{desc} {domain}".strip()
        else:
            query = f"{context_prefix} {desc}"

        query = " ".join(query.split())[:200]
        normalized = query.lower()
        if normalized not in seen:
            seen.add(normalized)
            queries.append(query)

    return queries
