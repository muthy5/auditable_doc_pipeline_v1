from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PASS_OUTPUT_FILES = [
    "00_normalize_request.json",
    "classify_document.json",
    "02_merge_global.json",
    "03_schema_audit.json",
    "04_dependency_audit.json",
    "05_assumption_audit.json",
    "06_evidence_audit.json",
    "07_synthesize.json",
    "09_generate_plan.json",
    "08_validate_final.json",
    "search_web_context.json",
]

PASS_SEQUENCE = [
    "00_normalize_request",
    "01_extract_chunk",
    "02_merge_global",
    "03_schema_audit",
    "04_dependency_audit",
    "05_assumption_audit",
    "06_evidence_audit",
    "07_synthesize",
    "09_generate_plan",
    "08_validate_final",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_final_sections(run_dir: Path) -> dict[str, Any]:
    """Parse final answer JSON into UI display sections."""
    final_answer_path = run_dir / "final" / "final_answer.json"
    final_payload = _read_json(final_answer_path)
    final_answer = final_payload.get("final_answer", {})

    return {
        "verified_content": final_answer.get("verified_content", []),
        "missing_information": final_answer.get("missing_information", []),
        "dependencies": final_answer.get("dependencies", []),
        "assumptions": final_answer.get("assumptions", []),
        "uncertainties": final_answer.get("uncertainties", []),
        "bottom_line": final_answer.get("bottom_line", {}).get("text", ""),
        "goal": final_answer.get("goal", ""),
    }




def parse_plan_output(run_dir: Path) -> dict[str, Any]:
    """Parse generated plan JSON for UI rendering."""
    plan_path = run_dir / "final" / "plan.json"
    if plan_path.exists():
        return _read_json(plan_path).get("plan", {})

    fallback_path = run_dir / "passes" / "09_generate_plan.json"
    if fallback_path.exists():
        return _read_json(fallback_path).get("plan", {})
    return {}


def format_gap_plain_english(gap: dict[str, Any]) -> str:
    """Convert a gap-like object into a readable sentence."""
    reason = str(gap.get("reason") or gap.get("text") or "Missing details were identified.").strip()
    section = str(gap.get("section") or gap.get("item") or "").strip()
    if section:
        return f"{reason} (Area: {section})"
    return reason


def get_status_color(synthesis: dict[str, Any]) -> str:
    """Return red/yellow/green status based on blocking gap signals."""
    final_answer = synthesis.get("final_answer", {}) if synthesis else {}
    dependency_count = len(final_answer.get("dependencies", []))
    missing_count = len(final_answer.get("missing_information", []))
    uncertainty_count = len(final_answer.get("uncertainties", []))

    if dependency_count > 0:
        return "red"
    if missing_count > 0 or uncertainty_count > 0:
        return "yellow"
    return "green"


def format_step_with_badge(step: dict[str, Any]) -> str:
    """Return step text prefixed with a status badge emoji."""
    badge_map = {
        "original": "✅",
        "added": "➕",
        "reordered": "🔁",
    }
    status = str(step.get("status", "")).lower()
    badge = badge_map.get(status, "•")
    text = str(step.get("text") or "").strip()
    return f"{badge} {text}" if text else badge


def format_plan_for_display(plan_data: dict[str, Any]) -> dict[str, Any]:
    """Convert raw plan JSON into display-friendly values."""
    materials = [
        {
            "Item": str(item.get("item", "")),
            "Quantity": str(item.get("quantity", "")),
            "Source": str(item.get("source", "unknown")).replace("_", " ").title(),
        }
        for item in plan_data.get("materials_and_quantities", [])
    ]

    steps = []
    for step in plan_data.get("steps", []):
        steps.append(
            {
                "number": step.get("step_number"),
                "text": format_step_with_badge(step),
                "status": str(step.get("status", "unknown")),
                "warning": str(step.get("warning", "")).strip(),
            }
        )

    checkpoints_by_step: dict[int, list[str]] = {}
    for checkpoint in plan_data.get("quality_checkpoints", []):
        step_number = int(checkpoint.get("after_step", 0))
        checkpoints_by_step.setdefault(step_number, []).append(str(checkpoint.get("check", "")).strip())

    return {
        "objective": str(plan_data.get("objective", {}).get("text", "")).strip(),
        "time_estimate": str(plan_data.get("time_estimates", {}).get("total_estimated", "")).strip(),
        "time_confidence": str(plan_data.get("time_estimates", {}).get("confidence", "")).strip(),
        "materials": materials,
        "steps": steps,
        "quality_checkpoints": checkpoints_by_step,
        "warnings": [str(item.get("text", "")).strip() for item in plan_data.get("warnings_and_safety", []) if item.get("text")],
        "assumptions": [str(item.get("text", "")).strip() for item in plan_data.get("assumptions_made", []) if item.get("text")],
        "blocking_items": [str(item.get("text", "")).strip() for item in plan_data.get("blocking_items", []) if item.get("text")],
        "contingencies": [
            {
                "if": str(item.get("if_condition", "")).strip(),
                "then": str(item.get("then_action", "")).strip(),
            }
            for item in plan_data.get("contingencies", [])
        ],
    }


def collect_pass_outputs(run_dir: Path) -> dict[str, Any]:
    """Collect JSON pass outputs for raw inspection in the UI."""
    passes_dir = run_dir / "passes"
    outputs: dict[str, Any] = {}

    for filename in PASS_OUTPUT_FILES:
        path = passes_dir / filename
        if path.exists():
            outputs[path.stem] = _read_json(path)

    extract_dir = passes_dir / "01_extract_chunk"
    if extract_dir.exists():
        chunk_outputs: dict[str, Any] = {}
        for chunk_file in sorted(extract_dir.glob("*.json")):
            chunk_outputs[chunk_file.name] = _read_json(chunk_file)
        outputs["01_extract_chunk"] = chunk_outputs

    return outputs


def collect_run_report(run_dir: Path) -> dict[str, Any]:
    """Return timing and validation stats from run artifacts."""
    report_path = run_dir / "report.json"
    timing_path = run_dir / "timing.json"
    validation_path = run_dir / "passes" / "08_validate_final.json"

    report = _read_json(report_path) if report_path.exists() else {}
    timing = _read_json(timing_path) if timing_path.exists() else {}
    validation = _read_json(validation_path) if validation_path.exists() else {}

    return {
        "report": report,
        "timing": timing,
        "validation": {
            "errors": validation.get("errors", []),
            "warnings": validation.get("warnings", []),
            "checks": validation.get("checks", []),
        },
    }


def format_item(item: dict[str, Any]) -> str:
    """Render one synthesis item as markdown bullet text."""
    text = item.get("text", "")
    support = item.get("support") or []
    if not support:
        return f"- {text}"
    support_text = ", ".join(str(ref) for ref in support)
    return f"- {text} _(support: {support_text})_"


def read_final_markdown(run_dir: Path) -> str:
    """Read the rendered final markdown answer."""
    return (run_dir / "final" / "final_answer.md").read_text(encoding="utf-8")


def is_streamlit_cloud_environment(env: dict[str, str] | None = None) -> bool:
    """Return True when running in a Streamlit Community Cloud-like environment."""
    active_env = env or os.environ
    markers = ["STREAMLIT_SHARING", "STREAMLIT_CLOUD", "STREAMLIT_RUNTIME_ENV"]
    return any(str(active_env.get(marker, "")).strip().lower() in {"1", "true", "cloud", "community"} for marker in markers)


def get_available_backends(cloud_mode: bool) -> list[str]:
    """Return UI backend choices based on deployment environment."""
    if cloud_mode:
        return ["demo", "claude"]
    return ["demo", "ollama", "claude"]
