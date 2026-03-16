from __future__ import annotations

import json
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
    if not plan_path.exists():
        return {}
    return _read_json(plan_path).get("plan", {})


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
