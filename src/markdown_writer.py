from __future__ import annotations

from typing import Any, Dict, List


def _render_supported_items(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "- None"
    return "\n".join(f"- {item['text']}  \n  Support: {', '.join(item.get('support', []))}" for item in items)


def render_final_answer_markdown(synthesis: Dict[str, Any]) -> str:
    final_answer = synthesis["final_answer"]
    sections = [
        "# Final Answer",
        "",
        "## Goal",
        final_answer["goal"],
        "",
        "## Verified Content",
        _render_supported_items(final_answer["verified_content"]),
        "",
        "## Missing Information",
        _render_supported_items(final_answer["missing_information"]),
        "",
        "## Dependencies",
        _render_supported_items(final_answer["dependencies"]),
        "",
        "## Assumptions",
        _render_supported_items(final_answer["assumptions"]),
        "",
        "## Uncertainties",
        _render_supported_items(final_answer["uncertainties"]),
        "",
        "## Organized Structure",
    ]

    organized = final_answer.get("organized_structure", [])
    if organized:
        for item in organized:
            sections.extend([f"### {item['section']}", item["content"], ""])
    else:
        sections.extend(["- None", ""])

    bottom_line = final_answer["bottom_line"]
    sections.extend(
        [
            "## Bottom Line",
            f"{bottom_line['text']}  ",
            f"Support: {', '.join(bottom_line.get('support', []))}",
            "",
        ]
    )

    return "\n".join(sections).strip() + "\n"


def _render_supported_bullets(items: List[Dict[str, Any]]) -> List[str]:
    if not items:
        return ["- None", ""]
    lines: List[str] = []
    for item in items:
        lines.append(f"- {item.get('text', '')}")
        support = ", ".join(item.get("support", []))
        lines.append(f"  - Support: {support}")
    lines.append("")
    return lines


def render_plan_markdown(plan_output: dict[str, Any]) -> str:
    plan = plan_output.get("plan", {})
    sections: List[str] = ["# Generated Plan", ""]

    objective = plan.get("objective", {})
    sections.extend(["## Objective", objective.get("text", ""), f"Support: {', '.join(objective.get('support', []))}", ""])

    sections.extend(["## Materials & Quantities", "| Item | Quantity | Source |", "| --- | --- | --- |"])
    for material in plan.get("materials_and_quantities", []):
        sections.append(f"| {material.get('item', '')} | {material.get('quantity', '')} | {material.get('source', '')} |")
    if not plan.get("materials_and_quantities"):
        sections.append("| None | None | unknown |")
    sections.append("")

    sections.append("## Equipment")
    equipment = plan.get("equipment_required", [])
    sections.extend([f"- {item.get('item', '')} ({item.get('source', '')})" for item in equipment] or ["- None"])
    sections.append("")

    sections.append("## Prerequisites")
    sections.extend(_render_supported_bullets(plan.get("prerequisites", [])))

    badge_map = {"original": "[ORIGINAL]", "added": "[ADDED]", "reordered": "[REORDERED]"}
    sections.append("## Steps")
    steps = plan.get("steps", [])
    if steps:
        for step in steps:
            badge = badge_map.get(step.get("status", ""), "[UNKNOWN]")
            sections.append(f"{step.get('step_number', '?')}. {badge} {step.get('text', '')}")
            sections.append(f"   - Support: {', '.join(step.get('support', []))}")
            if step.get("time_estimate"):
                sections.append(f"   - Time: {step['time_estimate']}")
            if step.get("warning"):
                sections.append(f"   - Warning: {step['warning']}")
    else:
        sections.append("- None")
    sections.append("")

    time_estimates = plan.get("time_estimates", {})
    sections.extend(["## Time Estimates", f"- Total estimated: {time_estimates.get('total_estimated', 'unknown')}", f"- Confidence: {time_estimates.get('confidence', 'unknown')}", ""])

    sections.append("## Warnings & Safety")
    severity_prefix = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
    warnings = plan.get("warnings_and_safety", [])
    if warnings:
        for warning in warnings:
            marker = severity_prefix.get(warning.get("severity", "info"), "🔵")
            sections.append(f"- {marker} **{warning.get('severity', 'info').upper()}**: {warning.get('text', '')}")
            sections.append(f"  - Support: {', '.join(warning.get('support', []))}")
    else:
        sections.append("- None")
    sections.append("")

    sections.append("## Quality Checkpoints")
    checkpoints = plan.get("quality_checkpoints", [])
    if checkpoints:
        for checkpoint in checkpoints:
            sections.append(f"- After step {checkpoint.get('after_step', '?')}: {checkpoint.get('check', '')}")
            sections.append(f"  - Support: {', '.join(checkpoint.get('support', []))}")
    else:
        sections.append("- None")
    sections.append("")

    sections.append("## Blocking Items")
    sections.extend(_render_supported_bullets(plan.get("blocking_items", [])))

    sections.append("## Assumptions")
    sections.extend(_render_supported_bullets(plan.get("assumptions_made", [])))

    sections.append("## Cost Indicators")
    cost_indicators = plan.get("cost_indicators", [])
    if cost_indicators:
        for indicator in cost_indicators:
            sections.append(f"- {indicator.get('item', '')}: {indicator.get('cost', '')} ({indicator.get('source', '')})")
    else:
        sections.append("- None")
    sections.append("")

    sections.append("## Contingencies")
    contingencies = plan.get("contingencies", [])
    if contingencies:
        for contingency in contingencies:
            sections.append(f"- If {contingency.get('if_condition', '')}, then {contingency.get('then_action', '')}.")
            sections.append(f"  - Support: {', '.join(contingency.get('support', []))}")
    else:
        sections.append("- None")
    sections.append("")

    return "\n".join(sections).strip() + "\n"
