from __future__ import annotations

from typing import Any, Dict, List


def _render_supported_items(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "- None"
    return "\n".join(f"- {item['text']}  \n  Support: {', '.join(item.get('support', []))}" for item in items)


def render_final_answer_markdown(synthesis: Dict[str, Any]) -> str:
    """Render final synthesis JSON into a readable markdown report.

    Args:
        synthesis: Final synthesis payload.

    Returns:
        Markdown representation of the final answer.
    """
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
