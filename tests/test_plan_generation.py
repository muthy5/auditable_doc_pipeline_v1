from __future__ import annotations

from pathlib import Path

from jsonschema import Draft202012Validator

from src.llm_interface import RuleBasedDemoBackend
from src.markdown_writer import render_plan_markdown
from src.schemas import load_schema


def test_demo_backend_generates_plan_with_added_juicing_step() -> None:
    backend = RuleBasedDemoBackend()
    output = backend.generate_json(
        "09_generate_plan",
        "",
        {
            "merge": {
                "doc_id": "doc_001",
                "global_entities": {"materials": ["8 lemons", "1 cup sugar"]},
                "all_outputs_produced": ["A pitcher of lemonade"],
                "all_steps": [{"step_id": "s1", "text": "Mix sugar and water."}],
            },
            "dependency_audit": {"blocking_dependencies": [{"dependency_id": "block_dep_001", "reason": "missing juicing"}]},
        },
    )

    steps = output["plan"]["steps"]
    assert any(step["status"] == "added" and "juice" in step["text"].lower() for step in steps)


def test_plan_schema_accepts_valid_plan() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    schema = load_schema(repo_root / "schemas", "09_generate_plan.schema.json")

    payload = RuleBasedDemoBackend().generate_json(
        "09_generate_plan",
        "",
        {
            "merge": {"doc_id": "doc_001"},
            "dependency_audit": {"blocking_dependencies": [{"dependency_id": "block_dep_001", "reason": "missing juicing"}]},
        },
    )

    Draft202012Validator(schema).validate(payload)


def test_plan_markdown_rendering_contains_expected_sections() -> None:
    payload = RuleBasedDemoBackend().generate_json(
        "09_generate_plan",
        "",
        {
            "merge": {"doc_id": "doc_001"},
            "dependency_audit": {"blocking_dependencies": [{"dependency_id": "block_dep_001", "reason": "missing juicing"}]},
        },
    )

    markdown = render_plan_markdown(payload)

    assert "## Objective" in markdown
    assert "## Materials & Quantities" in markdown
    assert "## Warnings & Safety" in markdown
    assert "## Contingencies" in markdown
