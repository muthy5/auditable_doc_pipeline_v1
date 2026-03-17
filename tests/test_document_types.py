from __future__ import annotations

import json
from pathlib import Path

from src.document_classifier import SUPPORTED_DOCUMENT_TYPES, classify_document
from src.llm_interface import RuleBasedDemoBackend
from src.pipeline import AuditablePipeline


def test_classifier_detects_procedural_plan_from_lemonade_example() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "examples" / "lemonade_plan_missing_juicing.txt").read_text(encoding="utf-8")
    backend = RuleBasedDemoBackend()
    document_type = classify_document(text, backend)
    assert document_type == "procedural_plan"


def test_document_type_schemas_are_valid_json() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for document_type in SUPPORTED_DOCUMENT_TYPES:
        payload = json.loads((repo_root / "schemas" / "document_types" / f"{document_type}.json").read_text(encoding="utf-8"))
        assert payload["document_type"] == document_type
        assert isinstance(payload["expected_sections"], list)
        assert payload["expected_sections"]


def test_procedural_plan_schema_is_domain_neutral() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    payload = json.loads((repo_root / "schemas" / "document_types" / "procedural_plan.json").read_text(encoding="utf-8"))

    expected = payload["expected_sections"]
    assert "ingredient_preparation" not in expected
    assert "inputs" in expected
    assert "constraints" in expected


def test_pipeline_runs_with_auto_classification(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "examples" / "lemonade_plan_missing_juicing.txt"
    pipeline = AuditablePipeline(repo_root=repo_root, backend_name="demo")
    run_dir = pipeline.run(input_path=input_path, runs_dir=tmp_path / "runs", document_type="auto")
    classification = json.loads((run_dir / "passes" / "classify_document.json").read_text(encoding="utf-8"))
    assert classification["selected_document_type"] == "procedural_plan"


def test_pipeline_runs_with_explicit_document_type(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "examples" / "lemonade_plan_missing_juicing.txt"
    pipeline = AuditablePipeline(repo_root=repo_root, backend_name="demo")
    run_dir = pipeline.run(input_path=input_path, runs_dir=tmp_path / "runs", document_type="technical_spec")
    schema_audit = json.loads((run_dir / "passes" / "03_schema_audit.json").read_text(encoding="utf-8"))
    assert schema_audit["document_type"] == "technical_spec"


def test_classifier_falls_back_on_malformed_backend_payload() -> None:
    class BrokenBackend:
        def generate_json(self, pass_name, prompt_text, payload, schema=None):  # noqa: ANN001
            return ["not", "a", "mapping"]

    document_type = classify_document("text", BrokenBackend())

    assert document_type == "procedural_plan"


def test_pipeline_auto_classification_uses_pipeline_repo_root(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "examples" / "lemonade_plan_missing_juicing.txt"
    pipeline = AuditablePipeline(repo_root=repo_root, backend_name="demo")

    captured_roots: list[Path | None] = []

    original = __import__("src.pipeline", fromlist=["classify_document_with_metadata"]).classify_document_with_metadata

    def wrapped(text, backend, *, repo_root=None):  # noqa: ANN001
        captured_roots.append(repo_root)
        return original(text, backend, repo_root=repo_root)

    monkeypatch.setattr("src.pipeline.classify_document_with_metadata", wrapped)

    pipeline.run(input_path=input_path, runs_dir=tmp_path / "runs", document_type="auto")

    assert captured_roots == [repo_root]
