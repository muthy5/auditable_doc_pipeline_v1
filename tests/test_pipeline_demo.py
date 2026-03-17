from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.pipeline import AuditablePipeline
from src.run_inspector import inspect_run


def test_demo_pipeline_creates_expected_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "examples" / "lemonade_plan_missing_juicing.txt"
    runs_dir = tmp_path / "runs"

    pipeline = AuditablePipeline(repo_root=repo_root, backend_name="demo")
    run_dir = pipeline.run(input_path=input_path, runs_dir=runs_dir)

    assert (run_dir / "timing.json").exists()
    assert (run_dir / "report.json").exists()
    assert (run_dir / "final" / "final_answer.json").exists()
    assert (run_dir / "final" / "plan.json").exists()
    assert (run_dir / "final" / "plan.md").exists()
    assert not (run_dir / "passes" / "search_web_context.json").exists()
    assert not (run_dir / "passes" / "retrieval_context.json").exists()


def test_dry_run_prints_plan_without_executing(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "examples" / "lemonade_plan_missing_juicing.txt"
    proc = subprocess.run([sys.executable, "-m", "src.cli", "--input", str(input_path), "--backend", "demo", "--dry-run"], cwd=repo_root, capture_output=True, text=True, check=True)
    assert "00_normalize_request" in proc.stdout


def test_run_inspector_output(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "examples" / "lemonade_plan_missing_juicing.txt"
    pipeline = AuditablePipeline(repo_root=repo_root, backend_name="demo")
    run_dir = pipeline.run(input_path=input_path, runs_dir=tmp_path / "runs")
    output = inspect_run(run_dir)
    assert "Pass status" in output
    assert "Final answer preview" in output


def test_demo_pipeline_fast_mode_skips_noncritical_passes(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "examples" / "lemonade_plan_missing_juicing.txt"
    runs_dir = tmp_path / "runs"

    pipeline = AuditablePipeline(repo_root=repo_root, backend_name="demo")
    run_dir = pipeline.run(input_path=input_path, runs_dir=runs_dir, fast=True)

    assert not (run_dir / "passes" / "05_assumption_audit.json").exists()
    assert not (run_dir / "passes" / "06_evidence_audit.json").exists()
    assert (run_dir / "passes" / "07_synthesize.json").exists()


def test_cli_fast_mode_skips_noncritical_passes(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "examples" / "lemonade_plan_missing_juicing.txt"
    runs_dir = tmp_path / "runs"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "--input",
            str(input_path),
            "--backend",
            "demo",
            "--runs-dir",
            str(runs_dir),
            "--fast",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    run_dir = Path(proc.stdout.strip())
    assert run_dir.exists()
    assert not (run_dir / "passes" / "05_assumption_audit.json").exists()
    assert not (run_dir / "passes" / "06_evidence_audit.json").exists()
    assert (run_dir / "passes" / "07_synthesize.json").exists()


def test_demo_pipeline_accepts_fast_mode_alias(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "examples" / "lemonade_plan_missing_juicing.txt"

    pipeline = AuditablePipeline(repo_root=repo_root, backend_name="demo")
    run_dir = pipeline.run(input_path=input_path, runs_dir=tmp_path / "runs", fast_mode=True)

    assert not (run_dir / "passes" / "05_assumption_audit.json").exists()
    assert not (run_dir / "passes" / "06_evidence_audit.json").exists()


def test_fast_mode_report_marks_skipped_passes(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "examples" / "lemonade_plan_missing_juicing.txt"
    pipeline = AuditablePipeline(repo_root=repo_root, backend_name="demo")
    run_dir = pipeline.run(input_path=input_path, runs_dir=tmp_path / "runs", fast=True)

    report = __import__("json").loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["per_pass_status"]["05_assumption_audit"] == "skipped"
    assert report["per_pass_status"]["06_evidence_audit"] == "skipped"
    assert report["per_pass_status"]["07_synthesize"] in {"completed", "resumed"}


def test_non_strict_schema_failure_writes_fallback_and_report_status(tmp_path: Path) -> None:
    import json

    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "examples" / "lemonade_plan_missing_juicing.txt"
    pipeline = AuditablePipeline(repo_root=repo_root, backend_name="demo")
    original_generate = pipeline.pass_runner.backend.generate_json

    def patched_generate_json(pass_name, prompt_text, payload, schema=None):
        if pass_name == "04_dependency_audit":
            return {"invalid": "payload"}
        return original_generate(pass_name, prompt_text, payload, schema)

    pipeline.pass_runner.backend.generate_json = patched_generate_json
    run_dir = pipeline.run(input_path=input_path, runs_dir=tmp_path / "runs", strict=False)

    failed_path = run_dir / "passes" / "04_dependency_audit.failed.json"
    canonical_path = run_dir / "passes" / "04_dependency_audit.json"
    assert failed_path.exists()
    assert canonical_path.exists()
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    assert canonical["_schema_validation_failed"] is True
    assert canonical["_fallback_generated"] is True

    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["per_pass_status"]["04_dependency_audit"] == "completed_with_fallback"


def test_resume_marks_fallback_pass_as_resumed(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "examples" / "lemonade_plan_missing_juicing.txt"
    pipeline = AuditablePipeline(repo_root=repo_root, backend_name="demo")
    original_generate = pipeline.pass_runner.backend.generate_json

    def patched_generate_json(pass_name, prompt_text, payload, schema=None):
        if pass_name == "03_schema_audit":
            return {"broken": True}
        return original_generate(pass_name, prompt_text, payload, schema)

    pipeline.pass_runner.backend.generate_json = patched_generate_json
    first_run_dir = pipeline.run(input_path=input_path, runs_dir=tmp_path / "runs", strict=False)

    second_pipeline = AuditablePipeline(repo_root=repo_root, backend_name="demo")
    resumed_run_dir = second_pipeline.run(input_path=input_path, runs_dir=tmp_path / "runs", run_dir=first_run_dir, resume=True, strict=False)
    report = json.loads((resumed_run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["per_pass_status"]["03_schema_audit"] == "completed_with_fallback"


def test_demo_lemonade_regression_still_reports_missing_juicing(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "examples" / "lemonade_plan_missing_juicing.txt"
    pipeline = AuditablePipeline(repo_root=repo_root, backend_name="demo")

    run_dir = pipeline.run(input_path=input_path, runs_dir=tmp_path / "runs")
    dependency = json.loads((run_dir / "passes" / "04_dependency_audit.json").read_text(encoding="utf-8"))

    assert any("juice" in item["item"].lower() for item in dependency["missing_prerequisites"])


def test_demo_non_lemonade_input_does_not_leak_sample_content(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "examples" / "non_procedural_dossier.txt"
    pipeline = AuditablePipeline(repo_root=repo_root, backend_name="demo")

    run_dir = pipeline.run(input_path=input_path, runs_dir=tmp_path / "runs")
    plan = json.loads((run_dir / "final" / "plan.json").read_text(encoding="utf-8"))["plan"]
    final_answer = (run_dir / "final" / "final_answer.md").read_text(encoding="utf-8").lower()
    plan_blob = json.dumps(plan).lower()
    source_text = input_path.read_text(encoding="utf-8").lower()

    forbidden = ["lemonade", "lemon", "lemons", "sugar", "water", "ice", "juicing"]
    for term in forbidden:
        if term not in source_text:
            assert term not in plan_blob
            assert term not in final_answer


def test_demo_current_run_isolation_uses_new_input(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pipeline = AuditablePipeline(repo_root=repo_root, backend_name="demo")

    lemonade_run = pipeline.run(input_path=repo_root / "examples" / "lemonade_plan_missing_juicing.txt", runs_dir=tmp_path / "runs")
    dossier_run = pipeline.run(input_path=repo_root / "examples" / "non_procedural_dossier.txt", runs_dir=tmp_path / "runs")

    lemonade_plan = (lemonade_run / "final" / "plan.md").read_text(encoding="utf-8").lower()
    dossier_plan = (dossier_run / "final" / "plan.md").read_text(encoding="utf-8").lower()

    assert "8 lemons" in lemonade_plan
    assert "8 lemons" not in dossier_plan
    assert "cannot reliably transform this document type" in dossier_plan


def test_demo_non_procedural_document_fails_safe_with_limitation_message(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "examples" / "non_procedural_dossier.txt"
    pipeline = AuditablePipeline(repo_root=repo_root, backend_name="demo")

    run_dir = pipeline.run(input_path=input_path, runs_dir=tmp_path / "runs")
    plan = json.loads((run_dir / "final" / "plan.json").read_text(encoding="utf-8"))["plan"]
    warnings_text = "\n".join(item["text"] for item in plan["warnings_and_safety"]).lower()

    assert "cannot reliably transform this document type" in warnings_text
    assert plan["steps"] == []
