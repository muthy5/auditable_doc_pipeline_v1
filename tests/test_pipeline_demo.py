from __future__ import annotations

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
