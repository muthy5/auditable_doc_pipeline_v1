from pathlib import Path

from src.pipeline import AuditablePipeline


def test_demo_pipeline_creates_expected_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "examples" / "lemonade_plan_missing_juicing.txt"
    runs_dir = tmp_path / "runs"

    pipeline = AuditablePipeline(repo_root=repo_root, backend_name="demo")
    run_dir = pipeline.run(input_path=input_path, runs_dir=runs_dir)

    assert run_dir.exists()
    assert (run_dir / "input" / "document.json").exists()
    assert (run_dir / "passes" / "00_normalize_request.json").exists()
    assert (run_dir / "passes" / "02_merge_global.json").exists()
    assert (run_dir / "passes" / "08_validate_final.json").exists()
    assert (run_dir / "final" / "final_answer.json").exists()
    assert (run_dir / "final" / "final_answer.md").exists()
