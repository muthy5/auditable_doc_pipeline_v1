from __future__ import annotations

from pathlib import Path

from src.run_exporter import export_all_runs, export_run


def test_exports_complete_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run_complete"
    (run_dir / "passes").mkdir(parents=True)
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "timing.json").write_text('{"seconds": 1}', encoding="utf-8")
    (run_dir / "report.json").write_text('{"status": "ok"}', encoding="utf-8")
    (run_dir / "passes" / "01_extract_chunk.json").write_text("{}", encoding="utf-8")
    (run_dir / "passes" / "03_schema_audit.failed.json").write_text("{}", encoding="utf-8")
    (run_dir / "logs" / "run.log").write_text("log", encoding="utf-8")

    export_path = export_run(run_dir=run_dir, export_dir=tmp_path / "exports")

    assert (export_path / "timing.json").exists()
    assert (export_path / "passes" / "01_extract_chunk.json").exists()
    assert (export_path / "passes" / "03_schema_audit.failed.json").exists()


def test_exports_incomplete_run(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_incomplete"
    (run_dir / "passes").mkdir(parents=True)
    (run_dir / "passes" / "01_extract_chunk.json").write_text("{}", encoding="utf-8")

    exported = export_all_runs(runs_dir=runs_dir, export_dir=tmp_path / "exports")

    assert len(exported) == 1
    assert (exported[0] / "passes" / "01_extract_chunk.json").exists()


def test_skips_input_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run_with_input"
    (run_dir / "input").mkdir(parents=True)
    (run_dir / "passes").mkdir(parents=True)
    (run_dir / "passes" / "01_extract_chunk.json").write_text("{}", encoding="utf-8")
    (run_dir / "input" / "large_source.txt").write_text("huge", encoding="utf-8")

    export_path = export_run(run_dir=run_dir, export_dir=tmp_path / "exports")

    assert not (export_path / "input").exists()


def test_never_raises_on_missing_dir(tmp_path: Path) -> None:
    missing_runs_dir = tmp_path / "does_not_exist"

    exported = export_all_runs(runs_dir=missing_runs_dir, export_dir=tmp_path / "exports")

    assert exported == []
