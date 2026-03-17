from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def export_run(run_dir: Path, export_dir: Path) -> Path:
    """Copy selected run artifacts into export_dir/runs/<run_id>."""
    export_run_dir = export_dir / "runs" / run_dir.name
    export_run_dir.mkdir(parents=True, exist_ok=True)

    for relative_path in [
        Path("timing.json"),
        Path("report.json"),
        Path("logs") / "run.log",
        Path("final") / "blocker_summary.json",
    ]:
        source = run_dir / relative_path
        if source.exists() and source.is_file():
            destination = export_run_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    source_passes_dir = run_dir / "passes"
    if source_passes_dir.exists() and source_passes_dir.is_dir():
        destination_passes_dir = export_run_dir / "passes"
        destination_passes_dir.mkdir(parents=True, exist_ok=True)
        for pass_file in source_passes_dir.glob("*.json"):
            if pass_file.is_file():
                shutil.copy2(pass_file, destination_passes_dir / pass_file.name)

    return export_run_dir


def export_all_runs(runs_dir: Path, export_dir: Path) -> list[Path]:
    """Export every run directory that contains timing, report, or pass artifacts."""
    if not runs_dir.exists() or not runs_dir.is_dir():
        return []

    exported: list[Path] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        has_timing = (run_dir / "timing.json").exists()
        has_report = (run_dir / "report.json").exists()
        passes_dir = run_dir / "passes"
        has_pass_json = any(passes_dir.glob("*.json")) if passes_dir.is_dir() else False
        if has_timing or has_report or has_pass_json:
            exported.append(export_run(run_dir=run_dir, export_dir=export_dir))
    return exported


def main() -> None:
    """CLI entrypoint for exporting run artifacts."""
    parser = argparse.ArgumentParser(
        description="Export pipeline run artifacts for sharing and analysis."
    )
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--export-dir", required=True)
    args = parser.parse_args()
    export_all_runs(runs_dir=Path(args.runs_dir), export_dir=Path(args.export_dir))


if __name__ == "__main__":
    main()
