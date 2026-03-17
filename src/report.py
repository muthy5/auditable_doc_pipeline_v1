from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_run_report(run_dir: Path, report: dict[str, Any]) -> Path:
    """Write a machine-readable run report."""
    path = run_dir / "report.json"
    report.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_partial_run_report(run_dir: Path, report: dict[str, Any]) -> Path:
    """Write a partial run report for interrupted or failed runs.

    This ensures that even incomplete runs persist enough metadata for the
    run advisor to learn from (timing, config, which passes completed, error info).
    Only writes if a full report.json does not already exist.
    """
    full_report_path = run_dir / "report.json"
    if full_report_path.exists():
        return full_report_path
    report.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    report.setdefault("incomplete", True)
    full_report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return full_report_path
