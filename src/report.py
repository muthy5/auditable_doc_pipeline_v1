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
