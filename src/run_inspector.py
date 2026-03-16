from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect_run(run_dir: Path) -> str:
    """Build a human-readable run summary."""
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    timing = json.loads((run_dir / "timing.json").read_text(encoding="utf-8"))
    final = json.loads((run_dir / "final" / "final_answer.json").read_text(encoding="utf-8"))
    preview = json.dumps(final, ensure_ascii=False)[:500]
    lines = [
        f"Run ID: {report.get('run_id')}",
        f"Backend: {report.get('backend')} ({report.get('model_name')})",
        f"Blocking gaps: {report.get('blocking_gap_count')}",
        f"Unsupported claims: {report.get('unsupported_claim_count')}",
        "Pass status:",
    ]
    for pass_name, status in report.get("per_pass_status", {}).items():
        lines.append(f"  - {pass_name}: {status}")
    lines.append(f"Timing total (s): {timing.get('total_pipeline_seconds')}")
    lines.append("Final answer preview:")
    lines.append(preview)
    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint for run inspection."""
    parser = argparse.ArgumentParser(description="Inspect an existing pipeline run directory.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(inspect_run(Path(args.run_dir)))


if __name__ == "__main__":
    main()
