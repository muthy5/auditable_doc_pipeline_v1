"""Standalone script to run the pipeline as a detached subprocess.

Reads job parameters from a JSON file, executes the pipeline, and writes
status/progress to a JSON file that the Streamlit UI can poll.

Usage:
    python background_runner.py <job_file.json>

The job file contains all parameters needed to reconstruct the pipeline and
run it.  The runner writes progress to <run_dir>/bg_status.json so the UI
can reconnect at any time.
"""
from __future__ import annotations

import json
import logging
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from src.config import PipelineConfig
from src.pipeline import AuditablePipeline


def _write_status(status_path: Path, data: dict) -> None:
    tmp = status_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(status_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    job_path = Path(sys.argv[1])
    job = json.loads(job_path.read_text(encoding="utf-8"))

    config_kwargs = job["config"]
    config = PipelineConfig(**config_kwargs)

    repo_root = Path(job["repo_root"])
    backend_name = job["backend_name"]
    input_path = Path(job["input_path"])
    runs_dir = Path(job["runs_dir"])
    user_goal = job["user_goal"]
    strict = job["strict"]
    document_type = job["document_type"]
    fast = job["fast"]
    parallel_chunks = job.get("parallel_chunks")

    # Write initial status
    status_path = runs_dir / "bg_status.json"
    _write_status(status_path, {"state": "running", "error": None, "run_dir": None})

    try:
        pipeline = AuditablePipeline(
            repo_root=repo_root, backend_name=backend_name, config=config
        )
        run_dir = pipeline.run(
            input_path=input_path,
            runs_dir=runs_dir,
            user_goal=user_goal,
            strict=strict,
            document_type=document_type,
            fast=fast,
            parallel_chunks=parallel_chunks,
        )
        _write_status(status_path, {
            "state": "completed",
            "error": None,
            "run_dir": str(run_dir),
        })
    except Exception as exc:
        _write_status(status_path, {
            "state": "failed",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            "run_dir": None,
        })
        sys.exit(1)


if __name__ == "__main__":
    main()
