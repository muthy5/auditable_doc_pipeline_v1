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
import os
import signal
import sys
import threading
import traceback
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

    # Create a cancel event that the pipeline checks between passes
    cancel_event = threading.Event()

    # Write initial status including PID so the UI can check liveness
    status_path = runs_dir / "bg_status.json"
    _write_status(status_path, {"state": "running", "error": None, "run_dir": None, "pid": os.getpid()})

    def _handle_stop_signal(signum: int, frame: object) -> None:
        """Handle SIGTERM/SIGINT by setting the cancel event."""
        logging.getLogger(__name__).info("Received signal %d, cancelling pipeline...", signum)
        cancel_event.set()

    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)

    # Also watch for a stop file written by the UI
    stop_file = runs_dir / "bg_stop.json"

    def _watch_stop_file() -> None:
        """Poll for a stop file and set the cancel event when found."""
        import time
        while not cancel_event.is_set():
            if stop_file.exists():
                logging.getLogger(__name__).info("Stop file detected, cancelling pipeline...")
                cancel_event.set()
                try:
                    stop_file.unlink(missing_ok=True)
                except OSError:
                    pass
                return
            time.sleep(0.5)

    watcher = threading.Thread(target=_watch_stop_file, daemon=True)
    watcher.start()

    try:
        pipeline = AuditablePipeline(
            repo_root=repo_root, backend_name=backend_name, config=config,
            cancel_event=cancel_event,
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
        from src.exceptions import PipelineCancelled
        if isinstance(exc, PipelineCancelled):
            _write_status(status_path, {
                "state": "cancelled",
                "error": "Pipeline stopped by user",
                "run_dir": None,
            })
        else:
            _write_status(status_path, {
                "state": "failed",
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                "run_dir": None,
            })
            sys.exit(1)


if __name__ == "__main__":
    main()
