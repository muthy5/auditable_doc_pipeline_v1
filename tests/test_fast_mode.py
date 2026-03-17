"""Tests for strict bool validation of the fast_mode kwarg."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline import AuditablePipeline


@pytest.fixture()
def pipeline() -> AuditablePipeline:
    repo_root = Path(__file__).resolve().parents[1]
    return AuditablePipeline(repo_root=repo_root, backend_name="demo")


@pytest.fixture()
def run_kwargs(tmp_path: Path) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    return {
        "input_path": repo_root / "examples" / "lemonade_plan_missing_juicing.txt",
        "runs_dir": tmp_path / "runs",
    }


def test_fast_mode_true_does_not_raise(pipeline: AuditablePipeline, run_kwargs: dict) -> None:
    run_dir = pipeline.run(**run_kwargs, fast_mode=True)
    assert run_dir.exists()


def test_fast_mode_false_does_not_raise(pipeline: AuditablePipeline, run_kwargs: dict) -> None:
    run_dir = pipeline.run(**run_kwargs, fast_mode=False)
    assert run_dir.exists()


def test_fast_mode_string_false_raises_type_error(pipeline: AuditablePipeline, run_kwargs: dict) -> None:
    with pytest.raises(TypeError, match="fast_mode must be a bool"):
        pipeline.run(**run_kwargs, fast_mode="false")


def test_fast_mode_string_true_raises_type_error(pipeline: AuditablePipeline, run_kwargs: dict) -> None:
    with pytest.raises(TypeError, match="fast_mode must be a bool"):
        pipeline.run(**run_kwargs, fast_mode="true")


def test_fast_mode_int_raises_type_error(pipeline: AuditablePipeline, run_kwargs: dict) -> None:
    with pytest.raises(TypeError, match="fast_mode must be a bool"):
        pipeline.run(**run_kwargs, fast_mode=1)
