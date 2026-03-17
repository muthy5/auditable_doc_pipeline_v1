from __future__ import annotations

import json
from pathlib import Path

from src.run_advisor import generate_run_advice


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_generate_run_advice_with_complete_and_incomplete_runs(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"

    run1 = runs_dir / "run_complete_1"
    _write_json(
        run1 / "timing.json",
        {
            "passes": {"05_assumption_audit": 30.0, "06_evidence_audit": 20.0},
            "total_pipeline_seconds": 100.0,
        },
    )
    _write_json(
        run1 / "report.json",
        {
            "backend": "demo",
            "total_duration_seconds": 100.0,
            "per_pass_status": {"05_assumption_audit": "completed_with_fallback", "06_evidence_audit": "completed"},
            "unsupported_claim_count": 1,
            "schema_validation_failure_list": [
                "Schema validation failed in pass '03_schema_audit': E_SYNTH_UNKNOWN_AS_FACT",
                "Schema validation failed in pass '03_schema_audit': another failure",
            ],
            "document_type": "technical_spec",
            "parallel_chunks": 4,
        },
    )
    _write_json(run1 / "input" / "chunks.json", [{"text": "word " * 3200}, {"text": "word " * 3200}])

    run2 = runs_dir / "run_complete_2"
    _write_json(
        run2 / "timing.json",
        {
            "passes": {"05_assumption_audit": 20.0, "06_evidence_audit": 15.0},
            "total_pipeline_seconds": 70.0,
        },
    )
    _write_json(
        run2 / "report.json",
        {
            "backend": "demo",
            "total_duration_seconds": 70.0,
            "per_pass_status": {"05_assumption_audit": "completed_with_fallback", "06_evidence_audit": "completed"},
            "unsupported_claim_count": 0,
            "schema_validation_failure_list": ["Schema validation failed in pass '03_schema_audit': repeated failure"],
            "document_type": "procedural_plan",
            "parallel_chunks": 2,
        },
    )
    _write_json(run2 / "input" / "chunks.json", [{"text": "word " * 3000}, {"text": "word " * 3000}])

    run3 = runs_dir / "run_incomplete_1"
    _write_json(run3 / "passes" / "00_normalize_request.json", {"ok": True})
    _write_json(run3 / "passes" / "03_schema_audit.json", {"ok": True})
    _write_json(run3 / "passes" / "checkpoint.json", {"byte_size": 1000, "url_count": 10})

    run4 = runs_dir / "run_incomplete_2"
    _write_json(run4 / "passes" / "03_schema_audit.json", {"ok": True})

    (runs_dir / "run_empty").mkdir(parents=True, exist_ok=True)

    report = generate_run_advice(runs_dir)

    assert report.runs_analyzed == 5
    assert report.complete_runs == 2
    assert report.incomplete_runs == 3
    assert report.crash_point_distribution["03_schema_audit"] >= 2
    assert any("fast=True" in item for item in report.speed_recommendations)
    assert any("parallel_chunks" in item for item in report.speed_recommendations)
    assert any("strict=True" in item for item in report.accuracy_recommendations)
    assert any("E_SYNTH_UNKNOWN_AS_FACT" in item for item in report.accuracy_recommendations)
    assert any("known failure pass" in item for item in report.accuracy_recommendations)
    assert "chunk_target_min_words" in report.suggested_config
    assert "chunk_target_max_words" in report.suggested_config


def test_generate_run_advice_all_incomplete_runs(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _write_json(runs_dir / "r1" / "passes" / "04_dependency_audit.json", {"ok": True})
    _write_json(runs_dir / "r2" / "passes" / "04_dependency_audit.json", {"ok": True})

    report = generate_run_advice(runs_dir)

    assert report.runs_analyzed == 2
    assert report.complete_runs == 0
    assert report.incomplete_runs == 2
    assert report.crash_point_distribution == {"04_dependency_audit": 2}
    assert any("known failure pass" in item for item in report.accuracy_recommendations)


def test_generate_run_advice_insufficient_history(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    (runs_dir / "only_one").mkdir(parents=True)

    report = generate_run_advice(runs_dir)

    assert report.runs_analyzed == 1
    assert report.speed_recommendations == []
    assert report.accuracy_recommendations == []
    assert report.warnings == ["Insufficient run history — minimum 2 runs required for reliable recommendations"]
