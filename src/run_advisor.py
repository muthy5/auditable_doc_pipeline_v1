from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

from .config import PipelineConfig


@dataclass(frozen=True)
class RunAdvisorReport:
    runs_analyzed: int
    complete_runs: int
    incomplete_runs: int
    crash_point_distribution: dict[str, int]
    speed_recommendations: list[str]
    accuracy_recommendations: list[str]
    suggested_config: dict[str, Any]
    warnings: list[str]


_PASS_FROM_FAILURE_RE = re.compile(r"pass '([^']+)':")


def _safe_load_json(path: Path) -> tuple[dict[str, Any] | list[Any] | None, bool]:
    if not path.exists():
        return None, False
    try:
        return json.loads(path.read_text(encoding="utf-8")), True
    except (OSError, json.JSONDecodeError):
        return None, False


def _run_dirs(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists():
        return []
    return sorted([path for path in runs_dir.iterdir() if path.is_dir()])


def _pass_order(pass_name: str) -> int:
    match = re.match(r"^(\d{2})_", pass_name)
    if match:
        return int(match.group(1))
    if pass_name == "classify_document":
        return 0
    return -1


def _extract_last_successful_pass(passes_dir: Path) -> str | None:
    if not passes_dir.exists():
        return None
    candidates = [
        path
        for path in passes_dir.rglob("*.json")
        if not path.name.endswith(".failed.json") and path.name != "checkpoint.json"
    ]
    if not candidates:
        return None
    valid_files: list[tuple[int, float, str]] = []
    for path in candidates:
        payload, ok = _safe_load_json(path)
        if not ok or payload is None:
            continue
        pass_name = path.stem if path.parent == passes_dir else path.parent.name
        valid_files.append((_pass_order(pass_name), path.stat().st_mtime, pass_name))
    if not valid_files:
        return None
    return sorted(valid_files, key=lambda item: (item[0], item[1], item[2]))[-1][2]


def _extract_chunk_stats(run_dir: Path) -> tuple[int | None, float | None]:
    chunks_payload, ok = _safe_load_json(run_dir / "input" / "chunks.json")
    if not ok or not isinstance(chunks_payload, list) or not chunks_payload:
        return None, None
    word_counts = [len(str(chunk.get("text", "")).split()) for chunk in chunks_payload if isinstance(chunk, dict)]
    if not word_counts:
        return len(chunks_payload), 0.0
    return len(chunks_payload), (sum(word_counts) / len(word_counts))


def _extract_failed_passes(schema_validation_failure_list: list[str]) -> list[str]:
    failed: list[str] = []
    for failure in schema_validation_failure_list:
        match = _PASS_FROM_FAILURE_RE.search(failure)
        if match:
            failed.append(match.group(1))
    return failed


def format_run_advice_summary(report: RunAdvisorReport) -> str:
    lines = [
        "Run Advisor Summary",
        f"Runs analyzed: {report.runs_analyzed} (complete={report.complete_runs}, incomplete={report.incomplete_runs})",
        "Crash point distribution:",
    ]
    if report.crash_point_distribution:
        for pass_name, count in sorted(report.crash_point_distribution.items()):
            lines.append(f"  - {pass_name}: {count}")
    else:
        lines.append("  - none")

    lines.append("Speed recommendations:")
    lines.extend([f"  - {item}" for item in report.speed_recommendations] or ["  - none"])
    lines.append("Accuracy recommendations:")
    lines.extend([f"  - {item}" for item in report.accuracy_recommendations] or ["  - none"])
    lines.append("Suggested config overrides:")
    lines.append(f"  - {json.dumps(report.suggested_config, ensure_ascii=False)}")
    if report.warnings:
        lines.append("Warnings:")
        lines.extend([f"  - {item}" for item in report.warnings])
    return "\n".join(lines)


def generate_run_advice(runs_dir: Path) -> RunAdvisorReport:
    run_dirs = _run_dirs(runs_dir)
    if len(run_dirs) < 2:
        report = RunAdvisorReport(
            runs_analyzed=len(run_dirs),
            complete_runs=0,
            incomplete_runs=len(run_dirs),
            crash_point_distribution={},
            speed_recommendations=[],
            accuracy_recommendations=[],
            suggested_config={},
            warnings=["Insufficient run history — minimum 2 runs required for reliable recommendations"],
        )
        print(format_run_advice_summary(report))
        return report

    crash_point_distribution: dict[str, int] = {}
    warnings: list[str] = []
    speed_recommendations: list[str] = []
    accuracy_recommendations: list[str] = []

    complete_runs = 0
    incomplete_runs = 0
    fallback_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    backend_fastest: dict[str, tuple[float, int]] = {}
    pass_times_05_06 = 0.0
    total_times = 0.0
    unsupported_seen = False
    unknown_fact_document_types: set[str] = set()
    schema_fail_pass_counts: dict[str, int] = {}
    no_chunk_error_averages: list[float] = []

    for run_dir in run_dirs:
        timing_payload, timing_ok = _safe_load_json(run_dir / "timing.json")
        report_payload, report_ok = _safe_load_json(run_dir / "report.json")
        complete = timing_ok and report_ok and isinstance(timing_payload, dict) and isinstance(report_payload, dict)

        if complete:
            complete_runs += 1
            backend = str(report_payload.get("backend", "unknown"))
            duration = float(report_payload.get("total_duration_seconds", timing_payload.get("total_pipeline_seconds", 0.0)) or 0.0)
            if duration > 0:
                total_times += duration
            passes = timing_payload.get("passes", {}) if isinstance(timing_payload, dict) else {}
            if isinstance(passes, dict):
                pass_times_05_06 += float(passes.get("05_assumption_audit", 0.0) or 0.0)
                pass_times_05_06 += float(passes.get("06_evidence_audit", 0.0) or 0.0)

            per_pass_status = report_payload.get("per_pass_status", {})
            if isinstance(per_pass_status, dict):
                for pass_name, status in per_pass_status.items():
                    status_counts[pass_name] = status_counts.get(pass_name, 0) + 1
                    if status == "completed_with_fallback":
                        fallback_counts[pass_name] = fallback_counts.get(pass_name, 0) + 1

            unsupported_claim_count = int(report_payload.get("unsupported_claim_count", 0) or 0)
            if unsupported_claim_count > 0:
                unsupported_seen = True

            document_type = str(report_payload.get("document_type", "unknown"))
            schema_validation_failures = report_payload.get("schema_validation_failure_list", [])
            if isinstance(schema_validation_failures, list):
                failure_text = "\n".join(str(item) for item in schema_validation_failures)
                if "E_SYNTH_UNKNOWN_AS_FACT" in failure_text:
                    unknown_fact_document_types.add(document_type)
                for pass_name in _extract_failed_passes([str(item) for item in schema_validation_failures]):
                    schema_fail_pass_counts[pass_name] = schema_fail_pass_counts.get(pass_name, 0) + 1

                has_chunk_range_error = any("E_CHUNK_RANGE_INVALID" in str(item) for item in schema_validation_failures)
                if not has_chunk_range_error:
                    _chunk_count, avg_chunk_words = _extract_chunk_stats(run_dir)
                    if avg_chunk_words:
                        no_chunk_error_averages.append(avg_chunk_words)

            if "parallel_chunks" in report_payload:
                candidate_parallel = int(report_payload.get("parallel_chunks") or 0)
                if candidate_parallel > 0:
                    current = backend_fastest.get(backend)
                    if current is None or duration < current[0]:
                        backend_fastest[backend] = (duration, candidate_parallel)
        else:
            incomplete_runs += 1
            last_pass = _extract_last_successful_pass(run_dir / "passes") or "unknown"
            crash_point_distribution[last_pass] = crash_point_distribution.get(last_pass, 0) + 1
            checkpoint_payload, checkpoint_ok = _safe_load_json(run_dir / "passes" / "checkpoint.json")
            if checkpoint_ok and isinstance(checkpoint_payload, dict):
                checkpoint_pass = checkpoint_payload.get("last_completed_pass")
                if isinstance(checkpoint_pass, str) and checkpoint_pass != "unknown":
                    # Prefer the checkpoint's pass name over filesystem scan
                    crash_point_distribution[last_pass] = crash_point_distribution.get(last_pass, 1) - 1
                    if crash_point_distribution.get(last_pass, 0) <= 0:
                        crash_point_distribution.pop(last_pass, None)
                    crash_point_distribution[checkpoint_pass] = crash_point_distribution.get(checkpoint_pass, 0) + 1
                byte_size = checkpoint_payload.get("byte_size")
                url_count = checkpoint_payload.get("url_count")
                if isinstance(byte_size, (int, float)) and isinstance(url_count, int):
                    warnings.append(
                        f"Run {run_dir.name} incomplete near '{last_pass}' (checkpoint byte_size={byte_size}, url_count={url_count})."
                    )

            # Learn from partial report.json written by interrupted runs
            if report_ok and isinstance(report_payload, dict) and report_payload.get("incomplete"):
                backend = str(report_payload.get("backend", "unknown"))
                duration = float(report_payload.get("total_duration_seconds", 0.0) or 0.0)
                if duration > 0:
                    total_times += duration
                per_pass_status = report_payload.get("per_pass_status", {})
                if isinstance(per_pass_status, dict):
                    for pass_name, status in per_pass_status.items():
                        if status not in ("not_started",):
                            status_counts[pass_name] = status_counts.get(pass_name, 0) + 1
                        if status == "completed_with_fallback":
                            fallback_counts[pass_name] = fallback_counts.get(pass_name, 0) + 1
                schema_validation_failures = report_payload.get("schema_validation_failure_list", [])
                if isinstance(schema_validation_failures, list):
                    for pass_name in _extract_failed_passes([str(item) for item in schema_validation_failures]):
                        schema_fail_pass_counts[pass_name] = schema_fail_pass_counts.get(pass_name, 0) + 1
                error_msg = report_payload.get("error")
                if isinstance(error_msg, str) and error_msg:
                    warnings.append(f"Run {run_dir.name} failed with error: {error_msg[:200]}")

    if complete_runs > 0:
        for pass_name, fallback_count in sorted(fallback_counts.items()):
            observed = status_counts.get(pass_name, 0)
            if observed >= 2 and fallback_count == observed:
                speed_recommendations.append(
                    f"Pass '{pass_name}' consistently returns completed_with_fallback; treat it as an instability candidate."
                )

        if total_times > 0 and (pass_times_05_06 / total_times) > 0.40:
            speed_recommendations.append("Enable fast=True because passes 05/06 account for more than 40% of total runtime.")

        if backend_fastest:
            for backend, (_duration, parallel_value) in sorted(backend_fastest.items()):
                speed_recommendations.append(
                    f"For backend '{backend}', use parallel_chunks={parallel_value} (fastest completed historical run)."
                )

        if no_chunk_error_averages:
            median_avg = float(median(no_chunk_error_averages))
            target_min = max(200, int(median_avg * 0.8))
            target_max = max(target_min + 100, int(median_avg * 1.2))
            speed_recommendations.append(
                f"Use chunk_target_min_words={target_min} and chunk_target_max_words={target_max} based on runs without E_CHUNK_RANGE_INVALID."
            )

        if unsupported_seen:
            accuracy_recommendations.append("Set strict=True because unsupported claims were observed in historical runs.")

        if unknown_fact_document_types:
            accuracy_recommendations.append(
                "E_SYNTH_UNKNOWN_AS_FACT appeared for document_type(s): "
                + ", ".join(sorted(unknown_fact_document_types))
                + "."
            )

        for pass_name, count in sorted(schema_fail_pass_counts.items()):
            if count >= 2:
                accuracy_recommendations.append(
                    f"Schema validation repeatedly failed in pass '{pass_name}'; review prompts/{pass_name}.txt and its schema."
                )

    for pass_name, count in sorted(crash_point_distribution.items()):
        if pass_name != "unknown" and count >= 2:
            accuracy_recommendations.append(
                f"Pass '{pass_name}' is a known failure pass across {count} incomplete runs; review its prompt and schema."
            )

    defaults = asdict(PipelineConfig())
    suggested_config: dict[str, Any] = {}
    chunk_match = next(
        (
            recommendation
            for recommendation in speed_recommendations
            if recommendation.startswith("Use chunk_target_min_words=")
        ),
        None,
    )
    if chunk_match:
        values = re.findall(r"chunk_target_(?:min|max)_words=(\d+)", chunk_match)
        if len(values) == 2:
            min_words, max_words = int(values[0]), int(values[1])
            if min_words != defaults["chunk_target_min_words"]:
                suggested_config["chunk_target_min_words"] = min_words
            if max_words != defaults["chunk_target_max_words"]:
                suggested_config["chunk_target_max_words"] = max_words

    report = RunAdvisorReport(
        runs_analyzed=len(run_dirs),
        complete_runs=complete_runs,
        incomplete_runs=incomplete_runs,
        crash_point_distribution=crash_point_distribution,
        speed_recommendations=speed_recommendations,
        accuracy_recommendations=accuracy_recommendations,
        suggested_config=suggested_config,
        warnings=warnings,
    )
    print(format_run_advice_summary(report))
    return report
