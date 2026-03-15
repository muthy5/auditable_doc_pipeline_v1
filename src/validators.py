from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from jsonschema import Draft202012Validator


class ValidationErrorCode:
    JSON_INVALID = "E_JSON_INVALID"
    SCHEMA_INVALID = "E_SCHEMA_INVALID"
    REQUIRED_FIELD_MISSING = "E_REQUIRED_FIELD_MISSING"
    EMPTY_CHUNK_TEXT = "E_EMPTY_CHUNK_TEXT"
    CHUNK_RANGE_INVALID = "E_CHUNK_RANGE_INVALID"
    SYNTH_UNSUPPORTED_CLAIM = "E_SYNTH_UNSUPPORTED_CLAIM"
    SYNTH_UNKNOWN_AS_FACT = "E_SYNTH_UNKNOWN_AS_FACT"
    SYNTH_MISSING_REQUIRED_SECTION = "E_SYNTH_MISSING_REQUIRED_SECTION"
    BLOCKING_GAP_NOT_SURFACED = "E_BLOCKING_GAP_NOT_SURFACED"
    BLOCKING_DEPENDENCY_IGNORED = "E_BLOCKING_DEPENDENCY_IGNORED"
    VALIDATION_CONTRADICTION = "E_VALIDATION_CONTRADICTION"
    CLAIM_STATUS_INVALID = "E_CLAIM_STATUS_INVALID"
    PROVENANCE_MISSING = "E_PROVENANCE_MISSING"
    RUN_BLOCKED = "E_RUN_BLOCKED"


def validate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    errors: List[Dict[str, str]] = []
    for chunk in chunks:
        if not chunk.get("text", "").strip():
            errors.append(
                {
                    "code": ValidationErrorCode.EMPTY_CHUNK_TEXT,
                    "message": f"Chunk {chunk.get('chunk_id')} has empty text.",
                }
            )
        start_char = chunk.get("start_char")
        end_char = chunk.get("end_char")
        if not isinstance(start_char, int) or not isinstance(end_char, int) or start_char < 0 or end_char < start_char:
            errors.append(
                {
                    "code": ValidationErrorCode.CHUNK_RANGE_INVALID,
                    "message": f"Chunk {chunk.get('chunk_id')} has invalid character offsets.",
                }
            )
    return errors


def _collect_supportable_ids(
    schema_audit: Dict[str, Any],
    dependency_audit: Dict[str, Any],
    assumption_audit: Dict[str, Any],
    evidence_audit: Dict[str, Any],
) -> Set[str]:
    ids: Set[str] = set()
    for item in schema_audit.get("blocking_gaps", []):
        ids.add(item["gap_id"])
    for item in schema_audit.get("nonblocking_gaps", []):
        ids.add(item["gap_id"])
    for item in dependency_audit.get("blocking_dependencies", []):
        ids.add(item["dependency_id"])
    for item in dependency_audit.get("ordering_constraints", []):
        if item.get("constraint_id"):
            ids.add(item["constraint_id"])
    for item in assumption_audit.get("implicit_assumptions_found", []):
        ids.add(item["assumption_id"])
    for item in assumption_audit.get("uncertainty_points", []):
        ids.add(item["uncertainty_id"])
    for item in assumption_audit.get("blocking_assumptions", []):
        ids.add(item["assumption_id"])
    for item in evidence_audit.get("claim_registry", []):
        ids.add(item["claim_id"])
    return ids


def _flatten_final_sections(final_answer: Dict[str, Any]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for section_name in [
        "verified_content",
        "missing_information",
        "dependencies",
        "assumptions",
        "uncertainties",
    ]:
        flattened.extend(final_answer.get(section_name, []))
    bottom_line = final_answer.get("bottom_line")
    if isinstance(bottom_line, dict):
        flattened.append(bottom_line)
    return flattened


def validate_final_output(
    synthesis: Dict[str, Any],
    task: Dict[str, Any],
    schema_audit: Dict[str, Any],
    dependency_audit: Dict[str, Any],
    assumption_audit: Dict[str, Any],
    evidence_audit: Dict[str, Any],
    synthesis_schema: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    errors: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []

    if task.get("blocked"):
        errors.append(
            {
                "code": ValidationErrorCode.RUN_BLOCKED,
                "message": "normalize_request marked the run as blocked.",
            }
        )

    if synthesis_schema is not None:
        try:
            Draft202012Validator(synthesis_schema).validate(synthesis)
        except Exception as exc:
            errors.append(
                {
                    "code": ValidationErrorCode.SCHEMA_INVALID,
                    "message": f"Synthesis failed JSON schema validation: {exc}",
                }
            )

    final_answer = synthesis.get("final_answer", {})
    required_sections = [
        "goal",
        "verified_content",
        "missing_information",
        "dependencies",
        "assumptions",
        "uncertainties",
        "organized_structure",
        "bottom_line",
    ]
    for key in required_sections:
        if key not in final_answer:
            errors.append(
                {
                    "code": ValidationErrorCode.SYNTH_MISSING_REQUIRED_SECTION,
                    "message": f"Final answer is missing required section: {key}",
                }
            )

    valid_support_ids = _collect_supportable_ids(
        schema_audit=schema_audit,
        dependency_audit=dependency_audit,
        assumption_audit=assumption_audit,
        evidence_audit=evidence_audit,
    )

    unsupported_statements_found = 0
    for item in _flatten_final_sections(final_answer):
        supports = item.get("support", [])
        if "text" in item and not isinstance(supports, list):
            errors.append(
                {
                    "code": ValidationErrorCode.PROVENANCE_MISSING,
                    "message": "Support field is missing or invalid.",
                }
            )
            continue
        for support_id in supports:
            if support_id not in valid_support_ids:
                unsupported_statements_found += 1
                errors.append(
                    {
                        "code": ValidationErrorCode.SYNTH_UNSUPPORTED_CLAIM,
                        "message": f"Support ID {support_id} is not present upstream.",
                    }
                )

    blocking_ids = [gap["gap_id"] for gap in schema_audit.get("blocking_gaps", [])]
    blocking_ids += [dep["dependency_id"] for dep in dependency_audit.get("blocking_dependencies", [])]
    blocking_ids += [assumption["assumption_id"] for assumption in assumption_audit.get("blocking_assumptions", [])]

    bottom_line_support = final_answer.get("bottom_line", {}).get("support", [])
    for blocking_id in blocking_ids:
        if blocking_id not in bottom_line_support:
            errors.append(
                {
                    "code": ValidationErrorCode.BLOCKING_GAP_NOT_SURFACED,
                    "message": f"Blocking ID {blocking_id} is not surfaced in bottom_line support.",
                }
            )

    bottom_line_text = final_answer.get("bottom_line", {}).get("text", "").lower()
    if dependency_audit.get("blocking_dependencies") and "complete" in bottom_line_text and "not" not in bottom_line_text:
        errors.append(
            {
                "code": ValidationErrorCode.BLOCKING_DEPENDENCY_IGNORED,
                "message": "Final answer claims completeness despite blocking dependencies.",
            }
        )

    allowed_statuses = {"supported", "inferred", "speculative", "unknown"}
    for claim in evidence_audit.get("claim_registry", []):
        if claim.get("status") not in allowed_statuses:
            errors.append(
                {
                    "code": ValidationErrorCode.CLAIM_STATUS_INVALID,
                    "message": f"Invalid claim status: {claim.get('status')}",
                }
            )

    claim_status_by_id = {claim["claim_id"]: claim["status"] for claim in evidence_audit.get("claim_registry", [])}
    for item in final_answer.get("verified_content", []):
        for support_id in item.get("support", []):
            if claim_status_by_id.get(support_id) in {"unknown", "speculative"}:
                errors.append(
                    {
                        "code": ValidationErrorCode.SYNTH_UNKNOWN_AS_FACT,
                        "message": f"Verified content relies on non-settled claim status: {support_id}",
                    }
                )

    if schema_audit.get("nonblocking_gaps"):
        warnings.extend(
            {
                "code": "W_PARTIAL_SECTION_PRESENT",
                "message": gap["reason"],
            }
            for gap in schema_audit["nonblocking_gaps"]
        )

    return {
        "doc_id": synthesis.get("doc_id"),
        "pass": not errors,
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "supported_statements_checked": len(_flatten_final_sections(final_answer)),
            "unsupported_statements_found": unsupported_statements_found,
            "missing_required_sections": sum(1 for key in required_sections if key not in final_answer),
        },
    }
