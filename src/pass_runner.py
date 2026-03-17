from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .exceptions import PassSchemaValidationError
from .llm_interface import LocalLLMBackend
from .prompts import load_prompt
from .schemas import load_schema

LOGGER = logging.getLogger(__name__)


class PassRunner:
    """Execute and validate pipeline passes."""

    def __init__(self, backend: LocalLLMBackend, prompts_dir: Path, schemas_dir: Path) -> None:
        """Initialize pass runner state."""
        self.backend = backend
        self.prompts_dir = prompts_dir
        self.schemas_dir = schemas_dir
        self.validation_failures: list[str] = []
        self.timings: dict[str, float] = {}
        self.pass_outcomes: dict[str, dict[str, Any]] = {}

    def _record_timing(self, pass_name: str, start_time: float, end_time: float) -> None:
        """Store elapsed seconds for a pass."""
        self.timings[pass_name] = end_time - start_time

    def _record_outcome(self, pass_name: str, status: str, detail: str | None = None, **extra: Any) -> None:
        outcome: dict[str, Any] = {"status": status}
        if detail:
            outcome["detail"] = detail
        outcome.update(extra)
        self.pass_outcomes[pass_name] = outcome

    def mark_pass_status(self, pass_name: str, status: str, detail: str | None = None, **extra: Any) -> None:
        """Allow pipeline orchestration code to record non-runner pass states."""
        self._record_outcome(pass_name, status, detail, **extra)

    def write_timings(self, output_path: Path, total_time_s: float) -> None:
        """Write timing summary to disk."""
        payload = {"passes": self.timings, "total_pipeline_seconds": total_time_s}
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _handle_schema_failure(
        self,
        pass_name: str,
        payload: dict[str, Any],
        output_path: Path,
        error: ValidationError,
        strict: bool,
    ) -> tuple[Path, str]:
        failed_path = output_path.with_suffix(".failed.json")
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        failed_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        message = (
            f"Schema validation failed in pass '{pass_name}': {error.message}. "
            f"Failed output written to {failed_path}."
        )
        LOGGER.error(message)
        self.validation_failures.append(message)
        if strict:
            self._record_outcome(pass_name, "failed", message, failed_output_path=str(failed_path), validation_error=error.message)
            raise PassSchemaValidationError(pass_name=pass_name, message=message) from error
        LOGGER.warning("Continuing because strict mode is disabled.")
        return failed_path, error.message

    def validate_with_schema(self, schema_filename: str, payload: dict[str, Any]) -> None:
        """Validate payload against schema without writing output."""
        Draft202012Validator(load_schema(self.schemas_dir, schema_filename)).validate(payload)

    def _build_fallback_from_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Build a minimal fallback payload from required schema fields."""

        def _default_for_schema(node: dict[str, Any]) -> Any:
            node_type = node.get("type")
            if isinstance(node_type, list):
                node_type = next((item for item in node_type if item != "null"), node_type[0] if node_type else None)
            if node_type == "object" or "properties" in node:
                fallback: dict[str, Any] = {}
                for key in node.get("required", []):
                    fallback[key] = _default_for_schema(node.get("properties", {}).get(key, {}))
                return fallback
            if node_type == "array":
                return []
            if node_type in {"number", "integer"}:
                return 0
            if node_type == "boolean":
                return False
            if node_type == "string":
                return ""
            return {}

        return _default_for_schema(schema)

    def run_model_pass(
        self,
        pass_name: str,
        prompt_filename: str,
        schema_filename: str,
        input_payload: dict[str, Any],
        output_path: Path,
        strict: bool = True,
        model_override: str | None = None,
    ) -> dict[str, Any]:
        """Run one model-backed pass and validate its output."""
        start = time.perf_counter()
        schema = load_schema(self.schemas_dir, schema_filename)
        output = self.backend.generate_json(
            pass_name=pass_name,
            prompt_text=load_prompt(self.prompts_dir, prompt_filename),
            payload=input_payload,
            schema=schema,
            model_override=model_override,
        )
        try:
            Draft202012Validator(schema).validate(output)
        except ValidationError as error:
            failed_path, validation_error = self._handle_schema_failure(pass_name, output, output_path, error, strict)
            fallback = self._build_fallback_from_schema(schema)
            fallback["_schema_validation_failed"] = True
            fallback["_fallback_generated"] = True
            fallback["_failed_output_path"] = str(failed_path)
            fallback["_validation_error"] = validation_error
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(fallback, indent=2, ensure_ascii=False), encoding="utf-8")
            self._record_timing(pass_name, start, time.perf_counter())
            self._record_outcome(
                pass_name,
                "completed_with_fallback",
                "Schema validation failed; fallback artifact generated.",
                failed_output_path=str(failed_path),
                validation_error=validation_error,
            )
            return fallback
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        self._record_timing(pass_name, start, time.perf_counter())
        self._record_outcome(pass_name, "completed")
        return output

    def write_validated_json(
        self,
        schema_filename: str,
        payload: dict[str, Any],
        output_path: Path,
        pass_name: str,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Validate and write JSON payload for non-model passes."""
        start = time.perf_counter()
        schema = load_schema(self.schemas_dir, schema_filename)
        try:
            Draft202012Validator(schema).validate(payload)
        except ValidationError as error:
            failed_path, validation_error = self._handle_schema_failure(pass_name, payload, output_path, error, strict)
            fallback = self._build_fallback_from_schema(schema)
            fallback["_schema_validation_failed"] = True
            fallback["_fallback_generated"] = True
            fallback["_failed_output_path"] = str(failed_path)
            fallback["_validation_error"] = validation_error
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(fallback, indent=2, ensure_ascii=False), encoding="utf-8")
            self._record_timing(pass_name, start, time.perf_counter())
            self._record_outcome(
                pass_name,
                "completed_with_fallback",
                "Schema validation failed; fallback artifact generated.",
                failed_output_path=str(failed_path),
                validation_error=validation_error,
            )
            return fallback
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self._record_timing(pass_name, start, time.perf_counter())
        self._record_outcome(pass_name, "completed")
        return payload
