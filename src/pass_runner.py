from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .llm_interface import LocalLLMBackend
from .prompts import load_prompt
from .schemas import load_schema


LOGGER = logging.getLogger(__name__)


class PassSchemaValidationError(Exception):
    def __init__(self, pass_name: str, message: str) -> None:
        super().__init__(message)
        self.pass_name = pass_name


class PassRunner:
    def __init__(self, backend: LocalLLMBackend, prompts_dir: Path, schemas_dir: Path) -> None:
        self.backend = backend
        self.prompts_dir = prompts_dir
        self.schemas_dir = schemas_dir
        self.validation_failures: list[str] = []

    def _handle_schema_failure(
        self,
        pass_name: str,
        payload: Dict[str, Any],
        output_path: Path,
        error: ValidationError,
        strict: bool,
    ) -> None:
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
            raise PassSchemaValidationError(pass_name=pass_name, message=message) from error
        LOGGER.warning("Continuing because strict mode is disabled.")

    def validate_with_schema(self, schema_filename: str, payload: Dict[str, Any]) -> None:
        schema = load_schema(self.schemas_dir, schema_filename)
        Draft202012Validator(schema).validate(payload)

    def _build_fallback_from_schema(self, schema: dict) -> dict:
        def _default_for_schema(node: Dict[str, Any]) -> Any:
            node_type = node.get("type")

            if isinstance(node_type, list):
                node_type = next((item for item in node_type if item != "null"), node_type[0] if node_type else None)

            if node_type == "object" or "properties" in node:
                fallback: Dict[str, Any] = {}
                properties = node.get("properties", {})
                for key in node.get("required", []):
                    child_schema = properties.get(key, {})
                    fallback[key] = _default_for_schema(child_schema)
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
        input_payload: Dict[str, Any],
        output_path: Path,
        strict: bool = True,
    ) -> Dict[str, Any]:
        prompt_text = load_prompt(self.prompts_dir, prompt_filename)
        schema = load_schema(self.schemas_dir, schema_filename)
        output = self.backend.generate_json(
            pass_name=pass_name, prompt_text=prompt_text, payload=input_payload, schema=schema
        )
        try:
            Draft202012Validator(schema).validate(output)
        except ValidationError as error:
            self._handle_schema_failure(
                pass_name=pass_name,
                payload=output,
                output_path=output_path,
                error=error,
                strict=strict,
            )
            fallback = self._build_fallback_from_schema(schema)
            fallback["_schema_validation_failed"] = True
            return fallback
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        return output

    def write_validated_json(
        self,
        schema_filename: str,
        payload: Dict[str, Any],
        output_path: Path,
        pass_name: str,
        strict: bool = True,
    ) -> Dict[str, Any]:
        schema = load_schema(self.schemas_dir, schema_filename)
        try:
            Draft202012Validator(schema).validate(payload)
        except ValidationError as error:
            self._handle_schema_failure(
                pass_name=pass_name,
                payload=payload,
                output_path=output_path,
                error=error,
                strict=strict,
            )
            return payload
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload
