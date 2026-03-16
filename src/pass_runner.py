from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .exceptions import PassValidationError
from .llm_interface import LocalLLMBackend
from .prompts import load_prompt
from .schemas import load_schema


class PassRunner:
    """Runs individual model passes and validates their JSON artifacts."""

    def __init__(self, backend: LocalLLMBackend, prompts_dir: Path, schemas_dir: Path) -> None:
        """Initialize a pass runner.

        Args:
            backend: Backend implementation that produces JSON outputs.
            prompts_dir: Directory containing pass prompts.
            schemas_dir: Directory containing pass schemas.
        """
        self.backend = backend
        self.prompts_dir = prompts_dir
        self.schemas_dir = schemas_dir

    def validate_with_schema(self, schema_filename: str, payload: Dict[str, Any]) -> None:
        """Validate payload against a schema.

        Args:
            schema_filename: Filename of the JSON schema.
            payload: Data to validate.

        Raises:
            PassValidationError: If validation fails.
        """
        schema = load_schema(self.schemas_dir, schema_filename)
        try:
            Draft202012Validator(schema).validate(payload)
        except ValidationError as exc:
            raise PassValidationError(f"Schema validation failed for {schema_filename}: {exc}") from exc

    def run_model_pass(
        self,
        pass_name: str,
        prompt_filename: str,
        schema_filename: str,
        input_payload: Dict[str, Any],
        output_path: Path,
    ) -> Dict[str, Any]:
        """Execute a model pass and persist the validated output.

        Args:
            pass_name: Logical pass name.
            prompt_filename: Prompt file to load.
            schema_filename: Output schema filename.
            input_payload: Input JSON payload.
            output_path: Output artifact path.

        Returns:
            Validated output object.

        Raises:
            PassValidationError: If the output fails schema validation.
        """
        prompt_text = load_prompt(self.prompts_dir, prompt_filename)
        schema = load_schema(self.schemas_dir, schema_filename)
        output = self.backend.generate_json(
            pass_name=pass_name, prompt_text=prompt_text, payload=input_payload, schema=schema
        )
        try:
            Draft202012Validator(schema).validate(output)
        except ValidationError as exc:
            raise PassValidationError(f"Pass {pass_name} produced invalid output: {exc}") from exc
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        return output

    def write_validated_json(
        self,
        schema_filename: str,
        payload: Dict[str, Any],
        output_path: Path,
    ) -> Dict[str, Any]:
        """Validate and write a JSON artifact.

        Args:
            schema_filename: Schema filename used for validation.
            payload: JSON payload to write.
            output_path: Destination artifact path.

        Returns:
            The original validated payload.
        """
        self.validate_with_schema(schema_filename=schema_filename, payload=payload)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload
