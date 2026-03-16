from **future** import annotations

import json
from pathlib import Path
from typing import Any, Dict

from jsonschema import Draft202012Validator

from .llm_interface import LocalLLMBackend
from .prompts import load_prompt
from .schemas import load_schema

class PassRunner:
def **init**(self, backend: LocalLLMBackend, prompts_dir: Path, schemas_dir: Path) -> None:
self.backend = backend
self.prompts_dir = prompts_dir
self.schemas_dir = schemas_dir

```
def validate_with_schema(self, schema_filename: str, payload: Dict[str, Any]) -> None:
    schema = load_schema(self.schemas_dir, schema_filename)
    Draft202012Validator(schema).validate(payload)

def run_model_pass(
    self,
    pass_name: str,
    prompt_filename: str,
    schema_filename: str,
    input_payload: Dict[str, Any],
    output_path: Path,
) -> Dict[str, Any]:
    prompt_text = load_prompt(self.prompts_dir, prompt_filename)
    schema = load_schema(self.schemas_dir, schema_filename)
    output = self.backend.generate_json(
        pass_name=pass_name, prompt_text=prompt_text, payload=input_payload, schema=schema,
    )
    Draft202012Validator(schema).validate(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output

def write_validated_json(
    self,
    schema_filename: str,
    payload: Dict[str, Any],
    output_path: Path,
) -> Dict[str, Any]:
    self.validate_with_schema(schema_filename=schema_filename, payload=payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
```
