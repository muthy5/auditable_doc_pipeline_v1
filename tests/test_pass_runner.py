from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.exceptions import SchemaLoadError
from src.llm_interface import LocalLLMBackend
from src.pass_runner import PassRunner
from src.schemas import load_schema


class _InvalidOutputBackend(LocalLLMBackend):
    def generate_json(self, pass_name, prompt_text, payload, schema=None, model_override=None):
        return {"title": 999, "extra": "ignored"}


def test_run_model_pass_non_strict_returns_schema_fallback(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    schemas_dir = tmp_path / "schemas"
    prompts_dir.mkdir()
    schemas_dir.mkdir()
    (prompts_dir / "test_prompt.txt").write_text("Prompt", encoding="utf-8")
    (schemas_dir / "test.schema.json").write_text('{"type":"object","required":["title"],"properties":{"title":{"type":"string"}}}', encoding="utf-8")

    runner = PassRunner(backend=_InvalidOutputBackend(), prompts_dir=prompts_dir, schemas_dir=schemas_dir)
    output_path = tmp_path / "out" / "result.json"
    output = runner.run_model_pass("test", "test_prompt.txt", "test.schema.json", {"input": "data"}, output_path, strict=False)

    assert output["title"] == ""
    assert output["_schema_validation_failed"] is True
    assert output["_fallback_generated"] is True
    assert Path(output["_failed_output_path"]).exists()
    assert output_path.exists()
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["_fallback_generated"] is True


def test_write_validated_json_non_strict_returns_fallback(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schemas_dir.mkdir()
    (schemas_dir / "test.schema.json").write_text('{"type":"object","required":["ok"],"properties":{"ok":{"type":"boolean"}}}', encoding="utf-8")
    runner = PassRunner(backend=_InvalidOutputBackend(), prompts_dir=tmp_path, schemas_dir=schemas_dir)
    output_path = tmp_path / "x.json"
    output = runner.write_validated_json("test.schema.json", {"ok": "nope"}, output_path, "p", strict=False)
    assert output["ok"] is False
    assert output["_schema_validation_failed"] is True
    assert output["_fallback_generated"] is True
    assert output_path.exists()


def test_load_schema_raises_schema_load_error_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SchemaLoadError):
        load_schema(tmp_path, "missing.schema.json")
