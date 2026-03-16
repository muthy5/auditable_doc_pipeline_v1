from __future__ import annotations

from pathlib import Path

import pytest
from src.llm_interface import LocalLLMBackend
from src.pass_runner import PassRunner
from src.schemas import load_schema
from src.exceptions import SchemaLoadError


class _InvalidOutputBackend(LocalLLMBackend):
    def generate_json(self, pass_name, prompt_text, payload, schema=None):
        return {"title": 999, "extra": "ignored"}


def test_run_model_pass_non_strict_returns_schema_fallback(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    schemas_dir = tmp_path / "schemas"
    prompts_dir.mkdir()
    schemas_dir.mkdir()
    (prompts_dir / "test_prompt.txt").write_text("Prompt", encoding="utf-8")
    (schemas_dir / "test.schema.json").write_text('{"type":"object","required":["title"],"properties":{"title":{"type":"string"}}}', encoding="utf-8")

    runner = PassRunner(backend=_InvalidOutputBackend(), prompts_dir=prompts_dir, schemas_dir=schemas_dir)
    output = runner.run_model_pass("test", "test_prompt.txt", "test.schema.json", {"input": "data"}, tmp_path / "out" / "result.json", strict=False)

    assert output["title"] == ""
    assert output["_schema_validation_failed"] is True


def test_write_validated_json_non_strict_returns_fallback(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schemas_dir.mkdir()
    (schemas_dir / "test.schema.json").write_text('{"type":"object","required":["ok"],"properties":{"ok":{"type":"boolean"}}}', encoding="utf-8")
    runner = PassRunner(backend=_InvalidOutputBackend(), prompts_dir=tmp_path, schemas_dir=schemas_dir)
    output = runner.write_validated_json("test.schema.json", {"ok": "nope"}, tmp_path / "x.json", "p", strict=False)
    assert output == {"ok": False, "_schema_validation_failed": True}


def test_load_schema_raises_schema_load_error_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SchemaLoadError):
        load_schema(tmp_path, "missing.schema.json")
