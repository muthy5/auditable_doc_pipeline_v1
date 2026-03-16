from pathlib import Path

from src.llm_interface import LocalLLMBackend
from src.pass_runner import PassRunner


class _InvalidOutputBackend(LocalLLMBackend):
    def generate_json(self, pass_name, prompt_text, payload, schema=None):
        return {"title": 999, "extra": "ignored"}


def test_run_model_pass_non_strict_returns_schema_fallback(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    schemas_dir = tmp_path / "schemas"
    prompts_dir.mkdir()
    schemas_dir.mkdir()

    (prompts_dir / "test_prompt.txt").write_text("Prompt", encoding="utf-8")
    (schemas_dir / "test.schema.json").write_text(
        """
        {
          "type": "object",
          "required": ["title", "tags", "meta", "count", "enabled"],
          "properties": {
            "title": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "meta": {
              "type": "object",
              "required": ["author"],
              "properties": {
                "author": {"type": "string"}
              }
            },
            "count": {"type": "integer"},
            "enabled": {"type": "boolean"}
          }
        }
        """,
        encoding="utf-8",
    )

    runner = PassRunner(backend=_InvalidOutputBackend(), prompts_dir=prompts_dir, schemas_dir=schemas_dir)
    output = runner.run_model_pass(
        pass_name="test",
        prompt_filename="test_prompt.txt",
        schema_filename="test.schema.json",
        input_payload={"input": "data"},
        output_path=tmp_path / "out" / "result.json",
        strict=False,
    )

    assert output["title"] == ""
    assert output["tags"] == []
    assert output["meta"] == {"author": ""}
    assert output["count"] == 0
    assert output["enabled"] is False
    assert output["_schema_validation_failed"] is True
    assert (tmp_path / "out" / "result.failed.json").exists()
