from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from src.schemas import load_schema


def test_document_schema_accepts_valid_artifact() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    schema = load_schema(repo_root / "schemas", "document.schema.json")

    payload = {
        "doc_id": "doc_001",
        "title": "Sample",
        "source_path": "examples/sample.txt",
        "content_type": "text/plain",
        "encoding": "utf-8",
        "text": "hello world",
        "metadata": {
            "author": None,
            "created_at": None,
            "user_goal": "analyze",
        },
    }

    Draft202012Validator(schema).validate(payload)


def test_document_schema_rejects_invalid_artifact() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    schema = load_schema(repo_root / "schemas", "document.schema.json")

    invalid_payload = {
        "doc_id": "doc_001",
        "text": "missing required fields",
    }

    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(invalid_payload)
