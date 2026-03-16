from __future__ import annotations

from pathlib import Path
from typing import Any

from .llm_interface import LocalLLMBackend
from .prompts import load_prompt
from .schemas import load_schema

SUPPORTED_DOCUMENT_TYPES = {
    "business_plan",
    "legal_contract",
    "project_proposal",
    "medical_protocol",
    "technical_spec",
    "procedural_plan",
}
DEFAULT_DOCUMENT_TYPE = "procedural_plan"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def classify_document_with_metadata(text: str, backend: LocalLLMBackend) -> dict[str, Any]:
    """Run LLM-backed classification and apply fallback selection rules."""
    repo_root = _repo_root()
    output = backend.generate_json(
        pass_name="classify_document",
        prompt_text=load_prompt(repo_root / "prompts", "classify_document.txt"),
        payload={"text": text},
        schema=load_schema(repo_root / "schemas", "classify_document.schema.json"),
    )

    document_type = output.get("document_type")
    confidence = output.get("confidence")
    selected_document_type = (
        document_type
        if document_type in SUPPORTED_DOCUMENT_TYPES and confidence in {"high", "medium"}
        else DEFAULT_DOCUMENT_TYPE
    )
    return {**output, "selected_document_type": selected_document_type}


def classify_document(text: str, backend: LocalLLMBackend) -> str:
    """Classify document text into one supported type; fallback to procedural_plan on uncertainty."""
    return str(classify_document_with_metadata(text, backend).get("selected_document_type", DEFAULT_DOCUMENT_TYPE))
