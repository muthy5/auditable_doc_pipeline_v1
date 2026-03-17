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


def classify_document_with_metadata(text: str, backend: LocalLLMBackend, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Run LLM-backed classification and apply fallback selection rules."""
    resolved_root = repo_root or _repo_root()
    try:
        output = backend.generate_json(
            pass_name="classify_document",
            prompt_text=load_prompt(resolved_root / "prompts", "classify_document.txt"),
            payload={"text": text},
            schema=load_schema(resolved_root / "schemas", "classify_document.schema.json"),
        )
    except Exception as exc:  # noqa: BLE001 - fallback to default type for non-strict pipeline behavior
        output = {
            "document_type": DEFAULT_DOCUMENT_TYPE,
            "confidence": "low",
            "reason": f"Classification fallback due to backend error: {type(exc).__name__}: {exc}",
            "_classification_fallback": True,
        }

    if not isinstance(output, dict):
        output = {
            "document_type": DEFAULT_DOCUMENT_TYPE,
            "confidence": "low",
            "reason": "Classification fallback due to invalid backend payload.",
            "_classification_fallback": True,
        }

    document_type = output.get("document_type")
    confidence = output.get("confidence")
    selected_document_type = (
        document_type
        if document_type in SUPPORTED_DOCUMENT_TYPES and confidence in {"high", "medium"}
        else DEFAULT_DOCUMENT_TYPE
    )
    return {**output, "selected_document_type": selected_document_type}


def classify_document(text: str, backend: LocalLLMBackend, *, repo_root: Path | None = None) -> str:
    """Classify document text into one supported type; fallback to procedural_plan on uncertainty."""
    return str(classify_document_with_metadata(text, backend, repo_root=repo_root).get("selected_document_type", DEFAULT_DOCUMENT_TYPE))
