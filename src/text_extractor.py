from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextExtractionResult:
    """Structured extraction status for supported file types."""

    ok: bool
    text: str
    error_code: str | None
    error_message: str | None
    file_type: str
    is_image_only_pdf: bool = False


def _suffix_to_file_type(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "unknown"


def extract_text_from_path(path: Path) -> TextExtractionResult:
    """Return structured text extraction status for supported local file types."""
    suffix = path.suffix.lower()
    file_type = _suffix_to_file_type(path)

    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            return TextExtractionResult(ok=True, text=text, error_code=None, error_message=None, file_type=file_type)
        return TextExtractionResult(
            ok=False,
            text=text,
            error_code="empty_document",
            error_message="The file contains no extractable text.",
            file_type=file_type,
        )

    if suffix == ".pdf":
        if importlib.util.find_spec("pypdf") is None:
            return TextExtractionResult(
                ok=False,
                text="",
                error_code="missing_pdf_parser",
                error_message="PDF parsing requires the optional 'pypdf' package.",
                file_type=file_type,
            )
        try:
            from pypdf import PdfReader
            from pypdf.errors import PdfReadError

            reader = PdfReader(str(path))
            page_texts = [(page.extract_text() or "") for page in reader.pages]
            text = "\n".join(page_texts)
            if text.strip():
                return TextExtractionResult(ok=True, text=text, error_code=None, error_message=None, file_type=file_type)
            if len(page_texts) > 0:
                return TextExtractionResult(
                    ok=False,
                    text="",
                    error_code="image_only_pdf",
                    error_message="The PDF appears to be scanned/image-only and has no extractable text.",
                    file_type=file_type,
                    is_image_only_pdf=True,
                )
            return TextExtractionResult(
                ok=False,
                text="",
                error_code="empty_document",
                error_message="The file contains no extractable text.",
                file_type=file_type,
            )
        except PdfReadError as exc:
            return TextExtractionResult(
                ok=False,
                text="",
                error_code="corrupted_document",
                error_message=f"PDF is malformed or unreadable: {exc}",
                file_type=file_type,
            )
        except Exception as exc:  # noqa: BLE001
            return TextExtractionResult(
                ok=False,
                text="",
                error_code="parser_error",
                error_message=f"PDF parsing failed: {exc}",
                file_type=file_type,
            )

    if suffix == ".docx":
        if importlib.util.find_spec("docx") is None:
            return TextExtractionResult(
                ok=False,
                text="",
                error_code="missing_docx_parser",
                error_message="DOCX parsing requires the optional 'python-docx' package.",
                file_type=file_type,
            )
        try:
            from docx import Document

            document = Document(str(path))
            text = "\n".join(paragraph.text for paragraph in document.paragraphs)
            if text.strip():
                return TextExtractionResult(ok=True, text=text, error_code=None, error_message=None, file_type=file_type)
            return TextExtractionResult(
                ok=False,
                text=text,
                error_code="empty_document",
                error_message="The file contains no extractable text.",
                file_type=file_type,
            )
        except (ValueError, OSError) as exc:
            return TextExtractionResult(
                ok=False,
                text="",
                error_code="corrupted_document",
                error_message=f"DOCX is malformed or unreadable: {exc}",
                file_type=file_type,
            )
        except Exception as exc:  # noqa: BLE001
            return TextExtractionResult(
                ok=False,
                text="",
                error_code="parser_error",
                error_message=f"DOCX parsing failed: {exc}",
                file_type=file_type,
            )

    return TextExtractionResult(
        ok=False,
        text="",
        error_code="unsupported_file_type",
        error_message=f"Unsupported file extension '{suffix or '<none>'}'.",
        file_type=file_type,
    )
