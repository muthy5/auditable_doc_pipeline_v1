from __future__ import annotations

import importlib.util
from pathlib import Path


def extract_text_from_path(path: Path) -> str:
    """Return best-effort extracted text for supported local file types."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        if importlib.util.find_spec("pypdf") is None:
            return ""
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix == ".docx":
        if importlib.util.find_spec("docx") is None:
            return ""
        from docx import Document

        document = Document(str(path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    return ""

