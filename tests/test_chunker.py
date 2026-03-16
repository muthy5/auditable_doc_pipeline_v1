from __future__ import annotations

import pytest

from src.chunker import chunk_document
from src.exceptions import ChunkingError


def test_chunker_splits_simple_text_into_multiple_chunks() -> None:
    text = "Section One\n\n" + " ".join(["alpha"] * 20) + "\n\nSection Two\n\n" + " ".join(["beta"] * 20)
    doc = {"doc_id": "doc_test", "text": text}

    chunks = chunk_document(doc=doc, target_min_words=5, target_max_words=15, hard_max_words=20, overlap_max_words=3)

    assert len(chunks) >= 2
    assert chunks[0]["chunk_id"] == "chunk_0001"
    assert chunks[1]["chunk_id"] == "chunk_0002"


def test_chunker_raises_error_on_empty_text() -> None:
    with pytest.raises(ChunkingError):
        chunk_document(doc={"doc_id": "d", "text": "   "}, target_min_words=5, target_max_words=15, hard_max_words=20, overlap_max_words=0)
