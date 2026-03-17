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


def test_paragraph_splitting_respects_target_max_words() -> None:
    paragraphs = [" ".join([f"p{i}"] * 9) for i in range(1, 6)]
    doc = {"doc_id": "doc_test", "text": "\n\n".join(paragraphs)}

    chunks = chunk_document(doc=doc, target_min_words=5, target_max_words=15, hard_max_words=20, overlap_max_words=0)

    assert len(chunks) >= 3
    assert all(len(chunk["text"].split()) <= 15 for chunk in chunks)


def test_oversized_paragraph_respects_hard_max_words() -> None:
    long_para = " ".join(["alpha"] * 53)
    doc = {"doc_id": "doc_test", "text": long_para}

    chunks = chunk_document(doc=doc, target_min_words=5, target_max_words=20, hard_max_words=25, overlap_max_words=0)

    assert len(chunks) >= 3
    assert all(len(chunk["text"].split()) <= 25 for chunk in chunks)


def test_oversized_paragraph_preserves_source_spans() -> None:
    text = "Para1  with   extra   spaces and\ttabs."
    doc = {"doc_id": "doc_test", "text": text}

    chunks = chunk_document(doc=doc, target_min_words=1, target_max_words=3, hard_max_words=4, overlap_max_words=0)

    for chunk in chunks:
        source_slice = text[chunk["start_char"] : chunk["end_char"]]
        assert chunk["text"] == source_slice.strip()
