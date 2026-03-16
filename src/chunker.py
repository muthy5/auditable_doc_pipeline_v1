from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .exceptions import ChunkingError

_HEADING_RE = re.compile(
    r"^(?:#{1,6}\s+.+|[A-Z][A-Z0-9\s\-]*:|(?:\d+(?:\.\d+)*)[\)\.]?\s+.+|(?:Goal|Ingredients|Materials|Plan|Steps|Notes|Expected Output|Objective):\s*)$"
)


@dataclass
class Section:
    """A contiguous section of the source document."""

    heading: str | None
    start_char: int
    end_char: int
    text: str


def _iter_lines_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Return document lines with start and end offsets."""
    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        start = cursor
        end = cursor + len(line)
        spans.append((line, start, end))
        cursor = end
    return spans or [("", 0, 0)]


def _is_heading(line: str) -> bool:
    """Return True when a line should be treated as a heading."""
    stripped = line.strip()
    return bool(stripped) and bool(_HEADING_RE.match(stripped))


def _last_n_words(text: str, n: int) -> str:
    """Return the last n whitespace-delimited words from text."""
    words = text.split()
    return " ".join(words[-n:]) if n > 0 else ""


def split_into_sections(text: str) -> list[Section]:
    """Split text into heading-based sections."""
    lines = _iter_lines_with_spans(text)
    sections: list[Section] = []
    current_heading: str | None = None
    current_start = 0

    for i, (line, start, _end) in enumerate(lines):
        if i == 0 and _is_heading(line):
            current_heading = line.strip().rstrip(":")
            current_start = 0
            continue
        if _is_heading(line) and start != 0:
            section_text = text[current_start:start]
            if section_text.strip():
                sections.append(Section(current_heading, current_start, start, section_text))
            current_heading = line.strip().rstrip(":")
            current_start = start

    final_text = text[current_start:]
    if final_text.strip():
        sections.append(Section(current_heading, current_start, len(text), final_text))
    return sections or [Section(None, 0, len(text), text)]


def _split_section_by_paragraphs(
    section: Section,
    doc_id: str,
    start_index: int,
    target_min_words: int,
    _target_max_words: int,
    hard_max_words: int,
    overlap_max_words: int,
) -> list[dict[str, Any]]:
    """Split a section into paragraph-based chunks with overlap metadata."""
    paragraphs = [p for p in re.split(r"\n\s*\n", section.text) if p.strip()] or [section.text]

    chunks: list[dict[str, Any]] = []
    buffer: list[str] = []
    chunk_index = start_index
    cursor = section.start_char

    def flush() -> None:
        nonlocal buffer, chunk_index
        if not buffer:
            return
        chunk_text = "\n\n".join(buffer).strip()
        word_count = len(chunk_text.split())
        end_char = cursor
        start_char = max(section.start_char, end_char - len(chunk_text))
        chunks.append(
            {
                "chunk_id": f"chunk_{chunk_index:04d}",
                "doc_id": doc_id,
                "index": chunk_index,
                "heading_path": [section.heading] if section.heading else [],
                "start_char": start_char,
                "end_char": end_char,
                "start_word": 1,
                "end_word": word_count,
                "overlap_prev_words": 0,
                "overlap_next_words": 0,
                "text": chunk_text,
            }
        )
        chunk_index += 1
        buffer = []

    for para in paragraphs:
        para_words = len(para.split())
        existing_words = len(" ".join(buffer).split()) if buffer else 0
        if buffer and existing_words + para_words > hard_max_words:
            flush()
        buffer.append(para)
        cursor += len(para)
        if len(" ".join(buffer).split()) >= target_min_words:
            flush()
        cursor += 2
    flush()

    for idx in range(1, len(chunks)):
        prev_text = chunks[idx - 1]["text"]
        overlap_text = _last_n_words(prev_text, overlap_max_words)
        overlap_words = len(overlap_text.split())
        if overlap_words:
            chunks[idx]["text"] = f"{overlap_text}\n\n{chunks[idx]['text']}"
            chunks[idx]["start_word"] = 1
            chunks[idx]["end_word"] = len(chunks[idx]["text"].split())
            chunks[idx]["overlap_prev_words"] = overlap_words
            chunks[idx - 1]["overlap_next_words"] = overlap_words
    return chunks


def chunk_document(
    doc: dict[str, Any],
    target_min_words: int,
    target_max_words: int,
    hard_max_words: int,
    overlap_max_words: int,
) -> list[dict[str, Any]]:
    """Chunk a document into bounded sections for downstream passes."""
    text = doc["text"]
    if not text.strip():
        raise ChunkingError("Document text is empty.")
    sections = split_into_sections(text)
    chunks: list[dict[str, Any]] = []
    chunk_index = 1
    for section in sections:
        section_words = len(section.text.split())
        if section_words <= target_max_words:
            chunks.append(
                {
                    "chunk_id": f"chunk_{chunk_index:04d}",
                    "doc_id": doc["doc_id"],
                    "index": chunk_index,
                    "heading_path": [section.heading] if section.heading else [],
                    "start_char": section.start_char,
                    "end_char": section.end_char,
                    "start_word": 1,
                    "end_word": section_words,
                    "overlap_prev_words": 0,
                    "overlap_next_words": 0,
                    "text": section.text.strip(),
                }
            )
            chunk_index += 1
            continue
        split_chunks = _split_section_by_paragraphs(
            section,
            doc_id=doc["doc_id"],
            start_index=chunk_index,
            target_min_words=target_min_words,
            _target_max_words=target_max_words,
            hard_max_words=hard_max_words,
            overlap_max_words=overlap_max_words,
        )
        chunks.extend(split_chunks)
        chunk_index += len(split_chunks)
    return chunks
