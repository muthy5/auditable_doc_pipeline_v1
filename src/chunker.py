from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List


_HEADING_RE = re.compile(
    r"^(?:#{1,6}\s+.+|[A-Z][A-Z0-9\s\-:]{3,}|(?:\d+(?:\.\d+)*)[\)\.]?\s+.+|(?:Goal|Ingredients|Materials|Plan|Steps|Notes|Expected Output|Objective):\s*)$"
)


@dataclass
class Section:
    heading: str | None
    start_char: int
    end_char: int
    text: str


def _iter_lines_with_spans(text: str) -> List[tuple[str, int, int]]:
    spans: List[tuple[str, int, int]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        start = cursor
        end = cursor + len(line)
        spans.append((line, start, end))
        cursor = end
    if not spans:
        spans.append(("", 0, 0))
    return spans


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and bool(_HEADING_RE.match(stripped))


def split_into_sections(text: str) -> List[Section]:
    lines = _iter_lines_with_spans(text)
    sections: List[Section] = []

    current_heading: str | None = None
    current_start = 0

    for i, (line, start, _end) in enumerate(lines):
        if i == 0:
            current_start = 0

        if _is_heading(line) and start != 0:
            prev_end = start
            section_text = text[current_start:prev_end]
            if section_text.strip():
                sections.append(
                    Section(
                        heading=current_heading,
                        start_char=current_start,
                        end_char=prev_end,
                        text=section_text,
                    )
                )
            current_heading = line.strip().rstrip(":")
            current_start = start

        elif i == 0 and _is_heading(line):
            current_heading = line.strip().rstrip(":")
            current_start = 0

    final_text = text[current_start:]
    if final_text.strip():
        sections.append(
            Section(
                heading=current_heading,
                start_char=current_start,
                end_char=len(text),
                text=final_text,
            )
        )

    return sections if sections else [Section(heading=None, start_char=0, end_char=len(text), text=text)]


def _split_section_by_paragraphs(
    section: Section,
    doc_id: str,
    start_index: int,
    target_min_words: int,
    target_max_words: int,
    hard_max_words: int,
    overlap_max_words: int,
) -> List[Dict[str, Any]]:
    text = section.text
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    paragraph_spans: List[tuple[str, int, int]] = []
    cursor = section.start_char
    remaining = text
    for para in paragraphs:
        idx = remaining.find(para)
        if idx < 0:
            idx = 0
        absolute_start = cursor + idx
        absolute_end = absolute_start + len(para)
        paragraph_spans.append((para, absolute_start, absolute_end))
        cursor = absolute_end
        remaining = text[(absolute_end - section.start_char):]

    chunks: List[Dict[str, Any]] = []
    buffer_texts: List[str] = []
    buffer_start: int | None = None
    buffer_end: int | None = None
    buffer_words = 0
    chunk_index = start_index

    def flush() -> None:
        nonlocal buffer_texts, buffer_start, buffer_end, buffer_words, chunk_index
        if not buffer_texts or buffer_start is None or buffer_end is None:
            return
        chunk_text = "\n\n".join(buffer_texts)
        chunk_words = len(chunk_text.split())
        chunks.append(
            {
                "chunk_id": f"chunk_{chunk_index:04d}",
                "doc_id": doc_id,
                "index": chunk_index,
                "heading_path": [section.heading] if section.heading else [],
                "start_char": buffer_start,
                "end_char": buffer_end,
                "start_word": 1,
                "end_word": chunk_words,
                "overlap_prev_words": 0,
                "overlap_next_words": 0,
                "text": chunk_text,
            }
        )
        chunk_index += 1
        buffer_texts = []
        buffer_start = None
        buffer_end = None
        buffer_words = 0

    for para, start_char, end_char in paragraph_spans:
        para_words = len(para.split())
        if buffer_start is None:
            buffer_start = start_char

        if buffer_words + para_words > hard_max_words and buffer_texts:
            flush()
            buffer_start = start_char

        buffer_texts.append(para)
        buffer_end = end_char
        buffer_words += para_words

        if buffer_words >= target_min_words:
            flush()

    flush()
    return chunks


def chunk_document(doc: Dict[str, Any], target_min_words: int, target_max_words: int, hard_max_words: int, overlap_max_words: int) -> List[Dict[str, Any]]:
    text = doc["text"]
    sections = split_into_sections(text)
    chunks: List[Dict[str, Any]] = []
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
        else:
            split_chunks = _split_section_by_paragraphs(
                section=section,
                doc_id=doc["doc_id"],
                start_index=chunk_index,
                target_min_words=target_min_words,
                target_max_words=target_max_words,
                hard_max_words=hard_max_words,
                overlap_max_words=overlap_max_words,
            )
            chunks.extend(split_chunks)
            chunk_index += len(split_chunks)

    return chunks
