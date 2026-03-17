from __future__ import annotations

import importlib.util
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .chunker import chunk_document
from .text_extractor import extract_text_from_path

_SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx"}
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class LocalFileRetriever:
    """Simple local-file retriever backed by in-memory TF-IDF vectors."""

    def __init__(
        self,
        reference_dir: str | Path,
        *,
        chunk_target_min_words: int,
        chunk_target_max_words: int,
        chunk_hard_max_words: int,
        chunk_overlap_max_words: int,
    ) -> None:
        self.reference_dir = Path(reference_dir)
        self._vectorizer: Any = None
        self._chunk_vectors: Any = None
        self._chunks: list[dict[str, Any]] = []
        self._chunk_target_min_words = chunk_target_min_words
        self._chunk_target_max_words = chunk_target_max_words
        self._chunk_hard_max_words = chunk_hard_max_words
        self._chunk_overlap_max_words = chunk_overlap_max_words
        self._tokenized_chunks: list[Counter[str]] = []
        self._idf: dict[str, float] = {}
        self._index_files()

    def _read_file(self, path: Path) -> str:
        result = extract_text_from_path(path)
        return result.text if result.ok else ""

    def _iter_reference_files(self) -> list[Path]:
        if not self.reference_dir.exists() or not self.reference_dir.is_dir():
            return []
        return [path for path in sorted(self.reference_dir.rglob("*")) if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES]

    def _tokenize(self, text: str) -> Counter[str]:
        return Counter(token.lower() for token in _TOKEN_RE.findall(text))

    def _build_fallback_index(self, chunk_texts: list[str]) -> None:
        self._tokenized_chunks = [self._tokenize(text) for text in chunk_texts]
        if not self._tokenized_chunks:
            return
        doc_freq: Counter[str] = Counter()
        for tokens in self._tokenized_chunks:
            doc_freq.update(tokens.keys())
        total_docs = len(self._tokenized_chunks)
        self._idf = {term: math.log((1 + total_docs) / (1 + freq)) + 1.0 for term, freq in doc_freq.items()}

    def _index_files(self) -> None:
        files = self._iter_reference_files()
        chunk_texts: list[str] = []
        for idx, file_path in enumerate(files, start=1):
            text = self._read_file(file_path).strip()
            if not text:
                continue
            doc = {"doc_id": f"reference_{idx:04d}", "text": text}
            try:
                chunks = chunk_document(doc, self._chunk_target_min_words, self._chunk_target_max_words, self._chunk_hard_max_words, self._chunk_overlap_max_words)
            except Exception:
                chunks = [{"text": text}]
            for chunk in chunks:
                chunk_text = chunk.get("text", "").strip()
                if not chunk_text:
                    continue
                self._chunks.append({"source_file": str(file_path), "text": chunk_text})
                chunk_texts.append(chunk_text)

        if not chunk_texts:
            return
        if importlib.util.find_spec("sklearn") is not None:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(stop_words="english")
            self._chunk_vectors = self._vectorizer.fit_transform(chunk_texts)
            return
        self._build_fallback_index(chunk_texts)

    def _fallback_similarity(self, query: str) -> list[float]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return [0.0 for _ in self._chunks]
        weighted_query = {term: count * self._idf.get(term, 0.0) for term, count in query_tokens.items()}
        query_norm = math.sqrt(sum(value * value for value in weighted_query.values())) or 1.0
        scores: list[float] = []
        for chunk_tokens in self._tokenized_chunks:
            weighted_chunk = {term: count * self._idf.get(term, 0.0) for term, count in chunk_tokens.items()}
            dot = sum(weighted_query.get(term, 0.0) * weighted_chunk.get(term, 0.0) for term in weighted_query)
            chunk_norm = math.sqrt(sum(value * value for value in weighted_chunk.values())) or 1.0
            scores.append(dot / (query_norm * chunk_norm))
        return scores

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return most relevant chunks for a query."""
        if not query.strip():
            return []

        if self._vectorizer is not None and self._chunk_vectors is not None:
            from sklearn.metrics.pairwise import cosine_similarity

            query_vector = self._vectorizer.transform([query])
            scores = cosine_similarity(query_vector, self._chunk_vectors).flatten().tolist()
        else:
            scores = self._fallback_similarity(query)

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results: list[dict[str, Any]] = []
        for index in ranked_indices:
            score = float(scores[index])
            if score <= 0:
                continue
            chunk = self._chunks[index]
            results.append({"source_file": chunk["source_file"], "text": chunk["text"], "similarity_score": score})
        return results
