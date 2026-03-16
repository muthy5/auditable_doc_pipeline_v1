from __future__ import annotations

from pathlib import Path

from src.retriever import LocalFileRetriever


def _make_retriever(reference_dir: Path) -> LocalFileRetriever:
    return LocalFileRetriever(
        reference_dir,
        chunk_target_min_words=20,
        chunk_target_max_words=40,
        chunk_hard_max_words=60,
        chunk_overlap_max_words=10,
    )


def test_retriever_indexes_and_retrieves_from_text_files(tmp_path: Path) -> None:
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    (ref_dir / "tea.txt").write_text("Green tea includes catechins and antioxidants for health benefits.", encoding="utf-8")
    (ref_dir / "coffee.md").write_text("Coffee beans are roasted seeds used to brew coffee beverages.", encoding="utf-8")

    retriever = _make_retriever(ref_dir)
    results = retriever.retrieve("antioxidants in green tea", top_k=3)

    assert results
    assert results[0]["source_file"].endswith("tea.txt")
    assert "antioxidants" in results[0]["text"].lower()


def test_retriever_returns_ranked_results(tmp_path: Path) -> None:
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    (ref_dir / "alpha.txt").write_text("Python is used for scripting and automation.", encoding="utf-8")
    (ref_dir / "beta.txt").write_text("Python data science uses pandas and scikit-learn extensively.", encoding="utf-8")

    retriever = _make_retriever(ref_dir)
    results = retriever.retrieve("python data science", top_k=2)

    assert len(results) == 2
    assert results[0]["similarity_score"] >= results[1]["similarity_score"]
    assert results[0]["source_file"].endswith("beta.txt")
