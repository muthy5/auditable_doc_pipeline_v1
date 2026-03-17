from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from src.cli import _ensure_preflight_or_exit
from src.preflight import check_ollama, run_preflight
from src.text_extractor import extract_text_from_path


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None


def test_extract_pdf_missing_parser(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.0")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "pypdf" else object())

    result = extract_text_from_path(path)

    assert not result.ok
    assert result.error_code == "missing_pdf_parser"


def test_extract_docx_missing_parser(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "doc.docx"
    path.write_bytes(b"PK\x03\x04")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "docx" else object())

    result = extract_text_from_path(path)

    assert not result.ok
    assert result.error_code == "missing_docx_parser"


def test_extract_pdf_malformed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"not a real pdf")

    class FakePdfReadError(Exception):
        pass

    class FakeReader:
        def __init__(self, _: str) -> None:
            raise FakePdfReadError("bad")

    fake_pypdf = type("FakePypdf", (), {"PdfReader": FakeReader})
    fake_errors = type("FakeErrors", (), {"PdfReadError": FakePdfReadError})
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setitem(__import__("sys").modules, "pypdf", fake_pypdf)
    monkeypatch.setitem(__import__("sys").modules, "pypdf.errors", fake_errors)

    result = extract_text_from_path(path)

    assert not result.ok
    assert result.error_code == "corrupted_document"


def test_extract_empty_text_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("   ", encoding="utf-8")

    result = extract_text_from_path(path)

    assert not result.ok
    assert result.error_code == "empty_document"


def test_extract_image_only_pdf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF-1.0")

    class FakePage:
        def extract_text(self) -> str:
            return ""

    class FakeReader:
        def __init__(self, _: str) -> None:
            self.pages = [FakePage(), FakePage()]

    fake_pypdf = type("FakePypdf", (), {"PdfReader": FakeReader})
    fake_errors = type("FakeErrors", (), {"PdfReadError": RuntimeError})
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setitem(__import__("sys").modules, "pypdf", fake_pypdf)
    monkeypatch.setitem(__import__("sys").modules, "pypdf.errors", fake_errors)

    result = extract_text_from_path(path)

    assert not result.ok
    assert result.error_code == "image_only_pdf"
    assert result.is_image_only_pdf


def test_preflight_missing_claude_key() -> None:
    statuses = run_preflight(
        backend="claude",
        enable_search=False,
        claude_api_key="",
        brave_api_key="",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="llama3",
    )
    assert not statuses["claude_backend"].available


def test_preflight_missing_anthropic_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "anthropic" else object())
    statuses = run_preflight(
        backend="claude",
        enable_search=False,
        claude_api_key="key",
        brave_api_key="",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="llama3",
    )
    assert not statuses["claude_backend"].available


def test_ollama_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("offline")))
    status = check_ollama("http://127.0.0.1:11434", "llama3")
    assert not status.available


def test_ollama_model_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: _FakeResponse({"models": [{"name": "other"}]}))
    status = check_ollama("http://127.0.0.1:11434", "llama3")
    assert not status.available

def test_ollama_model_matching_accepts_latest_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: _FakeResponse({"models": [{"name": "llama3:latest"}]}))
    status = check_ollama("http://127.0.0.1:11434", "llama3")
    assert status.available



def test_check_ollama_handles_non_mapping_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: _FakeResponse([]))

    status = check_ollama("http://127.0.0.1:11434", "llama3")

    assert not status.available
    assert "Unexpected Ollama /api/tags payload type" in status.message



def test_preflight_helper_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    statuses = run_preflight(
        backend="demo",
        enable_search=True,
        claude_api_key="",
        brave_api_key="key",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="llama3",
    )
    assert statuses["demo_backend"].available
    assert statuses["pdf_parsing"].available
    assert statuses["docx_parsing"].available
    assert statuses["web_search"].available


def test_preflight_openai_allows_local_base_url_without_api_key() -> None:
    statuses = run_preflight(
        backend="openai",
        enable_search=False,
        claude_api_key="",
        brave_api_key="",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="llama3",
        openai_api_key="",
        openai_base_url="http://localhost:8000/v1",
    )
    assert statuses["openai_backend"].available


def test_preflight_openai_requires_api_key_for_remote_base_url() -> None:
    statuses = run_preflight(
        backend="openai",
        enable_search=False,
        claude_api_key="",
        brave_api_key="",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="llama3",
        openai_api_key="",
        openai_base_url="https://api.openai.com/v1",
    )
    assert not statuses["openai_backend"].available


def test_cli_preflight_rejects_invalid_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    args = argparse.Namespace(
        backend="claude",
        enable_search=False,
        claude_api_key="",
        brave_api_key="",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="",
        ollama_timeout_s=1.0,
        openai_api_key="",
        openai_base_url="https://api.openai.com/v1",
    )

    class Parser:
        def error(self, message: str) -> None:
            raise ValueError(message)

    with pytest.raises(ValueError):
        _ensure_preflight_or_exit(args, Parser())
