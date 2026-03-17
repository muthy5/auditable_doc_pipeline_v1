from __future__ import annotations

import importlib

from app_utils import get_available_backends, is_streamlit_cloud_environment


def test_cloud_mode_hides_ollama() -> None:
    backends = get_available_backends(cloud_mode=True)
    assert "ollama" not in backends
    assert "demo" in backends
    assert "claude" in backends
    assert "openai" in backends


def test_local_mode_keeps_ollama() -> None:
    backends = get_available_backends(cloud_mode=False)
    assert "ollama" in backends
    assert "openai" in backends


def test_cloud_environment_detection() -> None:
    assert is_streamlit_cloud_environment({"STREAMLIT_SHARING": "1"}) is True
    assert is_streamlit_cloud_environment({"STREAMLIT_CLOUD": "true"}) is True
    assert is_streamlit_cloud_environment({}) is False


def test_app_import_smoke() -> None:
    importlib.import_module("app")
