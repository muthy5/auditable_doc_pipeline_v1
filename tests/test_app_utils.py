from __future__ import annotations

from app_utils import get_available_backends, is_streamlit_cloud_environment


def test_cloud_mode_hides_ollama() -> None:
    assert get_available_backends(cloud_mode=True) == ["demo", "claude"]


def test_local_mode_keeps_ollama() -> None:
    assert "ollama" in get_available_backends(cloud_mode=False)


def test_cloud_environment_detection() -> None:
    assert is_streamlit_cloud_environment({"STREAMLIT_SHARING": "1"}) is True
    assert is_streamlit_cloud_environment({"STREAMLIT_CLOUD": "true"}) is True
    assert is_streamlit_cloud_environment({}) is False
