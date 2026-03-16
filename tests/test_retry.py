from __future__ import annotations

import pytest

from src.retry import RetryConfig, retry_with_backoff


def test_retry_with_backoff_eventually_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("temp")
        return "ok"

    result = retry_with_backoff(fn, lambda exc: isinstance(exc, ConnectionError), RetryConfig(max_retries=3, base_delay_s=0.1, max_delay_s=1.0))
    assert result == "ok"
    assert sleeps == [0.1, 0.2]
