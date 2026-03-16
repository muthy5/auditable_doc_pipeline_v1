from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryConfig:
    """Retry configuration for exponential backoff."""

    max_retries: int = 2
    base_delay_s: float = 0.25
    max_delay_s: float = 5.0


def retry_with_backoff(
    fn: Callable[[], T],
    should_retry: Callable[[Exception], bool],
    config: RetryConfig,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> T:
    """Run a function with bounded exponential-backoff retries."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt >= config.max_retries or not should_retry(exc):
                raise
            if on_retry is not None:
                on_retry(attempt + 1, exc)
            delay = min(config.base_delay_s * (2**attempt), config.max_delay_s)
            time.sleep(delay)
            attempt += 1
