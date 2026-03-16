from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .retry import RetryConfig, retry_with_backoff

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BraveSearchClient:
    """Lightweight client for Brave Web Search API."""

    api_key: str
    timeout_s: float = 15.0
    max_retries: int = 2

    def search(self, query: str, count: int = 5) -> list[dict[str, str]]:
        """Search Brave and return normalized result objects.

        Returns list entries with title, url, and snippet.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        endpoint = "https://api.search.brave.com/res/v1/web/search"
        params = urllib.parse.urlencode({"q": cleaned_query, "count": max(1, count)})
        url = f"{endpoint}?{params}"

        def _call() -> list[dict[str, str]]:
            request = urllib.request.Request(
                url=url,
                method="GET",
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": self.api_key,
                },
            )
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return self._normalize_results(payload)

        def _should_retry(exc: Exception) -> bool:
            if isinstance(exc, urllib.error.HTTPError):
                return exc.code in {408, 425, 429, 500, 502, 503, 504}
            return isinstance(exc, (urllib.error.URLError, TimeoutError, json.JSONDecodeError))

        try:
            return retry_with_backoff(
                fn=_call,
                should_retry=_should_retry,
                config=RetryConfig(max_retries=self.max_retries, base_delay_s=0.25, max_delay_s=2.0),
                on_retry=lambda attempt, exc: LOGGER.warning("Retrying Brave search (%s): %s", attempt, exc),
            )
        except Exception as exc:  # noqa: BLE001 - graceful degradation on external API failures
            LOGGER.warning("Brave search failed for query '%s': %s", cleaned_query, exc)
            return []

    def _normalize_results(self, payload: dict[str, Any]) -> list[dict[str, str]]:
        web = payload.get("web") if isinstance(payload, dict) else {}
        raw_results = web.get("results", []) if isinstance(web, dict) else []

        normalized: list[dict[str, str]] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "title": str(item.get("title", "")).strip(),
                    "url": str(item.get("url", "")).strip(),
                    "snippet": str(item.get("description", "")).strip(),
                }
            )
        return [result for result in normalized if result["url"]]
