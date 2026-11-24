"""
POST JSON to a provider with retries, and turn failures into clear errors.

    200            -> parsed JSON
    429            -> "per day" / "per month" / "quota" wording  -> QuotaExhaustedError (no retry)
                      otherwise: wait Retry-After (or exponential backoff) and retry
    5xx, timeouts  -> exponential backoff (2, 4, 8 … ≤ 60 s) and retry
    401 / 403      -> ProviderError naming the API-key variable (no retry)
    other 4xx      -> ProviderError with the server's message (no retry: the request is wrong)
After max_retries: RateLimitedError (429) or ProviderError.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

import httpx

from finsight.errors import ProviderError, QuotaExhaustedError, RateLimitedError
from finsight.observability.logging import get_logger

log = get_logger(__name__)

# Wording providers use when a long-period quota (not the per-minute limit) is exhausted.
# Not "quota" or "billing" alone: Google's PER-MINUTE 429 also says "exceeded your current
# quota ... check your plan and billing"; its quotaId ("...PerMinute..." / "...PerDay...") decides.
_QUOTA_PATTERN = re.compile(
    r"per\s?day|perday|daily|per\s?month|permonth|monthly|insufficient_quota", re.IGNORECASE
)


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        return float(value) if value else None
    except ValueError:
        return None


def post_json(
    client: httpx.Client,
    url: str,
    *,
    body: dict[str, Any],
    headers: dict[str, str] | None = None,
    provider: str = "provider",
    key_hint: str = "",
    max_retries: int = 6,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    last_error = ""
    for attempt in range(max_retries + 1):
        delay = min(base_delay * 2**attempt, max_delay)
        try:
            response = client.post(url, json=body, headers=headers)
        except httpx.TransportError as exc:  # timeouts, connection resets, DNS
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            code = response.status_code
            if code == 200:
                data: dict[str, Any] = response.json()
                return data
            text = response.text[:500]
            if code == 429:
                if _QUOTA_PATTERN.search(text):
                    raise QuotaExhaustedError(
                        f"{provider}: quota exhausted (HTTP 429): {text}. Progress is cached; "
                        "re-run after the quota resets."
                    )
                last_error = f"HTTP 429: {text}"
                delay = _retry_after(response) or delay
                if attempt == max_retries:
                    raise RateLimitedError(
                        f"{provider}: still rate-limited after {max_retries} retries. Lower "
                        f"rate_limit.rpm/tpm for this preset. Last response: {text}"
                    )
            elif code in (401, 403):
                raise ProviderError(
                    f"{provider}: HTTP {code} (check {key_hint or 'the API key'}): {text}"
                )
            elif code >= 500:
                last_error = f"HTTP {code}: {text}"
            else:
                raise ProviderError(f"{provider}: HTTP {code}: {text}")
        if attempt < max_retries:
            log.warning(
                "provider request failed, retrying",
                extra={
                    "provider": provider,
                    "attempt": attempt + 1,
                    "wait_s": round(delay, 1),
                    "error": last_error[:200],
                },
            )
            sleep(delay)
    raise ProviderError(f"{provider}: failed after {max_retries} retries: {last_error}")
