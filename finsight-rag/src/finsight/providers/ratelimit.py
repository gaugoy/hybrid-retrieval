"""
Client-side rate limiting for free-tier APIs.

WHY ON THE CLIENT
  Free tiers allow few requests per minute (RPM), few tokens per minute (TPM) and a daily or
  monthly request cap (RPD / per month). Hitting the server limit costs a failed request and a
  back-off; hitting a daily cap mid-run used to mean starting over. So we:
    1. space requests at least 60/rpm seconds apart (smooth, no bursts)
    2. keep a 60-second window of tokens sent and wait while the next request would exceed tpm
    3. count requests per UTC day and per month in indexes/usage/<provider>.json and raise
       QuotaExhaustedError BEFORE sending a request over the cap
  Token counts are estimates (len(text) / 3.5, deliberately high), since each provider tokenizes
  differently. A provider's real 429 is still handled by the HTTP retry (http.py).

Clock and sleep are injectable, so tests run instantly.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from finsight.errors import QuotaExhaustedError
from finsight.observability.logging import get_logger

log = get_logger(__name__)

CHARS_PER_TOKEN = 3.5  # conservative: real tokenizers average ~4 chars/token on English


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


@dataclass
class UsageStore:
    """Requests/tokens per UTC day and calendar month, persisted as JSON (None path = memory)."""

    path: Path | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.path and self.path.is_file():
            self.data = json.loads(self.path.read_text())

    @staticmethod
    def _keys(now: datetime) -> tuple[str, str]:
        return now.strftime("%Y-%m-%d"), now.strftime("%Y-%m")

    def _bucket(self, kind: str, key: str) -> dict[str, int]:
        bucket = self.data.get(kind)
        if not bucket or bucket.get("key") != key:  # a new day / month starts from zero
            bucket = {"key": key, "requests": 0, "tokens": 0}
            self.data[kind] = bucket
        return bucket  # type: ignore[no-any-return]

    def today(self, now: datetime | None = None) -> dict[str, int]:
        return self._bucket("day", self._keys(now or datetime.now(UTC))[0])

    def this_month(self, now: datetime | None = None) -> dict[str, int]:
        return self._bucket("month", self._keys(now or datetime.now(UTC))[1])

    def add(self, requests: int, tokens: int, now: datetime | None = None) -> None:
        now = now or datetime.now(UTC)
        for bucket in (self.today(now), self.this_month(now)):
            bucket["requests"] += requests
            bucket["tokens"] += tokens
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True))


class RateLimiter:
    def __init__(
        self,
        *,
        rpm: int | None = None,
        tpm: int | None = None,
        rpd: int | None = None,
        rpmonth: int | None = None,
        usage: UsageStore | None = None,
        name: str = "provider",
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.rpm, self.tpm, self.rpd, self.rpmonth = rpm, tpm, rpd, rpmonth
        self.usage = usage or UsageStore()
        self.name = name
        self.clock, self.sleep = clock, sleep
        self._last: float | None = None
        self._window: deque[tuple[float, int]] = deque()  # (time, tokens) in the last 60 s
        self.waited_s = 0.0

    def _check_quotas(self) -> None:
        day, month = self.usage.today(), self.usage.this_month()
        if self.rpd and day["requests"] + 1 > self.rpd:
            raise QuotaExhaustedError(
                f"{self.name}: daily request cap reached ({day['requests']}/{self.rpd}, UTC day). "
                "Progress is cached: re-run tomorrow to continue."
            )
        if self.rpmonth and month["requests"] + 1 > self.rpmonth:
            raise QuotaExhaustedError(
                f"{self.name}: monthly request cap reached ({month['requests']}/{self.rpmonth})."
            )

    def _wait(self, seconds: float) -> None:
        if seconds > 0:
            self.waited_s += seconds
            self.sleep(seconds)

    def acquire(self, tokens: int = 0) -> None:
        """Block until one request carrying `tokens` may be sent, then record it."""
        self._check_quotas()
        if self.rpm and self._last is not None:
            self._wait(self._last + 60.0 / self.rpm - self.clock())
        if self.tpm:
            while True:
                now = self.clock()
                while self._window and now - self._window[0][0] >= 60.0:
                    self._window.popleft()
                used = sum(t for _, t in self._window)
                if not self._window or used + tokens <= self.tpm:
                    break
                self._wait(60.0 - (now - self._window[0][0]) + 0.01)
        now = self.clock()
        self._last = now
        self._window.append((now, tokens))
        self.usage.add(1, tokens)
