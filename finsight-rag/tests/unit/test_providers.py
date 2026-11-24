"""Rate limiter + HTTP retry policy, with a fake clock and a mock transport (instant, offline)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from finsight.errors import ProviderError, QuotaExhaustedError, RateLimitedError
from finsight.providers.http import post_json
from finsight.providers.ratelimit import RateLimiter, UsageStore


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


def test_rpm_spaces_requests_evenly() -> None:
    clock = FakeClock()
    rl = RateLimiter(rpm=30, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        rl.acquire()
    assert clock.slept == [pytest.approx(2.0), pytest.approx(2.0)]  # 60 / 30 s apart


def test_tpm_window_waits_for_old_tokens_to_expire() -> None:
    clock = FakeClock()
    rl = RateLimiter(tpm=1000, clock=clock, sleep=clock.sleep)
    rl.acquire(600)
    rl.acquire(300)  # 900 ≤ 1000: no wait
    assert clock.slept == []
    rl.acquire(300)  # would be 1200: wait until the first request leaves the 60 s window
    assert sum(clock.slept) == pytest.approx(60.01, abs=0.05)
    rl.acquire(5000)  # bigger than tpm on its own: allowed once the window is empty
    assert rl.usage.today()["requests"] == 4


def test_daily_cap_raises_before_sending_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "usage" / "gemini.json"
    rl = RateLimiter(rpd=2, usage=UsageStore(path), name="gemini")
    rl.acquire(10)
    rl.acquire(10)
    with pytest.raises(QuotaExhaustedError, match="daily request cap"):
        rl.acquire(10)
    again = RateLimiter(rpd=2, usage=UsageStore(path), name="gemini")  # a new process, same day
    with pytest.raises(QuotaExhaustedError):
        again.acquire()
    assert UsageStore(path).today()["tokens"] == 20


def test_usage_resets_on_a_new_day_and_month() -> None:
    usage = UsageStore()
    usage.add(5, 50, now=datetime(2026, 11, 30, tzinfo=UTC))
    assert usage.today(datetime(2026, 11, 30, tzinfo=UTC))["requests"] == 5
    assert usage.today(datetime(2026, 12, 1, tzinfo=UTC))["requests"] == 0
    assert usage.this_month(datetime(2026, 12, 1, tzinfo=UTC))["requests"] == 0


def _client(responses: list[httpx.Response]) -> tuple[httpx.Client, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses.pop(0)

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


def test_429_is_retried_honouring_retry_after() -> None:
    client, seen = _client(
        [
            httpx.Response(429, headers={"retry-after": "7"}, text="rate limit per minute"),
            httpx.Response(503, text="overloaded"),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    slept: list[float] = []
    assert post_json(client, "https://x/api", body={}, sleep=slept.append) == {"ok": True}
    assert len(seen) == 3 and slept == [7.0, 4.0]  # Retry-After, then backoff 2·2^1


def test_quota_429_stops_immediately() -> None:
    google_rpd = (
        '{"error": {"status": "RESOURCE_EXHAUSTED", "details": '
        '[{"quotaId": "EmbedRequestsPerDayPerProjectPerModel-FreeTier"}]}}'
    )
    client, seen = _client([httpx.Response(429, text=google_rpd)])
    with pytest.raises(QuotaExhaustedError):
        post_json(client, "https://x", body={}, sleep=lambda s: None)
    assert len(seen) == 1


def test_per_minute_429_mentioning_billing_is_not_a_quota_stop() -> None:
    per_minute = (
        "You exceeded your current quota, please check your plan and billing details. "
        "quotaId: PerMinute"
    )
    client, _ = _client([httpx.Response(429, text=per_minute)] * 3)
    with pytest.raises(RateLimitedError):
        post_json(client, "https://x", body={}, max_retries=2, sleep=lambda s: None)


def test_auth_and_bad_requests_fail_without_retry() -> None:
    client, seen = _client([httpx.Response(401, text="bad key")])
    with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
        post_json(client, "https://x", body={}, key_hint="GEMINI_API_KEY", sleep=lambda s: None)
    client, seen = _client([httpx.Response(400, text="unknown field")])
    with pytest.raises(ProviderError, match="400"):
        post_json(client, "https://x", body={}, sleep=lambda s: None)
    assert len(seen) == 1
