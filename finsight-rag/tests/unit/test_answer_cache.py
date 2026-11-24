"""AnswerCache: hits on the same key, misses when the config changes, expires after the TTL."""

from __future__ import annotations

from pathlib import Path

from finsight.cache import AnswerCache
from finsight.schemas import Answer


def _answer(text: str) -> Answer:
    return Answer(
        question="q",
        answer=text,
        citations=[],
        abstained=False,
        retrieved=[],
        pipeline="p",
        provider="fake",
        model="m",
        timings_ms={},
        request_id="r1",
    )


def test_hit_miss_and_ttl(tmp_path: Path) -> None:
    now = [1000.0]
    cache = AnswerCache(tmp_path / "a.sqlite", ttl_s=60, clock=lambda: now[0])
    k = AnswerCache.key("fp1", question="q", pipeline="p")
    assert cache.get(k) is None
    cache.put(k, _answer("A"))
    got = cache.get(k)
    assert got is not None and got.answer == "A"
    assert cache.get(AnswerCache.key("fp2", question="q", pipeline="p")) is None  # config changed
    assert cache.get(AnswerCache.key("fp1", question="q", pipeline="naive")) is None
    now[0] += 61
    assert cache.get(k) is None  # expired
