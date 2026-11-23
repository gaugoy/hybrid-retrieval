"""
Per-stage timers.

WHY
  "The RAG is slow" is not actionable; "rerank takes 900 ms of the 1.4 s" is. Every pipeline
  records how long each stage took into ``Answer.timings_ms``. The API logs it, the UI charts
  it, and the eval reports p50/p95 per stage.

USAGE
  timer = Timer()
  with timer.stage("dense"):
      ...
  timer.timings_ms  ->  {"dense": 12.3}
  timer.total_ms()  ->  sum of all stages
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


class Timer:
    def __init__(self) -> None:
        self.timings_ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time the ``with`` block. Repeated names accumulate (e.g. several retrieval rounds)."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            self.timings_ms[name] = round(self.timings_ms.get(name, 0.0) + elapsed, 2)

    def total_ms(self) -> float:
        return round(sum(v for k, v in self.timings_ms.items() if k != "total"), 2)

    def as_dict(self) -> dict[str, float]:
        """Timings plus a ``total`` key, ready for ``Answer.timings_ms``."""
        return {**self.timings_ms, "total": self.total_ms()}
