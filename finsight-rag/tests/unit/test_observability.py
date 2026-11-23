"""Request-ID logging and stage timers."""

from __future__ import annotations

import json
import logging
import time

import pytest

from finsight.observability.logging import (
    JsonFormatter,
    configure_logging,
    get_logger,
    get_request_id,
    request_context,
)
from finsight.observability.timing import Timer


def test_request_context_sets_and_resets() -> None:
    assert get_request_id() == "-"
    with request_context("abc123") as rid:
        assert rid == "abc123"
        assert get_request_id() == "abc123"
    assert get_request_id() == "-"


def test_json_log_line_has_request_id_and_extras(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json_format=True)
    with request_context("req-42"):
        get_logger("finsight.test").info("retrieved", extra={"n_chunks": 50})
    line = capsys.readouterr().err.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["msg"] == "retrieved"
    assert payload["request_id"] == "req-42"
    assert payload["n_chunks"] == 50
    configure_logging("INFO", json_format=False)


def test_json_formatter_includes_exception() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.getLogger("t").makeRecord(
            "t", logging.ERROR, __file__, 1, "failed", (), exc_info=True
        )
        import sys

        record.exc_info = sys.exc_info()
    assert "ValueError: boom" in json.loads(JsonFormatter().format(record))["exc"]


def test_timer_records_and_accumulates_stages() -> None:
    timer = Timer()
    with timer.stage("dense"):
        time.sleep(0.01)
    with timer.stage("dense"):
        time.sleep(0.01)
    with timer.stage("rerank"):
        pass
    out = timer.as_dict()
    assert out["dense"] >= 15  # two ~10 ms blocks accumulated
    assert set(out) == {"dense", "rerank", "total"}
    assert out["total"] == pytest.approx(out["dense"] + out["rerank"], abs=0.05)


def test_timer_records_even_when_block_raises() -> None:
    timer = Timer()
    with pytest.raises(RuntimeError), timer.stage("generate"):
        raise RuntimeError("provider down")
    assert "generate" in timer.timings_ms
