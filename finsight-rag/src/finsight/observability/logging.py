"""
Logging with a request ID on every line, in plain text or JSON.

WHY
  When an answer is wrong in production, you need to find *that* request's retrieval and LLM
  calls among thousands of log lines. A request ID kept in a ``contextvars`` variable is
  attached to every log record automatically, including records from code that knows nothing
  about the request. JSON output (one object per line) is what log platforms ingest.

USAGE
  configure_logging("INFO", json_format=False)      # once, at program start
  log = get_logger(__name__)
  with request_context("req-123"):
      log.info("retrieved", extra={"n_chunks": 50})

More info: https://docs.python.org/3/library/logging.html · https://docs.python.org/3/library/contextvars.html
"""

from __future__ import annotations

import contextvars
import json
import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# Attributes every LogRecord has; anything else was passed through ``extra=`` and gets logged.
_STANDARD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "request_id"}


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def get_request_id() -> str:
    return _request_id.get()


@contextmanager
def request_context(request_id: str | None = None) -> Iterator[str]:
    """Set the request ID for everything logged inside the ``with`` block."""
    rid = request_id or new_request_id()
    token = _request_id.set(rid)
    try:
        yield rid
    finally:
        _request_id.reset(token)


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Human-friendly format; appends any ``extra=`` fields as key=value."""

    def __init__(self) -> None:
        super().__init__(
            "%(asctime)s %(levelname)-7s %(name)s [%(request_id)s] %(message)s", "%H:%M:%S"
        )

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        extras = {k: v for k, v in vars(record).items() if k not in _STANDARD_ATTRS}
        if extras:
            line += " " + " ".join(f"{k}={v}" for k, v in extras.items())
        return line


def configure_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure the root logger once. Safe to call again: it replaces our handler."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_finsight", False):
            root.removeHandler(handler)
    handler = logging.StreamHandler()
    handler._finsight = True  # type: ignore[attr-defined]
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(JsonFormatter() if json_format else TextFormatter())
    root.addHandler(handler)
    root.setLevel(level)
    # Third-party libraries are chatty at INFO.
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "filelock"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
