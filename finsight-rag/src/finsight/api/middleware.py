"""
HTTP middleware: request id + one JSON log line per request, body size limit, per-IP rate limit.

The rate limiter is in-memory and per process: enough to stop a runaway client in a demo; in
production this belongs in the API gateway / load balancer.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from finsight.observability.logging import get_logger

log = get_logger("finsight.api")


class RequestContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, max_body_bytes: int = 16_384, per_minute: int = 60) -> None:
        super().__init__(app)
        self.max_body_bytes, self.per_minute = max_body_bytes, per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        if int(request.headers.get("content-length") or 0) > self.max_body_bytes:
            return JSONResponse({"detail": "request body too large"}, 413, {"X-Request-ID": rid})
        if request.url.path.startswith("/v1/") and request.method == "POST":
            ip = request.client.host if request.client else "?"
            now, q = time.monotonic(), self._hits[ip]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return JSONResponse(
                    {"detail": "rate limit: try again in a minute"},
                    429,
                    {"X-Request-ID": rid, "Retry-After": "60"},
                )
            q.append(now)
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        log.info(
            "request",
            extra={
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "ms": round((time.perf_counter() - t0) * 1000, 1),
            },
        )
        return response
