"""The HTTP API on the tiny corpus with FakeLLM: endpoints, validation, SSE, request ids, limits."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from finsight.api.main import create_app
from finsight.config import Settings
from finsight.container import Container

Q = "What is Acme's FY2020 capital expenditure?"


def _client(settings: Settings) -> Iterator[TestClient]:
    container = Container(settings)
    with TestClient(create_app(container=container)) as client:
        yield client


@pytest.fixture
def client(tiny: Settings) -> Iterator[TestClient]:
    yield from _client(tiny)


def test_health_ready_config(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["ready"] is True
    assert r.json()["checks"]["llm"].startswith("ok (fake")
    cfg = client.get("/v1/config").json()
    assert cfg["fingerprint"] and cfg["pipeline"] == "hybrid_rerank"
    assert "secrets" not in cfg["settings"] and "api_key" not in json.dumps(cfg).lower()


def test_query_returns_cited_answer_with_request_id(client: TestClient) -> None:
    r = client.post("/v1/query", json={"question": Q}, headers={"X-Request-ID": "abc123"})
    assert r.status_code == 200 and r.headers["X-Request-ID"] == "abc123"
    body = r.json()
    assert body["request_id"] == "abc123" and body["provider"] == "fake"
    assert body["citations"] and not body["abstained"]
    assert body["retrieved"] == []  # trimmed by default
    full = client.post("/v1/query", json={"question": Q, "include_retrieved": True}).json()
    assert full["retrieved"] and len(full["retrieved"]) <= 20


def test_request_id_generated_when_missing(client: TestClient) -> None:
    r = client.post("/v1/query", json={"question": Q})
    assert (
        len(r.headers["X-Request-ID"]) == 12 and r.json()["request_id"] == r.headers["X-Request-ID"]
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "hi"},
        {"question": "x" * 1001},
        {"question": Q, "top_k": 21},
        {"question": Q, "pipeline": "magic"},
    ],
)
def test_validation_rejects_bad_input(client: TestClient, payload: dict) -> None:
    assert client.post("/v1/query", json=payload).status_code == 422


def test_unknown_llm_preset_is_422(client: TestClient) -> None:
    r = client.post("/v1/query", json={"question": Q, "llm": "no-such-preset"})
    assert r.status_code == 422 and "no-such-preset" in r.json()["detail"]


def test_per_request_pipeline_override(client: TestClient) -> None:
    r = client.post("/v1/query", json={"question": Q, "pipeline": "naive", "top_k": 3})
    assert r.status_code == 200 and r.json()["pipeline"] == "naive"


def test_stream_emits_retrieved_tokens_done(client: TestClient) -> None:
    with client.stream("POST", "/v1/query/stream", json={"question": Q}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    events = [blk.split("\n")[0].removeprefix("event: ") for blk in text.strip().split("\n\n")]
    assert events[0] == "retrieved" and events[-1] == "done" and "token" in events
    done = json.loads(text.strip().split("\n\n")[-1].split("data: ", 1)[1])
    assert done["citations"] and "total" in done["timings_ms"]


def test_rate_limit_and_body_limit(tiny: Settings) -> None:
    s = tiny.model_copy(update={"api": tiny.api.model_copy(update={"rate_limit_per_minute": 2})})
    s.root, s.secrets = tiny.root, tiny.secrets
    for client in _client(s):
        codes = [client.post("/v1/query", json={"question": Q}).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        assert client.get("/health").status_code == 200  # GETs are not limited
        big = client.post(
            "/v1/query", content="x" * 50_000, headers={"content-type": "application/json"}
        )
        assert big.status_code == 413
