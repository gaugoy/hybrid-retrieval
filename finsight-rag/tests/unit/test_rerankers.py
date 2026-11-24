"""Rerankers: API request shape + ordering (mock transport), noop, missing key."""

from __future__ import annotations

import json

import httpx
import pytest
from _helpers import rc

from finsight.config import RerankCfg, Secrets
from finsight.errors import MissingAPIKeyError
from finsight.retrieval.reranker import APIReranker, NoopReranker, build_reranker


def test_noop_keeps_order() -> None:
    cands = [rc("a", 1), rc("b", 2), rc("c", 3)]
    assert [c.chunk.chunk_id for c in NoopReranker().rerank("q", cands, 2)] == ["a", "b"]


@pytest.mark.parametrize(
    ("provider", "top_field", "rows"), [("cohere", "top_n", "results"), ("voyage", "top_k", "data")]
)
def test_api_reranker_request_and_order(provider: str, top_field: str, rows: str) -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(
            200,
            json={
                rows: [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.5}]
            },
        )

    r = APIReranker(
        provider,
        RerankCfg(provider=provider),
        "k-1",  # type: ignore[arg-type]
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda s: None,
    )
    out = r.rerank("capex?", [rc("a", 1), rc("b", 2), rc("c", 3)], top_n=2)
    assert [o.chunk.chunk_id for o in out] == ["c", "a"] and out[0].stage == "rerank"
    assert out[0].scores_by_stage[provider] == 0.9
    assert seen[0]["query"] == "capex?" and seen[0]["documents"] == ["a", "b", "c"]
    assert seen[0][top_field] == 2


def test_missing_key() -> None:
    with pytest.raises(MissingAPIKeyError, match="COHERE_API_KEY"):
        build_reranker(RerankCfg(provider="cohere"), secrets=Secrets())
