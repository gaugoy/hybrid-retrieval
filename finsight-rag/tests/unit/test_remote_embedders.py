"""Online embedders: request shapes, parsing, batching, templates, caches, guards (offline)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from finsight.config import EmbeddingCfg, Secrets
from finsight.embedding.base import build_embedder
from finsight.embedding.cache import EmbeddingCache
from finsight.embedding.remote import (
    PROVIDERS,
    CohereEmbedder,
    GeminiEmbedder,
    OpenAIStyleEmbedder,
)
from finsight.errors import (
    MissingAPIKeyError,
    PaidProviderBlockedError,
    ProviderError,
    QuotaExhaustedError,
)

DIM = 4


class Recorder:
    def __init__(self, reply: Any) -> None:
        self.reply = reply  # function(body) -> json
        self.requests: list[httpx.Request] = []

    def client(self) -> httpx.Client:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(200, json=self.reply(json.loads(request.content)))

        return httpx.Client(transport=httpx.MockTransport(handler))

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]


def vec(i: int) -> list[float]:
    return [float(i + 1), 2.0, 0.0, 1.0]  # not normalized on purpose


def openai_reply(body: dict[str, Any]) -> dict[str, Any]:
    data = [{"index": i, "embedding": vec(i)} for i in range(len(body["input"]))]
    return {"data": list(reversed(data)), "usage": {"total_tokens": 10}}  # out of order on purpose


def make(cls: type, provider: str, reply: Any, **cfg: Any) -> tuple[Any, Recorder]:
    rec = Recorder(reply)
    c = EmbeddingCfg(provider=provider, model="m", dimensions=DIM, preset=None, **cfg)
    return cls(
        c, PROVIDERS[provider], "sk-test-key", client=rec.client(), sleep=lambda s: None
    ), rec


def test_openai_style_request_batching_roles_and_normalization() -> None:
    emb, rec = make(
        OpenAIStyleEmbedder,
        "jina",
        openai_reply,
        document_params={"task": "retrieval.passage"},
        query_params={"task": "retrieval.query"},
        rate_limit={"max_batch_texts": 2},
    )
    docs = emb.embed_documents(["a", "b", "c"])
    assert docs.shape == (3, DIM) and np.allclose(np.linalg.norm(docs, axis=1), 1.0)
    assert np.allclose(docs[0], np.array(vec(0)) / np.linalg.norm(vec(0)))  # index order restored
    assert len(rec.requests) == 2  # batches of 2
    first = rec.bodies()[0]
    assert first == {
        "model": "m",
        "input": ["a", "b"],
        "task": "retrieval.passage",
        "dimensions": 4,
    }
    assert str(rec.requests[0].url) == "https://api.jina.ai/v1/embeddings"
    assert rec.requests[0].headers["authorization"] == "Bearer sk-test-key"
    emb.embed_query("q")
    assert rec.bodies()[-1]["task"] == "retrieval.query"
    assert emb.stats.requests == 3 and emb.stats.billed_tokens == 30


def test_gemini_request_shape_and_templates() -> None:
    def reply(body: dict[str, Any]) -> dict[str, Any]:
        return {"embeddings": [{"values": vec(i)} for i in range(len(body["requests"]))]}

    emb, rec = make(
        GeminiEmbedder,
        "gemini",
        reply,
        query_template="task: search result | query: {text}",
        document_template="title: none | text: {text}",
    )
    emb.embed_documents(["capex"])
    emb.embed_query("What was capex?")
    url = str(rec.requests[0].url)
    assert url.endswith("/v1beta/models/m:batchEmbedContents")
    assert rec.requests[0].headers["x-goog-api-key"] == "sk-test-key"
    doc_req, q_req = rec.bodies()[0]["requests"][0], rec.bodies()[1]["requests"][0]
    assert doc_req == {
        "model": "models/m",
        "content": {"parts": [{"text": "title: none | text: capex"}]},
        "outputDimensionality": 4,
    }
    assert q_req["content"]["parts"][0]["text"] == "task: search result | query: What was capex?"


def test_cohere_request_shape() -> None:
    def reply(body: dict[str, Any]) -> dict[str, Any]:
        return {
            "embeddings": {"float": [vec(i) for i in range(len(body["texts"]))]},
            "meta": {"billed_units": {"input_tokens": 7}},
        }

    emb, rec = make(
        CohereEmbedder, "cohere", reply, document_params={"input_type": "search_document"}
    )
    emb.embed_documents(["x", "y"])
    body = rec.bodies()[0]
    assert str(rec.requests[0].url) == "https://api.cohere.com/v2/embed"
    assert body == {
        "model": "m",
        "texts": ["x", "y"],
        "embedding_types": ["float"],
        "input_type": "search_document",
        "output_dimension": 4,
    }
    assert emb.stats.billed_tokens == 7


def test_query_cache_makes_reruns_free(tmp_path: Path) -> None:
    emb, rec = make(OpenAIStyleEmbedder, "mistral", openai_reply)
    emb.query_cache = EmbeddingCache(tmp_path / "c.sqlite")
    first = emb.embed_queries(["q1", "q2"])
    second = emb.embed_queries(["q2", "q1", "q3"])
    assert len(rec.requests) == 2  # second call only sent q3
    assert rec.bodies()[1]["input"] == ["q3"]
    assert np.array_equal(first[0], second[1]) and emb.stats.query_cache_hits == 2


def test_budget_and_dimension_checks() -> None:
    emb, rec = make(OpenAIStyleEmbedder, "mistral", openai_reply, max_tokens_per_run=5)
    with pytest.raises(QuotaExhaustedError, match="max_tokens_per_run"):
        emb.embed_documents(["a long text that is well over five estimated tokens"])
    assert rec.requests == []  # stopped BEFORE sending
    wrong, _ = make(OpenAIStyleEmbedder, "mistral", openai_reply)
    wrong._dim = 8
    with pytest.raises(ProviderError, match="expected 8-dim"):
        wrong.embed_documents(["a"])


def test_token_limit_splits_batches() -> None:
    emb, rec = make(
        OpenAIStyleEmbedder,
        "mistral",
        openai_reply,
        rate_limit={"max_batch_texts": 100, "max_batch_tokens": 30},
    )
    emb.embed_documents(["x" * 70, "y" * 70, "z" * 70])  # ~21 estimated tokens each
    assert [len(b["input"]) for b in rec.bodies()] == [1, 1, 1]


def test_identity_depends_on_dims_and_document_presentation() -> None:
    a, _ = make(OpenAIStyleEmbedder, "mistral", openai_reply)
    b, _ = make(OpenAIStyleEmbedder, "mistral", openai_reply, document_template="doc: {text}")
    assert a.model_name == "mistral/m@4" and a.slug == "mistral-m-4"
    assert b.model_name.startswith("mistral/m@4#") and b.slug == a.slug


def test_guards_missing_key_paid_and_keyless() -> None:
    gem = EmbeddingCfg(provider="gemini", model="gemini-embedding-001", preset="gemini-001")
    with pytest.raises(MissingAPIKeyError, match="GEMINI_API_KEY"):
        build_embedder(gem, secrets=Secrets())
    ok = build_embedder(gem, secrets=Secrets.from_env({"GOOGLE_API_KEY": "g-123"}))
    assert isinstance(ok, GeminiEmbedder) and ok.api_key == "g-123"  # GOOGLE_API_KEY fallback

    paid = EmbeddingCfg(provider="openai", model="text-embedding-3-small")
    keys = Secrets.from_env({"OPENAI_API_KEY": "sk-1"})
    with pytest.raises(PaidProviderBlockedError, match=r"\$0.02"):
        build_embedder(paid, secrets=keys)
    allowed = paid.model_copy(update={"allow_paid": True})
    assert isinstance(build_embedder(allowed, secrets=keys), OpenAIStyleEmbedder)

    ollama = build_embedder(
        EmbeddingCfg(provider="ollama", model="nomic-embed-text"), secrets=Secrets()
    )
    assert ollama.api_key is None and ollama.base_url == "http://localhost:11434/v1"
