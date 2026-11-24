"""
Rerankers: re-score the hybrid candidates with a model that reads query AND chunk together.

BI-ENCODER vs CROSS-ENCODER
  Retrieval embedded query and chunk separately (fast, approximate). A cross-encoder feeds
  "[CLS] query [SEP] chunk" through one transformer and outputs a relevance score: far more
  accurate, far too slow for 31k chunks, fine for the top 50 candidates.

PROVIDERS (rerank.provider)
  none     keep the RRF order
  local    BAAI/bge-reranker-base via sentence-transformers CrossEncoder (free, offline, MPS)
  cohere   POST /v2/rerank   (trial key: counts toward the 1,000 calls/month)
  jina     POST /v1/rerank   (free token grant)
  voyage   POST /v1/rerank   (free tokens, low limits without a payment method)
  Online rerankers go through the same client-side rate limiter and retry policy as the
  embedders (finsight.providers). Rerank on chunk.text (what the LLM will read), not embed_text.
"""

from __future__ import annotations

import time
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from finsight.config import RerankCfg, Secrets
from finsight.errors import MissingAPIKeyError
from finsight.providers.http import post_json
from finsight.providers.ratelimit import RateLimiter, UsageStore, estimate_tokens
from finsight.retrieval.base import Reranker
from finsight.schemas import RetrievedChunk

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

API_RERANKERS: dict[str, dict[str, str]] = {
    "cohere": {
        "url": "https://api.cohere.com/v2/rerank",
        "model": "rerank-v3.5",
        "key_url": "https://dashboard.cohere.com/api-keys",
        "top": "top_n",
        "rows": "results",
    },
    "jina": {
        "url": "https://api.jina.ai/v1/rerank",
        "model": "jina-reranker-v2-base-multilingual",
        "key_url": "https://jina.ai/api-dashboard/",
        "top": "top_n",
        "rows": "results",
    },
    "voyage": {
        "url": "https://api.voyageai.com/v1/rerank",
        "model": "rerank-2.5-lite",
        "key_url": "https://dashboard.voyageai.com/",
        "top": "top_k",
        "rows": "data",
    },
}


def _reorder(
    candidates: list[RetrievedChunk], scores: list[tuple[int, float]], top_n: int, name: str
) -> list[RetrievedChunk]:
    ranked = sorted(scores, key=lambda s: (-s[1], candidates[s[0]].chunk.chunk_id))[:top_n]
    return [
        RetrievedChunk(
            chunk=candidates[i].chunk,
            score=float(score),
            rank=r + 1,
            stage="rerank",
            scores_by_stage={**candidates[i].scores_by_stage, name: float(score)},
        )
        for r, (i, score) in enumerate(ranked)
    ]


class NoopReranker(Reranker):
    name = "none"

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        return candidates[:top_n]


class LocalCrossEncoderReranker(Reranker):
    name = "local"

    def __init__(
        self, model: str, device: str = "auto", max_length: int = 512, batch_size: int = 16
    ):
        self.model_name, self._device = model, device
        self.max_length, self.batch_size = max_length, batch_size

    @cached_property
    def model(self) -> CrossEncoder:
        from sentence_transformers import CrossEncoder

        from finsight.embedding.sentence_transformer import resolve_device

        return CrossEncoder(
            self.model_name, max_length=self.max_length, device=resolve_device(self._device)
        )

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return _reorder(candidates, list(enumerate(map(float, scores))), top_n, self.name)


class APIReranker(Reranker):
    def __init__(
        self,
        provider: str,
        cfg: RerankCfg,
        api_key: str,
        *,
        usage_dir: Path | None = None,
        client: httpx.Client | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        self.name = provider
        self.spec = API_RERANKERS[provider]
        self.model_name = cfg.api_model or self.spec["model"]
        self.api_key, self._sleep, self.cfg = api_key, sleep, cfg
        self.client = client or httpx.Client(timeout=cfg.rate_limit.timeout_s)
        rl = cfg.rate_limit
        self.limiter = RateLimiter(
            rpm=rl.rpm,
            tpm=rl.tpm,
            rpd=rl.rpd,
            rpmonth=rl.rpmonth,
            usage=UsageStore(usage_dir / f"{provider}.json" if usage_dir else None),
            name=provider,
            sleep=sleep,
        )

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        docs = [c.chunk.text for c in candidates]
        self.limiter.acquire(estimate_tokens(query) * len(docs) + sum(map(estimate_tokens, docs)))
        body = {
            "model": self.model_name,
            "query": query,
            "documents": docs,
            self.spec["top"]: min(top_n, len(docs)),
        }
        data = post_json(
            self.client,
            self.spec["url"],
            body=body,
            headers={"Authorization": f"Bearer {self.api_key}"},
            provider=self.name,
            key_hint=f"{self.name.upper()}_API_KEY",
            max_retries=self.cfg.rate_limit.max_retries,
            sleep=self._sleep,
        )
        rows = data[self.spec["rows"]]
        return _reorder(
            candidates,
            [(int(r["index"]), float(r["relevance_score"])) for r in rows],
            top_n,
            self.name,
        )


def build_reranker(
    cfg: RerankCfg,
    *,
    secrets: Secrets | None = None,
    device: str = "auto",
    usage_dir: Path | None = None,
) -> Reranker:
    if cfg.provider == "none":
        return NoopReranker()
    if cfg.provider == "local":
        return LocalCrossEncoderReranker(cfg.model, device, cfg.max_length, cfg.batch_size)
    key = secrets.for_provider(cfg.provider) if secrets else None
    if key is None:
        raise MissingAPIKeyError(
            f"rerank.provider={cfg.provider} needs {cfg.provider.upper()}_API_KEY in .env "
            f"(free key: {API_RERANKERS[cfg.provider]['key_url']})"
        )
    return APIReranker(cfg.provider, cfg, key.get_secret_value(), usage_dir=usage_dir)
