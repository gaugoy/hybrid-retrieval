"""
Online embedding providers behind the same Embedder interface as the local model.

THREE WIRE FORMATS COVER SEVEN PROVIDERS
  openai-style  POST {base}/embeddings  {"model", "input": [...]} -> data[i].embedding
                OpenAI, Mistral, Jina, Voyage, and a local Ollama server all speak it; they differ
                only in the base URL, the key, and an optional role field
                (Jina "task": "retrieval.query", Voyage "input_type": "query")
  gemini        POST {base}/models/{model}:batchEmbedContents  {"requests": [...]}
                -> embeddings[i].values   (key in the x-goog-api-key header)
  cohere        POST {base}/embed  {"texts": [...], "input_type", "embedding_types": ["float"]}
                -> embeddings.float[i]
  Plain httpx, no vendor SDKs: one retry/rate-limit path for all, and nothing extra to install.

WHAT EVERY REMOTE CALL GOES THROUGH
  1. templates      query_template / document_template wrap the text (Gemini 2, nomic need them)
  2. batching       ≤ max_batch_texts texts and ≤ max_batch_tokens estimated tokens per request
  3. budget         max_tokens_per_run: stop before sending more (QuotaExhaustedError)
  4. rate limiter   rpm spacing, tpm window, daily/monthly caps (providers/ratelimit.py)
  5. post_json      retries 429/5xx with backoff, quota-429 stops the run (providers/http.py)
  6. L2-normalize   always on the client: some providers don't re-normalize truncated vectors
  Query vectors are cached too (same SQLite file as documents), so re-running an evaluation
  costs zero requests.

PAID GUARD
  OpenAI has no free tier. build_embedder refuses paid-only providers unless
  embedding.allow_paid is true, so a typo in .env can't spend money.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np
from tqdm import tqdm

from finsight.config import API_KEY_ENV, EmbeddingCfg
from finsight.embedding.base import Embedder, embedder_slug, l2_normalize
from finsight.errors import ProviderError, QuotaExhaustedError
from finsight.providers.http import post_json
from finsight.providers.ratelimit import RateLimiter, UsageStore, estimate_tokens

if TYPE_CHECKING:
    from finsight.embedding.cache import EmbeddingCache


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    wire: str  # "openai" | "gemini" | "cohere"
    base_url: str
    needs_key: bool = True
    key_url: str = ""  # where to create a (free) key
    paid_only: bool = False
    dims_param: str | None = None  # request field for the output size, if supported
    price_per_mtok: dict[str, float] = field(default_factory=dict)  # model -> USD / 1M tokens

    @property
    def key_env(self) -> tuple[str, ...]:
        return API_KEY_ENV.get(self.name, ())


PROVIDERS: dict[str, ProviderSpec] = {
    "gemini": ProviderSpec(
        "gemini",
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta",
        key_url="https://aistudio.google.com/apikey",
        dims_param="outputDimensionality",
    ),
    "mistral": ProviderSpec(
        "mistral", "openai", "https://api.mistral.ai/v1", key_url="https://console.mistral.ai/"
    ),
    "jina": ProviderSpec(
        "jina",
        "openai",
        "https://api.jina.ai/v1",
        key_url="https://jina.ai/api-dashboard/",
        dims_param="dimensions",
    ),
    "voyage": ProviderSpec(
        "voyage",
        "openai",
        "https://api.voyageai.com/v1",
        key_url="https://dashboard.voyageai.com/",
        dims_param="output_dimension",
    ),
    "cohere": ProviderSpec(
        "cohere",
        "cohere",
        "https://api.cohere.com/v2",
        key_url="https://dashboard.cohere.com/api-keys",
        dims_param="output_dimension",
    ),
    "ollama": ProviderSpec("ollama", "openai", "http://localhost:11434/v1", needs_key=False),
    "openai": ProviderSpec(
        "openai",
        "openai",
        "https://api.openai.com/v1",
        key_url="https://platform.openai.com/api-keys",
        paid_only=True,
        dims_param="dimensions",
        price_per_mtok={"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13},
    ),
}


def _short_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:8]


def remote_identity(cfg: EmbeddingCfg) -> str:
    """What the vectors depend on: provider, model, size, and how documents were presented."""
    ident = f"{cfg.provider}/{cfg.model}" + (f"@{cfg.dimensions}" if cfg.dimensions else "")
    if cfg.document_template != "{text}" or cfg.document_params:
        ident += "#" + _short_hash([cfg.document_template, cfg.document_params])
    return ident


@dataclass
class RemoteStats:
    requests: int = 0
    est_tokens: int = 0
    billed_tokens: int = 0  # when the provider reports usage
    query_cache_hits: int = 0


class RemoteEmbedder(Embedder):
    wire = "base"

    def __init__(
        self,
        cfg: EmbeddingCfg,
        spec: ProviderSpec,
        api_key: str | None,
        *,
        usage_dir: Path | None = None,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg, self.spec, self.api_key = cfg, spec, api_key
        self.provider = spec.name
        self.model_name = remote_identity(cfg)
        self.slug = embedder_slug(cfg)
        self.base_url = (cfg.base_url or spec.base_url).rstrip("/")
        self._dim = cfg.dimensions
        self._sleep = sleep
        self.client = client or httpx.Client(timeout=cfg.rate_limit.timeout_s)
        rl = cfg.rate_limit
        usage_path = usage_dir / f"{spec.name}.json" if usage_dir else None
        self.limiter = RateLimiter(
            rpm=rl.rpm,
            tpm=rl.tpm,
            rpd=rl.rpd,
            rpmonth=rl.rpmonth,
            usage=UsageStore(usage_path),
            name=spec.name,
            sleep=sleep,
        )
        self.stats = RemoteStats()
        self.query_cache: EmbeddingCache | None = None

    # ------------------------------------------------------------------ Embedder API
    @property
    def device(self) -> str:
        return f"remote:{self.provider}"

    @property
    def dim(self) -> int:  # type: ignore[override]
        if self._dim is None:  # unknown size: one probe request
            self._dim = int(self._embed(["dimension probe"], "query").shape[1])
        return self._dim

    @property
    def preferred_batch(self) -> int:
        """How many texts embed_with_cache should hand over (and cache) at a time."""
        return self.cfg.rate_limit.max_batch_texts * 4

    def embed_documents(self, texts: list[str], progress: bool = False) -> np.ndarray:
        tpl = self.cfg.document_template
        return self._embed([tpl.format(text=t) for t in texts], "document", progress)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_queries([text])[0]

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        from finsight.embedding.cache import cache_key

        tpl = self.cfg.query_template
        formatted = [tpl.format(text=t) for t in texts]
        ident = f"{self.model_name}|query|{_short_hash([tpl, self.cfg.query_params])}"
        keys = [cache_key(ident, t) for t in formatted]
        found = self.query_cache.get_many(keys) if self.query_cache is not None else {}
        self.stats.query_cache_hits += sum(k in found for k in keys)
        todo = [i for i, k in enumerate(keys) if k not in found]
        if todo:
            vecs = self._embed([formatted[i] for i in todo], "query")
            new = {keys[i]: v for i, v in zip(todo, vecs, strict=True)}
            found.update(new)
            if self.query_cache is not None:
                self.query_cache.put_many(new)
        if not keys:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([found[k] for k in keys]).astype(np.float32)

    # ------------------------------------------------------------------ batching + limits
    def _batches(self, texts: list[str]) -> list[list[str]]:
        rl = self.cfg.rate_limit
        out: list[list[str]] = []
        cur: list[str] = []
        cur_tokens = 0
        for t in texts:
            n = estimate_tokens(t)
            full = len(cur) >= rl.max_batch_texts or (
                rl.max_batch_tokens is not None and cur and cur_tokens + n > rl.max_batch_tokens
            )
            if full:
                out.append(cur)
                cur, cur_tokens = [], 0
            cur.append(t)
            cur_tokens += n
        if cur:
            out.append(cur)
        return out

    def _embed(self, texts: list[str], role: str, progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim or 0), dtype=np.float32)
        chunks: list[np.ndarray] = []
        batches = self._batches(texts)
        for batch in tqdm(
            batches, desc=f"{self.provider} {role}s", unit="req", disable=not progress
        ):
            est = sum(estimate_tokens(t) for t in batch)
            budget = self.cfg.max_tokens_per_run
            if budget is not None and self.stats.est_tokens + est > budget:
                raise QuotaExhaustedError(
                    f"{self.provider}: max_tokens_per_run={budget:,} reached "
                    f"({self.stats.est_tokens:,} sent). Progress is cached; raise the budget "
                    "and re-run to continue."
                )
            self.limiter.acquire(est)
            vectors = self._request(batch, role)
            if len(vectors) != len(batch):
                raise ProviderError(
                    f"{self.provider}: sent {len(batch)} texts, got {len(vectors)} vectors"
                )
            arr = np.asarray(vectors, dtype=np.float32)
            if self._dim is not None and arr.shape[1] != self._dim:
                raise ProviderError(
                    f"{self.provider}: expected {self._dim}-dim vectors, got {arr.shape[1]} "
                    "(set embedding.dimensions to the model's real size)"
                )
            self._dim = int(arr.shape[1])
            self.stats.requests += 1
            self.stats.est_tokens += est
            chunks.append(arr)
        return l2_normalize(np.concatenate(chunks))

    def _post(self, url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        return post_json(
            self.client,
            url,
            body=body,
            headers=headers,
            provider=self.provider,
            key_hint=" or ".join(self.spec.key_env),
            max_retries=self.cfg.rate_limit.max_retries,
            sleep=self._sleep,
        )

    def _role_params(self, role: str) -> dict[str, Any]:
        return dict(self.cfg.query_params if role == "query" else self.cfg.document_params)

    def _dims(self) -> dict[str, Any]:
        if self.spec.dims_param and self.cfg.dimensions:
            return {self.spec.dims_param: self.cfg.dimensions}
        return {}

    @abstractmethod
    def _request(self, batch: list[str], role: str) -> list[list[float]]: ...


class OpenAIStyleEmbedder(RemoteEmbedder):
    """OpenAI, Mistral, Jina, Voyage, Ollama: POST {base}/embeddings."""

    wire = "openai"

    def _request(self, batch: list[str], role: str) -> list[list[float]]:
        body = {"model": self.cfg.model, "input": batch, **self._role_params(role), **self._dims()}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = self._post(f"{self.base_url}/embeddings", body, headers)
        usage = data.get("usage") or {}
        self.stats.billed_tokens += int(
            usage.get("total_tokens") or usage.get("prompt_tokens") or 0
        )
        rows = sorted(data["data"], key=lambda r: r.get("index", 0))
        return [r["embedding"] for r in rows]


class GeminiEmbedder(RemoteEmbedder):
    """Google Gemini API: POST {base}/models/{model}:batchEmbedContents."""

    wire = "gemini"

    def _request(self, batch: list[str], role: str) -> list[list[float]]:
        extra = {**self._role_params(role), **self._dims()}
        body = {
            "requests": [
                {"model": f"models/{self.cfg.model}", "content": {"parts": [{"text": t}]}, **extra}
                for t in batch
            ]
        }
        headers = {"x-goog-api-key": self.api_key or ""}
        data = self._post(
            f"{self.base_url}/models/{self.cfg.model}:batchEmbedContents", body, headers
        )
        return [e["values"] for e in data["embeddings"]]


class CohereEmbedder(RemoteEmbedder):
    """Cohere v2: POST {base}/embed."""

    wire = "cohere"

    def _request(self, batch: list[str], role: str) -> list[list[float]]:
        body = {
            "model": self.cfg.model,
            "texts": batch,
            "embedding_types": ["float"],
            **self._role_params(role),
            **self._dims(),
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = self._post(f"{self.base_url}/embed", body, headers)
        billed = (data.get("meta") or {}).get("billed_units") or {}
        self.stats.billed_tokens += int(billed.get("input_tokens") or 0)
        return list(data["embeddings"]["float"])


WIRES: dict[str, type[RemoteEmbedder]] = {
    "openai": OpenAIStyleEmbedder,
    "gemini": GeminiEmbedder,
    "cohere": CohereEmbedder,
}
