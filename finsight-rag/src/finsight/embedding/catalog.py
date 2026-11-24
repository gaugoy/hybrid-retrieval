"""
The embedder catalog: every preset in configs/embedders.yaml, whether it can run right now
(key present? paid?), and what indexing would cost in requests, time and money.

ESTIMATES
  tokens    from the chunk texts (len / 3.5, the same conservative estimate the limiter uses)
  requests  number of batches (max_batch_texts / max_batch_tokens)
  minutes   the slowest of requests / rpm and tokens / tpm  (the rate limits, not the network)
  days      requests / rpd (or months for rpmonth): > 1 means the run spans several days, which
            is fine: the cache resumes it
  usd       price per 1M tokens × tokens (0 for free tiers and local models)
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from finsight.config import EmbeddingCfg, Settings, load_embedder_presets, resolve_embedding_preset
from finsight.embedding.base import embedder_slug
from finsight.providers.ratelimit import estimate_tokens

# measured on an M5 (MPS), page_aware_384 chunks; others unknown until you run them
LOCAL_CHUNKS_PER_S = {"BAAI/bge-small-en-v1.5": 106.0}


def preset_cfg(settings: Settings, name: str) -> EmbeddingCfg:
    generic = {
        k: getattr(settings.embedding, k)
        for k in ("device", "batch_size", "allow_paid", "max_tokens_per_run")
    }
    return EmbeddingCfg.model_validate(
        resolve_embedding_preset(settings.root, {"preset": name, **generic})
    )


def key_status(settings: Settings, cfg: EmbeddingCfg) -> str:
    if cfg.provider in ("local", "fake"):
        return "not needed"
    from finsight.embedding.remote import PROVIDERS

    spec = PROVIDERS[cfg.provider]
    if not spec.needs_key:
        return "not needed"
    return "set" if settings.secrets.for_provider(cfg.provider) else f"missing ({spec.key_env[0]})"


def estimate(cfg: EmbeddingCfg, texts: list[str]) -> dict[str, Any]:
    tokens = [estimate_tokens(cfg.document_template.format(text=t)) for t in texts]
    total = sum(tokens)
    rl = cfg.rate_limit
    requests, cur_n, cur_t = 0, 0, 0
    for n in tokens:
        if cur_n and (
            cur_n >= rl.max_batch_texts
            or (rl.max_batch_tokens is not None and cur_t + n > rl.max_batch_tokens)
        ):
            requests, cur_n, cur_t = requests + 1, 0, 0
        cur_n, cur_t = cur_n + 1, cur_t + n
    requests += 1 if cur_n else 0
    minutes = max(
        requests / rl.rpm if rl.rpm else 0.0,
        total / rl.tpm if rl.tpm else 0.0,
    )
    price = 0.0
    if cfg.provider not in ("local", "fake"):
        from finsight.embedding.remote import PROVIDERS

        price = PROVIDERS[cfg.provider].price_per_mtok.get(cfg.model, 0.0)
    return {
        "texts": len(texts),
        "est_tokens": total,
        "requests": requests,
        "minutes_at_limits": round(minutes, 1),
        "days_at_rpd": round(requests / rl.rpd, 2) if rl.rpd else None,
        "months_at_rpmonth": round(requests / rl.rpmonth, 2) if rl.rpmonth else None,
        "quota": (
            f"{requests / rl.rpd:.0%} of daily cap"
            if rl.rpd
            else f"{requests / rl.rpmonth:.0%} of monthly cap"
            if rl.rpmonth
            else "-"
        ),
        "usd": round(price * total / 1e6, 2),
    }


def catalog_df(settings: Settings, texts: list[str] | None = None) -> pd.DataFrame:
    """One row per preset: provider, model, dims, paid?, key status, (estimates), notes."""
    from finsight.embedding.remote import PROVIDERS

    rows = []
    for name in load_embedder_presets(settings.root):
        cfg = preset_cfg(settings, name)
        paid = cfg.provider in PROVIDERS and PROVIDERS[cfg.provider].paid_only
        row: dict[str, Any] = {
            "preset": name,
            "provider": cfg.provider,
            "model": cfg.model,
            "dims": cfg.dimensions,
            "cost": "PAID" if paid else "free",
            "key": key_status(settings, cfg),
            "slug": embedder_slug(cfg),
        }
        if texts is not None:
            est = estimate(cfg, texts)
            if cfg.provider in ("local", "ollama"):  # no API limits: compute-bound
                speed = LOCAL_CHUNKS_PER_S.get(cfg.model)
                est["requests"] = 0
                est["minutes_at_limits"] = round(len(texts) / speed / 60, 1) if speed else math.nan
            row.update(est)
        row["notes"] = cfg.notes
        rows.append(row)
    return pd.DataFrame(rows)
