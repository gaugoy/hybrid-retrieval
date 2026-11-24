"""Token -> USD estimate per LLM call. Prices live in configs/llms.yaml (price_in / price_out,
USD per 1M tokens), not in code; local models and free tiers cost 0."""

from __future__ import annotations

from finsight.config import LLMCfg


def estimate_cost(cfg: LLMCfg, input_tokens: int, output_tokens: int) -> float:
    return round((input_tokens * cfg.price_in + output_tokens * cfg.price_out) / 1e6, 6)
