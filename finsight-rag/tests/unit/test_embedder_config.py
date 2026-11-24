"""Embedding presets: resolution order, env selection, catalog estimates."""

from __future__ import annotations

from pathlib import Path

import pytest

from finsight.config import EmbeddingCfg, load_settings
from finsight.embedding.catalog import catalog_df, estimate
from finsight.errors import ConfigError


def test_default_preset_is_local_bge_small(project_root: Path) -> None:
    e = load_settings(root=project_root, env={}).embedding
    assert (e.preset, e.provider, e.model, e.dimensions) == (
        "bge-small",
        "local",
        "BAAI/bge-small-en-v1.5",
        384,
    )
    assert (
        e.query_template.startswith("Represent this sentence") and e.document_template == "{text}"
    )
    assert e.allow_paid is False


def test_env_selects_preset_and_explicit_values_win(project_root: Path) -> None:
    e = load_settings(
        root=project_root, env={"EMBEDDER": "gemini-001", "FINSIGHT__EMBEDDING__DIMENSIONS": "1536"}
    ).embedding
    assert e.provider == "gemini" and e.model == "gemini-embedding-001"
    assert e.dimensions == 1536  # explicit override beats the preset's 768
    assert e.query_params == {"taskType": "RETRIEVAL_QUERY"}
    assert e.rate_limit.rpd == 1000 and e.query_template == "{text}"  # no bge leftovers


def test_unknown_preset_and_bad_template(project_root: Path) -> None:
    with pytest.raises(ConfigError, match="Unknown embedding preset 'nope'"):
        load_settings(root=project_root, env={"EMBEDDER": "nope"})
    with pytest.raises(ValueError, match="text"):
        EmbeddingCfg(query_template="no placeholder")


def test_every_preset_in_the_catalog_is_valid(project_root: Path) -> None:
    s = load_settings(root=project_root, env={})
    df = catalog_df(s, ["some chunk text"] * 10)
    assert {
        "bge-small",
        "gemini-001",
        "gemini-2",
        "mistral-embed",
        "cohere-v4",
        "openai-3-small",
    } <= set(df["preset"])
    assert set(df.loc[df["provider"] == "openai", "cost"]) == {"PAID"}
    assert set(df.loc[df["provider"] != "openai", "cost"]) == {"free"}
    assert df.loc[df["preset"] == "gemini-001", "key"].item().startswith("missing")


def test_estimate_requests_minutes_and_cost() -> None:
    cfg = EmbeddingCfg(
        provider="openai",
        model="text-embedding-3-small",
        rate_limit={"rpm": 10, "max_batch_texts": 4},
    )
    est = estimate(cfg, ["x" * 350] * 10)  # 101 estimated tokens each
    assert est["requests"] == 3 and est["est_tokens"] == 1010
    assert est["minutes_at_limits"] == pytest.approx(0.3)
    assert est["usd"] == 0.0  # rounds to 0 at this size; price is applied
