"""chunk_corpus end to end on synthetic pages, with the whitespace tokenizer."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from finsight.chunking import pipeline as cp
from finsight.chunking.tokenizer import WhitespaceTokenizer
from finsight.config import Settings, load_settings
from finsight.data import financebench as fb

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "financebench"
TOK = WhitespaceTokenizer()


@pytest.fixture
def settings(project_root: Path) -> Settings:
    s = load_settings(root=project_root, env={})
    fb.raw_dir(s).mkdir(parents=True)
    for name in (fb.QA_FILE, fb.DOCS_FILE):
        shutil.copy(FIXTURES / name, fb.raw_dir(s) / name)
    return s


@pytest.fixture
def pages_df() -> pd.DataFrame:
    rows = []
    for doc, n_pages in (("ACME_2020_10K", 62), ("ACME_2021_10K", 43), ("BETA_2019_10Q", 15)):
        for i in range(n_pages):
            text = " ".join(f"{doc.lower()}w{i}x{j}" for j in range(150))
            if (doc, i) == ("ACME_2020_10K", 59):
                text = "Purchases of property, plant and equipment (PP&E) | (1,577)\n" + text
            rows.append(
                {
                    "doc_name": doc,
                    "page_idx": i,
                    "text": text,
                    "n_chars": len(text),
                    "has_table": False,
                }
            )
    return pd.DataFrame(rows)


def with_chunking(settings: Settings, **chunking: object) -> Settings:
    return load_settings(root=settings.root, env={}, overrides={"chunking": chunking})


def test_chunk_corpus_writes_parquet_and_skips_when_unchanged(
    settings: Settings, pages_df: pd.DataFrame
) -> None:
    s = with_chunking(settings, strategy="page_aware", chunk_tokens=100, overlap_tokens=10)
    first = cp.chunk_corpus(s, tokenizer=TOK, progress=False, pages_df=pages_df)
    assert first.name == "page_aware_100" and first.n_docs == 3 and first.n_chunks > 120
    df = cp.load_chunks_df(s)
    assert {"chunk_id", "page_start", "page_end", "text", "embed_text", "n_tokens"} <= set(
        df.columns
    )
    assert cp.chunk_corpus(s, tokenizer=TOK, progress=False, pages_df=pages_df).skipped
    assert not cp.chunk_corpus(
        s, tokenizer=TOK, progress=False, pages_df=pages_df, force=True
    ).skipped
    changed = with_chunking(settings, strategy="page_aware", chunk_tokens=100, overlap_tokens=20)
    assert not cp.chunk_corpus(changed, tokenizer=TOK, progress=False, pages_df=pages_df).skipped


def test_parent_child_writes_parents_file(settings: Settings, pages_df: pd.DataFrame) -> None:
    s = with_chunking(settings, strategy="parent_child", child_tokens=60, parent_tokens=120)
    report = cp.chunk_corpus(s, tokenizer=TOK, progress=False, pages_df=pages_df)
    assert report.n_parents == len(pages_df)
    parents = cp.load_chunks_df(s, parents=True)
    children = cp.load_chunks_df(s)
    assert set(children["parent_id"]) <= set(parents["chunk_id"])


def test_contextual_headers_change_name_and_embed_text(
    settings: Settings, pages_df: pd.DataFrame
) -> None:
    s = with_chunking(settings, strategy="page_aware", chunk_tokens=100, contextual_headers=True)
    report = cp.chunk_corpus(s, tokenizer=TOK, progress=False, pages_df=pages_df)
    assert report.name == "page_aware_100_ctx"
    df = cp.load_chunks_df(s)
    assert df["embed_text"].str.startswith("Acme | 10k FY2020 | page ").any()
    assert (df["n_tokens"] <= 100 - 24).all()  # budget reduced by the header reserve


def test_stats_and_evidence_coverage(settings: Settings, pages_df: pd.DataFrame) -> None:
    s = with_chunking(settings, strategy="fixed", chunk_tokens=100, overlap_tokens=10)
    cp.chunk_corpus(s, tokenizer=TOK, progress=False, pages_df=pages_df)
    df = cp.load_chunks_df(s)
    stats = cp.chunk_stats(df, TOK)
    assert stats["tokens_max"] <= 100 and stats["over_limit_share"] == 0.0
    assert 0 < stats["multi_page_share"] < 1  # fixed windows cross page boundaries
    coverage = cp.evidence_coverage(s, df)
    assert len(coverage) == 4  # the fixture has 4 evidence items
    summary = cp.coverage_summary(coverage)
    assert set(summary) == {
        "evidence_in_one_chunk",
        "best_chunk_overlap_mean",
        "union_overlap_mean",
        "chunks_per_evidence_page",
    }
