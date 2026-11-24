"""A tiny end-to-end corpus: fixture questions + synthetic pages, chunked and indexed with the
FakeEmbedder (preset "fake"), so retrieval/generation/API tests run offline in seconds."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from finsight.chunking import pipeline as cp
from finsight.chunking.tokenizer import WhitespaceTokenizer
from finsight.config import Settings, load_settings
from finsight.data import financebench as fb
from finsight.indexing import pipeline as ixp

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "financebench"
GOLD_TEXT = {
    ("ACME_2020_10K", 59): "Acme capital expenditure FY2020: purchases of property plant and "
    "equipment were 1,577 million dollars in the cash flow statement.",
    (
        "ACME_2021_10K",
        40,
    ): "Acme operating margin improved in FY2021 to 18 percent from 15 percent.",
    ("ACME_2021_10K", 0): "Acme 2021 annual report operating margin highlights.",
    ("BETA_2019_10Q", 12): "Beta debt maturing next year totals 250 million dollars of notes.",
}


def tiny_settings(root: Path, **overrides: object) -> Settings:
    base = {
        "embedding": {"preset": "fake"},
        "chunking": {"strategy": "page_aware", "chunk_tokens": 60, "overlap_tokens": 0},
        "rerank": {"provider": "none"},
        "retrieval": {"mode": "hybrid", "filter_fields": ["company"]},
    }
    for k, v in overrides.items():
        base[k] = {**base.get(k, {}), **v} if isinstance(v, dict) else v  # type: ignore[dict-item]
    return load_settings(root=root, env={}, overrides=base)


@pytest.fixture
def tiny(project_root: Path) -> Settings:
    with (project_root / "configs" / "embedders.yaml").open("a") as f:
        f.write("\nfake: {provider: fake, model: fake, dimensions: 32}\n")
    s = tiny_settings(project_root)
    fb.raw_dir(s).mkdir(parents=True)
    for name in (fb.QA_FILE, fb.DOCS_FILE):
        shutil.copy(FIXTURES / name, fb.raw_dir(s) / name)
    rows = []
    for doc, n in (("ACME_2020_10K", 62), ("ACME_2021_10K", 43), ("BETA_2019_10Q", 15)):
        for i in range(n):
            text = GOLD_TEXT.get((doc, i), f"{doc} page {i} general business discussion item {i}.")
            rows.append({"doc_name": doc, "page_idx": i, "text": text, "n_chars": len(text)})
    cp.chunk_corpus(s, tokenizer=WhitespaceTokenizer(), progress=False, pages_df=pd.DataFrame(rows))
    ixp.index_corpus(s, progress=False)
    return s
