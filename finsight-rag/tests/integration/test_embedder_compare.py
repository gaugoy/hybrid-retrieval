"""compare_embedders end to end with fake presets: indexes, metrics, skips, subset, report."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from finsight.chunking import pipeline as cp
from finsight.chunking.tokenizer import WhitespaceTokenizer
from finsight.config import Settings, load_settings
from finsight.data import financebench as fb
from finsight.evaluation.embedder_compare import compare_embedders, save_report
from finsight.indexing.qdrant_store import open_client

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "financebench"

PRESETS = """
fake-a: {provider: fake, model: fake, dimensions: 32}
fake-b: {provider: fake, model: fake, dimensions: 16}
voyage-nokey: {provider: voyage, model: voyage-3.5-lite, dimensions: 1024}
"""


@pytest.fixture
def settings(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    (project_root / "configs" / "embedders.yaml").write_text(PRESETS)
    s = load_settings(
        root=project_root,
        overrides={
            "embedding": {"preset": "fake-a"},
            "chunking": {"strategy": "page_aware", "chunk_tokens": 60, "overlap_tokens": 0},
        },
    )
    fb.raw_dir(s).mkdir(parents=True)
    for name in (fb.QA_FILE, fb.DOCS_FILE):
        shutil.copy(FIXTURES / name, fb.raw_dir(s) / name)
    rows = []
    for doc, n in (("ACME_2020_10K", 62), ("ACME_2021_10K", 43), ("BETA_2019_10Q", 15)):
        for i in range(n):
            text = f"{doc} page {i} " + " ".join(f"w{i}x{j}" for j in range(20))
            if i == 59:
                text = "capital expenditure purchases of property plant and equipment " + text
            rows.append({"doc_name": doc, "page_idx": i, "text": text, "n_chars": len(text)})
    cp.chunk_corpus(s, tokenizer=WhitespaceTokenizer(), progress=False, pages_df=pd.DataFrame(rows))
    return s


def test_compare_two_fakes_and_skip_missing_key(settings: Settings) -> None:
    client = open_client(settings)
    try:
        comp = compare_embedders(
            settings, ["fake-a", "fake-b", "voyage-nokey"], progress=False, client=client
        )
        n_questions = len(fb.load_eval_records(settings))
        assert list(comp.summary.index) == ["fake-a", "fake-b"], comp.skipped
        assert (comp.summary["n"] == n_questions).all()
        assert set(comp.summary["dims"]) == {32, 16}
        assert comp.summary[["hit@10", "mrr", "ndcg@10"]].le(1).all().all()
        assert "voyage-nokey" in comp.skipped and "VOYAGE_API_KEY" in comp.skipped["voyage-nokey"]

        sub = compare_embedders(
            settings, ["fake-a"], limit_questions=1, progress=False, client=client
        )
        assert sub.summary.loc["fake-a", "n"] == 1
        assert sub.reports["fake-a"]["collection"].endswith("__q1")
        assert sub.reports["fake-a"]["n_chunks"] < comp.reports["fake-a"]["n_chunks"]
    finally:
        client.close()

    out = save_report(settings, comp)
    text = (out / "report.md").read_text()
    assert "fake-a" in text and "Skipped" in text and (out / "per_question.jsonl").is_file()
