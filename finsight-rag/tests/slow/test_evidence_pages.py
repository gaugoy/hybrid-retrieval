"""Real-data check (slow): parsed pages contain FinanceBench's evidence at the gold page index.

Run locally after steps 01 and 03:  uv run pytest -m slow
"""

from __future__ import annotations

import pytest

from finsight.config import load_settings
from finsight.ingestion import pipeline as ip

pytestmark = pytest.mark.slow


def test_at_least_95_percent_of_evidence_found_on_gold_page() -> None:
    settings = load_settings()
    try:
        pages = ip.load_pages_df(settings)
    except Exception:
        pytest.skip("pages.parquet not built yet (run scripts/03_parse_pdfs.py)")
    ev = ip.evidence_check(settings, pages)
    assert len(ev) == 189
    assert (ev["overlap"] >= 0.8).mean() >= 0.95
    assert (ev["best_page_idx"] == ev["page_idx"]).all(), "no off-by-one page offsets"
