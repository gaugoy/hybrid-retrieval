"""run_pipeline_sample on the tiny corpus with FakeLLM."""

from __future__ import annotations

from finsight.config import Settings
from finsight.evaluation.pipeline_sample import run_pipeline_sample, sample_records


def test_sample_report(tiny: Settings) -> None:
    assert len(sample_records(tiny, n=3)) == 3  # no nulls in FinanceBench: all answerable
    out = run_pipeline_sample(tiny, ["hybrid_rerank", "naive"], n=3, progress=False)
    report = (out / "report.md").read_text()
    assert "hybrid_rerank" in report and "naive" in report and "evidence_hit" in report
