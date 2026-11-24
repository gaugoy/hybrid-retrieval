"""
End-to-end pipeline comparison on a question SAMPLE (P7; P8 adds the LLM judge).

The sample is seeded and stratified: `null_share` of it are unanswerable questions (MultiHop-RAG
null queries, gold_answer=None), where the right behaviour is to abstain.

Per question and pipeline:
  evidence_hit    a retrieved (or, for the agent, read) chunk comes from an evidence document
  cited_evidence  a CITED chunk comes from an evidence document
  answer_match    the gold answer appears in the answer (case-insensitive; short MultiHop answers)
  correct_abstain / false_abstain   abstained on a null / on an answerable question
  llm_calls, latency_ms, degraded fallbacks
"""

from __future__ import annotations

import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from finsight.config import Settings
from finsight.container import Container
from finsight.data import datasets as ds
from finsight.schemas import EvalRecord


def sample_records(
    settings: Settings, n: int, null_share: float = 0.2, seed: int = 7
) -> list[EvalRecord]:
    recs = ds.eval_records(settings)
    rng = random.Random(seed)
    nulls = [r for r in recs if r.gold_answer is None]
    rest = [r for r in recs if r.gold_answer is not None]
    k_null = min(len(nulls), round(n * null_share))
    picked = rng.sample(nulls, k_null) + rng.sample(rest, min(len(rest), n - k_null))
    rng.shuffle(picked)
    return picked


def run_pipeline_sample(
    settings: Settings,
    pipelines: list[str],
    *,
    n: int = 50,
    null_share: float = 0.2,
    seed: int = 7,
    progress: bool = True,
    out_dir: Path | None = None,
) -> Path:
    records = sample_records(settings, n, null_share, seed)
    rows: list[dict[str, Any]] = []
    for name in pipelines:
        s = settings.model_copy(update={"pipeline": name})
        s.root, s.secrets, s.paths = settings.root, settings.secrets, settings.paths
        c = Container(s)
        try:
            pipe = c.pipeline
            for rec in tqdm(records, desc=name, unit="q", disable=not progress):
                t0 = time.perf_counter()
                ans = pipe.answer(rec.question)
                gold_docs = {e.doc_name for e in rec.evidence}
                is_null = rec.gold_answer is None
                rows.append(
                    {
                        "pipeline": name,
                        "qid": rec.qid,
                        "question_type": rec.question_type,
                        "is_null": is_null,
                        "abstained": ans.abstained,
                        "correct_abstain": is_null and ans.abstained,
                        "false_abstain": (not is_null) and ans.abstained,
                        "evidence_hit": bool(gold_docs & {r.chunk.doc_name for r in ans.retrieved}),
                        "cited_evidence": bool(gold_docs & {ci.doc_name for ci in ans.citations}),
                        "answer_match": (not is_null)
                        and str(rec.gold_answer).lower() in ans.answer.lower(),
                        "llm_calls": int(ans.usage.get("llm_calls", 0))
                        or (getattr(pipe, "llm_calls", 0) + 1),
                        "latency_ms": round((time.perf_counter() - t0) * 1000),
                        "degraded": ans.degraded,
                        "answer": ans.answer[:300],
                        "gold": rec.gold_answer,
                    }
                )
        finally:
            c.close()
    df = pd.DataFrame(rows)
    out = out_dir or settings.paths.results_dir / (
        f"{datetime.now():%Y-%m-%d}_pipelines_{settings.dataset}_n{len(records)}"
    )
    out.mkdir(parents=True, exist_ok=True)
    df.to_json(out / "per_question.jsonl", orient="records", lines=True)
    (out / "report.md").write_text(sample_report(df, settings, len(records), seed))
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("pipeline", sort=False)
    ans = df[~df["is_null"]].groupby("pipeline", sort=False)
    nul = df[df["is_null"]].groupby("pipeline", sort=False)
    out = pd.DataFrame(
        {
            "n": g.size(),
            "evidence_hit": ans["evidence_hit"].mean(),
            "cited_evidence": ans["cited_evidence"].mean(),
            "answer_match": ans["answer_match"].mean(),
            "false_abstain": ans["false_abstain"].mean(),
            "correct_abstain (null)": nul["correct_abstain"].mean(),
            "LLM calls/q": g["llm_calls"].mean(),
            "p50 s": g["latency_ms"].median() / 1000,
            "fell back": g["degraded"].apply(lambda d: d.map(len).gt(0).mean()),
        }
    )
    return out.round(3)


def sample_report(df: pd.DataFrame, settings: Settings, n: int, seed: int) -> str:
    return "\n".join(
        [
            f"# Pipelines on {settings.dataset} (sample of {n}, seed {seed})",
            "",
            f"- LLM: `{settings.llm.preset}` · retrieval `{settings.retrieval.mode}` · "
            f"filters {settings.retrieval.filter_fields} · chunking "
            f"`{settings.chunking.strategy}_{settings.chunking.chunk_tokens}`",
            "- Answerable columns exclude null queries; `correct_abstain` is on null queries only.",
            "",
            summarize(df).to_markdown(),
            "",
        ]
    )
