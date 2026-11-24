"""
Compare embedders (local and online) on DENSE retrieval over the same chunks.

FAIR BY CONSTRUCTION
  Same chunks (sized with one tokenizer: chunking.tokenizer), same questions, same k, no
  filters, no BM25, no reranker: the embedder is the only variable. (P5 adds the full pipeline
  on top, with whichever embedder wins.)

FREE-TIER FRIENDLY
  --limit-questions N indexes only the filings that the first N questions need (index name
  gets "__qN"). Smaller corpus = fewer requests, and every embedder sees the same subset, so the
  comparison stays fair (absolute scores are higher than on the full corpus: fewer distractors).
  Each embedder's vectors are cached, so a re-run (or a run stopped by a daily quota) only sends
  what is missing. An embedder that can't run (no key, paid, quota) is reported and skipped.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from qdrant_client import QdrantClient

from finsight.config import Settings
from finsight.data import financebench as fb
from finsight.embedding.catalog import preset_cfg
from finsight.errors import FinsightError
from finsight.evaluation.retrieval_metrics import score_question
from finsight.indexing import pipeline as ixp
from finsight.indexing.qdrant_store import QdrantStore, open_client
from finsight.observability.logging import get_logger

log = get_logger(__name__)

KS = (1, 5, 10)


@dataclass
class Comparison:
    summary: pd.DataFrame
    per_question: pd.DataFrame
    reports: dict[str, dict[str, Any]] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    subset: str = "all"
    tag: str = "all"  # "all" or "q<N>": part of the report folder name


def settings_for(settings: Settings, preset: str) -> Settings:
    """The same settings (profile, chunking, overrides, paths, keys) with another embedder."""
    s = settings.model_copy(update={"embedding": preset_cfg(settings, preset)})
    s.root, s.config_path, s.secrets = settings.root, settings.config_path, settings.secrets
    return s


def evaluate_dense(
    settings: Settings,
    preset: str,
    *,
    client: QdrantClient,
    limit_questions: int | None = None,
    k: int = 10,
    force: bool = False,
    progress: bool = True,
    embedder: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Index (or reuse) and evaluate one preset. Returns per-question rows and the IndexReport."""
    s = settings_for(settings, preset)
    records = fb.load_eval_records(s)
    doc_names, tag = None, None
    if limit_questions:
        doc_names, tag = ixp.subset_for_questions(s, limit_questions)
        records = records[:limit_questions]
    embedder = embedder or ixp.make_embedder(s)
    report = ixp.index_corpus(
        s,
        embedder=embedder,
        client=client,
        only="dense",
        doc_names=doc_names,
        tag=tag,
        force=force,
        progress=progress,
    )
    start = time.perf_counter()
    qvecs = embedder.embed_queries([r.question for r in records])
    store = QdrantStore(client)
    rows = []
    for rec, qv in zip(records, qvecs, strict=True):
        hits = store.search(report.collection, qv, k=max(k, max(KS)))
        ranked = [
            (h.payload["doc_name"], h.payload["page_start"], h.payload["page_end"]) for h in hits
        ]
        gold = [(ev.doc_name, ev.page_idx) for ev in rec.evidence]
        rows.append(
            {
                "preset": preset,
                "qid": rec.qid,
                "question_type": rec.question_type,
                "top1": f"{ranked[0][0]} p{ranked[0][1] + 1}" if ranked else "",
                **score_question(ranked, gold, KS),
            }
        )
    info = report.as_dict()
    info["query_s"] = round(time.perf_counter() - start, 2)
    info["dims"] = embedder.dim
    info["provider"] = getattr(embedder, "provider", "local")
    return pd.DataFrame(rows), info


def compare_embedders(
    settings: Settings,
    presets: Sequence[str],
    *,
    limit_questions: int | None = None,
    k: int = 10,
    force: bool = False,
    progress: bool = True,
    client: QdrantClient | None = None,
) -> Comparison:
    own = client is None
    client = client or open_client(settings)
    frames, reports, skipped = [], {}, {}
    try:
        for preset in presets:
            try:
                df, info = evaluate_dense(
                    settings,
                    preset,
                    client=client,
                    limit_questions=limit_questions,
                    k=k,
                    force=force,
                    progress=progress,
                )
            except FinsightError as exc:  # no key, paid, quota: report and continue
                log.warning("embedder skipped", extra={"preset": preset, "reason": str(exc)})
                skipped[preset] = f"{type(exc).__name__}: {exc}"
                continue
            frames.append(df)
            reports[preset] = info
    finally:
        if own:
            client.close()
    per_q = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return Comparison(
        summary=summarize(per_q, reports),
        per_question=per_q,
        reports=reports,
        skipped=skipped,
        subset=f"first {limit_questions} questions" if limit_questions else "all questions",
        tag=f"q{limit_questions}" if limit_questions else "all",
    )


def summarize(per_q: pd.DataFrame, reports: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if per_q.empty:
        return pd.DataFrame()
    metrics = ["hit@1", "hit@5", "hit@10", "recall@10", "mrr", "ndcg@10", "doc_hit@10"]
    summary = per_q.groupby("preset", sort=False)[metrics].mean().round(3)
    summary.insert(0, "n", per_q.groupby("preset", sort=False).size())
    extra = pd.DataFrame.from_dict(
        {
            p: {
                "provider": r["provider"],
                "dims": r["dims"],
                "chunks": r["n_chunks"],
                "embedded_now": r["cache_misses"],
                "embed_s": r["embed_s"],
                "api_requests": r["api_requests"],
                "index_mb": r["qdrant_mb"],
            }
            for p, r in reports.items()
        },
        orient="index",
    )
    return summary.join(extra)


def by_question_type(per_q: pd.DataFrame, metric: str = "hit@10") -> pd.DataFrame:
    return per_q.pivot_table(index="question_type", columns="preset", values=metric).round(3)


def disagreements(per_q: pd.DataFrame, metric: str = "hit@10") -> pd.DataFrame:
    """Questions where at least one embedder hits and another misses."""
    wide = per_q.pivot_table(index="qid", columns="preset", values=metric)
    return wide[(wide.max(axis=1) > 0) & (wide.min(axis=1) == 0)]


def save_report(settings: Settings, comp: Comparison) -> Path:
    out = settings.paths.results_dir / f"{datetime.now():%Y-%m-%d}_embedder_comparison_{comp.tag}"
    out.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Embedder comparison (dense retrieval only)",
        "",
        f"- Questions: {comp.subset} · chunking `{ixp.cp.config_name(settings)}` · no filters, "
        "no BM25, no reranker",
        "- Relevant = a retrieved chunk on a gold evidence page. Metrics: see "
        "`src/finsight/evaluation/retrieval_metrics.py`.",
        f"- Config fingerprint (base settings): `{settings.fingerprint()}`",
        "",
        "## Summary",
        "",
        comp.summary.to_markdown() if not comp.summary.empty else "(no embedder ran)",
        "",
    ]
    if not comp.per_question.empty:
        lines += [
            "## hit@10 by question type",
            "",
            by_question_type(comp.per_question).to_markdown(),
            "",
        ]
    if comp.skipped:
        lines += (
            ["## Skipped", ""] + [f"- **{p}**: {why}" for p, why in comp.skipped.items()] + [""]
        )
    (out / "report.md").write_text("\n".join(lines))
    if not comp.per_question.empty:
        comp.per_question.to_json(out / "per_question.jsonl", orient="records", lines=True)
    return out
