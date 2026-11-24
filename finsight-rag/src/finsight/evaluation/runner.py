"""
Retrieval evaluation runner: experiment YAML -> results/<date>_<name>/.

    for each chunking config in `sweep.chunking`:
        one Container (indexes, embedder, filters; the reranker model is shared across configs)
        for each variant: RetrievalService with the variant's overrides
            for each question: retrieve top max(k) -> score against the gold pages

Outputs: per_question.jsonl (one row per chunking × variant × question, incl. the extracted
filters, the top-3 (doc, page) and per-stage timings), report.md, config.yaml.
Retrieval-only: no LLM calls, so the full grid runs in minutes on a laptop.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

from finsight.chunking import pipeline as cp
from finsight.config import Settings, deep_merge, load_settings
from finsight.container import Container
from finsight.data import datasets as ds
from finsight.evaluation.retrieval_metrics import score_question
from finsight.retrieval.base import Reranker
from finsight.retrieval.query_transform import transformed_retrieve
from finsight.retrieval.reranker import NoopReranker
from finsight.retrieval.service import RetrievalService


def load_experiment(path: str | Path) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    return data


def _with(settings: Settings, overrides: dict[str, Any]) -> Settings:
    base = {
        "chunking": settings.chunking.model_dump(),
        "embedding": {"preset": settings.embedding.preset},
    }
    s = load_settings(settings.config_path, deep_merge(base, overrides), root=settings.root)
    s.paths, s.secrets = settings.paths, settings.secrets
    return s


def run_retrieval_eval(
    settings: Settings,
    experiment: dict[str, Any],
    *,
    limit: int | None = None,
    progress: bool = True,
    out_dir: Path | None = None,
) -> Path:
    ks = tuple(experiment.get("k", [1, 5, 10, 20]))
    records = ds.eval_records(settings)[: limit or None]
    chunkings = experiment.get("sweep", {}).get("chunking") or [settings.chunking.model_dump()]
    rows: list[dict[str, Any]] = []
    rerankers: dict[str, Reranker] = {}
    llm = None  # built once, only if a variant uses a query transform (P7)

    for chunking in chunkings:
        s_chunk = _with(settings, {"chunking": chunking})
        c = Container(s_chunk)
        try:
            for variant in experiment["variants"]:
                overrides = {k: v for k, v in variant.items() if k != "name"}
                s_var = _with(s_chunk, {"chunking": chunking, **overrides})
                provider = s_var.rerank.provider
                if provider not in rerankers:
                    rerankers[provider] = (
                        NoopReranker() if provider == "none" else Container(s_var).reranker
                    )
                service = RetrievalService(
                    s_var,
                    dense=c.dense_retriever if s_var.retrieval.mode != "sparse" else None,
                    sparse=c.sparse_retriever if s_var.retrieval.mode != "dense" else None,
                    reranker=rerankers[provider],
                    extractor=c.filter_extractor,
                )
                label = f"{cp.config_name(s_chunk)} · {variant['name']}"
                technique = s_var.retrieval.query_transform
                if technique != "none" and llm is None:
                    llm = c.llm
                for rec in tqdm(records, desc=label, unit="q", disable=not progress):
                    t0 = time.perf_counter()
                    calls = 0
                    if technique == "none":
                        res = service.retrieve(rec.question, top_k=max(ks))
                    else:
                        assert llm is not None
                        res, trace = transformed_retrieve(
                            service,
                            llm,
                            rec.question,
                            technique,
                            top_k=max(ks),
                        )
                        calls = trace.llm_calls
                    total = (time.perf_counter() - t0) * 1000
                    ranked = [
                        (r.chunk.doc_name, r.chunk.page_start, r.chunk.page_end) for r in res.final
                    ]
                    gold = [(e.doc_name, e.page_idx) for e in rec.evidence]
                    rows.append(
                        {
                            "chunking": cp.config_name(s_chunk),
                            "variant": variant["name"],
                            "qid": rec.qid,
                            "question_type": rec.question_type,
                            "company": rec.company,
                            "filters": res.applied.model_dump() if res.applied else None,
                            "extracted_companies": res.extracted.companies,
                            "degraded": res.degraded,
                            "top3": [f"{d} p{p + 1}" for d, p, _ in ranked[:3]],
                            "gold": [f"{d} p{p + 1}" for d, p in sorted(set(gold))],
                            "latency_ms": round(total, 1),
                            "llm_calls": calls,
                            "top_score": round(res.final[0].score, 4) if res.final else None,
                            **{f"t_{k}": v for k, v in res.timings_ms.items()},
                            **score_question(ranked, gold, ks),
                        }
                    )
        finally:
            c.close()

    out = out_dir or settings.paths.results_dir / f"{datetime.now():%Y-%m-%d}_{experiment['name']}"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_json(out / "per_question.jsonl", orient="records", lines=True)
    (out / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment": experiment,
                "limit": limit,
                "fingerprint": settings.fingerprint(),
                "embedding": settings.embedding.preset,
            },
            sort_keys=False,
        )
    )
    from finsight.evaluation.report import retrieval_report

    (out / "report.md").write_text(retrieval_report(df, experiment, n_questions=len(records)))
    return out
