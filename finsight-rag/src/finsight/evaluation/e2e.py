"""
End-to-end evaluation: question -> pipeline -> answer -> metrics (+ LLM judge) -> report.

Per question and variant (a variant = config overrides: pipeline, llm preset, retrieval, ...):
  numeric_match      for numeric gold answers (metrics-generated questions mostly)
  contains_match     gold text inside the answer (short answers)
  judge_verdict      correct / partially_correct / incorrect / judge_error   (optional, cached)
  faithfulness       judge: supported claims / claims, against the cited chunks (optional)
  abstained, false_abstain (answerable question), cited, warnings, degraded
  latency_ms, input/output tokens, estimated cost (llm preset prices; 0 for local/free tiers)
Report: one row per variant (means, n), per question type, and a failure gallery.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

from finsight.config import Settings, deep_merge, load_settings
from finsight.container import Container
from finsight.data import datasets as ds
from finsight.evaluation.generation_metrics import contains_match, numeric_match
from finsight.evaluation.judge import Judge
from finsight.observability.cost import estimate_cost


def _variant_settings(settings: Settings, overrides: dict[str, Any]) -> Settings:
    base = {
        "chunking": settings.chunking.model_dump(),
        "embedding": {"preset": settings.embedding.preset},
        "llm": {"preset": settings.llm.preset},
        "judge": {"preset": settings.judge.preset},
    }
    s = load_settings(settings.config_path, deep_merge(base, overrides), root=settings.root)
    s.paths, s.secrets = settings.paths, settings.secrets
    return s


def run_end_to_end_eval(
    settings: Settings,
    experiment: dict[str, Any],
    *,
    limit: int | None = None,
    use_judge: bool | None = None,
    progress: bool = True,
    out_dir: Path | None = None,
) -> Path:
    records = ds.eval_records(settings)[: limit or None]
    use_judge = experiment.get("judge", True) if use_judge is None else use_judge
    faithfulness = experiment.get("faithfulness", False)
    judge: Judge | None = None
    rows: list[dict[str, Any]] = []
    for variant in experiment["variants"]:
        s = _variant_settings(settings, {k: v for k, v in variant.items() if k != "name"})
        c = Container(s)
        try:
            pipe = c.pipeline
            if use_judge and judge is None:
                judge = Judge(c.judge_llm, settings.paths.index_dir / "judge_cache.sqlite")
            for rec in tqdm(records, desc=variant["name"], unit="q", disable=not progress):
                t0 = time.perf_counter()
                ans = pipe.answer(rec.question)
                latency = (time.perf_counter() - t0) * 1000
                gold = rec.gold_answer or ""
                verdict, reason, faith = "", "", None
                if judge is not None and rec.gold_answer is not None:
                    verdict, reason = judge.correctness(rec.question, gold, ans.answer)
                    if faithfulness and not ans.abstained:
                        texts = [
                            r.chunk.text
                            for r in ans.retrieved
                            if r.chunk.chunk_id in {ci.chunk_id for ci in ans.citations}
                        ]
                        faith = judge.faithfulness(ans.answer, texts)
                usage = ans.usage
                rows.append(
                    {
                        "variant": variant["name"],
                        "qid": rec.qid,
                        "question_type": rec.question_type,
                        "question": rec.question,
                        "gold": gold,
                        "answer": ans.answer,
                        "numeric_match": numeric_match(ans.answer, gold) if gold else None,
                        "contains_match": contains_match(ans.answer, gold) if gold else None,
                        "judge_verdict": verdict,
                        "judge_reason": reason,
                        "faithfulness": faith,
                        "abstained": ans.abstained,
                        "false_abstain": ans.abstained and rec.gold_answer is not None,
                        "cited": len(ans.citations),
                        "gold_page_retrieved": any(
                            (r.chunk.doc_name, r.chunk.page_start)
                            in {(e.doc_name, e.page_idx) for e in rec.evidence}
                            for r in ans.retrieved
                        ),
                        "warnings": ans.warnings,
                        "degraded": ans.degraded,
                        "provider": ans.provider,
                        "model": ans.model,
                        "latency_ms": round(latency),
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "cost_usd": estimate_cost(
                            s.llm, usage.get("input_tokens", 0), usage.get("output_tokens", 0)
                        ),
                    }
                )
        finally:
            c.close()
    df = pd.DataFrame(rows)
    out = out_dir or settings.paths.results_dir / f"{datetime.now():%Y-%m-%d}_{experiment['name']}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_json(out / "per_question.jsonl", orient="records", lines=True)
    (out / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment": experiment,
                "limit": limit,
                "judge": judge.name if judge else None,
                "fingerprint": settings.fingerprint(),
            },
            sort_keys=False,
        )
    )
    (out / "report.md").write_text(
        e2e_report(df, experiment, judge.name if judge else None, len(records))
    )
    return out


def e2e_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("variant", sort=False)
    v = df["judge_verdict"]
    out = pd.DataFrame(
        {
            "n": g.size(),
            "numeric_match": g["numeric_match"].apply(lambda x: x.dropna().astype(float).mean()),
            "judge correct": df.assign(x=v == "correct").groupby("variant", sort=False)["x"].mean(),
            "judge partial": df.assign(x=v == "partially_correct")
            .groupby("variant", sort=False)["x"]
            .mean(),
            "judge errors": df.assign(x=v == "judge_error")
            .groupby("variant", sort=False)["x"]
            .mean(),
            "faithfulness": g["faithfulness"].mean(),
            "gold page retrieved": g["gold_page_retrieved"].mean(),
            "abstained": g["abstained"].mean(),
            "cited": g["cited"].apply(lambda x: (x > 0).mean()),
            "auto_cited": g["warnings"].apply(lambda w: w.map(lambda x: "auto_cited" in x).mean()),
            "p50 s": g["latency_ms"].median() / 1000,
            "p95 s": g["latency_ms"].quantile(0.95) / 1000,
            "tokens/q": (g["input_tokens"].mean() + g["output_tokens"].mean()).round(0),
            "$ total": g["cost_usd"].sum(),
        }
    )
    return out.round(3)


def e2e_report(df: pd.DataFrame, experiment: dict[str, Any], judge: str | None, n: int) -> str:
    best = e2e_summary(df)["numeric_match"].idxmax() if not df.empty else None
    by_type = df.pivot_table(
        index="question_type",
        columns="variant",
        values="numeric_match",
        aggfunc=lambda x: x.dropna().astype(float).mean(),
    ).round(3)
    fails = df[(df["variant"] == best) & (df["numeric_match"] == False)]  # noqa: E712
    gallery = fails.head(8)[
        ["qid", "question_type", "gold", "answer", "gold_page_retrieved"]
    ].copy()
    gallery["answer"] = gallery["answer"].str.replace("\n", " ").str[:160]
    gallery["gold"] = gallery["gold"].str[:80]
    return "\n".join(
        [
            f"# {experiment['name']}",
            "",
            f"- Questions: {n} · judge: `{judge or 'none'}` "
            "(validate with scripts/10b_label_judge_sample.py)",
            "- numeric_match: any number in the answer within 1% of the gold's first number "
            "(numeric golds only).",
            "- Differences under ~0.05 on 150 questions are noise (std of a 0/1 metric ≈ 0.5).",
            "",
            "## Summary",
            "",
            e2e_summary(df).to_markdown(),
            "",
            "## numeric_match by question type",
            "",
            by_type.to_markdown(),
            "",
            f"## Failure gallery ({best}: numeric answers that did not match)",
            "",
            gallery.to_markdown(index=False),
            "",
        ]
    )
