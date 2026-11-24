"""Markdown reports for evaluation runs (mean ± std, n, per question type, worst questions)."""

from __future__ import annotations

from typing import Any

import pandas as pd

HEADLINE = ["hit@5", "hit@10", "recall@10", "mrr", "ndcg@10", "doc_hit@10"]


def summary_table(df: pd.DataFrame, metrics: list[str] = HEADLINE) -> pd.DataFrame:
    g = df.groupby(["chunking", "variant"], sort=False)
    mean, std = g[metrics].mean(), g[metrics].std(ddof=1)
    out = pd.DataFrame(index=mean.index)
    for m in metrics:
        out[m] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(mean[m], std[m].fillna(0), strict=True)]
    out["p50 ms"] = g["latency_ms"].median().round(0).astype(int)
    out["n"] = g.size()
    return out


def best_config(df: pd.DataFrame, metric: str = "hit@10") -> tuple[str, str]:
    means = df.groupby(["chunking", "variant"])[[metric, "mrr"]].mean()
    best = means.sort_values([metric, "mrr"], ascending=False).index[0]
    return str(best[0]), str(best[1])


def retrieval_report(df: pd.DataFrame, experiment: dict[str, Any], n_questions: int) -> str:
    chunking, variant = best_config(df)
    best = df[(df["chunking"] == chunking) & (df["variant"] == variant)]
    by_type = (
        df[df["chunking"] == chunking]
        .pivot_table(index="question_type", columns="variant", values="hit@10")
        .round(3)
    )
    worst = best.sort_values(["hit@10", "mrr"]).head(10)[["qid", "question_type", "top3", "gold"]]
    filt = df[df["variant"] == "hybrid+filter"]
    filter_rate = (
        filt["extracted_companies"].map(len).gt(0).mean() if not filt.empty else float("nan")
    )
    lines = [
        f"# {experiment['name']}",
        "",
        f"- Questions: {n_questions} (FinanceBench) · relevant = a chunk on a gold evidence page",
        "- Values: mean ± std over questions. Differences under ~0.05 on 150 questions are noise.",
        f"- Best by hit@10 (then MRR): **{chunking} · {variant}**",
        f"- Company extracted from the question: {filter_rate:.0%} of questions",
        "",
        "## All configurations",
        "",
        summary_table(df).to_markdown(),
        "",
        f"## hit@10 by question type ({chunking})",
        "",
        by_type.to_markdown(),
        "",
        f"## 10 hardest questions for the best configuration ({chunking} · {variant})",
        "",
        worst.to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)
