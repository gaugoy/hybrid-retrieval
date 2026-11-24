# %% [markdown]
# # Step 07 — Retrieval evaluation: which configuration wins?   (Phase P5)
#
# **What this step does:** runs the ablation in `configs/experiments/retrieval_ablation.yaml`:
# every retrieval variant (dense, BM25, hybrid, + company filter, + year filter, + rerank) on
# every chunking strategy, for the FinanceBench questions. No LLM is called.
#
# Metrics per question (relevant = a chunk on a gold evidence page): hit@k, recall@k, MRR,
# nDCG@k, doc_hit@k. Reported as mean ± std; on 150 questions, differences under ~0.05 are noise.
#
# **Three ways to run it**
# - Terminal: `uv run python scripts/07_eval_retrieval.py [--limit 20]` (or `make eval-retrieval`)
# - VS Code, cell by cell: click **Run Cell** above each `# %%`
# - Jupyter: `make notebooks`, then open `notebooks/07_eval_retrieval.ipynb`
#
# **Variables to explore afterwards:** `df` (one row per chunking × variant × question),
# `summary_df`, `by_type_df`, `filter_df`.

# %%
import pandas as pd

from finsight.config import load_settings
from finsight.evaluation import report as rp
from finsight.evaluation.runner import load_experiment, run_retrieval_eval
from finsight.notebook import Param, configure_pandas, in_notebook, script_args, show, step
from finsight.observability.logging import configure_logging

configure_pandas()

# %% Parameters: in a notebook, edit the defaults here; in the terminal, use the flags.
P = script_args(
    "Step 07: retrieval ablation",
    config=Param(None, str, "profile YAML (default: configs/local.yaml)"),
    experiment=Param("configs/experiments/retrieval_ablation.yaml", str, "experiment YAML"),
    limit=Param(0, int, "first N questions only (0 = all 150)"),
)
P

# %%
settings = load_settings(P.config)
configure_logging(settings.logging.level, settings.logging.json_format)
experiment = load_experiment(settings.root / P.experiment)
show(pd.DataFrame(experiment["variants"]).set_index("name"), "Variants")
show(pd.DataFrame(experiment["sweep"]["chunking"]), "Chunking configs")

# %% [markdown]
# ## 1. Run (reranking dominates the time: ~0.5–2 s per question per chunking config)

# %%
step("1. Run the ablation")
out = run_retrieval_eval(settings, experiment, limit=P.limit or None, progress=not in_notebook())
df = pd.read_json(out / "per_question.jsonl", lines=True)
print(f"{len(df):,} rows → {out.relative_to(settings.root)}\n")

# %% [markdown]
# ## 2. The table

# %%
step("2. Summary")
summary_df = rp.summary_table(df)
show(summary_df, "mean ± std per chunking × variant")
best = rp.best_config(df)
print(f"Best by hit@10 (then MRR): {best[0]} · {best[1]}\n")

# %% [markdown]
# ## 3. By question type, and how often the filter fired

# %%
step("3. Breakdown")
by_type_df = (
    df[df["chunking"] == best[0]]
    .pivot_table(index="question_type", columns="variant", values="hit@10")
    .round(3)
)
show(by_type_df, f"hit@10 by question type ({best[0]})")
filter_df = (
    df[df["variant"] == "hybrid+filter"]
    .assign(company_found=lambda d: d["extracted_companies"].map(len) > 0)
    .groupby("company_found")[["hit@10", "doc_hit@10"]]
    .mean()
    .round(3)
)
show(filter_df, "hybrid+filter: questions with vs without an extracted company")

# %% Summary
step("Summary")
print(f"Report: {out.relative_to(settings.root)}/report.md")
print("Next: put the winner in configs/base.yaml; P6 generates answers on top of it.")
