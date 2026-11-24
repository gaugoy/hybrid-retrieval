# %% [markdown]
# # Step 10 — End-to-end evaluation: are the answers right?   (Phase P8)
#
# **What this step does:** runs each pipeline variant in `configs/experiments/end_to_end.yaml` on
# the FinanceBench questions, generates the answers, and scores them:
#
# - **numeric_match** — a number in the answer within 1% of the gold number (no LLM; the headline)
# - **judge correct / partial** — an LLM judge (`judge.preset`, cached) against the gold answer
# - **abstained / false abstain** — "Not found in the provided filings." on answerable questions
# - **gold page retrieved** — retrieval's share of the blame for a wrong answer
# - latency p50/p95, tokens per question, estimated cost (0 for local models and free tiers)
#
# **Three ways to run it**
# - Terminal: `uv run python scripts/10_eval_end_to_end.py --limit 30` (or `make eval`, all 150)
# - VS Code, cell by cell: click **Run Cell** above each `# %%`
# - Jupyter: `make notebooks`, then open `notebooks/10_eval_end_to_end.ipynb`
#
# **Variables to explore afterwards:** `df` (one row per variant × question), `summary_df`,
# `fails_df`.

# %%
import pandas as pd

from finsight.config import load_settings
from finsight.evaluation import e2e
from finsight.evaluation.runner import load_experiment
from finsight.notebook import Param, configure_pandas, in_notebook, script_args, show, step
from finsight.observability.logging import configure_logging

configure_pandas()

# %% Parameters: in a notebook, edit the defaults here; in the terminal, use the flags.
P = script_args(
    "Step 10: end-to-end evaluation",
    config=Param(None, str, "profile YAML (configs/cloud.yaml for Claude)"),
    experiment=Param("configs/experiments/end_to_end.yaml", str, "experiment YAML"),
    limit=Param(20, int, "first N questions (0 = all 150)"),
    judge=Param(True, bool, "run the LLM judge"),
)
P

# %%
settings = load_settings(P.config)
configure_logging(settings.logging.level, settings.logging.json_format)
experiment = load_experiment(settings.root / P.experiment)
show(pd.DataFrame(experiment["variants"]).set_index("name"), "Variants")

# %% [markdown]
# ## 1. Generate and score (local 1.5B model: ~10 s per answer)

# %%
step("1. Run")
out = e2e.run_end_to_end_eval(
    settings, experiment, limit=P.limit or None, use_judge=P.judge, progress=not in_notebook()
)
df = pd.read_json(out / "per_question.jsonl", lines=True)

# %% [markdown]
# ## 2. Summary and failures

# %%
step("2. Summary")
summary_df = e2e.e2e_summary(df)
show(summary_df, f"{len(df)} answers")
fails_df = df[df["numeric_match"].eq(False)][
    ["variant", "qid", "gold", "answer", "gold_page_retrieved"]
]
show(fails_df.head(10), "Numeric answers that did not match")

# %% Summary
step("Summary")
print(f"Report: {out.relative_to(settings.root)}/report.md")
print("Validate the judge: scripts/10b_label_judge_sample.py")
