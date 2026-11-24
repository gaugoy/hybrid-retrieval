# %% [markdown]
# # Step 10b — Validate the LLM judge against YOUR labels   (Phase P8)
#
# **What this step does:** shows 20 random (question, gold, prediction) triples from an
# end-to-end run. You label each one **c**orrect / **p**artial / **i**ncorrect; the script prints
# your agreement with the judge and a confusion matrix, and saves your labels so you only do it
# once (`results/<run>/human_labels.json`).
#
# Why: an LLM judge has biases (its own model family, long answers). A number from a judge is
# only as good as its agreement with a human on a sample.
#
# **Run it in a terminal** (it asks for input): `uv run python scripts/10b_label_judge_sample.py`
# Non-interactive: `--labels path.json` with `{"qid|variant": "c|p|i", ...}`.

# %%
import json
import random
from pathlib import Path

import pandas as pd

from finsight.config import load_settings
from finsight.notebook import Param, configure_pandas, script_args, show, step

configure_pandas()

# %% Parameters
P = script_args(
    "Step 10b: label a judge sample",
    run=Param("", str, "results folder of an end-to-end run (default: the latest)"),
    n=Param(20, int, "how many answers to label"),
    labels=Param("", str, "JSON file of labels instead of asking"),
    seed=Param(3, int, "sampling seed"),
)
P

# %%
settings = load_settings()
runs = sorted(settings.paths.results_dir.glob("*_end_to_end*"))
run = Path(P.run) if P.run else runs[-1]
df = pd.read_json(run / "per_question.jsonl", lines=True)
df = df[df["judge_verdict"].isin(["correct", "partially_correct", "incorrect"])]
sample = df.sample(min(P.n, len(df)), random_state=random.Random(P.seed).randint(0, 10**6))
print(f"Run: {run.name} · {len(sample)} answers to label\n")

# %%
step("1. Label")
MAP = {"c": "correct", "p": "partially_correct", "i": "incorrect"}
saved = run / "human_labels.json"
labels: dict[str, str] = (
    json.loads(Path(P.labels).read_text())
    if P.labels
    else (json.loads(saved.read_text()) if saved.is_file() else {})
)
for _, r in sample.iterrows():
    key = f"{r.qid}|{r.variant}"
    if key in labels:
        continue
    print(f"\nQ: {r.question}\nGOLD: {r.gold}\nANSWER: {r.answer[:600]}")
    ans = ""
    while ans not in MAP:
        ans = input("c / p / i ? ").strip().lower()
    labels[key] = ans
saved.write_text(json.dumps(labels, indent=2))

# %%
step("2. Agreement")
sample = sample.assign(
    human=[
        MAP.get(labels.get(f"{q}|{v}", "")) for q, v in zip(sample.qid, sample.variant, strict=True)
    ]
)
sample = sample.dropna(subset=["human"])
agree = (sample["human"] == sample["judge_verdict"]).mean()
show(
    pd.crosstab(sample["human"], sample["judge_verdict"]),
    f"human (rows) vs judge (columns): agreement {agree:.0%}",
)

# %% Summary
step("Summary")
print(
    f"{len(sample)} labels saved to {saved}. Agreement {agree:.0%}: write it in docs/evaluation.md."
)
