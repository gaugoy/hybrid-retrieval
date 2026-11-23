# %% [markdown]
# # Step 02 — Explore FinanceBench   (Phase P1)
#
# **What this step does:** loads the downloaded data into our schemas (`EvalRecord`,
# `DocumentMeta`), turns them into pandas DataFrames, and explores what a question, its gold
# answer and its evidence look like. Ends with consistency checks between questions, metadata and
# the PDFs on disk.
#
# **Three ways to run it**
# - Terminal, all at once: `uv run python scripts/02_explore_data.py [--samples 5 --seed 7]`
# - VS Code, cell by cell: click **Run Cell** above each `# %%`
# - Jupyter: `make notebooks`, then open `notebooks/02_explore_data.ipynb`
#
# **Variables to explore afterwards:** `records` (list of EvalRecord), `meta` (dict of
# DocumentMeta), `qa_df` (one row per question), `ev_df` (one row per evidence page), `meta_df`.
# Try: `qa_df[qa_df.company == "3M"]`, `ev_df.groupby("doc_type").page_idx.describe()`.

# %%
import pandas as pd

from finsight.config import load_settings
from finsight.data import financebench as fb
from finsight.notebook import (
    Param,
    configure_pandas,
    in_notebook,
    preview,
    script_args,
    show,
    step,
)
from finsight.observability.logging import configure_logging

configure_pandas()

# %% Parameters: in a notebook, edit the defaults here; in the terminal, use the flags.
P = script_args(
    "Step 02: explore FinanceBench",
    config=Param(None, str, "profile YAML (default: configs/local.yaml)"),
    samples=Param(3, int, "how many random questions to show in full"),
    seed=Param(42, int, "random seed for the samples"),
)
P

# %% [markdown]
# ## 1. Load into our schemas
# `load_eval_records()` turns each raw row into a validated `EvalRecord`; page numbers stay
# **0-based** (`page_idx`), exactly as the benchmark stores them.

# %%
step("1. Load into our schemas")
settings = load_settings(P.config)
configure_logging(settings.logging.level, settings.logging.json_format)
records = fb.load_eval_records(settings)
meta = fb.load_document_meta(settings)
print(f"{len(records)} EvalRecord objects, {len(meta)} DocumentMeta objects\n")
show(preview(records[0].model_dump()), "records[0] as a dict (long text shortened)")

# %% [markdown]
# ## 2. Questions as a DataFrame (one row per question)

# %%
step("2. Questions")
qa_df = pd.DataFrame(
    [
        {
            "qid": r.qid,
            "question_type": r.question_type,
            "reasoning": (r.question_reasoning or "(not labelled)").split(" OR ")[0],
            "company": r.company,
            "doc_name": r.doc_name,
            "n_evidence": len(r.evidence),
            "question": r.question,
            "gold_answer": r.gold_answer,
        }
        for r in records
    ]
)
show(qa_df.head(), "qa_df.head()")
show(qa_df["question_type"].value_counts(), "By question_type")
show(qa_df["reasoning"].value_counts(), "By primary reasoning")
show(pd.crosstab(qa_df["question_type"], qa_df["reasoning"]), "question_type × reasoning")
show(qa_df["company"].value_counts().head(10), f"Top 10 of {qa_df['company'].nunique()} companies")

# %% [markdown]
# ## 3. Evidence as a DataFrame (one row per evidence page)
# This is the ground truth for retrieval: a chunk is relevant if it comes from `doc_name` and
# covers `page_idx`. `printed_page` is what you'd see in a PDF viewer.

# %%
step("3. Evidence pages")
meta_df = pd.DataFrame([m.model_dump() for m in meta.values()])
ev_df = pd.DataFrame(
    [
        {
            "qid": r.qid,
            "doc_name": ev.doc_name,
            "page_idx": ev.page_idx,
            "printed_page": ev.page_idx + 1,
            "evidence_chars": len(ev.text),
        }
        for r in records
        for ev in r.evidence
    ]
).merge(meta_df[["doc_name", "doc_type", "doc_period"]], on="doc_name", how="left")
show(ev_df.head(), "ev_df.head()")
show(qa_df["n_evidence"].value_counts().sort_index(), "Questions by number of evidence pages")
show(ev_df["page_idx"].describe().round(1), "Where the evidence sits (page_idx)")
show(ev_df.groupby("doc_type")["page_idx"].describe().round(1), "Evidence position by doc_type")

# %% [markdown]
# ## 4. Read a few questions in full
# Open the PDF at the printed page to check the 0-based convention yourself.

# %%
step("4. Sample questions")
for rec in qa_df.sample(n=min(P.samples, len(qa_df)), random_state=P.seed).itertuples():
    full = next(r for r in records if r.qid == rec.qid)
    ev = full.evidence[0]
    m = meta.get(ev.doc_name)
    print(f"── {rec.qid} ({rec.question_type} · {rec.reasoning})")
    print(f"Q: {rec.question}")
    print(f"Gold answer: {rec.gold_answer}")
    print(
        f"Evidence: {ev.doc_name} ({m.company} · {m.doc_type} · FY{m.doc_period}) · "
        f"page_idx {ev.page_idx} = printed page {ev.page_idx + 1}"
        + (f" · +{len(full.evidence) - 1} more page(s)" if len(full.evidence) > 1 else "")
        if m
        else f"Evidence: {ev.doc_name}"
    )
    print(f"Evidence text (whitespace collapsed): {' '.join(ev.text.split())[:300]}…\n")

# %% [markdown]
# ## 5. Consistency checks
# Every evidence document must have metadata and a PDF on disk; IDs unique; answers non-empty.

# %%
step("5. Consistency checks")
problems = fb.validate_dataset(records, meta, settings.paths.pdf_dir)
if problems:
    show(pd.Series(problems, name="problem"), f"{len(problems)} problems")
    if not in_notebook():
        raise SystemExit(1)
else:
    print(
        f"0 problems: {len(records)} questions, {len(ev_df)} evidence items, every evidence "
        "document has metadata and a PDF on disk."
    )

# %% Summary
step("Summary")
print("Next phase: P2 (parse the PDFs into clean pages).")
