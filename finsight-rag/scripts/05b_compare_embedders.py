# %% [markdown]
# # Step 05b — Compare embedders: local vs online   (Phase P4)
#
# **What this step does:** indexes the SAME chunks with several embedding models and measures
# dense retrieval on FinanceBench's gold evidence pages. The embedder is the only variable:
# no metadata filter, no BM25, no reranker (those come in P5).
#
# Models are **presets** in `configs/embedders.yaml`: local (bge-small, bge-base, Ollama) and
# online free tiers (Gemini, Mistral, Cohere trial, Jina, Voyage). OpenAI is paid-only and is
# refused unless `embedding.allow_paid: true`. Put the keys you have in `.env`
# (`GEMINI_API_KEY=...`); presets without a key are skipped and reported, never fatal.
#
# **Free tiers:** `limit_questions=30` indexes only the filings those 30 questions need, so
# a rate-limited API finishes in minutes. Every request is rate-limited on the client, daily and
# monthly caps are tracked in `indexes/usage/`, and vectors are cached: a run stopped by a quota
# resumes where it stopped.
#
# **Three ways to run it**
# - Terminal: `uv run python scripts/05b_compare_embedders.py --embedders bge-small,gemini-001`
# - VS Code, cell by cell: click **Run Cell** above each `# %%`
# - Jupyter: `make notebooks`, then open `notebooks/05b_compare_embedders.ipynb`
#
# **Variables to explore afterwards:** `catalog`, `plan_df`, `comp.summary`, `comp.per_question`,
# `by_type_df`, `disagree_df`.

# %%
import pandas as pd

from finsight.chunking import pipeline as cp
from finsight.config import load_settings
from finsight.embedding.catalog import catalog_df, estimate, preset_cfg
from finsight.evaluation import embedder_compare as ec
from finsight.indexing import pipeline as ixp
from finsight.indexing.qdrant_store import open_client
from finsight.notebook import Param, configure_pandas, in_notebook, script_args, show, step
from finsight.observability.logging import configure_logging

configure_pandas()

# %% Parameters: in a notebook, edit the defaults here; in the terminal, use the flags.
P = script_args(
    "Step 05b: compare embedders (local vs online)",
    config=Param(None, str, "profile YAML (default: configs/local.yaml)"),
    embedders=Param("bge-small,bge-small-noinstr,bge-base", str, "comma-separated presets"),
    limit_questions=Param(30, int, "first N questions and only their filings (0 = all 150)"),
    force=Param(False, bool, "rebuild the indexes"),
)
P

# %%
settings = load_settings(P.config)
configure_logging(settings.logging.level, settings.logging.json_format)
presets = [p.strip() for p in P.embedders.split(",") if p.strip()]
limit = P.limit_questions or None
cp.chunk_corpus(settings, progress=not in_notebook())  # skipped if up to date
chunks_df = cp.load_chunks_df(settings)

# %% [markdown]
# ## 1. The catalog: what can run right now?
# `key`: is the API key in `.env`? `cost`: free tier / local, or PAID (blocked by default).

# %%
step("1. Embedding presets")
catalog = catalog_df(settings)
show(catalog.drop(columns=["notes"]), "configs/embedders.yaml")

# %% [markdown]
# ## 2. The plan: requests, time and quota for the chosen subset
# Estimated from the chunk texts with the presets' rate limits (minutes = the slower of RPM and
# TPM). The same chunks are used for every embedder.

# %%
step("2. Plan")
if limit:
    docs, tag = ixp.subset_for_questions(settings, limit)
    texts = chunks_df.loc[chunks_df["doc_name"].isin(docs), "embed_text"].tolist()
    print(f"subset {tag}: {len(docs)} filings, {len(texts):,} of {len(chunks_df):,} chunks\n")
else:
    texts = chunks_df["embed_text"].tolist()
plan_df = pd.DataFrame(
    [{"preset": p, **estimate(preset_cfg(settings, p), texts)} for p in presets]
).set_index("preset")
show(plan_df, "Estimated cost of indexing (cached vectors are not re-sent)")

# %% [markdown]
# ## 3. Index + evaluate each embedder
# Per preset: build (or reuse) its Qdrant collection, embed the questions (one batched call
# for online providers, cached), search top 10, score against the gold pages.

# %%
step("3. Run the comparison")
client = open_client(settings)
try:
    comp = ec.compare_embedders(
        settings,
        presets,
        limit_questions=limit,
        force=P.force,
        progress=not in_notebook(),
        client=client,
    )
finally:
    client.close()
for preset, why in comp.skipped.items():
    print(f"skipped {preset}: {why}")
show(comp.summary, f"Dense retrieval, {comp.subset}")

# %% [markdown]
# ## 4. Where do they differ?
# By question type (FinanceBench has 3 × 50: domain-relevant, metrics-generated, novel), and the
# questions where one embedder finds the gold page in the top 10 and another doesn't.

# %%
step("4. Breakdown")
if comp.per_question.empty:
    by_type_df = disagree_df = pd.DataFrame()
    print("No embedder ran: add a key to .env or choose local presets.")
else:
    by_type_df = ec.by_question_type(comp.per_question)
    disagree_df = ec.disagreements(comp.per_question)
    show(by_type_df, "hit@10 by question type")
    show(disagree_df.head(15), f"{len(disagree_df)} questions where embedders disagree (hit@10)")

# %% Summary
step("Summary")
out = ec.save_report(settings, comp)
print(f"Report: {out.relative_to(settings.root)}/report.md")
if not comp.summary.empty:
    best = comp.summary["hit@10"].idxmax()
    print(f"Best hit@10 on {comp.subset}: {best} ({comp.summary.loc[best, 'hit@10']:.3f})")
print("Next: P5 combines the chosen embedder with BM25, filters and a reranker.")
