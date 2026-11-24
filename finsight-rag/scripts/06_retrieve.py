# %% [markdown]
# # Step 06 — Retrieve: BM25 vs dense vs hybrid vs reranked   (Phase P5)
#
# **What this step does:** takes ONE question through every retrieval stage and shows the
# top results side by side, so you can see what each stage adds:
#
# | stage | how it ranks | strong at |
# |---|---|---|
# | filters | company (and optionally year) from the question | dropping look-alike pages |
# | BM25 | exact word overlap | line-item names, tickers, years |
# | dense | embedding similarity | paraphrases ("capex" ≈ "purchases of PP&E") |
# | hybrid (RRF) | Σ 1/(60 + rank) over both lists | chunks that both like |
# | rerank | a cross-encoder reads question + chunk together | the final order |
#
# With a FinanceBench `qid`, gold-page hits are marked ✅.
#
# **Three ways to run it**
# - Terminal: `uv run python scripts/06_retrieve.py --qid financebench_id_03029`
#   or `--question "What was Netflix's FY2017 revenue?"`
# - VS Code, cell by cell: click **Run Cell** above each `# %%`
# - Jupyter: `make notebooks`, then open `notebooks/06_retrieve.ipynb`
#
# **Variables to explore afterwards:** `res` (all stages), `side_df`, `timings_df`.

# %%
import pandas as pd

from finsight.config import load_settings
from finsight.container import Container
from finsight.data import financebench as fb
from finsight.notebook import Param, configure_pandas, preview, script_args, show, step
from finsight.observability.logging import configure_logging
from finsight.schemas import RetrievedChunk

configure_pandas()

# %% Parameters: in a notebook, edit the defaults here; in the terminal, use the flags.
P = script_args(
    "Step 06: one question through every retrieval stage",
    config=Param(None, str, "profile YAML (default: configs/local.yaml)"),
    qid=Param("financebench_id_03029", str, "FinanceBench id (marks gold pages); '' to skip"),
    question=Param("", str, "your own question (overrides qid's question)"),
    k=Param(5, int, "rows per stage"),
    rerank=Param(True, bool, "run the reranker"),
)
P

# %%
settings = load_settings(P.config)
configure_logging(settings.logging.level, settings.logging.json_format)
records = {r.qid: r for r in fb.load_eval_records(settings)} if P.qid else {}
rec = records.get(P.qid)
question = P.question or (rec.question if rec else "What is 3M's FY2018 capital expenditure?")
gold = {(e.doc_name, e.page_idx) for e in rec.evidence} if rec and not P.question else set()
c = Container(settings)
print(f"Question: {question}")
if gold and rec:
    print(f"Gold: {sorted((d, p + 1) for d, p in gold)} · answer: {rec.gold_answer}\n")

# %% [markdown]
# ## 1. Filters extracted from the question
# Applied fields: `retrieval.filter_fields` (default: company only, see the P5 ablation).

# %%
step("1. Filters")
res = c.retrieval.retrieve(question, rerank=P.rerank, top_k=P.k)
show(
    pd.Series(
        {
            "extracted companies": res.extracted.companies,
            "extracted years": res.extracted.doc_periods,
            "applied": res.applied.model_dump() if res.applied else None,
            "degraded": res.degraded or "-",
        },
        dtype=object,
    ),
    "Filters",
)

# %% [markdown]
# ## 2. Top results per stage
# `page` is the printed page (page_idx + 1). ✅ = a gold evidence page.


# %%
def cell(r: RetrievedChunk) -> str:
    mark = "✅ " if (r.chunk.doc_name, r.chunk.page_start) in gold else ""
    return f"{mark}{r.chunk.doc_name} p{r.chunk.page_start + 1} ({r.score:.3g})"


step("2. Side by side")
stages = [s for s in ("sparse", "dense", "hybrid", "rerank") if s in res.stages]
side_df = pd.DataFrame({s: [cell(r) for r in res.stages[s][: P.k]] for s in stages})
side_df.index = range(1, len(side_df) + 1)
show(
    side_df.rename(columns={"sparse": "BM25", "hybrid": "hybrid (RRF)", "rerank": "reranked"}),
    f"Top {P.k} per stage",
)
final = res.final[0] if res.final else None
if final:
    show(preview({"top chunk": final.chunk.chunk_id, "text": final.chunk.text}, 400), "Final #1")

# %% [markdown]
# ## 3. Where did the time go?

# %%
step("3. Timings")
timings_df = pd.DataFrame([res.timings_ms]).T.rename(columns={0: "ms"})
show(timings_df, "Per-stage latency (first call includes model loading)")

# %% Summary
step("Summary")
hit = any((r.chunk.doc_name, r.chunk.page_start) in gold for r in res.final)
c.close()
print(f"Gold page in the final top {P.k}: {hit}" if gold else "No gold (custom question).")
print("Next: scripts/07_eval_retrieval.py measures every configuration on all 150 questions.")
