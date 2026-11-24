# %% [markdown]
# # Step 09 — Advanced RAG: query rewriting, corrective and agentic   (Phase P7)
#
# **What this step does:** runs ONE question through an advanced technique and shows every
# intermediate step, then compares it with the P6 baseline on the same question.
#
# - `multi_query` — the LLM writes 3 rewrites in filing vocabulary; fuse all results (RRF)
# - `hyde` — the LLM writes a fake filing passage; embed THAT for the dense search
# - `decompose` — split a multi-part question into sub-questions; retrieve for each
# - `corrective` — grade the retrieved context; if weak, rewrite and retry; still weak: abstain
# - `agentic` — the LLM drives `search` / `read` / `answer` tools for up to 5 steps
#
# Every technique costs extra LLM calls (shown). The P7 report measures them on all questions.
#
# **Three ways to run it**
# - Terminal: `uv run python scripts/09_advanced_rag.py --technique multi_query`
#   or `--technique agentic -c configs/multihop.yaml --qid mh_0000`
# - VS Code, cell by cell: click **Run Cell** above each `# %%`
# - Jupyter: `make notebooks`, then open `notebooks/09_advanced_rag.ipynb`
#
# **Variables to explore afterwards:** `answer`, `baseline`, `steps_df`, `compare_df`.

# %%
import pandas as pd

from finsight.config import load_settings
from finsight.container import Container
from finsight.data import datasets as ds
from finsight.notebook import Param, configure_pandas, preview, script_args, show, step
from finsight.observability.logging import configure_logging
from finsight.schemas import Answer

configure_pandas()

# %% Parameters: in a notebook, edit the defaults here; in the terminal, use the flags.
P = script_args(
    "Step 09: advanced RAG techniques",
    config=Param(None, str, "profile YAML (configs/multihop.yaml for MultiHop-RAG)"),
    technique=Param("multi_query", str, "multi_query | hyde | decompose | corrective | agentic"),
    qid=Param("financebench_id_03029", str, "question id (FinanceBench or mh_NNNN)"),
    question=Param("", str, "your own question (overrides qid)"),
    llm=Param(None, str, "LLM preset (default: config / auto)"),
)
P

# %%
TRANSFORMS = {"multi_query", "hyde", "decompose"}
overrides: dict = {"llm": {"preset": P.llm}} if P.llm else {}
if P.technique in TRANSFORMS:
    overrides["retrieval"] = {"query_transform": P.technique}
    overrides["pipeline"] = "hybrid_rerank"
else:
    overrides["pipeline"] = P.technique
settings = load_settings(P.config, overrides)
configure_logging(settings.logging.level, settings.logging.json_format)
rec = {r.qid: r for r in ds.eval_records(settings)}.get(P.qid)
question = P.question or (rec.question if rec else "What is 3M's FY2018 capital expenditure?")
c = Container(settings)
print(f"Question: {question}\nTechnique: {P.technique} · LLM {c.llm.provider}:{c.llm.model}\n")

# %% [markdown]
# ## 1. Run the technique and show its intermediate steps

# %%
step(f"1. {P.technique}")
pipe = c.pipeline
answer = pipe.answer(question)
rows = []
trace = getattr(pipe, "last_trace", None)
if trace is not None:
    rows += [{"step": "query", "detail": q} for q in trace.queries]
    if trace.hyde_passage:
        rows.append({"step": "hyde passage", "detail": preview(trace.hyde_passage, 300)})
if P.technique == "corrective":
    rows.append({"step": "branch", "detail": getattr(pipe, "branch", "?")})
for t in getattr(pipe, "trace", []) if P.technique == "agentic" else []:
    rows.append({"step": f"{t['step']}. {t['tool']}", "detail": preview(str(t.get("args")), 120)})
steps_df = pd.DataFrame(rows or [{"step": "-", "detail": "(no intermediate steps)"}])
show(steps_df, "Intermediate steps")
print(answer.answer + "\n")
show(
    pd.Series(
        {
            "citations": [f"{ci.doc_name} p{ci.page_idx + 1}" for ci in answer.citations],
            "abstained": answer.abstained,
            "warnings": answer.warnings or "-",
            "degraded": answer.degraded or "-",
            "llm calls": answer.usage.get("llm_calls", getattr(pipe, "llm_calls", 0) + 1),
            "total ms": answer.timings_ms.get("total"),
        },
        dtype=object,
    ),
    "Result",
)

# %% [markdown]
# ## 2. Compare with the baseline pipeline (P6) on the same question

# %%
step("2. Baseline")
c.close()
base_settings = load_settings(
    P.config, {**({"llm": {"preset": P.llm}} if P.llm else {}), "pipeline": "hybrid_rerank"}
)
b = Container(base_settings)
baseline = b.pipeline.answer(question)
b.close()
gold = {(e.doc_name, e.page_idx) for e in rec.evidence} if rec and not P.question else set()


def gold_hit(ans: Answer) -> bool:
    return any((r.chunk.doc_name, r.chunk.page_start) in gold for r in ans.retrieved)


compare_df = pd.DataFrame(
    [
        {
            "pipeline": n,
            "answer": preview(a.answer, 90),
            "cited": len(a.citations),
            "gold page retrieved": gold_hit(a) if gold else "-",
            "ms": a.timings_ms.get("total"),
        }
        for n, a in ((P.technique, answer), ("baseline", baseline))
    ]
)
show(compare_df, "Technique vs baseline")
if rec and not P.question:
    print(f"Gold answer: {rec.gold_answer}")

# %% Summary
step("Summary")
print("All questions: results/<date>_advanced_query_transforms/report.md (make eval-advanced)")
