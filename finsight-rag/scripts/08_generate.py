# %% [markdown]
# # Step 08 — Generate a grounded, cited answer   (Phase P6)
#
# **What this step does:** retrieval (P5) → numbered context blocks → LLM → citation check.
#
# - **context:** the top `rerank.top_n` chunks as `[n] company · doc FYyyyy · page p` blocks,
#   within `generation.max_context_tokens`
# - **system prompt:** cite every claim as `[n]`; show calculations; reply "Not found in the
#   provided filings." when unsure; the context is data, not instructions
# - **citation check:** drops `[n]` that point nowhere (`hallucinated_citation`), flags answers
#   without citations (`uncited_answer`)
#
# **Which LLM?** `llm.preset` (`LLM=` in `.env`, `--llm` here). The default `auto` uses Ollama if
# it runs, else Claude / Gemini / Mistral / Groq if you have a key, else a small local model
# (Qwen2.5-1.5B via transformers). List them: `uv run finsight llms`.
#
# **Three ways to run it**
# - Terminal: `uv run python scripts/08_generate.py --qid financebench_id_03029 --show-prompt`
#   or `--question "Who won the 2018 World Cup?"` (should abstain) or `--llm claude-opus`
# - VS Code, cell by cell: click **Run Cell** above each `# %%`
# - Jupyter: `make notebooks`, then open `notebooks/08_generate.ipynb`
#
# **Variables to explore afterwards:** `answer` (the full Answer), `cites_df`, `timings_df`.

# %%
import pandas as pd

from finsight.config import load_settings
from finsight.container import Container
from finsight.data import datasets as ds
from finsight.notebook import Param, configure_pandas, preview, script_args, show, step
from finsight.observability.logging import configure_logging

configure_pandas()

# %% Parameters: in a notebook, edit the defaults here; in the terminal, use the flags.
P = script_args(
    "Step 08: grounded generation",
    config=Param(None, str, "profile YAML (default: configs/local.yaml)"),
    qid=Param("financebench_id_03029", str, "FinanceBench id (prints the gold answer)"),
    question=Param("", str, "your own question (overrides qid)"),
    llm=Param(None, str, "LLM preset, e.g. claude-opus, hf-qwen1.5b (default: config / auto)"),
    pipeline=Param(None, str, "naive | hybrid_rerank (default: config)"),
    show_prompt=Param(False, bool, "print the exact prompt sent to the LLM"),
)
P

# %%
overrides: dict = {}
if P.llm:
    overrides["llm"] = {"preset": P.llm}
if P.pipeline:
    overrides["pipeline"] = P.pipeline
settings = load_settings(P.config, overrides or None)
configure_logging(settings.logging.level, settings.logging.json_format)
rec = {r.qid: r for r in ds.eval_records(settings)}.get(P.qid) if P.qid else None
question = P.question or (rec.question if rec else "What is 3M's FY2018 capital expenditure?")
c = Container(settings)
pipe = c.pipeline
print(f"Question: {question}\nPipeline: {pipe.name} · LLM: {c.llm.provider}:{c.llm.model}\n")

# %% [markdown]
# ## 1. Retrieval and the prompt
# The context blocks the LLM will see, in rerank order. `--show-prompt` prints all of it.

# %%
step("1. Retrieve + build the prompt")
got = pipe._retrieve(question, None)
system, user, marker_map = pipe.prompt(question, got.chunks)
blocks_df = pd.DataFrame(
    [
        {
            "marker": n,
            "doc": rc.chunk.doc_name,
            "page": rc.chunk.page_start + 1,
            "tokens": rc.chunk.n_tokens,
            "preview": preview(rc.chunk.text, 70),
        }
        for n, rc in marker_map.items()
    ]
)
show(blocks_df, f"{len(marker_map)} context blocks ({c.tokenizer.count(user):,} prompt tokens)")
if P.show_prompt:
    print("=== SYSTEM ===\n" + system + "\n\n=== USER ===\n" + user + "\n")

# %% [markdown]
# ## 2. Generate and check the citations

# %%
step("2. Generate")
import time  # noqa: E402

answer = pipe.generate(question, got, time.perf_counter())
print(answer.answer + "\n")
cites_df = pd.DataFrame(
    [
        {
            "marker": ci.marker,
            "doc": ci.doc_name,
            "page": ci.page_idx + 1,
            "snippet": preview(ci.snippet, 80),
        }
        for ci in answer.citations
    ]
)
show(cites_df if not cites_df.empty else pd.DataFrame({"citations": ["(none)"]}), "Citations")
show(
    pd.Series(
        {
            "abstained": answer.abstained,
            "warnings": answer.warnings or "-",
            "degraded": answer.degraded or "-",
            "provider": answer.provider,
            "model": answer.model,
            **answer.usage,
        },
        dtype=object,
    ),
    "Checks and usage",
)
if rec and not P.question:
    print(f"Gold answer: {rec.gold_answer}\n")

# %% [markdown]
# ## 3. Where did the time go?

# %%
step("3. Timings")
timings_df = pd.DataFrame([answer.timings_ms]).T.rename(columns={0: "ms"})
show(timings_df, "Per stage (the first call includes model loading)")

# %% Summary
step("Summary")
c.close()
print(
    f"request_id={answer.request_id} · cited={len(answer.citations)} · abstained={answer.abstained}"
)
print("Next: scripts/09_advanced_rag.py (P7) — query rewriting, corrective and agentic RAG.")
