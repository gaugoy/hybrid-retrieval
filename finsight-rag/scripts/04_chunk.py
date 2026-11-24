# %% [markdown]
# # Step 04 — Chunk the pages   (Phase P3)
#
# **What this step does:** splits the 12,013 clean pages into chunks small enough to embed
# (≤ 510 tokens for bge), with four interchangeable strategies:
#
# | strategy | idea | crosses pages? |
# |---|---|---|
# | `fixed` | slide a window of N tokens, step N - overlap | yes |
# | `recursive` | split at paragraph → line → sentence → word, then pack | yes |
# | `page_aware` | recursive, but per page; carries table column headers | no |
# | `parent_child` | small page-aware children for search, whole page as parent for the LLM | no |
#
# The first cells take ONE page (3M's 2018 cash-flow statement) through every strategy. Then the
# configured strategy chunks the corpus, and (optionally) a sweep compares all strategies × sizes.
#
# **Three ways to run it**
# - Terminal: `uv run python scripts/04_chunk.py [--strategy recursive --tokens 256] [--sweep]`
# - VS Code, cell by cell: click **Run Cell** above each `# %%`
# - Jupyter: `make notebooks`, then open `notebooks/04_chunk.ipynb`
#
# **Variables to explore afterwards:** `tokens_df`, `page_chunks_df`, `chunks_df`,
# `coverage_df`, `sweep_df`.

# %%
import pandas as pd

from finsight.chunking import pipeline as cp
from finsight.chunking.base import DocText
from finsight.chunking.page_aware import PageAwareChunker, is_column_header
from finsight.chunking.parent_child import ParentChildChunker
from finsight.chunking.registry import CHUNKERS
from finsight.chunking.tokenizer import get_tokenizer
from finsight.config import MAX_EMBED_TOKENS, load_settings
from finsight.data import financebench as fb
from finsight.ingestion import pipeline as ip
from finsight.notebook import Param, configure_pandas, in_notebook, preview, script_args, show, step
from finsight.observability.logging import configure_logging
from finsight.schemas import Page

configure_pandas()

# %% Parameters: in a notebook, edit the defaults here; in the terminal, use the flags.
P = script_args(
    "Step 04: chunk the pages",
    config=Param(None, str, "profile YAML (default: configs/local.yaml)"),
    strategy=Param(None, str, "fixed | recursive | page_aware | parent_child (default: config)"),
    tokens=Param(None, int, "chunk size (child size for parent_child; default: config)"),
    doc=Param("3M_2018_10K", str, "document for the walkthrough"),
    page_idx=Param(59, int, "0-based page for the walkthrough (59 = printed page 60)"),
    sweep=Param(False, bool, "compare all strategies × sizes (~2-3 min) and save a report"),
    force=Param(False, bool, "re-chunk even if outputs are up to date"),
)
P

# %%
overrides = {}
if P.strategy:
    overrides["strategy"] = P.strategy
if P.tokens:
    overrides["child_tokens" if P.strategy == "parent_child" else "chunk_tokens"] = P.tokens
settings = load_settings(P.config, {"chunking": overrides} if overrides else None)
configure_logging(settings.logging.level, settings.logging.json_format)
show(pd.Series(settings.chunking.model_dump(), dtype=object), "Chunking settings")
tokenizer = get_tokenizer(settings.embedding.model)
pages_df = ip.load_pages_df(settings)
meta = fb.load_document_meta(settings)[P.doc]
doc_pages = [Page(**r) for r in pages_df[pages_df["doc_name"] == P.doc].to_dict("records")]
page_text = next(p.text for p in doc_pages if p.page_idx == P.page_idx)

# %% [markdown]
# ## 1. Tokens: what the size limit is measured in
# The embedding model's own WordPiece tokenizer. Numbers are expensive: `(1,577)` is 6 tokens.
# bge accepts 512 tokens including 2 special ones, so **510** tokens of text; more is silently cut.

# %%
step("1. Tokens")
samples = [
    "Purchases of property, plant and equipment (PP&E) | (1,577) | (1,373) | (1,420)",
    "Capital expenditures were primarily for manufacturing capacity.",
]
tokens_df = pd.DataFrame(
    [{"text": t, "tokens": tokenizer.count(t), "words": len(t.split())} for t in samples]
)
show(tokens_df, "Tokens vs words")
print(
    f"This page: {tokenizer.count(page_text)} tokens; model limit {MAX_EMBED_TOKENS} text tokens\n"
)

# %% [markdown]
# ## 2. One page, four strategies
# Same page, same budget. Look at `pages` (does a chunk span two pages?), where chunks start and
# end, and whether table chunks carry the `(Millions) | 2018 | 2017 | 2016` header.

# %%
step(f"2. {P.doc} page_idx {P.page_idx} through every strategy")
walk_rows = []
for name in ("fixed", "recursive", "page_aware", "parent_child"):
    cfg = settings.chunking.model_copy(update={"strategy": name})
    chunker = cp.build_chunker(cfg, tokenizer)
    for c in chunker.chunk(doc_pages, meta):
        if c.page_start <= P.page_idx <= c.page_end:
            lines = c.text.splitlines()
            walk_rows.append(
                {
                    "strategy": name,
                    "pages": f"{c.page_start}-{c.page_end}",
                    "tokens": c.n_tokens,
                    "starts_with_header": is_column_header(lines[0]),
                    "first_line": lines[0][:55],
                    "last_line": lines[-1][:55],
                }
            )
page_chunks_df = pd.DataFrame(walk_rows)
show(page_chunks_df, "Chunks touching this page")
show(page_chunks_df.groupby("strategy").size().rename("chunks"), "Chunks per strategy")

# %% [markdown]
# ## 3. Overlap: the end of one chunk is repeated at the start of the next
# So a fact sitting exactly on a boundary appears whole in at least one chunk.

# %%
step("3. Overlap between neighbouring page-aware chunks")
pa = PageAwareChunker(tokenizer, settings.chunking.chunk_tokens, settings.chunking.overlap_tokens)
doc = DocText.from_pages(doc_pages, tokenizer)
page_chunks = pa.chunk_page(doc, P.page_idx, meta)
if len(page_chunks) > 1:
    a_lines, b_lines = page_chunks[0].text.splitlines(), page_chunks[1].text.splitlines()
    shared = [ln for ln in b_lines if ln in a_lines and not is_column_header(ln)]
    print(f"chunk 0 ends with : {a_lines[-1][:80]}")
    print(f"chunk 1 starts    : {b_lines[1 if is_column_header(b_lines[0]) else 0][:80]}")
    print(
        f"{len(shared)} lines appear in both chunks "
        f"(overlap {settings.chunking.overlap_tokens} tokens)\n"
    )
else:
    print("This page fits in one chunk; pick a longer page to see overlap.\n")

# %% [markdown]
# ## 4. Carrying the table header
# With small chunks a statement splits into several pieces. Without the header, a piece reads
# `Inventories | (509) | (387) | 57`: which year is which? Compare carry on vs off.

# %%
step("4. Table header carry: on vs off")
for carry in (True, False):
    ch = PageAwareChunker(tokenizer, 150, 0, carry_table_header=carry).chunk_page(
        doc, P.page_idx, meta
    )
    with_header = sum(any(is_column_header(ln) for ln in c.text.splitlines()) for c in ch)
    print(
        f"carry_table_header={carry!s:5}: {with_header}/{len(ch)} chunks contain the column header"
    )
print()

# %% [markdown]
# ## 5. Parent-child: small children for search, the page for the LLM

# %%
step("5. Parent-child")
pc = ParentChildChunker(
    tokenizer, settings.chunking.child_tokens, 30, settings.chunking.parent_tokens
)
children = [c for c in pc.chunk(doc_pages, meta) if c.page_start == P.page_idx]
parent = next(p for p in pc.parents(doc_pages, meta) if p.page_start == P.page_idx)
show(
    pd.DataFrame(
        [{"chunk_id": c.chunk_id, "tokens": c.n_tokens, "parent_id": c.parent_id} for c in children]
    ),
    f"{len(children)} children of this page",
)
show(
    preview({"parent_id": parent.chunk_id, "tokens": parent.n_tokens, "text": parent.text}, 300),
    "Their parent",
)

# %% [markdown]
# ## 6. Chunk the whole corpus with the configured strategy
# Writes `data/chunks/<strategy>_<size>.parquet` (skipped when inputs and settings are unchanged).
# Quality checks:
# - `over_limit_share` recounts every chunk with the real tokenizer and must be 0.
# - `evidence_in_one_chunk` is the share of the 189 evidence passages that ONE chunk on their page
#   contains (≥ 0.8 overlap).
# - `union_overlap_mean` covers all chunks of that page together (< 1 means text was lost).

# %%
step(f"6. Chunk the corpus: {cp.config_name(settings)}")
report = cp.chunk_corpus(settings, tokenizer=tokenizer, force=P.force, progress=not in_notebook())
chunks_df = cp.load_chunks_df(settings)
coverage_df = cp.evidence_coverage(settings, chunks_df)
show(
    pd.Series(
        {
            "file": str(report.path.relative_to(settings.root)) if report.path else "-",
            "documents": report.n_docs,
            "chunks": report.n_chunks,
            "parents": report.n_parents,
            "skipped (up to date)": report.skipped,
            "time (s)": report.elapsed_s,
            **cp.chunk_stats(chunks_df, tokenizer),
            **cp.coverage_summary(coverage_df),
        },
        dtype=object,
    ),
    "ChunkReport + quality",
)
show(
    chunks_df[["chunk_id", "page_start", "page_end", "n_tokens", "text"]].head(3),
    "chunks_df.head(3)",
)
show(
    coverage_df.sort_values("best_chunk_overlap").head(5),
    "Evidence least contained in a single chunk",
)

# %% [markdown]
# ## 7. (optional) Sweep: every strategy × size on the same pages
# Set `sweep=True` (or `--sweep`). Saves `results/<date>_chunking_sweep/report.md`.

# %%
sweep_df = pd.DataFrame()
if P.sweep:
    step("7. Sweep")
    sweep_df = cp.run_sweep(
        settings, tokenizer=tokenizer, force=P.force, progress=not in_notebook()
    )
    out = cp.save_sweep_report(settings, sweep_df)
    cols = [
        "config",
        "chunks",
        "tokens_p50",
        "multi_page_share",
        "evidence_in_one_chunk",
        "union_overlap_mean",
    ]
    show(sweep_df[cols], f"Sweep (full table in {out.relative_to(settings.root)}/report.md)")

# %% Summary
step("Summary")
stats = cp.chunk_stats(chunks_df, tokenizer)
if stats.get("over_limit_share", 0) > 0:
    print("Some chunks exceed the model limit: check the tokenizer and chunk_tokens.")
    if not in_notebook():
        raise SystemExit(1)
else:
    print(
        f"P3 done: {report.n_chunks} chunks in {cp.config_name(settings)}. "
        f"Strategies available: {sorted(CHUNKERS)}"
    )
    print("Next: scripts/05_embed_and_index.py (P4)")
