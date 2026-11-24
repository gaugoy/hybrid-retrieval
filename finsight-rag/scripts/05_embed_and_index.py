# %% [markdown]
# # Step 05 — Embed the chunks and build the indexes   (Phase P4)
#
# **What this step does:** turns every chunk into two searchable indexes.
#
# | index | stores per chunk | good at |
# |---|---|---|
# | **dense** (Qdrant, `indexes/qdrant/`) | 384-dim vector + chunk | meaning ("capex" ≈ "PP&E") |
# | **sparse** (BM25, `indexes/bm25/`) | word statistics | exact words: tickers, line items, years |
#
# P5 combines them (hybrid search). Both are rebuilt only when the chunks or the model change
# (`indexes/manifest.json`), and vectors are cached in SQLite, so re-runs are cheap.
#
# **Three ways to run it**
# - Terminal: `uv run python scripts/05_embed_and_index.py [--strategy recursive --tokens 384]`
#   (add `--force` to rebuild)
# - VS Code, cell by cell: click **Run Cell** above each `# %%`
# - Jupyter: `make notebooks`, then open `notebooks/05_embed_and_index.ipynb`
#
# ⚠️ Embedded Qdrant locks `indexes/qdrant/`: close other scripts/notebooks using it first.
#
# **Variables to explore afterwards:** `sim_df`, `instruction_df`, `report`, `bm25_tokens_df`,
# `manifest_df`, `dense_df`, `sparse_df`, `filter_df`, `gold_df`.

# %%
import numpy as np
import pandas as pd

from finsight.chunking import pipeline as cp
from finsight.config import load_settings
from finsight.container import Container
from finsight.embedding.cache import EmbeddingCache, cache_key
from finsight.indexing import bm25_store
from finsight.indexing import pipeline as ixp
from finsight.notebook import Param, configure_pandas, in_notebook, preview, script_args, show, step
from finsight.observability.logging import configure_logging
from finsight.schemas import QueryFilters

configure_pandas()

# %% Parameters: in a notebook, edit the defaults here; in the terminal, use the flags.
P = script_args(
    "Step 05: embed chunks, build Qdrant + BM25",
    config=Param(None, str, "profile YAML (default: configs/local.yaml)"),
    strategy=Param(None, str, "fixed | recursive | page_aware | parent_child (default: config)"),
    tokens=Param(None, int, "chunk size (child size for parent_child; default: config)"),
    embedder=Param(None, str, "embedding preset, e.g. gemini-001 (default: config / EMBEDDER)"),
    query=Param("3M capital expenditure 2018", str, "sanity query"),
    company=Param("3M", str, "company for the filter demo"),
    filter_query=Param(
        "capital expenditure 2018", str, "query (no company name) for the filter demo"
    ),
    k=Param(3, int, "results per index in the sanity query"),
    gold_question=Param(
        "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response "
        "to the question by relying on the details shown in the cash flow statement.",
        str,
        "FinanceBench question (financebench_id_03029)",
    ),
    force=Param(False, bool, "rebuild even if the manifest says up to date"),
)
P

# %%
overrides = {}
if P.strategy:
    overrides["strategy"] = P.strategy
if P.tokens:
    overrides["child_tokens" if P.strategy == "parent_child" else "chunk_tokens"] = P.tokens
all_overrides: dict = {"chunking": overrides} if overrides else {}
if P.embedder:
    all_overrides["embedding"] = {"preset": P.embedder}
settings = load_settings(P.config, all_overrides or None)
configure_logging(settings.logging.level, settings.logging.json_format)
c = Container(settings)
chunk_report = cp.chunk_corpus(settings, progress=not in_notebook())  # skipped if up to date
chunks_df = cp.load_chunks_df(settings)
show(
    pd.Series(
        {
            **settings.embedding.model_dump(),
            "chunks file": str(cp.chunks_path(settings).relative_to(settings.root)),
            "chunks": len(chunks_df),
            "qdrant collection": c.collection,
            "qdrant mode": settings.vector_store.qdrant_url or "embedded (indexes/qdrant/)",
            "bm25 dir": str(ixp.bm25_dir(settings).relative_to(settings.root)),
        },
        dtype=object,
    ),
    "Embedding settings",
)

# %% [markdown]
# ## 1. The embedder: text → a unit-length vector
# A **bi-encoder** encodes each text on its own. Vectors are L2-normalized (length 1), so
# **cosine similarity = dot product**. Related texts get a high score even with different words.

# %%
step("1. Embeddings and cosine similarity")
emb = c.embedder
samples = [
    "Purchases of property, plant and equipment (PP&E) | (1,577) | (1,373) | (1,420)",
    "Capital expenditures were primarily for manufacturing capacity.",
    "Net sales | 32,765 | 31,657 | 30,109",
]
vecs = emb.embed_documents(samples)
print(
    f"model {emb.model_name} on {emb.device}: dim {emb.dim}, "
    f"max_seq_length {getattr(emb, 'max_seq_length', 'n/a')}, "
    f"norms {np.round(np.linalg.norm(vecs, axis=1), 3)}\n"
)
labels = ["PP&E row", "capex prose", "net sales row"]
sim_df = pd.DataFrame(vecs @ vecs.T, index=labels, columns=labels).round(3)
show(sim_df, "Cosine similarity (= dot product of unit vectors)")

# %% [markdown]
# ## 2. Query form vs document form
# Retrieval models embed queries differently from documents: bge prepends
# `"Represent this sentence for searching relevant passages: "` (query_template), Gemini 2 uses
# `"task: search result | query: ..."`, Cohere/Voyage/Jina send an input type. Compare the
# query→passage scores when the query is embedded as a QUERY vs as a plain document.
# (`scripts/05b_compare_embedders.py` measures bge-small vs bge-small-noinstr on all questions.)

# %%
step("2. Query form vs document form")
q_with = emb.embed_query(P.query)
q_without = emb.embed_documents([P.query])[0]  # same text, embedded as a document
instruction_df = pd.DataFrame(
    {
        "passage": labels,
        "as query (template)": np.round(vecs @ q_with, 3),
        "as document": np.round(vecs @ q_without, 3),
    }
)
show(instruction_df, f"Query: {P.query!r}")

# %% [markdown]
# ## 3. The embedding cache
# Key = `sha256(model + "\x00" + text)` → vector bytes, in `indexes/embed_cache/<model>.sqlite`.
# Only texts missing from the cache are sent to the model.

# %%
step("3. Embedding cache")
cache_file = ixp.embed_cache_path(settings, emb.model_name)
cache = EmbeddingCache(cache_file)
texts = chunks_df["embed_text"].tolist()
keys = [cache_key(emb.model_name, t) for t in texts]
found = cache.get_many(list(set(keys)))
cache.close()
print(f"key of chunk 0: {keys[0][:16]}…")
print(
    f"{sum(k in found for k in keys):,} / {len(keys):,} chunks already cached in "
    f"{cache_file.name} ({len(set(keys)):,} distinct texts: "
    f"{len(keys) - len(set(keys))} chunks repeat another chunk's text)\n"
)

# %% [markdown]
# ## 4. Build both indexes (skipped when the manifest says they're up to date)
# Dense: embed misses → recreate the Qdrant collection → upsert points in batches of 256
# (id = uuid5(chunk_id), payload = the whole chunk). Sparse: tokenize → BM25 index → save.

# %%
step(f"4. Index {cp.config_name(settings)}")
report = ixp.index_corpus(
    settings,
    embedder=emb,
    client=c.qdrant_client,
    chunks_df=chunks_df,
    force=P.force,
    progress=not in_notebook(),
)
show(pd.Series(report.as_dict(), dtype=object), "IndexReport")

# %% [markdown]
# ## 5. What BM25 actually sees
# Lowercased words of ≥ 2 characters, English stopwords removed, no stemming. Note `1,577`
# becomes `577`: numbers split at the comma (questions rarely contain the answer's digits).

# %%
step("5. BM25 tokenization")
bm25 = c.bm25
bm25_tokens_df = pd.DataFrame(
    [{"text": t, "bm25 tokens": bm25_store.tokenize([t])[0]} for t in [*samples[:2], P.query]]
)
show(bm25_tokens_df, f"Tokenization (vocabulary: {bm25.vocab_size:,} words)")

# %% [markdown]
# ## 6. The manifest: what each index was built from

# %%
step("6. indexes/manifest.json")
manifest_df = pd.DataFrame.from_dict(ixp.manifest(settings).load(), orient="index")
show(manifest_df.drop(columns=["chunks_sha256"], errors="ignore"), "Manifest entries")

# %% [markdown]
# ## 7. Sanity query: top results from each index
# `page` is the **printed** page number (page_idx + 1). For "3M capital expenditure 2018" the
# cash-flow statement (3M_2018_10K, page 60) should be near the top.

# %%
step(f"7. Sanity query: {P.query!r}")
by_id = chunks_df.set_index("chunk_id")


def row(rank: int, doc: str, page_idx: int, score: float, text: str) -> dict:
    return {
        "rank": rank,
        "doc": doc,
        "page": page_idx + 1,
        "score": round(score, 3),
        "text": preview(text, 80),
    }


dense_hits = c.qdrant.search(c.collection, emb.embed_query(P.query), k=P.k)
dense_df = pd.DataFrame(
    [
        row(i + 1, h.payload["doc_name"], h.payload["page_start"], h.score, h.payload["text"])
        for i, h in enumerate(dense_hits)
    ]
)
sparse_hits = bm25.search(P.query, k=P.k)
sparse_df = pd.DataFrame(
    [
        row(
            i + 1,
            by_id.at[h.chunk_id, "doc_name"],
            int(by_id.at[h.chunk_id, "page_start"]),
            h.score,
            by_id.at[h.chunk_id, "text"],
        )
        for i, h in enumerate(sparse_hits)
    ]
)
show(dense_df, "Dense (cosine)")
show(sparse_df, "BM25 (unbounded score)")

# %% [markdown]
# ## 8. Metadata filter: search only one company's filings
# Qdrant applies the filter **during** the search, so all k results match.

# %%
step(f"8. Dense search for {P.filter_query!r}, without / with company = {P.company!r}")
fq = emb.embed_query(P.filter_query)
unfiltered = c.qdrant.search(c.collection, fq, k=5)
filtered = c.qdrant.search(c.collection, fq, k=5, filters=QueryFilters(companies=[P.company]))


def where(hits: list) -> list[str]:
    return [f"{h.payload['doc_name']} p{h.payload['page_start'] + 1}" for h in hits]


filter_df = pd.DataFrame(
    {"no filter": where(unfiltered), f"company = {P.company}": where(filtered)}
)
show(filter_df, "Top-5 documents, with and without the filter")

# %% [markdown]
# ## 9. Where is the gold page for a real FinanceBench question?
# The question says **"capital expenditure"**; 3M's cash-flow statement says **"Purchases of
# property, plant and equipment"**. BM25 needs the exact words, and the dense model only partly
# bridges the gap. This **vocabulary mismatch** is what hybrid search, reranking (P5) and query
# rewriting (P7) are for. Rank = position of the first chunk from the gold page (None = > 1000).
# Try the short form "What is the FY2018 capital expenditure amount for 3M?": BM25 loses the
# page entirely, because "cash flow statement" was the only bridge to the right words.

# %%
step("9. Gold-page rank for a real question")


def gold_rank(chunk_ids: list[str], doc: str, page_idx: int) -> int | None:
    prefix = f"{doc}::p{page_idx}-"
    return next((i + 1 for i, cid in enumerate(chunk_ids) if cid.startswith(prefix)), None)


GOLD_Q, GOLD_DOC, GOLD_PAGE = P.gold_question, "3M_2018_10K", 59
only_3m = QueryFilters(companies=["3M"])
mask_3m = (chunks_df["company"] == "3M").to_numpy()
variants = {
    "dense, query form": emb.embed_query(GOLD_Q),
    "dense, document form": emb.embed_documents([GOLD_Q])[0],
}
gold_rows = []
for name, vec in variants.items():
    for label, flt in (("all filings", None), ("3M only", only_3m)):
        hits = c.qdrant.search(c.collection, vec, k=1000, filters=flt)
        ids = [h.payload["chunk_id"] for h in hits]
        gold_rows.append(
            {"search": name, "scope": label, "rank": gold_rank(ids, GOLD_DOC, GOLD_PAGE)}
        )
for label, m in (("all filings", None), ("3M only", mask_3m)):
    ids = [h.chunk_id for h in bm25.search(GOLD_Q, k=1000, mask=m)]
    gold_rows.append(
        {"search": "bm25", "scope": label, "rank": gold_rank(ids, GOLD_DOC, GOLD_PAGE)}
    )
gold_df = pd.DataFrame(gold_rows)
show(gold_df, f"{GOLD_Q!r} → gold {GOLD_DOC} page {GOLD_PAGE + 1}")

# %% Summary
step("Summary")
ok = report.qdrant_points == len(chunks_df) and report.bm25_docs == len(chunks_df)
c.close()  # release the embedded-Qdrant folder lock
if not ok:
    print("Index counts don't match the chunk count: re-run with --force.")
    if not in_notebook():
        raise SystemExit(1)
else:
    print(
        f"P4 done: {report.qdrant_points:,} vectors in {report.collection} "
        f"({report.qdrant_mb} MB) and BM25 over {report.bm25_docs:,} chunks "
        f"({report.bm25_vocab:,} words, {report.bm25_mb} MB)."
    )
    print("Next: scripts/06_retrieve.py (P5)")
