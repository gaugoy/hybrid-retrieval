# Evaluation log

Every measured comparison in the project, newest phase last. Each table links to the committed report it came from.
Numbers are on the FinanceBench open-source sample (150 questions, 189 evidence items, 84 filings, 12,013 pages).

## P2 — Parsing quality

Source: `scripts/03_parse_pdfs.py`, details in [`data_card.md`](data_card.md#parsing-results-p2-scripts03_parse_pdfspy).

| Measure | Plain PyMuPDF `get_text()` | Ours (row reconstruction) |
|---|---|---|
| Evidence found on the gold page (overlap ≥ 0.8) | n/a | **189 / 189** (mean 0.998, 0 page offsets) |
| Label–number adjacency (160 numeric evidence pages) | 0.28 | **0.97** |
| Time for 12,013 pages | ~8 ms/page | ~6 ms/page (70 s sequential, 17 s on 6 processes) |
| `find_tables()` (rejected) | 1.5 s/page (≈ 5 h), cells lose spaces | n/a |

## P3 — Chunking

Source: [`results/2026-11-24_chunking_sweep/report.md`](../results/2026-11-24_chunking_sweep/report.md) (`make chunk-sweep`).
Tokenizer `BAAI/bge-small-en-v1.5`, overlap 64 tokens, table-header carry on.

| config | chunks | tokens p50 | chunks spanning 2 pages | evidence in one chunk | union overlap |
|---|---|---|---|---|---|
| fixed_256 | 44,998 | 256 | 34.0% | 43.9% | 0.998 |
| fixed_384 | 27,013 | 384 | 49.6% | 68.3% | 0.998 |
| fixed_480 | 20,793 | 480 | 60.1% | **86.8%** | 0.998 |
| recursive_256 | 45,333 | 242 | 29.7% | 44.4% | 0.998 |
| recursive_384 | 27,927 | 368 | 43.1% | 68.8% | 0.998 |
| recursive_480 | 21,789 | 462 | 52.5% | 80.4% | 0.998 |
| page_aware_256 | 47,847 | 236 | **0%** | 47.6% | 0.998 |
| page_aware_384 (default) | 31,114 | 355 | **0%** | 72.0% | 0.998 |
| page_aware_480 | 25,337 | 419 | **0%** | 82.5% | 0.998 |
| parent_child_128 | 95,174 (+11,944 parents) | 113 | 0% | 28.6% (parent: the whole page) | 0.998 |
| parent_child_200 | 60,593 (+11,944 parents) | 182 | 0% | 38.6% (parent: the whole page) | 0.998 |
| parent_child_256 | 47,847 (+11,944 parents) | 236 | 0% | 47.6% (parent: the whole page) | 0.998 |

All configurations: 0 chunks over the 510-token limit (recounted with the real tokenizer).

**What the numbers say**
- **Size dominates single-chunk containment:** ~45% → ~70% → ~83% from 256 → 384 → 480 tokens. Evidence passages in filings
  are long (often a whole statement), and numbers are token-expensive.
- **Only page-aware chunking gives exact citations:** 0% of chunks span two pages, against 30–60% for fixed/recursive. That
  costs a few points of containment at 480 tokens (82.5% vs 86.8% for fixed).
- **Table header carry:** on pages with a column-header row, table chunks that contain the header go from **64.1% → 94.1%**,
  so a chunk like `Inventories | (509) | (387) | 57` still says which year each number belongs to.
- **No text is lost** by any strategy (union overlap 0.998 = parsing level).

**Caveat (why P3 doesn't pick the winner):** "evidence in one chunk" favours big chunks by construction. Bigger chunks also make
blurrier embeddings (one vector for more topics) and send more noise to the LLM. The choice is made in **P5** with retrieval
metrics (Recall@k, MRR, NDCG on the gold pages). Default until then: `page_aware`, 384 tokens.

## P4 — Indexing

Source: `scripts/05_embed_and_index.py` (`make index`), config `page_aware_384`, model `BAAI/bge-small-en-v1.5` on MPS (M5).

| Measure | Value |
|---|---|
| Chunks indexed | 31,114 (30,875 distinct texts) |
| Embedding | 293 s, **106 chunks/s** (batch 64, ~355 tokens/chunk) |
| Qdrant upsert / BM25 build | 11 s / 1.9 s |
| Total first run / re-run | 5 min 6 s / 0.07 s (manifest skip) |
| On disk | Qdrant 244 MB (vectors + full chunk payloads) · BM25 25 MB (27,097 words) · embedding cache 66 MB |
| Peak memory | 1.2 GB RSS |

**Gold-page rank for one real question** (`financebench_id_03029`, 3M FY2018 capex; gold = 3M_2018_10K page 60). Rank of the
first chunk from the gold page, out of 31,114:

| Query wording | dense + instruction | dense, no instruction | BM25 | dense, 3M only (no instr.) | BM25, 3M only |
|---|---|---|---|---|---|
| Full question (mentions "cash flow statement") | 47 | 36 | 121 | 14 | 66 |
| Short: "What is the FY2018 capital expenditure amount for 3M?" | 77 | 55 | > 1000 | 44 | 666 |
| Filing's own words ("purchases of property, plant and equipment" …) | – | – | 2 | 2 (with instr.) | – |

**What it says**
- **Vocabulary mismatch:** the question says "capital expenditure", the statement says "Purchases of property, plant and
  equipment". BM25 can't bridge that at all, and the small dense model only partly does. Neither index alone puts the gold
  page in a top 10, so P5 (hybrid + company filter + cross-encoder rerank) and P7 (query rewriting, HyDE) have real work.
- **The company filter helps a lot** (dense 36 → 14), because most near-misses are other companies' capex pages.
- **Query instruction:** on this one question it *hurts* (36 vs 47). One question is an anecdote; P5 measures it on all 150.

## P4b — Embedder comparison (dense retrieval only)

Source: [`results/2026-11-24_embedder_comparison_all/report.md`](../results/2026-11-24_embedder_comparison_all/report.md) and
[`..._q30/report.md`](../results/2026-11-24_embedder_comparison_q30/report.md) (`make compare-embedders`, step 05b). Same
`page_aware_384` chunks for every embedder; no metadata filter, no BM25, no reranker; relevant = a chunk on a gold evidence page.

**All 150 questions, full corpus (31,114 chunks)**

| preset | hit@1 | hit@5 | hit@10 | recall@10 | MRR | nDCG@10 | doc_hit@10 |
|---|---|---|---|---|---|---|---|
| bge-small (with query instruction) | 0.127 | 0.340 | **0.460** | 0.433 | 0.227 | 0.272 | **0.893** |
| bge-small-noinstr | 0.133 | 0.360 | 0.453 | 0.423 | 0.230 | 0.269 | 0.867 |

**First 30 questions, their 18 filings only (5,325 chunks)**: the free-tier-friendly setting

| preset | dims | hit@1 | hit@5 | hit@10 | recall@10 | MRR | nDCG@10 | doc_hit@10 | embed time |
|---|---|---|---|---|---|---|---|---|---|
| bge-small | 384 | 0.100 | 0.267 | 0.433 | 0.383 | 0.192 | 0.229 | 0.967 | (cached; ~50 s fresh) |
| bge-small-noinstr | 384 | 0.067 | 0.267 | 0.367 | 0.300 | 0.166 | 0.183 | 0.967 | (same index) |
| bge-base | 768 | 0.100 | **0.400** | **0.500** | 0.417 | 0.218 | 0.246 | 0.933 | 184 s (29 chunks/s) |

**What it says**
- **Dense alone finds the gold page in the top 10 for less than half the questions** (0.46), but the right *filing* for 89%. The
  gap is page-level: within a 300-page 10-K, many pages look alike to a small embedding. That is P5's job (BM25 for exact line
  items, company filter, cross-encoder rerank).
- **The query instruction:** +0.7 pt hit@10 on 150 questions (noise), +6.7 pt on the 30-question subset. Small, consistently
  non-negative: keep it on.
- **bge-base vs bge-small:** +6.7 pt hit@10 / +13 pt hit@5 on 30 questions (2–4 questions: suggestive, not significant) at 3.6×
  the embedding time. Worth a full-corpus run (~18 min) in P5 before deciding.
- **Online models:** presets for Gemini, Mistral, Cohere, Jina, Voyage and OpenAI are ready (`finsight embedders`); add a free
  key and run `make compare-embedders E=bge-small,bge-base,gemini-001 Q=30` to extend this table.
