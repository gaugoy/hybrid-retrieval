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

## P5 — Retrieval ablation

Source: [`results/2026-11-24_retrieval_ablation/report.md`](../results/2026-11-24_retrieval_ablation/report.md) (+
[`_extra`](../results/2026-11-24_retrieval_ablation_extra/report.md)), `make eval-retrieval`. 150 questions, bge-small,
candidate_k 50, metrics on the final top 10. Values are means; std ≈ 0.5 for hit@k, so differences under ~0.05 are noise.

**hit@10 (gold evidence page in the top 10)**

| variant | page_aware_384 | fixed_384 | recursive_384 | parent_child_200 |
|---|---|---|---|---|
| dense | 0.460 | 0.407 | 0.467 | 0.473 |
| bm25 | 0.147 | 0.153 | 0.160 | 0.147 |
| hybrid (RRF) | 0.320 | 0.300 | 0.387 | 0.347 |
| hybrid + company filter | 0.447 | 0.393 | 0.473 | 0.440 |
| hybrid + company + year filter | 0.520 | 0.533 | **0.553** | 0.500 |
| hybrid + company filter + rerank | 0.527 | 0.507 | 0.513 | 0.520 |
| **dense + company + year filter** (extra run) | 0.593 | 0.560 | **0.613** | 0.587 |
| hybrid + company + year filter + rerank (extra run) | 0.607 | 0.600 | 0.580 | 0.573 |

**Chosen default (`configs/base.yaml`): recursive_384 · dense · company + year filter · no reranker** — hit@10 **0.613**,
hit@5 0.527, MRR 0.353, nDCG@10 0.385, doc_hit@10 0.993, ~110 ms per question. BM25 and the reranker stay available by config.

**doc_hit@10 (right filing in the top 10):** dense 0.89–0.91 · hybrid 0.78–0.85 · with the company filter **0.96–0.99**.
**Latency (p50 per question):** dense ~20 ms · hybrid + filter ~110–220 ms · + rerank (50 pairs, bge-reranker-base, MPS) ~1.6–2.8 s.

**What it says**
- **Filters are the biggest lever.** The company filter (found in 98% of questions) takes doc_hit from ~0.83 to ~0.98: most
  dense errors were *other companies'* look-alike pages. The year filter adds another +4–14 pts hit@10; it can drop prior-year
  data in principle, but on FinanceBench it helps everywhere.
- **Hybrid without filters is worse than dense alone** (0.32 vs 0.46). FinanceBench questions are long and instruction-heavy
  ("Assume that you are a public equities analyst. Answer the following question by primarily using information that is shown in
  the balance sheet: ..."), so BM25 mostly matches boilerplate, and RRF gives that weak list equal weight. Inside one company's
  filings BM25 becomes useful again.
- **Dense + filters beats hybrid + filters + rerank** (0.61 vs 0.58 on recursive_384), at 1/25 of the latency. Inside one company,
  BM25 still adds noise, and bge-reranker-base (512-token window, trained on web text) doesn't sort financial tables better
  than the embedding does. "Add BM25 and a reranker" is the usual advice; on this data it is measurably not worth it.
- **Chunkers are within noise of each other** once filters are on (0.56–0.61); recursive_384 is best by 2–5 pts. Containment
  (P3) favoured big chunks; retrieval says strategy matters less than filters and the query.
- **The remaining gap is page-level:** with doc_hit ≈ 0.99 and hit@10 ≈ 0.55, the right filing is found but the right page often
  isn't (vocabulary mismatch, near-duplicate pages). P7 targets this with query rewriting / HyDE.

## P7 — Advanced RAG

### Contextual chunk headers (150 questions, dense, `results/2026-11-24_contextual_headers/`)

Each chunk's *embedded* text gets a header `company | doc FY | page` (the LLM still reads the plain text).

| variant | hit@10 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| recursive_384 + company/year filter | 0.613 | 0.580 | 0.353 | 0.385 |
| **recursive_384_ctx + company/year filter** | **0.673** | **0.629** | 0.350 | **0.400** |
| recursive_384, no filter | 0.467 | — | — | — |
| recursive_384_ctx, no filter | 0.567 | — | — | — |

doc_hit@10 without a filter: 0.913 → 0.973.

**What it says**
- Contextual headers are the **largest single gain since filters**: +6 pts hit@10 with filters, +10 pts without.
- The header puts the company and year into the vector, so it partly substitutes for the metadata filter. That makes it useful
  when the filter extractor misses.
- MRR is flat. The header helps get the right page *into* the top 10, but not to rank 1.
- It costs nothing at query time. The only cost is a one-off re-embed.

### Query transforms (multi_query / HyDE / decompose)

`configs/experiments/advanced.yaml` (`make eval-advanced`) compares them on recursive_384 with the local Qwen2.5-1.5B as the
helper LLM. **Not measured yet:** the 60-question run was stopped before it finished, because each question needs 1–4 LLM calls
on CPU/MPS. Run `make eval-advanced` to fill this in. The report adds an "LLM calls/q" column for the cost side.

## P8 — End-to-end answer quality (30 questions, local Qwen2.5-1.5B, `results/2026-11-24_end_to_end/`)

The generator and the judge are both the local `hf-qwen1.5b`: no keys on this machine, and nothing is spent.

| pipeline | numeric match | gold page retrieved | abstained | cited (model) | auto-cited | p50 s | p95 s | tokens/q | $ |
|---|---|---|---|---|---|---|---|---|---|
| naive (dense, no filter) | 0.24 | 0.17 | 0.37 | 0.43 | 0.30 | 3.8 | 17.3 | 3,837 | 0 |
| **baseline** (hybrid_rerank: dense + company/year filter) | **0.31** | **0.47** | 0.27 | 0.57 | 0.43 | 4.8 | 11.2 | 3,817 | 0 |
| corrective (CRAG) | 0.28 | 0.47 | 0.33 | 0.57 | 0.43 | 12.8 | 50.3 | 3,440 | 0 |

`numeric_match` = a number in the answer within 1% of the gold's first number (87 of the 90 rows have a numeric gold).
n = 30, so the standard error of a 0/1 rate is about ±0.08: only the naive → baseline gap in retrieval (0.17 → 0.47) is clearly real.

**What it says**
- **Retrieval drives answer quality.** The filtered baseline finds the gold page about 3× as often as naive and answers more
  numbers correctly. The 1.5B generator is now the bottleneck: it gets about 0.31 even though the page is found 0.47 of the time.
- **Corrective RAG didn't pay off here.** It abstained more and cost 2.6× the p50 latency (4.5× the p95) for no accuracy gain. With a 1.5B model,
  the grader and rewrite steps are as weak as the answerer.
- **The local judge is unusable.** The 1.5B judge rated **90/90 answers correct**, including 63 of the 87 whose number did not
  match the gold. It is also the generator's own model (self-preference bias). "Judge correct" here is **not a metric**; use
  `numeric_match` until a stronger judge from a different model family (e.g. `JUDGE=claude-haiku` or a free-tier Gemini) is
  configured and validated against human labels (`scripts/10b_label_judge_sample.py`, 20 labels). Human validation: **not done yet**.
- **Cost:** $0 (local). The token counts give the paid-model estimate once prices are set in `configs/llms.yaml`.
