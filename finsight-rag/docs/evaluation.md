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
