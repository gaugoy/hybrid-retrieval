# Data card: FinanceBench (open-source sample)

> Every number below was measured on 2026-11-23 by `scripts/01_download_data.py` and
> `scripts/02_explore_data.py`. Re-run them to reproduce.

## Source and license

| | |
|---|---|
| Benchmark | FinanceBench (Islam et al., 2023, [arXiv 2311.11944](https://arxiv.org/abs/2311.11944)), by Patronus AI |
| Repository | <https://github.com/patronus-ai/financebench> (Q&A and metadata in `data/`, filings in `pdfs/`) |
| Subset | The open-source sample (`dataset_subset_label = OPEN_SOURCE`). The full set is licensed separately. |
| License | **CC-BY-NC-4.0**: attribution, non-commercial use only. |
| Our handling | Files are downloaded by script into the gitignored `data/` folder and **never committed**. SHA-256 checksums are in `data/raw/financebench/download_manifest.json`. |

## What was downloaded

| Item | Value |
|---|---|
| Questions | **150** |
| Evidence items | **189** (115 questions need 1 page, 31 need 2, 4 need 3) |
| Documents referenced | **84** of the 368 PDFs in the repo |
| Document types | 64 × 10-K · 8 × 10-Q · 6 × 8-K · 6 × earnings release |
| Companies | **32** (most questions: PepsiCo 11, Amcor 9, Johnson & Johnson 9, 3M 8, AMD 8, Best Buy 8, Boeing 8) |
| Size on disk | **166.5 MB** (JSONL + PDFs) |
| Pages | **12,013** in total; per filing min 4 (`FOOTLOCKER_2022_8K_dated-2022-05-20`), median 128.5, max 549 (`PEPSICO_2021_10K`) |
| Download time | ~23 s for 84 PDFs (4 parallel workers); a re-run takes 0.1 s (all skipped) |

## Question mix

| question_type | n | What it tests |
|---|---|---|
| metrics-generated | 50 | Pull or compute a metric from the financial statements (e.g. capex, DPO). Often numerical. |
| domain-relevant | 50 | Analyst-style questions (e.g. "does the company have debt maturing next year?"). |
| novel-generated | 50 | Free-form, often harder or multi-step questions written by annotators. |

Primary reasoning label: numerical 53 · information extraction 33 · logical (based on numerical) 14 · **unlabelled 50**.
The unlabelled 50 are exactly the **novel-generated** questions (`pd.crosstab` in step 02). Domain-relevant questions
mix all three labels; metrics-generated are numerical (36) or extraction (14).

## Schema mapping (raw → `finsight.schemas`)

| Raw field | Ours | Notes |
|---|---|---|
| `financebench_id` | `EvalRecord.qid` | unique |
| `question`, `answer` | `question`, `gold_answer` | answers are free text, e.g. `"$1577.00"`, `"63.86"`, full sentences |
| `question_type`, `question_reasoning` | same names | reasoning may be `None` |
| `company`, `doc_name` | same names | the company always matches the document metadata |
| `evidence[*].evidence_page_num` | `Evidence.page_idx` | **0-based**, kept as-is (verified against PyMuPDF on `3M_2018_10K`) |
| `evidence[*].evidence_text` | `Evidence.text` | |
| doc info `doc_type`, `doc_period`, `gics_sector`, `doc_link` | `DocumentMeta.doc_type`, `doc_period`, `gics_sector`, `source_url` | |

## Known quirks (read before P2)

1. **Duplicate metadata row:** `FOOTLOCKER_2023_annualreport` appears twice in the document-information file, with
   `doc_period` 2023 and 2022. No question uses it. The loader keeps the first row and logs it at INFO.
2. **Tables arrive one cell per line** in the evidence text, so a number is separated from its row label. P2 must
   rebuild table rows (`| label | 2023 | 2022 |`) so numbers stay next to their labels.
3. **Some evidence text has lost its spaces:** e.g. Corning's evidence reads `Netsales`, `Costofsales`. **Resolved in P2:**
   PyMuPDF 2.1 extracts these pages correctly (`Net sales | $ 11,303`); the problem is only in FinanceBench's stored
   evidence text, so the evidence check compares space-insensitively.
4. **Evidence is deep inside long filings:** page index median 51, max 303. By type: 10-K median page 56 (financial
   statements sit mid-document), 10-Q median 18, 8-K and earnings releases pages 0–13. Retrieval can't rely on the first pages.
5. **Multi-evidence questions** (35 of 150) need two or three pages; Recall@k counts each evidence page.
6. **Question wording varies** (`"What Was AMCOR's Adjusted Non GAAP EBITDA"`), with abbreviations and tickers
   (`MMM`). This is a reason for hybrid retrieval and company-alias filters (P5).

## How evaluation uses it

- **Retrieval (P5):** a retrieved chunk is relevant if `doc_name` matches and `page_start ≤ page_idx ≤ page_end`.
- **Generation (P8):** numeric match against `gold_answer` for metric questions, plus an LLM judge for free-text answers.
- **Small-sample caution:** 150 questions means a difference of 1–2 points is noise. Always report n per row.

## Parsing results (P2, `scripts/03_parse_pdfs.py`)

| Measure | Value |
|---|---|
| Pages parsed | 12,013 (every page kept, 0-based `page_idx`) |
| Empty pages (no text layer) | 69, mostly `WALMART_2019_10K` (29) and `AMCOR_2020_10K` (10). No evidence page is empty |
| Pages with tables (heuristic) | 5,098 (42%) |
| Characters | 40.4 M (median 3,344 per page) |
| Header/footer patterns removed | 96 across 84 documents (page numbers `##`, "table of contents", "item #", "part ii", …) |
| **Evidence found on the gold page** | **189 / 189** (overlap ≥ 0.8; mean 0.998, worst 0.973); 0 page offsets |
| Label–number adjacency on numeric evidence pages | plain PyMuPDF **0.28** → ours **0.97** (160 pages with ≥ 5 numbers) |
| Time | ~70 s sequential, ~17 s with 6 processes (M5) |

Known parsing limitations: two-column prose could merge into one row; some multi-line table labels still split from their
numbers (e.g. `VERIZON_2021_10K` page 93, adjacency 0.57); `has_table` can fire on cover pages with addresses.
