# FinSight RAG — Interview Guide

> **Who this is for:** you, preparing to present this project and answer RAG / LLM-engineering questions in interviews
> (Python developer, 7 years of experience, targeting AI/ML engineer and GenAI engineer roles).
>
> **How to use it:** after finishing each phase in [`DEVELOPMENT_GUIDE.md`](DEVELOPMENT_GUIDE.md), read the matching section here.
> Answer each question **out loud**, in your own words, before reading the model answer.
> Replace every **📌** placeholder with *your* measured numbers from `results/`. Real numbers from your own runs
> are what separate you from candidates who only watched tutorials.

---

## Contents

1. [Your pitch (30 s, 2 min, 5 min)](#1-your-pitch)
2. [Formula cheat sheet](#2-formula-cheat-sheet)
3. [Phase by phase: concepts, most-asked questions, what to study](#3-phase-by-phase)
4. [Rapid-fire RAG questions](#4-rapid-fire-rag-questions)
5. [System design: "Design a RAG system for …"](#5-system-design-template)
6. [Behavioral questions using this project](#6-behavioral-questions)
7. [Gaps to study beyond this project (prioritized)](#7-gaps-to-study-beyond-this-project)
8. [Red flags to avoid](#8-red-flags-to-avoid)
9. [Final-week checklist](#9-final-week-checklist)

---

## 1. Your pitch

**30 seconds (the elevator version)**
> "I built FinSight RAG, a question-answering system over real SEC filings that gives cited, page-level answers, and says 'not found'
> instead of guessing. It runs fully locally at $0 (bge embeddings, a local LLM; Ollama, free-tier APIs or Claude by config).
> I scored every design choice on FinanceBench against page-level ground truth. For example, metadata filters plus contextual chunk
> headers raised hit@10 from 0.46 to 0.67 over plain dense search, while the textbook "hybrid + reranker" measurably did *not* help,
> so I didn't ship it. CI blocks retrieval regressions."

**2 minutes (structure: problem → approach → evidence → trade-offs)**
1. *Problem:* analysts search 100–300-page filings; plain LLMs make up numbers; wrong financial answers are expensive.
2. *Learning first:* I rebuilt BM25, dense retrieval, RRF and reranking on FiQA to understand each one on its own.
3. *Product:* PDF parsing with table handling → 4 chunking strategies → hybrid retrieval with metadata filters → cross-encoder rerank → a grounded prompt with citation checks → abstention.
4. *Evidence:* a 150-question ablation (best: recursive_384 + dense + company/year filter, hit@10 0.613; 0.673 with contextual headers), end-to-end numeric match 0.31 with a 1.5B local model (30 Q), retrieval p50 ~110 ms, answer p50 2.5 s. The local LLM judge proved unreliable (90/90 "correct"), so I report exact numeric match instead.
5. *Trade-offs:* the local reranker adds ~2.6 s per question for −3 to +7 pts hit@10, so it's off by default. Corrective RAG cost 2.6× the latency for no gain with a small model. Query transforms cost 1–4 extra LLM calls per question.
6. *Production:* typed config with a fingerprint, FastAPI with SSE streaming, a Streamlit UI, caches (query vectors, rerank scores, answers), fallbacks with `degraded` flags, a CI regression gate, a runbook and ADRs. It runs as three independent processes (LLM, API, UI).

**5 minutes:** the 2-minute version plus a live demo of the Streamlit "under the hood" panel. Show one question where BM25 wins (an exact line-item name), one where dense wins (a paraphrase), and one correct abstention.

---

## 2. Formula cheat sheet

| Concept | Formula / rule | Say this in plain words |
|---|---|---|
| **BM25** | `score(q,d) = Σ_t IDF(t) · f(t,d)(k1+1) / (f(t,d) + k1(1 − b + b·|d|/avgdl))` | Term frequency with saturation (k1≈1.2–2.0), normalized for document length (b≈0.75), weighted by how rare the term is |
| **IDF** | `log((N − n_t + 0.5)/(n_t + 0.5) + 1)` | Rare terms matter more |
| **Cosine similarity** | `a·b / (‖a‖‖b‖)`, which is just `a·b` when the vectors are normalized | The angle between meanings |
| **RRF** | `score(d) = Σ_r 1/(k + rank_r(d))`, k=60 | Fuse *ranks*, not scores: no need to calibrate BM25 scores against cosine scores |
| **Recall@k** | relevant items found in the top-k / total relevant | Did we find it at all? |
| **Precision@k** | relevant items in the top-k / k | How much noise was in the context? |
| **MRR** | mean over queries of 1/rank of the first relevant item | How quickly the first good hit appears |
| **NDCG@k** | `DCG/IDCG`, with `DCG = Σ rel_i / log2(i+1)` | Rewards relevant items ranked higher; 1.0 is a perfect ranking |
| **Faithfulness** | supported claims / total claims | Did the answer stay inside the evidence? |
| **Token cost** | `in_tok × price_in + out_tok × price_out` | Know your cost per question |

---

## 3. Phase by phase

Each phase has the same parts: **Concepts** (plain words) · **Most-asked questions** with strong answers · **📌 Your numbers** · **Go deeper** (what to study, with links).

---

### P0 — Engineering foundation

**Concepts**
- **src layout:** the package lives in `src/` so tests import the *installed* package, not whatever happens to be in the current folder. This catches packaging bugs early.
- **Config precedence:** defaults → YAML → environment → CLI. One typed `Settings` object; secrets only come from the environment (the 12-factor rule).
- **Dependency injection with a container:** the API, CLI, scripts and eval all build the same objects from config, so the score you report is the system you actually serve.

**Most-asked questions**
1. *"How do you structure a Python ML project for production?"*
   → An installable package (`src/`), interfaces plus registries for swappable parts, typed config, no side effects at import, tests split into unit tests (pure logic) and integration tests (with fake models), pinned dependencies (`uv.lock`), and CI running lint, type checks and tests.
2. *"How do you manage secrets?"*
   → Environment variables or `.env` locally (gitignored), a secret manager in the cloud. Never in YAML or logs; keys are masked in diagnostics. The config fingerprint excludes secrets.
3. *"Why do tests need fake models?"*
   → They are deterministic, fast and free in CI. Real-model checks run as `slow` tests locally. You test *your* logic, not the model's weights.

**Go deeper:** [12-Factor App](https://12factor.net/) · [Python packaging: src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) · [Hypermodern Python (series)](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/)

---

### P1 — Data & benchmark

**Concepts**
- **A benchmark has three parts:** a corpus (documents), queries, and relevance labels (qrels or evidence). FinanceBench also gives **gold answers and the evidence page**, which lets you score retrieval *and* generation.
- **Why FinanceBench:** long, table-heavy, real documents; realistic analyst questions; small enough for a laptop (84 PDFs, ~165 MB, 150 questions).
- **Data card:** document where the data comes from, its license (CC-BY-NC), its size and its quirks (for example, the 0-based page numbers).

**Most-asked questions**
1. *"Why this dataset instead of making up questions?"*
   → Expert-written questions with evidence pages let me measure retrieval objectively. Synthetic questions tend to copy the source's wording, which inflates the scores, especially for BM25.
2. *"150 questions is small. Can you trust the results?"*
   → For large differences, yes; for small ones, no. I report n and the standard deviation per row, break the results down by question type, and treat differences of 1–2 points as noise. I added MultiHop-RAG (2,556 queries) to check that the findings generalize. A production system would grow the eval set from real user queries that fail.
3. *"How would you build an eval set for a new domain without labels?"*
   → Sample real user questions, have experts label the evidence, add LLM-generated questions *written from* chunks (with a filter), include adversarial and unanswerable questions, and grow it with every bug found in production.

**📌 Your numbers:** questions 150 · evidence items 189 · docs 84 (64 10-K, 8 10-Q, 6 8-K, 6 earnings) · **12,013 pages** · 166.5 MB · 32 companies · question types 50/50/50 · 35 questions need 2–3 evidence pages

**A story worth telling:** "Before writing any retrieval code I verified the page convention: the benchmark's evidence pages are 0-based and match PyMuPDF's index. Getting that wrong by one page would have silently made every retrieval metric wrong." Also mention the quirks in `docs/data_card.md` (tables one cell per line, missing spaces in some text): they show you looked at the data before modelling it.

**Go deeper:** [FinanceBench paper](https://arxiv.org/abs/2311.11944) (read §3 and the results comparing a shared vector store with per-document stores) · [BEIR](https://arxiv.org/abs/2104.08663) · [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)

---

### P2 — Ingestion (PDF parsing)

**Concepts**
- **Parsing quality caps retrieval quality.** If the number "1,577" is separated from "Purchases of property, plant and equipment", no retriever can recover it.
- **Tables:** keep each row together, rendered as `| label | 2018 | 2017 |` text, so the numbers stay next to their labels and column headers.
- **Boilerplate:** headers and footers repeated on every page add noise to BM25 and embeddings, so remove them by frequency.
- **Validation:** I checked automatically that the gold evidence text appears on the page with the gold page number: **189/189 (100%)**, mean overlap 0.998, no page offsets.

**Most-asked questions**
1. *"How do you handle tables in RAG?"*
   → Options: (a) table text linearized row by row, as I did; (b) table-to-Markdown or HTML chunks kept whole; (c) generate a summary of the table to embed, while giving the raw table to the LLM; (d) vision models or layout parsers (unstructured, Docling, LlamaParse) for complex layouts. Choose by measuring evidence coverage.
2. *"What about scanned PDFs?"*
   → Detect pages with no text layer, then run OCR (Tesseract or a cloud OCR service) or use a vision-language model. Track OCR confidence and flag pages with low confidence.
3. *"How did you verify that parsing works?"*
   → The evidence-page overlap test against the ground truth, plus a manual look at the worst 5 cases. That is a measurable check, not just eyeballing.

**📌 Your numbers (P2):** 12,013 pages parsed in ~70 s (17 s on 6 cores) · evidence on gold page 189/189 · label–number adjacency **0.28 → 0.97** vs plain extraction · `find_tables()` rejected at 1.5 s/page.

**A story worth telling (decision backed by measurement):** "The obvious tool was PyMuPDF's `find_tables()`. I measured it: 1.5 seconds per page, about 5 hours for the corpus, and its cells lost their spaces. I rebuilt table rows from word coordinates instead, ~3 ms per page, and defined a metric for what mattered: the share of numbers that stay on the same line as their label. It went from 0.28 to 0.97. I also found that the benchmark's own evidence text had lost spaces ('Netsales'), so I made the evidence check space-insensitive rather than penalise correct parsing."

**Go deeper:** [PyMuPDF recipes](https://pymupdf.readthedocs.io/en/latest/recipes-text.html) · [Docling (IBM layout-aware parsing)](https://github.com/docling-project/docling) · [Unstructured](https://docs.unstructured.io/) · learn about **layout detection** and **multimodal RAG** (§7)

---

### P3 — Chunking ⭐ (very frequently asked)

**Concepts**
- **Why chunk at all:** embedding models have a token limit (512 for bge); smaller units retrieve more precisely; the LLM context is limited and costs money.
- **The trade-off:** small chunks are precise but lose context; large chunks keep context but dilute the embedding (one vector for many topics) and send more noise to the LLM.
- **Overlap:** prevents a fact from being cut in half at a boundary. The cost is more chunks and some duplication.
- **Strategies:** fixed tokens · recursive (by structure: paragraph → sentence) · **page-aware** (exact citations) · **parent-child / small-to-big** (search small chunks, give the LLM the big parent) · semantic chunking (split where the embedding similarity drops) · contextual chunking (prepend document context).

**What this project uses, and why** (say this in one breath)

> "**Page-aware recursive chunking, 384 tokens, 64-token overlap, with the table column header carried into every table chunk.** Parent-child is built and switchable from config, and the final choice is made on retrieval metrics in P5."

| Choice | What it means | Why (the reason you give) |
|---|---|---|
| **Page-aware** | Recursive splitting *inside* one page; a chunk never crosses a page boundary | Every answer must cite a page ("3M 10-K FY2018, p. 60"). With fixed/recursive, **30–60% of chunks span two pages**, so the citation is ambiguous. Page-aware: **0%**. Filing pages are also natural units (one statement, one note). |
| **Recursive inside the page** | Cut at `\n\n` → `\n` → `. ` → space, then pack greedily | Keeps paragraphs and table rows whole. A table row (`label \| 2018 \| 2017`) is one line, so it is never split mid-row. |
| **384 tokens** | Middle of the sweep (256 / 384 / 480) | 72% of evidence fits in one chunk (vs 48% at 256), while the vector stays focused. 480 gives 82.5% but blurrier embeddings and more LLM noise; that trade-off is decided with Recall@k in P5. |
| **64-token overlap** | The tail of one chunk repeats at the start of the next | A sentence on the boundary is whole in at least one chunk; ~17% duplication is cheap. |
| **Table header carry** | If a chunk holds table rows but not their header row, prepend `(Millions) \| 2018 \| 2017 \| 2016` | Without it, `Inventories \| (509) \| (387)` doesn't say which year. Table chunks with their header: **64% → 94%**. The header's tokens are **reserved** in the budget, so the 510 limit still holds. |
| **Model's own tokenizer** | Count with bge's WordPiece, cap at 510 | Over-long chunks are silently truncated by the embedder. 0 chunks over the limit, recounted. |

**Benefits of this choice (vs the alternatives)**
- **Exact, verifiable citations**: page_start == page_end on every chunk, so the UI and the faithfulness check (P8) can point at one page.
- **Numbers keep their meaning**: header carry plus whole-row splitting means a retrieved number always comes with its label and its year. This is the #1 failure mode in financial RAG.
- **Better containment than the naive baselines at the same size**: at 384 tokens, page-aware 72.0% vs fixed 68.3% vs recursive 68.8%, *despite* never crossing a page.
- **No text lost**: union overlap 0.998 = the parsing ceiling.
- **Swappable, measurable**: all four strategies sit behind a registry and one config key, so P5 can compare them on Recall@10/MRR without code changes.
- **Cost of the choice (say it before they do):** a sentence continuing onto the next page is split in two, and at 480 tokens page-aware contains 4 points less evidence than fixed (82.5% vs 86.8%). Parent-child fixes the context side if P5 shows it matters.

**Most-asked questions**
1. *"Which chunking strategy did you use and why?"*
   → The quote above. Then the three reasons: exact page citations (0% cross-page), tables stay readable (header carry 64% → 94%), and it beat fixed/recursive on containment at the same size. Finish with: "I keep the others behind a registry and choose on retrieval metrics."
2. *"How do you choose the chunk size?"*
   → Not by rule of thumb. I swept 4 strategies × 3 sizes and measured page-crossing, evidence containment and token limits; the final pick uses Recall@10/MRR on the gold pages (P5). It also depends on the embedding model's limit and how dense the documents are: tables want larger units, prose can be smaller.
3. *"Why not just use fixed-size chunks? They scored highest at 480."*
   → Containment favours big chunks by construction, so it can't pick the winner alone. Fixed chunks cross pages 60% of the time at 480, which breaks citations, and they cut tables mid-row. A 4-point containment gain doesn't pay for wrong citations.
4. *"What is parent-child retrieval, and why does it help?"*
   → Embed small child chunks (~200 tokens) for precise matching, then send the LLM the parent (here, the whole page, capped at 1,500 tokens) for context. It separates *what matches well* from *what reads well*. Deduplicate parents when several children hit the same one. Cost: more vectors (95k children at 128 tokens) and a bigger LLM prompt.
5. *"What happens if a chunk is longer than the embedding model's limit?"*
   → It is **silently truncated**: the tail is never embedded. bge's 512 includes `[CLS]`/`[SEP]`, so 510 text tokens. I count with the model's own tokenizer and recount every chunk in a check that fails the run.
6. *"How do you handle tables when chunking?"*
   → Rows are one line each (from P2), the splitter never cuts inside a line, and the column-header row is carried into every chunk of that table. Alternatives: keep each table as one chunk, or embed an LLM-written table summary and give the raw table to the LLM.
7. *"How much overlap, and why?"*
   → 64 tokens (~17% of 384). Enough to keep a boundary sentence whole; more just duplicates text and inflates the index. Overlap matters less for page-aware chunks because pages already break at natural boundaries.
8. *"How do you evaluate chunking without running the whole pipeline?"*
   → Intrinsic checks: share of gold evidence contained in one chunk, union coverage (is text lost?), share of chunks crossing pages, token distribution. Then the real test: retrieval Recall@k/MRR with the chunker as the only variable.
9. *"Semantic chunking: worth it?"*
   → Sometimes. It costs an embedding pass at ingest time and research results are mixed; structure-aware chunking often matches it on well-structured documents like filings. I'd test it on the same eval before adopting it.
10. *"What is contextual chunking?"*
    → Prepend document context (company, filing, year, page) to the text that gets *embedded*, so a chunk saying "revenue grew 5%" is findable for "3M 2018 revenue". Implemented here as an option: it writes only `embed_text` (the LLM still sees clean text) and reserves 24 tokens for the header. Anthropic's Contextual Retrieval goes further with an LLM-written sentence per chunk.
11. *"How would this change for millions of documents or frequent updates?"*
    → Chunk IDs are deterministic (`doc::p{page}::c{n}`), so re-ingesting a changed document replaces only its chunks. A manifest with a config fingerprint skips unchanged work. Chunking is per-document, so it parallelizes trivially across workers.
12. *"Chunks lose context: the word 'it' or 'the Company' with no name. How do you fix that?"*
    → Contextual headers (metadata in the embedded text), metadata filters (company/year at query time), and parent-child to give the LLM the surrounding page.

**📌 Your numbers (P3, `docs/evaluation.md`):** page_aware_384 = 31,114 chunks, 0% span two pages vs 30–60% for fixed/recursive · evidence in one chunk ~45% @256 → ~70% @384 → ~83% @480 · @384: page-aware 72.0% vs fixed 68.3% vs recursive 68.8% · table chunks containing their column header **64% → 94%** with header carry · 0 chunks over the 510-token limit · Recall@10 difference between chunkers: 📌 (P5)

**Stories worth telling:**
- *"512 is not 512."* bge's limit includes `[CLS]` and `[SEP]`, so 510 text tokens; and numbers are expensive: `(1,577)` is 6 WordPiece tokens. I count with the model's own tokenizer and recount every chunk to prove none is truncated.
- *"Which year is this number?"* Splitting a balance sheet loses the `2018 | 2017 | 2016` header row in later chunks. I carry the column header into every table chunk on the page (64% → 94%), reserving room for it so the size limit still holds.
- *"I didn't pick the winner on a proxy."* My containment metric favours big chunks by construction, so I deferred the choice to retrieval metrics (P5).

**Go deeper:** [Chroma: evaluating chunking](https://research.trychroma.com/evaluating-chunking) · [Pinecone: chunking strategies](https://www.pinecone.io/learn/chunking-strategies/) · [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) · [Lost in the Middle](https://arxiv.org/abs/2307.03172)

---

### P4 — Embeddings & vector database ⭐

**Concepts**
- **Bi-encoder:** the query and the document are embedded *separately*, so document vectors can be computed once, ahead of time. Search is then a fast nearest-neighbour lookup.
- **Normalize the vectors**, and cosine similarity becomes a dot product.
- **Query instruction:** bge v1.5 expects queries prefixed with "Represent this sentence for searching relevant passages: ". Asymmetric retrieval treats questions and passages differently.
- **ANN / HNSW:** a graph index with layers. Search walks it greedily, giving roughly logarithmic search time, with a recall/speed trade-off controlled by `ef` and `M`. Exact search is fine at small scale; ANN matters at millions of vectors.
- **Payload filtering:** Qdrant applies metadata filters *during* the vector search, rather than filtering afterwards and ending up with fewer than k results.
- **Sparse vs dense:** BM25 scores exact word overlap (weighted by rarity, saturated by term frequency, normalized by length); dense vectors score meaning. They fail on different questions, which is why P5 fuses them.
- **Idempotent indexing:** a manifest of hashes means re-runs skip unchanged work, and an embedding cache means re-chunking only embeds new text.

**What this project uses, and why** (say this in one breath)

> "**bge-small-en-v1.5 on the Apple GPU, normalized vectors in Qdrant with the full chunk as payload, plus a BM25 index over the same text. Both are idempotent: a manifest skips unchanged work and a SQLite cache means no text is embedded twice.**"

| Choice | Why (the reason you give) |
|---|---|
| **bge-small-en-v1.5** (384 dims, 33M params) | Strong on MTEB retrieval for its size, runs at **106 chunks/s** on a laptop GPU, 512-token limit fits the chunks. bge-base / larger models are a measured upgrade in P5, not a guess. |
| **L2-normalized vectors, COSINE** | Cosine = dot product, so scores are comparable across queries, and any distance setting ranks the same. |
| **Query instruction on queries only** | It's how bge v1.5 was trained for short queries. Configurable, because on our first test question it *hurt* (rank 36 → 47); P5 measures it on all 150. |
| **Qdrant** | Filters applied during search, the same API embedded (no Docker for dev) and as a server (production), sparse-vector support if needed later. |
| **Full chunk as payload** | A search hit is a complete, citable chunk (text, doc, page, company) with no second lookup. Costs disk (244 MB for 31k chunks). |
| **uuid5(chunk_id) as point id** | Qdrant needs int/UUID ids; uuid5 is deterministic, so re-indexing a chunk overwrites it instead of duplicating it. |
| **BM25 (bm25s) beside the vectors** | Exact words (tickers, line items, years) that embeddings blur. 1.9 s to build, 25 MB. |
| **Manifest + embedding cache** | Rebuild only when chunk content, model or index code changed. Re-run: 0.07 s. A forced rebuild takes the vectors from the cache. |
| **Pluggable embedders (local + online), chosen by a preset** | `EMBEDDER=gemini-001` swaps in Google's model; Mistral, Cohere, Jina, Voyage, Ollama and OpenAI are one line away. Same chunks, separate collections, so any two can be compared fairly. |
| **Client-side rate limits + a paid guard** | Free tiers only: requests are paced to RPM/TPM, daily/monthly caps are tracked on disk, quota 429s stop cleanly, and paid-only providers are refused unless explicitly allowed. |

**Benefits**
- **Laptop-scale, production-shaped:** the same code talks to embedded Qdrant (dev) and a Qdrant server (prod) by changing one URL.
- **Cheap experiments:** every chunking config gets its own collection and BM25 folder, so P5 compares 4 chunkers without rebuilding anything.
- **Reproducible:** the manifest records chunk hash, model, dimensions and config fingerprint for every index.
- **Testable without models:** a hash-based `FakeEmbedder` runs the whole index → search → filter path in CI in under a second.
- **Vendor-neutral and zero-spend:** comparing a hosted model is a config change, costs nothing on free tiers, and survives rate limits and daily caps (the cache resumes the run).

**Most-asked questions**
1. *"Which embedding model did you use and why?"*
   → bge-small-en-v1.5: good retrieval quality for 33M parameters, 384 dims, 512-token limit, MIT license, fast on Apple's GPU (106 chunks/s, 31k chunks in under 5 minutes). I start small, measure Recall@k, and only upgrade if the gain is worth the cost.
2. *"How do you choose an embedding model?"*
   → Start from the MTEB retrieval leaderboard, filtered by size, latency and license. Then **evaluate on my own data**, because leaderboard rank doesn't transfer reliably between domains. Also consider the max sequence length, multilingual needs and domain fit.
3. *"Which vector DB, and why?"*
   → Qdrant: payload filtering during search, local and server modes with the same API, it runs in Docker, and it supports sparse vectors if needed. Alternatives: pgvector (when you already run Postgres and the scale is moderate), Elasticsearch or OpenSearch (when you want BM25 and vectors in one system), Pinecone (managed), FAISS (a library, not a database).
4. *"Explain HNSW."*
   → A multi-layer proximity graph: the top layers are sparse long-range links, the bottom layer is dense. Search starts at the top and greedily descends toward the query. `M` sets the links per node (memory vs. recall); `ef_search` sets the size of the candidate list (latency vs. recall). (Embedded Qdrant searches by brute force, which is exact and fast enough at 31k vectors.)
5. *"Why normalize embeddings?"*
   → With unit vectors, cosine similarity equals the dot product, so scores are comparable and any distance metric gives the same ranking. Mixing normalized and unnormalized vectors silently breaks ranking.
6. *"Why keep BM25 if you have embeddings?"*
   → They fail differently. Embeddings blur exact tokens (tickers, years, line-item names); BM25 can't handle synonyms. In my data, BM25 ranks the 3M capex page **2nd** with the filing's wording but misses it with the question's wording; dense is the reverse. Fusion (P5) gets the best of both.
7. *"How do you update the index when documents change?"*
   → Use deterministic chunk IDs (`doc::page::seq`) and UUIDs derived from them. Upsert the changed docs, and delete chunks whose doc version changed. The manifest tracks the hashes. Re-embedding the whole corpus is only needed when the embedding model changes (a new collection, a blue/green switch).
8. *"How do you avoid paying to re-embed?"*
   → A content-addressed cache: key = sha256(model + text). Re-chunking or a forced rebuild only embeds texts never seen before. It also de-duplicates (239 of my chunks repeat another chunk's text).
9. *"How do metadata filters work with vector search?"*
   → Pre-filtering during the search (Qdrant): every result matches and you still get k. Post-filtering (search, then drop) can return fewer than k, or zero for a small company. For BM25 I pass a document mask before ranking for the same reason. In my test the company filter moved the gold page from rank 36 to 14.
10. *"What about embedding dimension and cost?"*
    → Memory is roughly n × dim × 4 bytes for float32, before the index overhead: 31k × 384 × 4 ≈ 48 MB here; the rest of the 244 MB is payload. Options to shrink it: smaller dimensions, Matryoshka embeddings, scalar or product quantization.
11. *"How would this scale to 10 million chunks?"*
    → A Qdrant cluster with HNSW and payload indexes, scalar quantization (4× smaller), vectors on disk with the HNSW graph in RAM, sharding by tenant/company, and embedding as a batch job on GPUs with the same cache. BM25 moves to OpenSearch or Qdrant sparse vectors.
12. *"Local or hosted embedding model?"*
    → Measure both on your own data, same chunks. Local: free, private, no rate limits, predictable latency, but you host it. Hosted: often stronger and bigger context, but per-token cost, rate limits, data leaves your network, and the vendor can deprecate the model (re-embed everything). My code makes it a config switch (`EMBEDDER=gemini-001`) and a one-command comparison, so the decision is a table, not an opinion.
13. *"How do you deal with rate limits and quotas?"*
    → On the client: pace requests (60/RPM apart), a sliding token window for TPM, daily/monthly counters persisted to disk and checked *before* sending. On the server's 429: honour `Retry-After`, else exponential backoff; but a *daily-quota* 429 stops the run instead of retrying for hours. Everything is cached per batch, so a stopped run resumes. Subtle bug avoided: Google's per-minute 429 also says "exceeded your current quota … billing", so only explicit per-day/per-month wording means stop.
14. *"How do you make sure an experiment never costs money?"*
    → Paid-only providers are refused unless `allow_paid` is set, a `max_tokens_per_run` budget is checked before every request, the catalog estimates requests/time/cost before a run, and caching means nothing is ever sent twice (query vectors included).
15. *"How do you compare embedders fairly?"*
    → One variable at a time: identical chunks (sized with one tokenizer regardless of the model), same questions, dense-only search with no filter or reranker, same metrics (hit@k, recall@k, MRR, nDCG, doc-hit). Report n and treat small differences on 30 questions as suggestive. Use the model's documented query/document formats (bge instruction, Gemini 2 task prefixes, Cohere input types), or you're testing the wrong thing.
16. *"Your retrieval missed the gold page. Why?"*
    → Vocabulary mismatch: the question says "capital expenditure", the statement says "Purchases of property, plant and equipment". Dense ranks it 36th, BM25 121st. Fixes, in order: hybrid fusion, company filter, cross-encoder reranking (P5), then query rewriting or HyDE (P7), measured on all 150 questions.

**📌 Your numbers (P4b, embedder comparison):** dense-only, 150 questions: hit@10 **0.46**, MRR 0.23, doc_hit@10 **0.89** (right filing, wrong page is the main failure) · query instruction: +0.7 pt (150 Q), +6.7 pt (30 Q) · bge-base vs bge-small on 30 questions: hit@10 0.50 vs 0.43 at 3.6× embedding time · online presets: Gemini, Mistral, Cohere, Jina, Voyage (free tiers), OpenAI (paid, blocked) · online vs local result 📌 (add a free key)

**📌 Your numbers (P4, `docs/evaluation.md`):** 31,114 chunks indexed in **5 min 6 s** on M5 · **106 chunks/s** (bge-small, MPS, batch 64) · Qdrant 244 MB (vectors ~48 MB + payload), BM25 25 MB (27,097 words), cache 66 MB · re-run 0.07 s · gold page for the 3M capex question: dense 36 (14 with company filter), BM25 121 · query instruction on that question: 36 → 47 (worse; all-150 effect 📌 P5)

**Stories worth telling:**
- *"The cache that never cached."* My first cache check was `if cache:`. The cache class defines `__len__`, so an **empty** cache is falsy, and it silently never stored anything. A unit test that counts model calls caught it (`is not None` fixed it). Lesson: test the *behaviour* (calls avoided), not just the output.
- *"Measure before you assert."* I wanted the slow test to assert the gold page is in the dense top 20. It isn't (rank 36): the question and the filing use different words. Instead of weakening the test silently, I recorded the ranks per index and wording as the baseline P5 must beat.
- *"Free tiers, real constraints."* "I wanted to compare hosted embedding models without spending anything. Free tiers allow ~30k tokens/minute and ~1,000 requests/day, and the corpus is 11M tokens: 7 hours for one model. So I built client-side rate limiting with persisted daily counters, a cache that resumes after a quota stop, a 30-question subset (17% of the corpus) for quick comparisons, and a guard that refuses paid-only providers. The comparison is now one command per model."
- *"Know your tool's limits."* Embedded Qdrant ignores payload indexes, searches by brute force and warns above 20k points. Fine for dev at 31k; I create payload indexes only on a server and documented the production path.

**Go deeper:** [Sentence-BERT](https://arxiv.org/abs/1908.10084) · [BGE / C-Pack](https://arxiv.org/abs/2309.07597) · [HNSW paper](https://arxiv.org/abs/1603.09320) · [Qdrant filtering](https://qdrant.tech/documentation/concepts/filtering/) · [Qdrant quantization](https://qdrant.tech/documentation/guides/quantization/) · [MTEB](https://huggingface.co/spaces/mteb/leaderboard) · [bm25s](https://github.com/xhluca/bm25s) · study **product quantization and IVF** and **Matryoshka embeddings** (§7)

---

### P5 — Hybrid retrieval & reranking ⭐⭐ (the most asked)

**Concepts**
- **BM25 wins** on exact tokens: tickers, line-item names ("purchases of property, plant and equipment"), years, rare words.
- **Dense search wins** on paraphrase and meaning ("capex" ↔ "purchases of PP&E").
- **Hybrid with RRF** takes the best of both. Ranks are fused, so you don't have to put BM25 scores (unbounded) and cosine scores (bounded) on one scale.
- **Cross-encoder reranking:** the query and the document go into *one* model together, so attention works across both and gives a sharper relevance score. It is too slow for the whole corpus, which is why it runs on the top 50: a **retrieve-then-rerank** cascade.
- **Metadata filters:** pull the company from the question and restrict the search to it. This removes cross-company confusion, which is a big deal when 84 filings have similar wording.

**Most-asked questions**
1. *"Why hybrid search? Isn't dense retrieval enough?"*
   → My honest answer is a table (page-aware 384, 150 questions, hit@10): dense **0.46**, BM25 **0.15**, hybrid **0.32**, hybrid + company filter 0.45, + year filter **0.52**, + reranker 0.53. Hybrid alone was *worse* than dense: FinanceBench questions are long and instruction-heavy ("Assume you are a public equities analyst…"), so BM25 matches boilerplate words, and RRF gives that weak list equal weight. Hybrid pays off once the filter narrows the corpus to one company, where exact line-item words matter again. The lesson: don't assume hybrid > dense, measure it on your data.
2. *"Why RRF rather than a weighted score sum?"*
   → The scores live on different scales and distributions, and those change from query to query. RRF needs no calibration, is robust, and k=60 is the standard value. A weighted sum needs normalization (min-max or z-score) and tuning, and can win if it is tuned on held-out data.
3. *"Bi-encoder vs. cross-encoder?"*
   → A bi-encoder embeds separately: fast and indexable, but coarse. A cross-encoder reads the pair jointly: accurate but O(n) model calls per query. So: bi-encoder for recall, cross-encoder for precision on the top candidates.
4. *"How many candidates should you rerank?"*
   → It's a trade-off: more candidates give more recall, but latency grows linearly. I rerank 50 with bge-reranker-base on the Mac GPU: **~2.5 s per question** (p50) versus ~120 ms without it, for +1–12 points hit@10 depending on the chunker. Measure 20/50/100 and pick the knee of the curve; in production, a smaller reranker or an API reranker.
5. *"Your retrieval is bad. How do you debug it?"*
   → (1) Is the answer in the corpus, and was it parsed? (the evidence check) (2) Is it inside one chunk? (chunk coverage) (3) Does it appear in the top-50 of either retriever? (4) Is it lost at fusion or at reranking? (5) Did a filter remove it? Each stage is logged, so I can tell which one failed.
6. *"What's the difference between Recall@k and NDCG?"*
   → Recall asks whether it was found; NDCG asks how high it was ranked. For RAG, recall at the context size matters most (the LLM sees all k chunks), but rank still matters because of "lost in the middle".
7. *"Filtering: before or after the vector search?"*
   → Filtering during the search (Qdrant's filterable HNSW) avoids getting fewer than k results. Filtering afterwards needs over-fetching. For BM25 I pass a document mask before ranking (bm25s `weight_mask`), so it's exact too.
8. *"Which retrieval setup did you choose and why?"*
   → Recursive 384-token chunks, **dense** search, company **and** year filters extracted from the question, **no** BM25 and **no** reranker by default: hit@10 **0.613** at ~110 ms. I expected hybrid + rerank to win; the ablation said otherwise: hybrid + both filters + rerank scored 0.580 at ~2.7 s. Inside one company BM25 still adds noise, and bge-reranker-base (512-token window, web-trained) doesn't order financial tables better than the embedding. Both stay one config switch away, and online rerankers (Cohere/Jina/Voyage) are wired in for the next comparison. That's the senior answer: choose by measurement, keep the alternatives cheap to re-test.
9. *"How do you extract metadata filters, and what if it's wrong?"*
   → Rules first: company aliases from the document metadata plus a ticker/nickname map (98% of questions get a company), years by regex. If a filter returns nothing, retry without it. A hard year filter is risky (a FY2017 figure often lives in the FY2018 10-K), which is why it's an ablation, not an assumption; here it helped by +4–14 pts. An LLM "self-query" extractor is the upgrade path for messy questions.
10. *"Your filter improved doc_hit to 99% but hit@10 is only ~55%. What's left?"*
    → The right filing, wrong page: inside one 300-page 10-K many pages look alike to a small embedder, and the question's wording differs from the filing's ("capex" vs "purchases of PP&E"). Next levers: query rewriting/HyDE (P7), contextual chunk headers, a stronger embedder (bge-base was +7 pts on a subset).

**📌 Your numbers (P5, `results/2026-11-24_retrieval_ablation/`):** hit@10 on 150 questions, page-aware 384: dense 0.46 · BM25 0.15 · hybrid 0.32 · +company filter 0.45 (doc_hit 0.83 → **0.99**) · +year filter 0.52 · +rerank 0.53 · hybrid + company/year filter 0.553 (recursive) · rerank cost ~2.5 s/question (p50) vs ~120 ms · company extracted from 98% of questions · dense + company/year filter **0.613** (recursive) vs hybrid + filters + rerank 0.580 at ~25× the latency

**Go deeper:** [RRF paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) · [BEIR](https://arxiv.org/abs/2104.08663) · [ColBERT, late interaction](https://arxiv.org/abs/2004.12832) · [SPLADE, learned sparse retrieval](https://arxiv.org/abs/2107.05720) · [SBERT cross-encoders](https://sbert.net/examples/cross_encoder/applications/README.html)

---

### P6 — Grounded generation ⭐

**Concepts**
- **Grounding prompt:** answer only from the numbered context, cite as [n], show calculations, use one exact abstain phrase, and treat the context as data (this is the prompt-injection defense).
- **Context assembly:** rerank order, a token budget, parent expansion, deduplication. Put the most relevant material first and keep it short ("lost in the middle").
- **Citation checks:** parse the [n] markers, drop the ones that don't exist, and flag uncited answers. This is a cheap, deterministic guardrail.
- **Abstention:** in finance, a wrong answer is worse than "not found". It is measured as the false-abstain rate (on answerable questions) and the correct-abstain rate (on unanswerable ones).
- **Provider abstraction:** one `LLM` interface; Ollama or a small in-process HF model locally, free-tier APIs (Gemini, Mistral, Groq), Claude in the cloud; one preset switch (`LLM=claude-opus`) or `auto` (first that can run right now).

**What this project uses, and why**

| Choice | Why |
|---|---|
| Numbered context blocks `[n] company · doc FY · page` in rerank order, ≤ 6k tokens | The model can cite precisely; the header tells it which year/filing each number comes from |
| One exact abstain phrase | Abstention becomes measurable (string match), not a vibe |
| Deterministic citation check + `auto_cite` repair | Small models answer right but ignore the format; code catches invented `[7]`s and repairs missing citations only when the figure is verbatim in a block (flagged) |
| `auto` LLM selection | Works on any machine: Ollama, a free key, Claude, or fully offline with a 1.5B model |
| Claude via the SDK with effort + refusal fallbacks, no sampling params | Current API: effort is the depth control; sampling params are rejected; fallbacks keep a declined request alive |

**Most-asked questions**
1. *"How do you reduce hallucinations in RAG?"*
   → Better retrieval first (most hallucinations come from missing context), a strict grounding prompt, required citations with automatic checks, an abstain option, CRAG-style evidence grading, a faithfulness eval, and a lower-temperature or stronger model. I measure it with citation checks on every answer (hallucinated-citation and uncited-answer warnings) and numeric match against gold (0.31 with the local 1.5B). LLM-judged faithfulness is wired (`faithfulness: true`) but needs a judge from another model family, validated on human labels, before I'd quote it.
2. *"How do you handle the context window?"*
   → A token budget during assembly, reranking so the best chunks come first, deduplicating parents, and, for long-context models, testing whether *more* context actually helps (it often doesn't). Also watch for silent truncation, such as Ollama's default `num_ctx`.
3. *"Local LLM vs. API LLM: how do you decide?"*
   → Privacy and data residency, cost at your volume, latency, the quality bar (my local 1.5B scored 0.31 numeric match on 30 questions; the 7B/Claude comparison is one env var away, `LLM=claude-opus`, but I haven't run it: no paid usage), operations (GPU serving with vLLM vs. a managed API). Often the answer is a hybrid: local for cheap steps, the API for the hard ones.
4. *"How do you make the LLM do arithmetic correctly?"*
   → Ask it to show the formula with cited inputs. Better still, extract the numbers and compute them in Python (a tool call or calculator). Verify with the numeric-match metric.
5. *"Your small model wouldn't cite. What did you do?"*
   → Measured, then fixed it in code. Qwen2.5-1.5B gave the right number ($1,577M) without `[n]`; adding a one-line example to the prompt made it abstain instead. So citation repair is deterministic: each figure in the answer is matched verbatim against the blocks and that block is cited, flagged `auto_cited` so metrics can separate model-cited from repaired. Small-model prompts are brittle; guardrails belong in code.
6. *"How do you choose between local and API LLMs in one codebase?"*
   → One interface, presets per model, and an `auto` resolver with an explicit order and reasons ("no GEMINI_API_KEY", "Ollama not running"). Paid-only providers need an explicit flag. Every answer records provider and model, so eval numbers are always attributable.
7. *"What is prompt injection in RAG, and how do you defend against it?"*
   → Retrieved text can contain instructions ("ignore previous instructions"). Defenses: tell the model the context is data, use delimiters, give the model the least privilege (no dangerous tools), filter the output, and keep humans in the loop for actions. See the OWASP LLM Top 10.

**📌 Your numbers (P6):** 3M capex with the local 1.5B model: correct ($1,577M), ~3.9k prompt tokens, 117 output tokens, ~11 s on M5 · off-topic question abstains · Ollama/Claude correctness and hallucinated-citation / false-abstain rates: P8

**Go deeper:** [RAG paper](https://arxiv.org/abs/2005.11401) · [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) · [Claude docs: prompt engineering and citations](https://platform.claude.com/docs)

---

### P7 — Advanced RAG ⭐

**Concepts**
- **Multi-query:** several paraphrases, each retrieved, then fused. It improves recall for vague questions at the cost of N× retrieval and one extra LLM call.
- **HyDE:** embed a *hypothetical answer* instead of the question. Answers look more like documents than questions do. It is risky when the model invents details.
- **Decomposition:** split "compare FY2015 vs. FY2017 margins" into sub-questions, retrieve for each, and combine.
- **Contextual retrieval:** prepend document and section context to each chunk before embedding and BM25 indexing, so a chunk saying "revenue grew 5%" also says *whose* revenue and *which year*.
- **Corrective RAG (CRAG):** grade the evidence; if it is weak, rewrite the query and retry, or abstain.
- **Agentic RAG:** the LLM decides its own search, read and answer steps with tools. It is flexible for multi-hop questions, but costs more, is slower, and is less predictable, so cap the steps.

**Most-asked questions**
1. *"When would you use agentic RAG instead of a fixed pipeline?"*
   → For multi-hop or open-ended questions where you can't know the retrieval plan in advance. For simple lookups a fixed pipeline is cheaper, faster and easier to test. My MultiHop-RAG result: the loader and the agentic-vs-baseline harness are built (`make multihop`), but not run yet; with a 1.5B model the agent loop is slow and weak.
2. *"Does HyDE always help?"*
   → No. It helps when there is a big vocabulary gap between questions and documents, and hurts when the generated answer adds wrong specifics. My result: not measured yet. The harness is ready (`make eval-advanced`), but with a 1.5B helper model the transforms are slow. Contextual headers, a cheaper fix for the same vocabulary gap, gave +6 pts. Always A/B test it.
3. *"What is contextual retrieval?"*
   → A technique Anthropic published: before indexing, add a short, document-aware context to each chunk, for both the embeddings and BM25. My cheap version (a metadata header, embeddings only) gave hit@10 0.613 → 0.673 on 150 questions; the LLM-written version costs one LLM call per chunk at ingest time.
4. *"How do you keep advanced techniques from blowing up latency and cost?"*
   → Route by query type (a classifier or rules), run the extra steps only when the first pass is weak (the CRAG idea), cache LLM outputs, use small models for the helper steps, and stream the final answer.
5. *"Re-running your evaluation was slow and gave different numbers each time. What was wrong?"*
   → Two things, both now fixed. (1) Query embeddings and cross-encoder rerank scores were recomputed every run — ~19 s per question — and on Apple's GPU (MPS) the float results wobble slightly, so the ranking order changed between runs. I cache both on disk keyed by (model, text) and (model, query, chunk); a replay is ~0.1 s and byte-identical. (2) Document vectors were never the problem — the index manifest already skips re-embedding when the chunk content, model and index version are unchanged. Lesson: make retrieval deterministic by caching, and measure before blaming the expensive-looking step.

**📌 Your numbers (P7):** contextual chunk headers on 150 Q (dense + company/year filter): hit@10 **0.613 → 0.673**, recall@10 0.580 → 0.629, nDCG@10 0.385 → 0.400, MRR flat (0.353 → 0.350). Without a filter: 0.467 → 0.567. The header partly replaces the metadata filter. Query transforms (multi_query / HyDE / decompose): `make eval-advanced`, not yet measured (needs 1–4 local-LLM calls per question). MultiHop-RAG agentic vs baseline: `make multihop`.

**Go deeper:** [HyDE](https://arxiv.org/abs/2212.10496) · [RAG-Fusion](https://arxiv.org/abs/2402.03367) · [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) · [CRAG](https://arxiv.org/abs/2401.15884) · [Self-RAG](https://arxiv.org/abs/2310.11511) · [ReAct](https://arxiv.org/abs/2210.03629) · [GraphRAG](https://arxiv.org/abs/2404.16130)

---

### P8 — Evaluation ⭐⭐ (what separates senior candidates)

**Concepts**
- **Evaluate in layers:** retrieval metrics (free, deterministic) → generation metrics (correctness, faithfulness) → system metrics (latency, cost).
- **Reference-based** metrics compare against a gold answer (numeric match, judge correctness). **Reference-free** metrics need no gold answer (faithfulness to the context, answer relevance).
- **LLM-as-judge:** scalable, but biased (towards position, verbosity and its own model family). Validate it against human labels. My own lesson: a local 1.5B judge (the same model as the generator) rated 90/90 answers correct, including 63 whose number was wrong. So it was useless, and I caught that by cross-checking against an exact numeric match before trusting it. Human-label agreement (20 labels, script 10b) is still to be measured.
- **Noise:** with 150 questions, small differences aren't significant. Report n, the standard deviation, and the breakdown by question type.
- **Regression gate:** a small, fixed set in CI that blocks merges which break retrieval.

**Most-asked questions**
1. *"How do you evaluate a RAG system?"*
   → Use my three layers, with the same pipeline objects as production, a fixed eval set, a stored config fingerprint, a failure gallery, and a validated judge. Online, add user feedback, citation click-through and the abstain rate.
2. *"How do you know your LLM judge is reliable?"*
   → Hand-label a sample, measure agreement or Cohen's kappa, use a judge from a different model or provider than the generator, give it a clear rubric with JSON output, and vary the answer order to check for position bias.
3. *"What did your failure analysis show?"*
   → My top 3: (1) **right filing, wrong page**: doc_hit@10 0.99 but hit@10 0.61. Near-identical pages inside a 300-page 10-K. Contextual headers gave +6 pts. (2) **wrong line item**: the 1.5B model found the right page but computed from the wrong row (e.g. used Total Debt for net PP&E), so the model is now the bottleneck (gold page retrieved 0.47, numeric match 0.31). (3) **over-abstaining**: 27% abstentions on answerable questions. **Story:** the vocabulary gap ("capex" vs "purchases of property, plant and equipment": the gold page ranked 36th) led to filters, then contextual headers.
4. *"How do you evaluate in production without labels?"*
   → Faithfulness scored on a sample of traffic, user thumbs up or down, how often citations are clicked, retrieval-score drift, abstain-rate drift, and turning failures into new eval cases.

**📌 Your numbers (P8, 30 Q, local 1.5B, `results/2026-11-24_end_to_end/`):** numeric match naive 0.24 → **baseline 0.31** → corrective 0.28 · gold page retrieved 0.17 → **0.47** · abstained 0.37 / 0.27 / 0.33 · p50 3.8 / 4.8 / 12.8 s · $0. The judge was useless: 90/90 marked correct, with 63 disagreements with numeric match. Top failures: the right filing but a neighbouring page; the small model computing a ratio from the wrong line items; over-abstaining.

**Go deeper:** [LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) · [RAGAS](https://arxiv.org/abs/2309.15217) · [ARES](https://arxiv.org/abs/2311.09476) · [Anthropic: statistical approach to evals](https://www.anthropic.com/research/statistical-approach-to-model-evals) · learn **bootstrap confidence intervals** and **paired significance tests** (§7)

---

### P9 — Serving

**Concepts:** health vs. readiness checks; warm-up in the app lifespan (load models once); streaming over SSE for perceived latency; request IDs for tracing; input validation.

**Most-asked questions**
1. *"How do you reduce perceived latency?"* → Stream tokens; send the citations as soon as retrieval finishes; warm up the models; cache embeddings and frequent answers; run dense and BM25 in parallel.
2. *"Health vs. readiness?"* → Health means the process is alive (restart it if not). Ready means it can serve, with indexes loaded and dependencies reachable (keep it out of the load balancer until it is). Kubernetes uses both.
3. *"Sync or async FastAPI here?"* → Model inference is CPU/GPU-bound, so run it in a threadpool or separate worker processes. The external LLM calls are I/O-bound, so async helps there. Don't block the event loop.

**📌 Your numbers (P9, local Qwen2.5-1.5B on an M5):** 1 user p50 **4.3 s** / p95 4.9 s · 5 concurrent users p50 **17.3 s** / p95 20.4 s with the **same total throughput** (10 requests in 35 s either way), because one local model on one GPU serves one request at a time · time to first token **1.5 s** with SSE streaming vs ~1.8 s for the whole short answer. **Story:** the 5-user load test crashed the server. The SQLite caches I added in P7 were bound to the thread that opened them, and FastAPI runs sync endpoints in a thread pool. I fixed it with shared connections plus a lock, a process-wide lock for the GPU models, and a regression test. Unit tests with fakes didn't catch it; a load test did.

**Go deeper:** [FastAPI concurrency](https://fastapi.tiangolo.com/async/) · [SSE (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)

---

### P10 — Production hardening

**Concepts:** containers (multi-stage, non-root); graceful degradation (reranker → RRF order, Claude → Ollama); caching layers; observability (logs, metrics, traces); cost tracking; security.

**Most-asked questions**
1. *"What breaks in production that doesn't break in a notebook?"* → Timeouts and rate limits, cold starts, index and model version skew, concurrent writes, memory pressure, bad inputs, prompt injection, cost spikes, and silent quality drift.
2. *"How would you deploy this on AWS, GCP or Azure?"* → A container on ECS, Cloud Run or AKS; a managed Qdrant or Qdrant on Kubernetes with persistent volumes; models served with vLLM or TGI on GPU nodes, or an API LLM; secrets in a secret manager; OpenTelemetry traces; a separate batch ingestion job with a queue (SQS, Pub/Sub); blue/green index switching.
3. *"How do you control LLM cost?"* → Cache answers, route cheap models to the easy steps, enforce token budgets, use prompt caching for stable prefixes, use batch APIs for offline jobs, and track the cost of each request.

4. *"What happens when a dependency goes down?"* → Every stage degrades instead of failing, and says so. Vector store down → BM25 only (`dense_failed`). Reranker error → RRF order (`reranker_failed`). Filter matches nothing → retry unfiltered. LLM provider error or quota → the next preset in `llm.fallbacks` (`llm_fallback`). Only *runtime* errors trigger a fallback: a missing key is a config error and raises, because retrying can't fix it. Degraded answers are flagged in the response and **never cached**, so an outage doesn't stick.
5. *"How does your caching work, and how do you avoid stale answers?"* → Three layers: query vectors, rerank scores, and whole answers. The answer key includes the **config fingerprint** (a hash of chunking, embedding, retrieval, rerank, generation and LLM settings), so any config change misses automatically. There's also a TTL. A repeat question takes 0.01 s instead of 2.5–14 s.

**📌 Your numbers (P10, local 1.5B, `scripts/12_load_test.py`):** 1 user p50 2.5 s / p95 7.8 s · 5 users p50 14.4 s / p95 20.5 s · throughput ~17 req/min either way (one GPU model) · answer-cache hit p50 **0.01 s** · 0 errors in 30 requests. **Docker story:** the API container was OOM-killed on its first question. The 1.5B model in float32 is ~6 GB and Docker's VM had 8 GB, shared with everything else. bfloat16 fit, but took >15 min per answer on a CPU without bf16 kernels. Fixes: load the weights at startup so the health check catches it, keep the LLM outside the container (Ollama on the host GPU, or an API), and use a 0.5B CPU default. Lesson: size the model to the deployment target, and make failures appear at startup, not on the first user's request.

**Go deeper:** [OpenTelemetry](https://opentelemetry.io/docs/) · [vLLM](https://docs.vllm.ai/) · [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)

---

## 4. Rapid-fire RAG questions

Practice answering each in two sentences or fewer.

1. **RAG vs. fine-tuning?** RAG adds *knowledge* (fresh, citable, cheap to update); fine-tuning changes *behavior and format*. You can combine them.
2. **RAG vs. long-context models?** Long context costs more per query, suffers from "lost in the middle", and has no access control. RAG scales to any corpus size and gives citations. Hybrid: retrieve, then use a generous context.
3. **What's a reranker?** A cross-encoder that rescores the top candidates jointly with the query.
4. **Sparse vs. dense vectors?** Sparse = vocabulary dimensions with term weights (BM25, SPLADE); dense = learned semantic dimensions.
5. **What's k in RRF?** A smoothing constant (60) that limits how much the very top ranks dominate.
6. **Cosine vs. dot product vs. L2?** They are the same ranking on normalized vectors.
7. **What causes RAG hallucinations?** Missing or irrelevant context, conflicting chunks, a weak prompt, or a model that relies on its prior knowledge.
8. **How do you handle multi-tenant access control?** Store the tenant or ACL in the payload and filter at query time, *never* after generation.
9. **What's a semantic cache?** It returns a cached answer when a new query's embedding is close to a past one. Risk: a near-duplicate question with a different meaning (for example, a different year).
10. **What's HyDE?** Embedding a hypothetical answer instead of the query.
11. **What's GraphRAG?** Extract entities and relations into a graph, then retrieve along graph paths and community summaries. Good for "global" questions.
12. **How do you choose top-k for the context?** By recall at k against cost and noise, measured. Mine: 10 blocks. hit@5 0.53 → hit@10 0.61 on the best config, and 10 blocks fit in ~3.9k prompt tokens for a small model.
13. **What's query routing?** Classify the query, then pick the pipeline, index or model.
14. **How do you keep the index fresh?** Incremental upserts with deterministic IDs, delete old versions, and reindex in the background.
15. **What is embedding drift?** A new model's vectors are incompatible with the old ones, so re-embed into a new collection and switch atomically.
16. **What's Matryoshka representation learning?** Embeddings whose first-n dimensions still work on their own, so you can truncate them to save memory.
17. **Precision vs. recall for RAG context?** Recall first (you need the fact), then precision (less noise, lower cost).
18. **What's ColBERT?** Late interaction: per-token embeddings with MaxSim scoring. Accuracy close to a cross-encoder at near bi-encoder speed.
19. **What's a good chunk overlap?** Typically 10–20% of the chunk size. Measure it.
20. **How do you evaluate without gold answers?** Faithfulness and relevance judged by an LLM, plus user signals.

---

## 5. System design template

**Prompt:** *"Design a document Q&A system for a bank: 10M documents, 1,000 users, strict access control."*

Answer in this order and say each heading out loud:

1. **Clarify:** document types and their rate of change, query volume (QPS), latency SLO, accuracy and compliance needs, access-control model, languages, budget, on-premises vs. cloud.
2. **Ingestion (offline, async):** connectors → a queue → parse/OCR workers → chunk → embed (GPU batch) → upsert. Deterministic IDs, versioning, dead-letter queue, data lineage.
3. **Storage:** a vector DB (sharded and replicated, HNSW, with ACL metadata), a BM25 engine (OpenSearch), an object store for the originals, and a metadata DB.
4. **Query path:** auth → query understanding (filters, routing) → hybrid retrieval *with ACL filters* → rerank → a grounded LLM with citations → validation → streamed response.
5. **Scale math:** 10M docs × ~20 chunks = 200M vectors × 384 dims × 4 bytes ≈ 300 GB raw, before quantization (about 4× less with int8). It must be sharded.
6. **Quality:** an eval set, an offline gate, online feedback, monitoring of the abstain rate and faithfulness.
7. **Safety and compliance:** PII redaction, audit logs (who asked what and which docs they saw), prompt-injection defenses, data residency.
8. **Cost and latency:** caching, model routing, a GPU serving plan or API budgets, autoscaling.
9. **Trade-offs and next steps:** say explicitly what you'd do first versus later.

*Tie it back:* "In FinSight I built a small version of steps 2–6; here is how each one changes at bank scale…"

---

## 6. Behavioral questions

Use the **STAR** format (Situation, Task, Action, Result), with real numbers.

| Question | Story from this project |
|---|---|
| A technical decision you made with data | Choosing the chunker and filter policy from the 150-question ablation: the company filter took doc_hit from 0.83 to 0.99, and the year filter added +4–14 pts. I had expected to reject the year filter (10-Ks contain prior-year data), but the data said otherwise, so I kept it. I dropped hybrid + rerank (0.58 vs 0.61, 25× slower) |
| A bug you found that others would miss | The 5-user load test crashed the API: the SQLite caches were bound to one thread, and FastAPI uses a thread pool. Unit tests with fakes passed; only a load test found it. Also: re-runs gave different rankings because Apple MPS floats wobble, which I fixed by caching query vectors and rerank scores (19 s → 0.1 s, byte-identical) |
| Balancing quality vs. cost | Reranker depth vs. latency; Haiku as judge; local by default |
| Learning something fast | Tutorial experiment → production rebuild in N weeks; reading the papers behind RRF, HyDE and CRAG |
| Something that didn't work | (1) Hybrid BM25 + dense without filters scored *worse* than dense alone (0.32 vs 0.46): FinanceBench questions are full of boilerplate instructions. (2) The local LLM judge rated 90/90 answers correct. (3) The local LLM in Docker on a Mac: OOM in fp32, >15 min per answer in bf16, so I run it as a separate process instead |

---

## 7. Gaps to study beyond this project

Prioritized for GenAI and AI-engineer interviews. Tick them off as you go.

**High priority (very likely to come up)**
- [ ] **Vector index internals:** HNSW parameters, IVF, product quantization, scalar quantization. *Read:* [HNSW paper](https://arxiv.org/abs/1603.09320) · [FAISS wiki](https://github.com/facebookresearch/faiss/wiki) · [Pinecone: product quantization](https://www.pinecone.io/learn/series/faiss/product-quantization/)
- [ ] **LLM fundamentals:** tokenization, the context window, attention cost (O(n²)), the KV cache, temperature and sampling, why models hallucinate. *Read:* [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) · [Andrej Karpathy: Intro to LLMs (video)](https://www.youtube.com/watch?v=zjkBMFhNj_g)
- [ ] **Fine-tuning vs. RAG, and LoRA/QLoRA basics.** *Read:* [LoRA](https://arxiv.org/abs/2106.09685) · [QLoRA](https://arxiv.org/abs/2305.14314)
- [ ] **Agents and tool use:** ReAct, function calling, MCP, guardrails, step limits. *Read:* [ReAct](https://arxiv.org/abs/2210.03629) · [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) · [Model Context Protocol](https://modelcontextprotocol.io/)
- [ ] **Eval statistics:** bootstrap confidence intervals, paired tests, inter-annotator agreement (Cohen's kappa). *Read:* [Anthropic: statistical approach to evals](https://www.anthropic.com/research/statistical-approach-to-model-evals)

**Medium priority**
- [ ] **Embedding fine-tuning** with contrastive loss and hard negatives. *Read:* [sentence-transformers training](https://sbert.net/docs/sentence_transformer/training_overview.html)
- [ ] **LLM serving:** vLLM, continuous batching, quantization (GGUF/AWQ/GPTQ), throughput vs. latency. *Read:* [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)
- [ ] **GraphRAG and knowledge graphs.** *Read:* [GraphRAG](https://arxiv.org/abs/2404.16130)
- [ ] **Multimodal RAG:** charts, scanned documents, vision-language models for PDFs. *Read:* [ColPali](https://arxiv.org/abs/2407.01449)
- [ ] **Learned sparse retrieval:** SPLADE. *Read:* [SPLADE](https://arxiv.org/abs/2107.05720)
- [ ] **Prompt caching and cost optimization** with API LLMs. *Read:* [Claude docs: prompt caching](https://platform.claude.com/docs)

**Nice to have**
- [ ] Semantic caching · query routing classifiers · PII redaction (e.g. Presidio) · LLM observability tools (Langfuse, Arize Phoenix) · frameworks (know what LangChain and LlamaIndex do, and when you'd use one, even though you didn't here)

---

## 8. Red flags to avoid

- ❌ "I used LangChain and it just worked." ✅ Explain each stage and show its metric.
- ❌ Quoting public leaderboard numbers as if they were yours. ✅ "On *my* eval set, I measured…"
- ❌ Saying "chunk size 512 is best" as a rule. ✅ "On this data, measured, the trade-off was…"
- ❌ Claiming that hallucination is solved. ✅ "It's reduced and measured: answers must cite a page (citation checks plus auto-cite repair), off-topic questions abstain, and numeric match is 0.31 with a 1.5B model. Faithfulness scoring needs a validated judge, which is the next step."
- ❌ No failure stories. ✅ Have two things that didn't work and what you learned.
- ❌ Overstating scale. ✅ Be honest: "laptop-scale; here's what changes at 10M documents."

---

## 9. Final-week checklist

- [x] Every 📌 is filled in with real numbers and linked to `results/` (where a number wasn't measured, the guide says so)
- [ ] I can draw the architecture from memory in 2 minutes
- [ ] I can do the live demo offline (Ollama running, indexes built, 5 preset questions)
- [ ] I can explain BM25, RRF, cross-encoders, HNSW, NDCG and LLM-as-judge without notes
- [ ] I've practiced the system-design template twice, out loud, with a timer
- [ ] I have 2 failure stories and 1 "decision backed by data" story in STAR form
- [ ] The README and spec are polished; the GitHub repo is public (license notes included); CI is green
