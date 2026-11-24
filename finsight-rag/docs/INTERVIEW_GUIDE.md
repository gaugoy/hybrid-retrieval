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
> instead of guessing. It runs fully locally (bge embeddings, a local reranker, and a 7B model on Ollama), with Claude available by config.
> I scored every design choice on FinanceBench against page-level ground truth. For example, hybrid search plus reranking raised
> Recall@10 from 📌 to 📌 compared with dense-only, and CI blocks retrieval regressions."

**2 minutes (structure: problem → approach → evidence → trade-offs)**
1. *Problem:* analysts search 100–300-page filings; plain LLMs make up numbers; wrong financial answers are expensive.
2. *Learning first:* I rebuilt BM25, dense retrieval, RRF and reranking on FiQA to understand each one on its own.
3. *Product:* PDF parsing with table handling → 4 chunking strategies → hybrid retrieval with metadata filters → cross-encoder rerank → a grounded prompt with citation checks → abstention.
4. *Evidence:* an ablation table (📌 best row), judge-scored correctness 📌%, faithfulness 📌, p50 latency 📌 ms.
5. *Trade-offs:* the reranker adds 📌 ms for +📌 recall; query transforms help comparison questions but cost 2–4× the LLM calls.
6. *Production:* typed config, FastAPI with streaming, Docker Compose, a CI regression gate, fallbacks.

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
- **Idempotent indexing:** a manifest of hashes means re-runs skip unchanged work, and an embedding cache means re-chunking only embeds new text.

**Most-asked questions**
1. *"How do you choose an embedding model?"*
   → Start from the MTEB retrieval leaderboard, filtered by size, latency and license. Then **evaluate on my own data**, because leaderboard rank doesn't transfer reliably between domains. I chose bge-small for speed on a laptop and its 384 dimensions; bge-base is the next experiment. Also consider the max sequence length, multilingual needs and domain fit.
2. *"Which vector DB, and why?"*
   → Qdrant: payload filtering during search, local and server modes with the same API, it runs in Docker, and it supports sparse vectors if needed. Alternatives: pgvector (when you already run Postgres and the scale is moderate), Elasticsearch or OpenSearch (when you want BM25 and vectors in one system), Pinecone (managed), FAISS (a library, not a database).
3. *"Explain HNSW."*
   → A multi-layer proximity graph: the top layers are sparse long-range links, the bottom layer is dense. Search starts at the top and greedily descends toward the query. `M` sets the links per node (memory vs. recall); `ef_search` sets the size of the candidate list (latency vs. recall).
4. *"How do you update the index when documents change?"*
   → Use deterministic chunk IDs (`doc::page::seq`) and UUIDs derived from them. Upsert the changed docs, and delete chunks whose doc version changed. The manifest tracks the hashes. Re-embedding the whole corpus is only needed when the embedding model changes (a new collection, a blue/green switch).
5. *"What about embedding dimension and cost?"*
   → Memory is roughly n × dim × 4 bytes for float32, before the index overhead. Options to shrink it: smaller dimensions, Matryoshka embeddings, scalar or product quantization.

**📌 Your numbers:** chunks indexed 📌 · embedding throughput 📌 chunks/s on M5 · index size 📌 MB · effect of the query instruction 📌

**Go deeper:** [Sentence-BERT](https://arxiv.org/abs/1908.10084) · [HNSW paper](https://arxiv.org/abs/1603.09320) · [Qdrant filtering](https://qdrant.tech/documentation/concepts/filtering/) · [MTEB](https://huggingface.co/spaces/mteb/leaderboard) · study **product quantization and IVF** and **Matryoshka embeddings** (§7)

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
   → Show the table: dense 📌, BM25 📌, hybrid 📌. Then give a concrete example query where BM25 found the line item and dense didn't, and the reverse. For context, my FiQA experiment cited public baselines of roughly BM25 ~24 and `text-embedding-3-small` ~31 NDCG@10, with cross-encoder reranking above 40. Quote your own measured numbers first.
2. *"Why RRF rather than a weighted score sum?"*
   → The scores live on different scales and distributions, and those change from query to query. RRF needs no calibration, is robust, and k=60 is the standard value. A weighted sum needs normalization (min-max or z-score) and tuning, and can win if it is tuned on held-out data.
3. *"Bi-encoder vs. cross-encoder?"*
   → A bi-encoder embeds separately: fast and indexable, but coarse. A cross-encoder reads the pair jointly: accurate but O(n) model calls per query. So: bi-encoder for recall, cross-encoder for precision on the top candidates.
4. *"How many candidates should you rerank?"*
   → It's a trade-off: more candidates give more recall, but latency grows linearly. I measured top-20/50/100 against latency: 📌. Pick the knee of the curve.
5. *"Your retrieval is bad. How do you debug it?"*
   → (1) Is the answer in the corpus, and was it parsed? (the evidence check) (2) Is it inside one chunk? (chunk coverage) (3) Does it appear in the top-50 of either retriever? (4) Is it lost at fusion or at reranking? (5) Did a filter remove it? Each stage is logged, so I can tell which one failed.
6. *"What's the difference between Recall@k and NDCG?"*
   → Recall asks whether it was found; NDCG asks how high it was ranked. For RAG, recall at the context size matters most (the LLM sees all k chunks), but rank still matters because of "lost in the middle".
7. *"Filtering: before or after the vector search?"*
   → Filtering during the search (Qdrant's filterable HNSW) avoids getting fewer than k results. Filtering afterwards needs over-fetching. I do that for BM25 (fetch 4k, then filter).

**📌 Your numbers:** the ablation table (copy it here) · rerank latency 📌 ms per 50 candidates · effect of the filter 📌

**Go deeper:** [RRF paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) · [BEIR](https://arxiv.org/abs/2104.08663) · [ColBERT, late interaction](https://arxiv.org/abs/2004.12832) · [SPLADE, learned sparse retrieval](https://arxiv.org/abs/2107.05720) · [SBERT cross-encoders](https://sbert.net/examples/cross_encoder/applications/README.html)

---

### P6 — Grounded generation ⭐

**Concepts**
- **Grounding prompt:** answer only from the numbered context, cite as [n], show calculations, use one exact abstain phrase, and treat the context as data (this is the prompt-injection defense).
- **Context assembly:** rerank order, a token budget, parent expansion, deduplication. Put the most relevant material first and keep it short ("lost in the middle").
- **Citation checks:** parse the [n] markers, drop the ones that don't exist, and flag uncited answers. This is a cheap, deterministic guardrail.
- **Abstention:** in finance, a wrong answer is worse than "not found". It is measured as the false-abstain rate (on answerable questions) and the correct-abstain rate (on unanswerable ones).
- **Provider abstraction:** one `LLM` interface; Ollama locally, Claude in the cloud; a config switch; fallbacks.

**Most-asked questions**
1. *"How do you reduce hallucinations in RAG?"*
   → Better retrieval first (most hallucinations come from missing context), a strict grounding prompt, required citations with automatic checks, an abstain option, CRAG-style evidence grading, a faithfulness eval, and a lower-temperature or stronger model. I measure it: faithfulness 📌.
2. *"How do you handle the context window?"*
   → A token budget during assembly, reranking so the best chunks come first, deduplicating parents, and, for long-context models, testing whether *more* context actually helps (it often doesn't). Also watch for silent truncation, such as Ollama's default `num_ctx`.
3. *"Local LLM vs. API LLM: how do you decide?"*
   → Privacy and data residency, cost at your volume, latency, the quality bar (my 7B vs. Claude on the same retrieval: 📌 vs. 📌 correct), operations (GPU serving with vLLM vs. a managed API). Often the answer is a hybrid: local for cheap steps, the API for the hard ones.
4. *"How do you make the LLM do arithmetic correctly?"*
   → Ask it to show the formula with cited inputs. Better still, extract the numbers and compute them in Python (a tool call or calculator). Verify with the numeric-match metric.
5. *"What is prompt injection in RAG, and how do you defend against it?"*
   → Retrieved text can contain instructions ("ignore previous instructions"). Defenses: tell the model the context is data, use delimiters, give the model the least privilege (no dangerous tools), filter the output, and keep humans in the loop for actions. See the OWASP LLM Top 10.

**📌 Your numbers:** Ollama vs. Claude correctness 📌 · hallucinated-citation rate 📌 · false-abstain rate 📌 · tokens per question 📌

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
   → For multi-hop or open-ended questions where you can't know the retrieval plan in advance. For simple lookups a fixed pipeline is cheaper, faster and easier to test. My MultiHop-RAG result: 📌.
2. *"Does HyDE always help?"*
   → No. It helps when there is a big vocabulary gap between questions and documents, and hurts when the generated answer adds wrong specifics. My result: 📌. Always A/B test it.
3. *"What is contextual retrieval?"*
   → A technique Anthropic published: before indexing, add a short, document-aware context to each chunk, for both the embeddings and BM25. My cheap version (a metadata header) gave 📌; the LLM-written version costs one LLM call per chunk at ingest time.
4. *"How do you keep advanced techniques from blowing up latency and cost?"*
   → Route by query type (a classifier or rules), run the extra steps only when the first pass is weak (the CRAG idea), cache LLM outputs, use small models for the helper steps, and stream the final answer.

**📌 Your numbers:** a row per technique (Δ recall, Δ correctness, extra latency, LLM calls)

**Go deeper:** [HyDE](https://arxiv.org/abs/2212.10496) · [RAG-Fusion](https://arxiv.org/abs/2402.03367) · [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) · [CRAG](https://arxiv.org/abs/2401.15884) · [Self-RAG](https://arxiv.org/abs/2310.11511) · [ReAct](https://arxiv.org/abs/2210.03629) · [GraphRAG](https://arxiv.org/abs/2404.16130)

---

### P8 — Evaluation ⭐⭐ (what separates senior candidates)

**Concepts**
- **Evaluate in layers:** retrieval metrics (free, deterministic) → generation metrics (correctness, faithfulness) → system metrics (latency, cost).
- **Reference-based** metrics compare against a gold answer (numeric match, judge correctness). **Reference-free** metrics need no gold answer (faithfulness to the context, answer relevance).
- **LLM-as-judge:** scalable, but biased (towards position, verbosity and its own model family). Validate it against human labels: my agreement was 📌% on 20 labels.
- **Noise:** with 150 questions, small differences aren't significant. Report n, the standard deviation, and the breakdown by question type.
- **Regression gate:** a small, fixed set in CI that blocks merges which break retrieval.

**Most-asked questions**
1. *"How do you evaluate a RAG system?"*
   → Use my three layers, with the same pipeline objects as production, a fixed eval set, a stored config fingerprint, a failure gallery, and a validated judge. Online, add user feedback, citation click-through and the abstain rate.
2. *"How do you know your LLM judge is reliable?"*
   → Hand-label a sample, measure agreement or Cohen's kappa, use a judge from a different model or provider than the generator, give it a clear rubric with JSON output, and vary the answer order to check for position bias.
3. *"What did your failure analysis show?"*
   → 📌 Your top 3 failure categories (for example: the right doc but a neighbouring page, a table that was split, a wrong fiscal year) and what you changed because of them. **Have one concrete story ready.**
4. *"How do you evaluate in production without labels?"*
   → Faithfulness scored on a sample of traffic, user thumbs up or down, how often citations are clicked, retrieval-score drift, abstain-rate drift, and turning failures into new eval cases.

**📌 Your numbers:** headline table · judge agreement · top failure categories

**Go deeper:** [LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) · [RAGAS](https://arxiv.org/abs/2309.15217) · [ARES](https://arxiv.org/abs/2311.09476) · [Anthropic: statistical approach to evals](https://www.anthropic.com/research/statistical-approach-to-model-evals) · learn **bootstrap confidence intervals** and **paired significance tests** (§7)

---

### P9 — Serving

**Concepts:** health vs. readiness checks; warm-up in the app lifespan (load models once); streaming over SSE for perceived latency; request IDs for tracing; input validation.

**Most-asked questions**
1. *"How do you reduce perceived latency?"* → Stream tokens; send the citations as soon as retrieval finishes; warm up the models; cache embeddings and frequent answers; run dense and BM25 in parallel.
2. *"Health vs. readiness?"* → Health means the process is alive (restart it if not). Ready means it can serve, with indexes loaded and dependencies reachable (keep it out of the load balancer until it is). Kubernetes uses both.
3. *"Sync or async FastAPI here?"* → Model inference is CPU/GPU-bound, so run it in a threadpool or separate worker processes. The external LLM calls are I/O-bound, so async helps there. Don't block the event loop.

**📌 Your numbers:** p50/p95 latency 📌 at 5 concurrent users · time to first token 📌

**Go deeper:** [FastAPI concurrency](https://fastapi.tiangolo.com/async/) · [SSE (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)

---

### P10 — Production hardening

**Concepts:** containers (multi-stage, non-root); graceful degradation (reranker → RRF order, Claude → Ollama); caching layers; observability (logs, metrics, traces); cost tracking; security.

**Most-asked questions**
1. *"What breaks in production that doesn't break in a notebook?"* → Timeouts and rate limits, cold starts, index and model version skew, concurrent writes, memory pressure, bad inputs, prompt injection, cost spikes, and silent quality drift.
2. *"How would you deploy this on AWS, GCP or Azure?"* → A container on ECS, Cloud Run or AKS; a managed Qdrant or Qdrant on Kubernetes with persistent volumes; models served with vLLM or TGI on GPU nodes, or an API LLM; secrets in a secret manager; OpenTelemetry traces; a separate batch ingestion job with a queue (SQS, Pub/Sub); blue/green index switching.
3. *"How do you control LLM cost?"* → Cache answers, route cheap models to the easy steps, enforce token budgets, use prompt caching for stable prefixes, use batch APIs for offline jobs, and track the cost of each request.

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
12. **How do you choose top-k for the context?** By recall at k against cost and noise, measured. Mine: 📌.
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
| A technical decision you made with data | Choosing the chunker and filter policy from the ablation (📌 numbers), and rejecting the hard year filter because 10-Ks contain prior-year data |
| A bug you found that others would miss | 📌 for example, the 0-based page index, silent truncation at 512 tokens, or Ollama `num_ctx` truncation |
| Balancing quality vs. cost | Reranker depth vs. latency; Haiku as judge; local by default |
| Learning something fast | Tutorial experiment → production rebuild in N weeks; reading the papers behind RRF, HyDE and CRAG |
| Something that didn't work | 📌 a technique that didn't help (HyDE or semantic chunking?) and why you dropped it. Interviewers value this. |

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
- ❌ Claiming that hallucination is solved. ✅ "It's reduced and measured: faithfulness 📌, plus abstention."
- ❌ No failure stories. ✅ Have two things that didn't work and what you learned.
- ❌ Overstating scale. ✅ Be honest: "laptop-scale; here's what changes at 10M documents."

---

## 9. Final-week checklist

- [ ] Every 📌 is filled in with real numbers and linked to `results/`
- [ ] I can draw the architecture from memory in 2 minutes
- [ ] I can do the live demo offline (Ollama running, indexes built, 5 preset questions)
- [ ] I can explain BM25, RRF, cross-encoders, HNSW, NDCG and LLM-as-judge without notes
- [ ] I've practiced the system-design template twice, out loud, with a timer
- [ ] I have 2 failure stories and 1 "decision backed by data" story in STAR form
- [ ] The README and spec are polished; the GitHub repo is public (license notes included); CI is green
