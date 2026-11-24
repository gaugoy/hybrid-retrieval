# FinSight RAG: architecture

How the pieces fit together, from PDFs to a cited answer. The diagrams are Mermaid, so GitHub renders them.

## 1. Runtime: three independent processes

```mermaid
flowchart LR
    U([User / browser]) --> UI["Frontend<br/>Streamlit :8501<br/>ui/streamlit_app.py"]
    C([Any HTTP client]) --> API
    UI -- "HTTP JSON / SSE" --> API["Backend<br/>FastAPI :8000<br/>finsight.api"]
    API --> CT["Container<br/>(built once from Settings)"]
    CT --> EMB["Embedder<br/>bge-small (MPS)"]
    CT --> QD[("Qdrant<br/>embedded folder<br/>indexes/qdrant")]
    CT --> BM[("BM25<br/>indexes/bm25")]
    CT --> CACHE[("SQLite caches<br/>query vectors · rerank · answers")]
    CT --> LLM{"LLM preset<br/>auto"}
    LLM --> OL["Ollama :11434<br/>(optional, own process)"]
    LLM --> HF["Local HF Qwen2.5-1.5B<br/>(in the API process)"]
    LLM --> APIS["Free-tier APIs / Claude<br/>(key in .env)"]
```

- No Docker is needed: `ollama serve` (optional) + `make serve` + `make ui` ([runbook](runbook.md) §1).
- The **UI loads no models**. It calls the API, so the demo shows exactly what an API client gets.
- The **Container** (`src/finsight/container.py`) builds every component lazily from one `Settings`. The CLI, scripts,
  evaluation and API all use it, so the pipeline you evaluate is the pipeline you serve.

## 2. Offline: building the indexes

```mermaid
flowchart LR
    A["FinanceBench<br/>150 Q · 84 PDFs"] -->|make data| B["data/raw"]
    B -->|"make ingest (PyMuPDF,<br/>tables as rows, header/footer strip)"| C["pages.parquet<br/>12,013 pages"]
    C -->|"chunk: recursive 384 tok<br/>(+ contextual header)"| D["data/chunks/recursive_384.parquet<br/>27,927 chunks"]
    D -->|"embed (cache: sha256(model+text))"| E[("Qdrant<br/>384-d cosine<br/>payload: company, year, page")]
    D -->|tokenize| F[("BM25 (bm25s)")]
    E & F --> M["manifest.json<br/>skip if chunks + model + version unchanged"]
```

Re-running is free: the manifest skips unchanged indexes, and the embedding cache means only *new* chunk texts are
embedded (indexing into a fresh Qdrant server: 27,927/27,927 cache hits, 8 s).

## 3. Online: one `/v1/query` request

```mermaid
sequenceDiagram
    autonumber
    participant Cl as Client / UI
    participant MW as Middleware
    participant R as Route /v1/query
    participant AC as Answer cache
    participant RS as RetrievalService
    participant L as LLM (FallbackLLM)
    Cl->>MW: POST {question, pipeline?, filters?, top_k?}
    MW->>MW: X-Request-ID · body ≤ limit · rate limit per IP (429)
    MW->>R: validated QueryRequest (422 on bad input)
    R->>AC: key = sha256(question, filters, pipeline, llm, top_k, config fingerprint)
    alt cache hit (0.01 s)
        AC-->>R: Answer (new request_id), X-Cache: hit
    else miss
        R->>RS: retrieve(question)
        RS->>RS: extract company/year filter (rules)
        RS->>RS: dense search (query vector cached) → [BM25] → [RRF] → [rerank]
        Note over RS: degrade, don't fail: dense down → BM25 (dense_failed),<br/>reranker error → RRF order, empty filter → unfiltered
        RS-->>R: top-k chunks + timings
        R->>L: numbered context blocks + grounding prompt
        Note over L: provider error → next preset in llm.fallbacks (llm_fallback)
        L-->>R: text
        R->>R: parse [n] citations, auto-cite repair, abstain check
        R->>AC: store (only if not degraded)
    end
    R-->>Cl: Answer {answer, citations[page], abstained, timings_ms, usage, degraded, request_id}
```

`/v1/query/stream` does the same, but emits Server-Sent Events: `retrieved` (the sources), then `token` deltas, then
`done` (citations and timings). The first token arrives after ~1.5 s with the local model.

## 4. Configuration

`configs/base.yaml` ← profile (`local.yaml` / `cloud.yaml` / `multihop.yaml`) ← env (`FINSIGHT__SECTION__KEY`, plus
aliases `LLM`, `EMBEDDER`, `QDRANT_URL`) ← CLI overrides. Models are **presets** (`configs/llms.yaml`,
`configs/embedders.yaml`). Paid-only providers are refused unless `allow_paid: true`, and free-tier rate limits are
enforced on the client side. The **config fingerprint** (a hash of the settings that change answers) is stored with
every result and keys the answer cache.

## 5. Where the code lives

| Concern | Module |
|---|---|
| Data, parsing | `finsight.data` (`financebench`, `multihop`, `datasets` dispatch), `finsight.ingestion` |
| Chunking | `finsight.chunking` (fixed, recursive, page_aware, parent_child; exact bge token counts) |
| Embeddings, indexes | `finsight.embedding` (local + online, SQLite cache), `finsight.indexing` (Qdrant, BM25, manifest) |
| Retrieval | `finsight.retrieval` (dense, sparse, RRF, filters, rerankers + score cache, query transforms, service) |
| Generation | `finsight.generation` (prompts, context blocks, citations, LLM providers, `FallbackLLM`) |
| Pipelines | `finsight.pipelines` (naive, hybrid_rerank, corrective, agentic) |
| Evaluation | `finsight.evaluation` (retrieval metrics + runner, end-to-end, judge, reports) |
| Serving | `finsight.api` (routes, middleware, models), `finsight.cache` (answers), `ui/streamlit_app.py` |

Design decisions with evidence: [ADRs](adr/). Measurements: [evaluation.md](evaluation.md).
