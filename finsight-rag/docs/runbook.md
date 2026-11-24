# FinSight RAG: runbook

Operating the system: start it, rebuild indexes, switch providers, rotate keys, debug a bad answer.
Everything runs locally and needs no paid service.

## 1. Run the app (no Docker needed)

The app runs as **independent processes on your machine**, one terminal each. Docker / `docker compose` is **not
needed** (and not recommended on a Mac: containers can't use the Apple GPU, see the appendix).

| # | Process | Start | Port | Notes |
|---|---|---|---|---|
| 1 | **LLM server** (optional) | `ollama serve` (then once: `ollama pull qwen2.5:7b-instruct`) | 11434 | Free, uses the Mac GPU. Select it with `LLM=ollama-qwen7b` in `.env`. **Skip this step** to use the built-in local model (`hf-qwen1.5b`, loaded inside the API process) or a free-tier API key (`LLM=gemini-flash`, ...) |
| 2 | **Backend API** (FastAPI) | `make serve` (or `uv run finsight serve`) | 8000 | Loads the embedder, the indexes and the LLM client. Docs at http://127.0.0.1:8000/docs |
| 3 | **Frontend UI** (Streamlit) | `make ui` | 8501 | A thin client of the API (`FINSIGHT_API_URL`, default `http://127.0.0.1:8000`). It loads no models |

Start them in that order (LLM → API → UI). Each can be restarted on its own: restarting the UI never reloads
models, and switching the LLM is just `.env` + restarting the API.

- Embedded Qdrant lives in `indexes/qdrant` and allows **one process at a time**: close notebook kernels that opened
  it, or give the API a copy (`cp -R indexes/qdrant indexes/qdrant_work && rm indexes/qdrant_work/.lock`, then
  `FINSIGHT__PATHS__QDRANT_DIR=indexes/qdrant_work make serve`).
- Probes: `GET /health` (process alive) · `GET /ready` (index loaded + LLM resolvable; 503 lists the reasons) ·
  `GET /v1/config` (what is running, plus the config fingerprint).
- Smoke test: `uv run python scripts/11_smoke_api.py`. Load test: `uv run python scripts/12_load_test.py --users 5`.
- Stop: Ctrl+C in each terminal.

## 2. First-time setup (once)

```bash
make setup data ingest index     # deps -> FinanceBench PDFs -> pages -> chunks -> Qdrant (embedded) + BM25
```

## 3. Rebuild indexes

| Change | Do | Cost |
|---|---|---|
| Chunking settings | `uv run finsight chunk && make index` | Re-embeds only new chunk texts (embedding cache) |
| Embedding model / preset | `EMBEDDER=<preset> make index` | Full re-embed with the new model (new collection; the old one stays) |
| Code change in indexing | bump `INDEX_VERSION` in `indexing/pipeline.py`, then `make index` | Vectors come from the cache |
| Corrupt or missing index | `uv run finsight index --force` | Same as above |

The manifest (`indexes/manifest.json`) skips the work when the chunk content, model and version are unchanged.
Query vectors, rerank scores and API answers are cached in `indexes/embed_cache/`. Answers are keyed by the
config fingerprint, so any config change misses the cache automatically. Delete `answers.sqlite` to clear them.

## 4. Switch LLM / embedding / reranker provider

Set it in `.env`, without code changes: `LLM=gemini-flash`, `JUDGE=claude-haiku`, `EMBEDDER=gemini-001`,
`FINSIGHT__RERANK__PROVIDER=local`. List presets with key status: `uv run finsight llms` / `uv run finsight embedders`.
Paid-only providers are refused unless `allow_paid: true`. Free-tier limits are enforced client-side
(`indexes/usage/`). Fallbacks: `llm.fallbacks: [ollama-qwen7b, hf-qwen1.5b]` (the cloud profile does this).

## 5. Rotate an API key

1. Create the new key in the provider console. 2. Replace it in `.env` (never committed; `.dockerignore` keeps it
out of images). 3. Restart the API (Ctrl+C, then `make serve`; keys are read at startup).
4. Check `/ready`, then revoke the old key. Keys are `SecretStr` and never logged.

## 6. Debug a bad answer

1. Find the `request_id` (response header `X-Request-ID`, the UI caption, or the answer JSON) and grep the API log for it.
2. Check `degraded` in the answer: `dense_failed`, `reranker_failed`, `filter_returned_nothing`, `llm_fallback`.
   Degraded answers are never cached.
3. Re-run retrieval alone: `uv run python scripts/06_retrieve.py --question "<question>"`. Is the gold page in the top 10?
   No → a retrieval problem (filters? vocabulary? try contextual headers / a query transform). Yes → go on.
4. Look at the prompt and answer: `uv run python scripts/08_generate.py --question "<question>"`. Is the number in a
   context block? Did the model pick the wrong line item (common with small models)? Was it cited?
5. Is it a stale answer? `X-Cache: hit` means it came from the cache. A config change or deleting `answers.sqlite` clears it.

## 7. Common errors

| Symptom | Cause → fix |
|---|---|
| `Storage folder ... is already accessed by another instance` | Embedded Qdrant locked by another process (notebook kernel). Close it, use a copy (§1), or use the Qdrant server |
| `/ready` 503 `dense_index: empty` | Index not built for this config → `make index` (Docker: with `QDRANT_URL`) |
| 429 `rate limit` from the API | More than `api.rate_limit_per_minute` POSTs per IP → raise it or slow down |
| `QuotaExhaustedError` | Free-tier daily/monthly quota used up → wait, or switch preset; fallbacks take over if configured |
| `PaidProviderBlockedError` | A paid-only provider was selected → pick a free preset, or set `allow_paid: true` deliberately |
| Docker container exits 137 / `OOMKilled=true` | The local LLM didn't fit Docker's memory → `LLM=ollama-qwen7b` (host), a free-tier key, or `hf-qwen0.5b`. `docker compose down` frees everything |
| Very slow answers (~5–20 s) | Local 1.5B on MPS/CPU, one request at a time → Ollama with a GPU, or a hosted free tier |
| `SQLite objects created in a thread...` | Fixed in P9 (shared connection + lock). If it reappears, a new cache missed `check_same_thread=False` |

## Appendix: Docker (optional, not used)

`Dockerfile` and `docker-compose.yml` exist for a Linux server with enough memory (or a GPU). Verified on this Mac:
the image builds, the Qdrant v1.20.2 server runs, and indexing into it took 8 s (embedding cache). But **the local
LLM inside Docker didn't work here**: the 1.5B model was OOM-killed in float32 (8 GB VM), and bfloat16 used 10 cores
for >15 min per answer. If you ever use it: `docker compose up -d qdrant`, `QDRANT_URL=http://localhost:6333 make index`,
`docker compose up -d`, with `LLM=ollama-qwen7b` (Ollama on the host) or an API key. `docker compose down` stops everything.
