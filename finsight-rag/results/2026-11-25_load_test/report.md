# load_test

- Source: `scripts/12_load_test.py --users 5 --n 10` against `make serve` (local process, no Docker), 2026-11-25.
- Setup: MacBook (M5), config `configs/local.yaml`: recursive_384 · dense + company/year filter · bge-small (MPS) ·
  LLM `hf-qwen1.5b` (Qwen2.5-1.5B-Instruct, float16 on MPS, one generation at a time per process).
- Questions: FinanceBench questions 21–40 (distinct per pass, so the first two passes miss the answer cache).

| pass | users | requests | errors | cache hits | p50 s | p95 s | throughput (req/min) |
|---|---|---|---|---|---|---|---|
| sequential | 1 | 10 | 0 | 0 | 2.50 | 7.83 | 17.6 |
| concurrent | 5 | 10 | 0 | 0 | 14.36 | 20.49 | 17.1 |
| cached (same questions again) | 5 | 10 | 0 | 10 | 0.01 | 0.01 | n/a |

**Reading it:** throughput doesn't change with 5 users. One local model on one GPU serves one request at a time
(`_GPU_LOCK`), so concurrency only adds queueing. Scaling needs a batching inference server (Ollama/vLLM) or a hosted API.
Streaming (`/v1/query/stream`, P9 smoke test): first token ~1.5 s.

An earlier 5-user run (P9) crashed on thread-bound SQLite caches. It was fixed with shared connections + locks and a regression test.
