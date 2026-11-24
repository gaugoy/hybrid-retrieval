# ADR 0001: Qdrant (embedded for development, server in Docker) + a separate BM25 index

- **Status:** accepted (P4, revised P10) · **Date:** 2026-11-24

## Context
About 31k chunks × 384 dims. It must run on a laptop with no services, use payload filters (company, fiscal year),
and grow into a server deployment. Candidates: FAISS, Chroma, pgvector, Qdrant, plain numpy.

## Decision
Qdrant through `qdrant-client`: **embedded mode** (a folder, no server) for development and notebooks, and the
**Qdrant server** (`QDRANT_URL`) in Docker. The code is the same; only the config changes. BM25 is a separate `bm25s` index on disk.

## Consequences
- ✅ No infrastructure for development. Filters live in the store. A one-line switch to a server.
- ⚠️ Embedded mode locks its folder per process, so there's one API worker, and notebooks block the CLI (runbook §1).
  Embedded search is brute force (fine at 31k: ~20 ms); payload indexes are ignored there.
- ⚠️ BM25 lives outside Qdrant, so there are two indexes to keep in sync. The manifest records both.
- Revisit if the corpus passes ~1M chunks (HNSW tuning, sharding) or hybrid search moves into Qdrant (sparse vectors).
