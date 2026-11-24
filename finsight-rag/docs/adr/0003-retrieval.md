# ADR 0003: Dense + company/year metadata filter by default; BM25 and rerank by config

- **Status:** accepted (P5) · **Date:** 2026-11-24

## Context
The usual advice is "hybrid BM25 + dense, then a cross-encoder reranker". We measured it on all 150 FinanceBench
questions (4 chunkers × 8 variants) instead of assuming it.

## Decision
Default `retrieval.mode: dense` with rule-based **company + fiscal-year filters** extracted from the question
(the extractor falls back to no filter if nothing matches). Hybrid (RRF k=60) and rerankers (local bge-reranker-base,
Cohere/Jina/Voyage) stay one config switch away.

## Evidence
| variant (recursive_384) | hit@10 | latency |
|---|---|---|
| dense, no filter | 0.46 | ~20 ms |
| hybrid, no filter | 0.32 | ~100 ms |
| **dense + company/year filter** | **0.613** | ~110 ms |
| hybrid + filter + local rerank | 0.580 | ~2.7 s |

## Consequences
- ✅ The filter is the biggest lever (doc_hit 0.83 → 0.99) at almost no cost.
- ⚠️ FinanceBench questions are long and full of instructions, so BM25 matches boilerplate. The conclusion is
  dataset-specific: re-measure on new data (MultiHop-RAG support exists for that).
- P10: if the dense search fails, BM25 answers (`degraded: dense_failed`).
