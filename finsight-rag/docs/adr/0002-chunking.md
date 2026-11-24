# ADR 0002: Recursive 384-token chunks with contextual headers

- **Status:** accepted (P3, P5, P7) · **Date:** 2026-11-24

## Context
10-K/10-Q pages mix prose and tables. The embedder (bge-small) reads at most 512 tokens. Answers need an exact page citation.

## Decision
`recursive` chunking at 384 tokens (bge tokenizer, hard limit 510) with table-header carry. In P7 we added
**contextual headers**: `company · doc type FY · page` is prepended to the *embedded* text only (the LLM reads the plain chunk).

## Evidence
- P3: evidence held in one chunk rises 44–48% (256) → 68–72% (384) → 80–87% (480). Bigger chunks dilute the embedding.
- P5: with filters, the chunkers are within 2–5 pts of each other. recursive_384 is best: hit@10 0.613.
- P7: contextual headers raise hit@10 to **0.673** (+6 pts with the filter; +10 pts without, 0.467 → 0.567).

## Consequences
- ✅ The headers put company and year into the vector, which partly backs up the metadata filter.
- ⚠️ A header change means a re-embed (the embedding cache key includes the text). page_aware is kept for
  exact single-page citations.
