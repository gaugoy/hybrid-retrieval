"""Shared plumbing for ONLINE providers (embeddings now; rerankers in P5, LLMs in P6):
client-side rate limiting, persisted daily/monthly usage, HTTP retries with backoff, and
classification of 429s into "slow down" vs "quota used up"."""
