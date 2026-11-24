# Embedder comparison (dense retrieval only)

- Questions: first 30 questions · chunking `page_aware_384` · no filters, no BM25, no reranker
- Relevant = a retrieved chunk on a gold evidence page. Metrics: see `src/finsight/evaluation/retrieval_metrics.py`.
- Config fingerprint (base settings): `c52d8790fa77`

## Summary

| preset            |   n |   hit@1 |   hit@5 |   hit@10 |   recall@10 |   mrr |   ndcg@10 |   doc_hit@10 | provider   |   dims |   chunks |   embedded_now |   embed_s |   api_requests |   index_mb |
|:------------------|----:|--------:|--------:|---------:|------------:|------:|----------:|-------------:|:-----------|-------:|---------:|---------------:|----------:|---------------:|-----------:|
| bge-small         |  30 |   0.1   |   0.267 |    0.433 |       0.383 | 0.192 |     0.229 |        0.967 | local      |    384 |     5325 |              0 |         0 |              0 |       41.5 |
| bge-small-noinstr |  30 |   0.067 |   0.267 |    0.367 |       0.3   | 0.166 |     0.183 |        0.967 | local      |    384 |     5325 |              0 |         0 |              0 |       41.5 |
| bge-base          |  30 |   0.1   |   0.4   |    0.5   |       0.417 | 0.218 |     0.246 |        0.933 | local      |    768 |     5325 |              0 |         0 |              0 |       60.1 |

## hit@10 by question type

| question_type     |   bge-base |   bge-small |   bge-small-noinstr |
|:------------------|-----------:|------------:|--------------------:|
| domain-relevant   |      0.273 |       0.182 |               0.091 |
| metrics-generated |      0.833 |       0.833 |               0.667 |
| novel-generated   |      0.286 |       0.143 |               0.286 |
