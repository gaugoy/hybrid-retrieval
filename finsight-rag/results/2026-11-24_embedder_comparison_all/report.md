# Embedder comparison (dense retrieval only)

- Questions: all questions · chunking `page_aware_384` · no filters, no BM25, no reranker
- Relevant = a retrieved chunk on a gold evidence page. Metrics: see `src/finsight/evaluation/retrieval_metrics.py`.
- Config fingerprint (base settings): `c52d8790fa77`

## Summary

| preset            |   n |   hit@1 |   hit@5 |   hit@10 |   recall@10 |   mrr |   ndcg@10 |   doc_hit@10 | provider   |   dims |   chunks |   embedded_now |   embed_s |   api_requests |   index_mb |
|:------------------|----:|--------:|--------:|---------:|------------:|------:|----------:|-------------:|:-----------|-------:|---------:|---------------:|----------:|---------------:|-----------:|
| bge-small         | 150 |   0.127 |    0.34 |    0.46  |       0.433 | 0.227 |     0.272 |        0.893 | local      |    384 |    31114 |              0 |         0 |              0 |        244 |
| bge-small-noinstr | 150 |   0.133 |    0.36 |    0.453 |       0.423 | 0.23  |     0.269 |        0.867 | local      |    384 |    31114 |              0 |         0 |              0 |        244 |

## hit@10 by question type

| question_type     |   bge-small |   bge-small-noinstr |
|:------------------|------------:|--------------------:|
| domain-relevant   |        0.3  |                0.28 |
| metrics-generated |        0.64 |                0.6  |
| novel-generated   |        0.44 |                0.48 |
