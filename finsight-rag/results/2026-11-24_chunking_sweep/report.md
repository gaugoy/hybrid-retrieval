# Chunking sweep

- Tokenizer: `BAAI/bge-small-en-v1.5` (limit 510 text tokens)
- Overlap: 64 tokens · carry_table_header: True · base fingerprint `25a7e33aec43`
- `evidence_in_one_chunk`: share of the 189 evidence items that a single chunk covering the evidence page contains (overlap ≥ 0.8).
- `union_overlap_mean`: all chunks of that page together (text lost if < 1).

| config | strategy | size | chunks | chunks_per_doc | tokens_p50 | tokens_p95 | tokens_max | multi_page_share | over_limit_share | evidence_in_one_chunk | best_chunk_overlap_mean | union_overlap_mean | chunks_per_evidence_page | build_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed_256 | fixed | 256 | 44998 | 535.7 | 256 | 256 | 256 | 0.34 | 0.0 | 0.439 | 0.791 | 0.998 | 5.1 | 9.37 |
| fixed_384 | fixed | 384 | 27013 | 321.6 | 384 | 384 | 384 | 0.496 | 0.0 | 0.683 | 0.866 | 0.998 | 3.4 | 9.2 |
| fixed_480 | fixed | 480 | 20793 | 247.5 | 480 | 480 | 480 | 0.601 | 0.0 | 0.868 | 0.905 | 0.998 | 2.9 | 9.1 |
| recursive_256 | recursive | 256 | 45333 | 539.7 | 242 | 255 | 256 | 0.297 | 0.0 | 0.444 | 0.78 | 0.998 | 5.1 | 9.64 |
| recursive_384 | recursive | 384 | 27927 | 332.5 | 368 | 383 | 384 | 0.431 | 0.0 | 0.688 | 0.864 | 0.998 | 3.6 | 9.54 |
| recursive_480 | recursive | 480 | 21789 | 259.4 | 462 | 479 | 480 | 0.525 | 0.0 | 0.804 | 0.907 | 0.998 | 3.1 | 9.47 |
| page_aware_256 | page_aware | 256 | 47847 | 569.6 | 236 | 255 | 256 | 0.0 | 0.0 | 0.476 | 0.793 | 0.998 | 4.1 | 9.89 |
| page_aware_384 | page_aware | 384 | 31114 | 370.4 | 355 | 382 | 384 | 0.0 | 0.0 | 0.72 | 0.88 | 0.998 | 2.7 | 9.76 |
| page_aware_480 | page_aware | 480 | 25337 | 301.6 | 419 | 477 | 480 | 0.0 | 0.0 | 0.825 | 0.921 | 0.998 | 2.1 | 9.61 |
| parent_child_128 | parent_child | 128 | 95174 | 1133.0 | 113 | 127 | 128 | 0.0 | 0.0 | 0.286 | 0.633 | 0.998 | 8.3 | 19.26 |
| parent_child_200 | parent_child | 200 | 60593 | 721.3 | 182 | 199 | 200 | 0.0 | 0.0 | 0.386 | 0.733 | 0.998 | 5.2 | 19.11 |
| parent_child_256 | parent_child | 256 | 47847 | 569.6 | 236 | 255 | 256 | 0.0 | 0.0 | 0.476 | 0.793 | 0.998 | 4.1 | 19.05 |
