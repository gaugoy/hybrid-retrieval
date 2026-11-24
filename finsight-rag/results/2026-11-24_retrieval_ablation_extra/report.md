# retrieval_ablation_extra

- Questions: 150 (FinanceBench) · relevant = a chunk on a gold evidence page
- Values: mean ± std over questions. Differences under ~0.05 on 150 questions are noise.
- Best by hit@10 (then MRR): **recursive_384 · dense+filter+year**
- Company extracted from the question: nan% of questions

## All configurations

|                                                   | hit@5         | hit@10        | recall@10     | mrr           | ndcg@10       | doc_hit@10    |   p50 ms |   n |
|:--------------------------------------------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|---------:|----:|
| ('page_aware_384', 'dense+filter+year')           | 0.467 ± 0.501 | 0.593 ± 0.493 | 0.567 ± 0.484 | 0.358 ± 0.402 | 0.393 ± 0.396 | 0.980 ± 0.140 |      121 | 150 |
| ('page_aware_384', 'hybrid+filter+year+rerank')   | 0.420 ± 0.495 | 0.607 ± 0.490 | 0.540 ± 0.465 | 0.306 ± 0.372 | 0.343 ± 0.363 | 0.987 ± 0.115 |     2649 | 150 |
| ('fixed_384', 'dense+filter+year')                | 0.433 ± 0.497 | 0.560 ± 0.498 | 0.530 ± 0.485 | 0.308 ± 0.372 | 0.341 ± 0.371 | 0.987 ± 0.115 |      105 | 150 |
| ('fixed_384', 'hybrid+filter+year+rerank')        | 0.453 ± 0.499 | 0.600 ± 0.492 | 0.567 ± 0.480 | 0.332 ± 0.385 | 0.370 ± 0.378 | 0.987 ± 0.115 |     2858 | 150 |
| ('recursive_384', 'dense+filter+year')            | 0.527 ± 0.501 | 0.613 ± 0.489 | 0.580 ± 0.478 | 0.353 ± 0.385 | 0.385 ± 0.375 | 0.993 ± 0.082 |      108 | 150 |
| ('recursive_384', 'hybrid+filter+year+rerank')    | 0.447 ± 0.499 | 0.580 ± 0.495 | 0.550 ± 0.484 | 0.280 ± 0.346 | 0.327 ± 0.350 | 0.993 ± 0.082 |     2732 | 150 |
| ('parent_child_200', 'dense+filter+year')         | 0.440 ± 0.498 | 0.587 ± 0.494 | 0.550 ± 0.480 | 0.359 ± 0.410 | 0.390 ± 0.402 | 0.987 ± 0.115 |      220 | 150 |
| ('parent_child_200', 'hybrid+filter+year+rerank') | 0.413 ± 0.494 | 0.573 ± 0.496 | 0.520 ± 0.474 | 0.284 ± 0.360 | 0.326 ± 0.359 | 0.973 ± 0.162 |     1606 | 150 |

## hit@10 by question type (recursive_384)

| question_type     |   dense+filter+year |   hybrid+filter+year+rerank |
|:------------------|--------------------:|----------------------------:|
| domain-relevant   |                0.42 |                        0.38 |
| metrics-generated |                0.82 |                        0.74 |
| novel-generated   |                0.6  |                        0.62 |

## 10 hardest questions for the best configuration (recursive_384 · dense+filter+year)

| qid                   | question_type     | top3                                                                                 | gold                          |
|:----------------------|:------------------|:-------------------------------------------------------------------------------------|:------------------------------|
| financebench_id_04672 | metrics-generated | ['3M_2018_10K p100', '3M_2018_10K p44', '3M_2018_10K p46']                           | ['3M_2018_10K p58']           |
| financebench_id_01226 | domain-relevant   | ['3M_2022_10K p19', '3M_2022_10K p20', '3M_2022_10K p121']                           | ['3M_2022_10K p27']           |
| financebench_id_01865 | novel-generated   | ['3M_2022_10K p19', '3M_2022_10K p32', '3M_2022_10K p27']                            | ['3M_2022_10K p25']           |
| financebench_id_00807 | domain-relevant   | ['3M_2023Q2_10Q p71', '3M_2023Q2_10Q p61', '3M_2023Q2_10Q p53']                      | ['3M_2023Q2_10Q p5']          |
| financebench_id_00799 | domain-relevant   | ['AMCOR_2023Q4_EARNINGS p7', 'AMCOR_2023_10K p35', 'AMCOR_2023Q4_EARNINGS p2']       | ['AMCOR_2023_10K p52']        |
| financebench_id_01079 | domain-relevant   | ['AMCOR_2023_10K p52', 'AMCOR_2023Q4_EARNINGS p2', 'AMCOR_2023Q2_10Q p31']           | ['AMCOR_2023_10K p64']        |
| financebench_id_01936 | novel-generated   | ['AMCOR_2023Q2_10Q p31', 'AMCOR_2023Q2_10Q p9', 'AMCOR_2023_10K p69']                | ['AMCOR_2023Q2_10Q p15']      |
| financebench_id_01930 | novel-generated   | ['AMCOR_2023Q4_EARNINGS p2', 'AMCOR_2023Q4_EARNINGS p2', 'AMCOR_2023Q4_EARNINGS p1'] | ['AMCOR_2023Q4_EARNINGS p10'] |
| financebench_id_00222 | domain-relevant   | ['AMD_2015_10K p3', 'AMD_2015_10K p36', 'AMD_2015_10K p4']                           | ['AMD_2022_10K p56']          |
| financebench_id_01279 | domain-relevant   | ['AMD_2015_10K p34', 'AMD_2015_10K p36', 'AMD_2015_10K p3']                          | ['AMD_2022_10K p58']          |
