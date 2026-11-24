# contextual_headers

- Questions: 150 (FinanceBench) · relevant = a chunk on a gold evidence page
- Values: mean ± std over questions. Differences under ~0.05 on 150 questions are noise.
- Best by hit@10 (then MRR): **recursive_384_ctx · dense+filter+year**
- Company extracted from the question: nan% of questions

## All configurations

|                                            | hit@5         | hit@10        | recall@10     | mrr           | ndcg@10       | doc_hit@10    |   p50 ms |   n |
|:-------------------------------------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|---------:|----:|
| ('recursive_384', 'dense+filter+year')     | 0.527 ± 0.501 | 0.613 ± 0.489 | 0.580 ± 0.478 | 0.353 ± 0.385 | 0.385 ± 0.375 | 0.993 ± 0.082 |      112 | 150 |
| ('recursive_384', 'dense (no filter)')     | 0.367 ± 0.484 | 0.467 ± 0.501 | 0.440 ± 0.484 | 0.257 ± 0.359 | 0.281 ± 0.354 | 0.913 ± 0.282 |       10 | 150 |
| ('recursive_384_ctx', 'dense+filter+year') | 0.540 ± 0.500 | 0.673 ± 0.471 | 0.629 ± 0.462 | 0.350 ± 0.365 | 0.400 ± 0.360 | 0.987 ± 0.115 |      105 | 150 |
| ('recursive_384_ctx', 'dense (no filter)') | 0.413 ± 0.494 | 0.567 ± 0.497 | 0.520 ± 0.477 | 0.267 ± 0.332 | 0.314 ± 0.344 | 0.973 ± 0.162 |        9 | 150 |

## hit@10 by question type (recursive_384_ctx)

| question_type     |   dense (no filter) |   dense+filter+year |
|:------------------|--------------------:|--------------------:|
| domain-relevant   |                0.42 |                0.56 |
| metrics-generated |                0.64 |                0.8  |
| novel-generated   |                0.64 |                0.66 |

## 10 hardest questions for the best configuration (recursive_384_ctx · dense+filter+year)

| qid                   | question_type     | top3                                                                           | gold                                       |
|:----------------------|:------------------|:-------------------------------------------------------------------------------|:-------------------------------------------|
| financebench_id_04672 | metrics-generated | ['3M_2018_10K p98', '3M_2018_10K p44', '3M_2018_10K p51']                      | ['3M_2018_10K p58']                        |
| financebench_id_01865 | novel-generated   | ['3M_2022_10K p60', '3M_2022_10K p19', '3M_2022_10K p21']                      | ['3M_2022_10K p25']                        |
| financebench_id_00807 | domain-relevant   | ['3M_2023Q2_10Q p61', '3M_2023Q2_10Q p70', '3M_2023Q2_10Q p83']                | ['3M_2023Q2_10Q p5']                       |
| financebench_id_01858 | novel-generated   | ['3M_2018_10K p43', '3M_2023Q2_10Q p70', '3M_2022_10K p37']                    | ['3M_2023Q2_10Q p62']                      |
| financebench_id_10420 | metrics-generated | ['AES_2022_10K p111', 'AES_2022_10K p153', 'AES_2022_10K p157']                | ['AES_2022_10K p130', 'AES_2022_10K p132'] |
| financebench_id_00799 | domain-relevant   | ['AMCOR_2023_10K p2', 'AMCOR_2023Q4_EARNINGS p1', 'AMCOR_2023_10K p35']        | ['AMCOR_2023_10K p52']                     |
| financebench_id_00684 | domain-relevant   | ['AMCOR_2023Q4_EARNINGS p7', 'AMCOR_2023_10K p35', 'AMCOR_2023Q4_EARNINGS p2'] | ['AMCOR_2023_10K p50']                     |
| financebench_id_01928 | novel-generated   | ['AMCOR_2023_10K p35', 'AMCOR_2023Q4_EARNINGS p1', 'AMCOR_2023Q4_EARNINGS p2'] | ['AMCOR_2023Q4_EARNINGS p12']              |
| financebench_id_00222 | domain-relevant   | ['AMD_2015_10K p73', 'AMD_2015_10K p72', 'AMD_2015_10K p53']                   | ['AMD_2022_10K p56']                       |
| financebench_id_00563 | novel-generated   | ['AMD_2015_10K p4', 'AMD_2022_10K p67', 'AMD_2022_10K p103']                   | ['AMD_2022_10K p48']                       |
