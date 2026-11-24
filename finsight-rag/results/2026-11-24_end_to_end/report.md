# end_to_end

- Questions: 30 · judge: `hf:Qwen/Qwen2.5-1.5B-Instruct` (validate with scripts/10b_label_judge_sample.py)
- numeric_match: any number in the answer within 1% of the gold's first number (numeric golds only).
- Differences under ~0.05 on 150 questions are noise (std of a 0/1 metric ≈ 0.5).

## Summary

| variant    |   n |   numeric_match |   judge correct |   judge partial |   judge errors |   faithfulness |   gold page retrieved |   abstained |   cited |   auto_cited |   p50 s |   p95 s |   tokens/q |   $ total |
|:-----------|----:|----------------:|----------------:|----------------:|---------------:|---------------:|----------------------:|------------:|--------:|-------------:|--------:|--------:|-----------:|----------:|
| naive      |  30 |           0.241 |               1 |               0 |              0 |            nan |                 0.167 |       0.367 |   0.433 |        0.3   |   3.827 |  17.251 |       3837 |         0 |
| baseline   |  30 |           0.31  |               1 |               0 |              0 |            nan |                 0.467 |       0.267 |   0.567 |        0.433 |   4.839 |  11.184 |       3817 |         0 |
| corrective |  30 |           0.276 |               1 |               0 |              0 |            nan |                 0.467 |       0.333 |   0.567 |        0.433 |  12.796 |  50.263 |       3440 |         0 |

## numeric_match by question type

| question_type     |   baseline |   corrective |   naive |
|:------------------|-----------:|-------------:|--------:|
| domain-relevant   |      0.3   |        0.2   |   0.3   |
| metrics-generated |      0.25  |        0.25  |   0.083 |
| novel-generated   |      0.429 |        0.429 |   0.429 |

## Failure gallery (baseline: numeric answers that did not match)

| qid                   | question_type     | gold                                                                             | answer                                                                                                                                                           | gold_page_retrieved   |
|:----------------------|:------------------|:---------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------|
| financebench_id_04672 | metrics-generated | $8.70                                                                            | To calculate the Year End FY2018 Net PPNE for 3M, we need to follow these steps:  1. Identify the Total Debt figure for FY2018 from the Balance Sheet.    - From | False                 |
| financebench_id_00499 | domain-relevant   | No, the company is managing its CAPEX and Fixed Assets pretty efficiently, which | Yes, 3M is a capital-intensive business based on FY2022 data. The company reported a significant amount of net debt relative to its total debt, indicating a hig | False                 |
| financebench_id_01865 | novel-generated   | The consumer segment shrunk by 0.9% organically.                                 | To determine which segment has dragged down 3M's overall growth in 2022 after excluding the impact of M&A, we need to analyze the information provided in the co | False                 |
| financebench_id_00941 | domain-relevant   | Following debt securities registered under 3M's name are listed to trade on the  | Not found in the provided filings.                                                                                                                               | True                  |
| financebench_id_02987 | metrics-generated | 24.26                                                                            | Not found in the provided filings.                                                                                                                               | True                  |
| financebench_id_07966 | metrics-generated | 1.9%                                                                             | To calculate the 3-year average of CapEx as a percentage of Revenue for Activision Blizzard, I will use the following steps:  1. Identify the relevant informati | False                 |
| financebench_id_04735 | metrics-generated | 0.66                                                                             | To calculate the FY2015 operating cash flow ratio for Adobe, we need to determine both the cash from operations and total current liabilities from the provided  | True                  |
| financebench_id_07507 | metrics-generated | 65.4%                                                                            | To calculate Adobe's year-over-year change in unadjusted operating income from FY2015 to FY2016, we need to look at the "Operating Expenses" column in the Incom | True                  |
