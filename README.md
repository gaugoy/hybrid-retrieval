# hybrid-retrieval

This repository has two parts:

| Folder | What it is |
|---|---|
| `/` (this folder: `1-explore-data.py` … `6-evaluate.py`, `utils/`) | **Learning experiment.** BM25, dense retrieval, RRF fusion, reranking and evaluation, each built by hand on FiQA to understand it on its own |
| [`finsight-rag/`](finsight-rag/) | **Production project.** FinSight RAG: grounded, cited Q&A over SEC filings (FinanceBench), with measured ablations, an API + UI, fallbacks, caches, CI and docs. Start at [`finsight-rag/README.md`](finsight-rag/README.md) |

The two are independent: `finsight-rag/` doesn't import from the experiment.
