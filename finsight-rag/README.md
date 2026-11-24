# FinSight RAG

**Grounded, cited question answering over SEC financial filings.** Runs fully on a MacBook with local
embeddings and a local LLM; Claude is optional, by config.

> Status: **P7 (advanced RAG) complete**: query rewriting (multi-query, HyDE, decomposition), corrective and agentic pipelines, contextual chunk headers, and a second benchmark (MultiHop-RAG, cross-document + unanswerable questions). Query and reranker caches make evaluations fast and reproducible. Retrieval (P5): gold page in the top 10 for 61% of FinanceBench questions ([`docs/evaluation.md`](docs/evaluation.md)).
> Results tables and a demo arrive in later phases.

## Docs

| Doc | For |
|---|---|
| [`docs/PROJECT_SPEC.html`](docs/PROJECT_SPEC.html) | What and why: problem, dataset, architecture, roadmap (open in a browser) |
| [`docs/DEVELOPMENT_GUIDE.md`](docs/DEVELOPMENT_GUIDE.md) | How to build it, phase by phase, with a hand-run script per step |
| [`docs/INTERVIEW_GUIDE.md`](docs/INTERVIEW_GUIDE.md) | Concepts, most-asked interview questions, what to study further |

## Quickstart

Requires macOS (Apple Silicon recommended), [uv](https://docs.astral.sh/uv/), and, from P6 on, [Ollama](https://ollama.com/download).

```bash
cd finsight-rag
uv python install 3.12
make setup                 # install dependencies + git hooks
cp .env.example .env       # optional: add API keys for the cloud profile
make check-env             # step 00: verify Python, Apple GPU (MPS), packages, Ollama, keys
make test                  # fast test suite
```

```bash
make data                                   # step 01: download FinanceBench (~166 MB, ~25 s)
uv run python scripts/02_explore_data.py   # step 02: explore questions, evidence and consistency checks
uv run python scripts/03_parse_pdfs.py     # step 03: parse PDFs into clean pages + evidence check (~70 s)
uv run python scripts/04_chunk.py          # step 04: chunk pages (4 strategies), quality stats; --sweep compares all
uv run python scripts/05_embed_and_index.py  # step 05: embed chunks (~5 min first time), build Qdrant + BM25, sanity queries
uv run python scripts/05b_compare_embedders.py   # step 05b: compare embedders (local vs online) on the same chunks
uv run python scripts/06_retrieve.py            # step 06: one question through BM25 / dense / hybrid / rerank
uv run python scripts/07_eval_retrieval.py      # step 07: the retrieval ablation (~30 min with rerank variants)
uv run python scripts/08_generate.py --show-prompt   # step 08: a cited answer, the exact prompt, timings
uv run finsight query "What was 3M's FY2018 capex?"  # the same from the CLI (finsight llms: which LLM)
uv run python scripts/09_advanced_rag.py --technique multi_query   # step 09: query rewriting / corrective / agentic
```

### Choosing the embedding model (local or online)

```bash
uv run finsight embedders                    # presets, key status, free/paid, requests/time/quota estimates
echo "GEMINI_API_KEY=..." >> .env            # any free key you have (see .env.example for the links)
uv run finsight index --embedder gemini-001 --limit-questions 30   # its own collection; nothing overwritten
make compare-embedders E=bge-small,gemini-001 Q=30   # compare on 30 questions' filings (free-tier friendly)
```

### Run any step three ways

| How | Command |
|---|---|
| Terminal, all at once | `uv run python scripts/01_download_data.py --limit-docs 3` |
| VS Code, cell by cell | open a script, click **Run Cell** above each `# %%`; variables stay in the Interactive Window |
| Jupyter | `make notebooks && make jupyter`, then open `notebooks/01_download_data.ipynb` |
| VS Code notebook | `make kernel` once, open a `.ipynb`, **Select Kernel** → Jupyter Kernel → **Python 3.12 (finsight-rag)** |

Each cell runs one internal step and shows the result as a pandas DataFrame.

Further steps are added phase by phase. `make ingest` parses and chunks on all cores; `make index` embeds and builds both indexes (skipped when up to date). Dataset details: [`docs/data_card.md`](docs/data_card.md).

## Project layout (so far)

```
configs/          base.yaml (defaults) + local.yaml / cloud.yaml profiles
src/finsight/     the package: config, schemas, container, CLI, data, ingestion, chunking, embedding, indexing
scripts/          cell-based learning steps (00_check_env … 09_advanced_rag)
notebooks/        the same steps as Jupyter notebooks (generated: make notebooks)
tests/            unit, integration (fake models) and slow (real data/models) tests
docs/             spec, development guide, interview guide
```

## Configuration in one paragraph

`configs/base.yaml` holds every default. A profile (`configs/local.yaml` by default, or `configs/cloud.yaml`)
is deep-merged on top, then environment variables (`LLM_PROVIDER=anthropic`, or any
`FINSIGHT__SECTION__KEY=value`), then overrides passed in code. API keys come only from the environment
or `.env`. `uv run finsight show-config` prints the resolved config and its **fingerprint**, a hash
stored with every index and eval result so each number traces back to its exact settings.

## License notes

FinanceBench is CC-BY-NC-4.0 (non-commercial). The PDFs are downloaded by script and never committed.
PyMuPDF is AGPL-licensed.
