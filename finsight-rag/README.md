# FinSight RAG

**Grounded, cited question answering over SEC financial filings.** Runs fully on a MacBook with local
embeddings and a local LLM; Claude is optional, by config.

> Status: **P0 (scaffold) complete**. See the [progress tracker](docs/DEVELOPMENT_GUIDE.md#6-progress-tracker).
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

Further steps are added phase by phase (`make data`, `make ingest`, `make index`, ...).

## Project layout (so far)

```
configs/          base.yaml (defaults) + local.yaml / cloud.yaml profiles
src/finsight/     the package: config, schemas, errors, container, CLI, observability, diagnostics
scripts/          hand-run learning steps (00_check_env.py, ...)
tests/            unit tests (more integration tests from P4 on)
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
