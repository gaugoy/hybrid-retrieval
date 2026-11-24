# FinSight RAG — Development Guide

> **Who this is for:** you, the developer, and any coding agent (for example Claude Code) building this project.
> **What it is:** the single source of truth for *how* to build FinSight RAG, one phase at a time.
> Every phase ends with a script you run by hand, output you inspect, and a check that tells you the phase is done.
>
> Companion docs in this folder:
> - [`PROJECT_SPEC.html`](PROJECT_SPEC.html): the *what* and *why* (problem, dataset, architecture, roadmap). Open it in a browser.
> - [`INTERVIEW_GUIDE.md`](INTERVIEW_GUIDE.md): the concepts behind each phase, the questions interviewers ask most, and what to study further.

---

## Table of contents

0. [How to use this guide](#0-how-to-use-this-guide)
1. [Verified facts about the data](#1-verified-facts-about-the-data)
2. [Machine setup](#2-machine-setup-one-time)
3. [Engineering conventions](#3-engineering-conventions)
4. [Core contracts (schemas and interfaces)](#4-core-contracts)
5. [Phase-by-phase build](#5-phase-by-phase-build)
   - [P0 Scaffold](#p0--scaffold--engineering-foundation) · [P1 Data](#p1--data-acquisition--exploration) · [P2 Ingestion](#p2--ingestion-pdf--clean-pages--metadata) · [P3 Chunking](#p3--chunking-strategies) · [P4 Indexing](#p4--embedding--indexing) · [P5 Retrieval](#p5--retrieval--retrieval-evaluation) · [P6 Generation](#p6--grounded-generation) · [P7 Advanced RAG](#p7--advanced-rag-techniques) · [P8 Evaluation](#p8--end-to-end-evaluation-harness) · [P9 Serving](#p9--serving-api--demo-ui) · [P10 Production](#p10--production-hardening--deployment) · [P11 Docs](#p11--documentation--showcase)
6. [Progress tracker (update after every phase)](#6-progress-tracker)
7. [Decision and deviation log](#7-decision-and-deviation-log)
8. [Troubleshooting](#8-troubleshooting)
9. [Reference index](#9-reference-index)

---

## 0. How to use this guide

### For you (the human)

1. Read the phase section here, then the matching section in [`INTERVIEW_GUIDE.md`](INTERVIEW_GUIDE.md).
2. Build the phase yourself, or ask the agent: *"Implement phase P2 from docs/DEVELOPMENT_GUIDE.md."*
3. **Run the phase's step script yourself**: whole from the terminal (`uv run python scripts/NN_*.py`), cell by cell in VS Code (**Run Cell**), or as a notebook (`make notebooks && make jupyter`). Each cell does one internal step and shows its result as a DataFrame (see §3.3).
4. Run the phase's checks (`make test`, plus the phase-specific command).
5. Tick the phase in the [progress tracker](#6-progress-tracker), then commit.

### For a coding agent (rules to follow)

- **Work on one phase at a time**, in order. Don't start phase N+1 until the "Done when" checks for phase N pass.
- **Follow the contracts in §4 exactly.** If a contract has to change, update §4 and add a row to the [decision log](#7-decision-and-deviation-log) in the same change.
- **Keep the learning-script pattern (§3.3):** every phase has a cell-based `scripts/NN_*.py` that runs in the terminal, cell by cell in VS Code, or as a generated notebook. One cell per internal step, results as DataFrames. Library code lives in `src/finsight/`; scripts are thin. Run `make notebooks` after changing a script.
- **No LangChain or LlamaIndex** (ADR-0002). Plain Python plus the libraries listed in §3.
- **Nothing loads at import time:** no module reads data, loads models or opens connections when imported.
- **Local by default:** every phase must work with no API keys. Cloud providers (Anthropic, Cohere) are optional and switched on by config.
- **Never commit** `.env`, `data/`, `indexes/` or model weights. Never print full API keys (mask them as `sk-...abcd`).
- **When a phase ends:** update §6 (status, date, key numbers) and §7 (any deviations). Update the README if the user-facing commands changed. Then propose a commit message and wait. Commit or push only when the user asks.
- **Verify, don't guess,** third-party APIs. If a library call in this guide fails because the API changed, check the library's current docs (links in §9) and record the fix in §7.

---

## 1. Verified facts about the data

Checked on 2026-11-23 against the live sources, so code can rely on them.

| Fact | Value |
|---|---|
| Q&A file | `https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_open_source.jsonl` (~0.9 MB, **150 rows**) |
| Document metadata file | `https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_document_information.jsonl` (**361 rows**) |
| PDF URL pattern | `https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{doc_name}.pdf` |
| PDFs in the repo | 368 files, about 705 MB in total. **We download only the 84 the questions reference, about 165 MB.** |
| Doc types among those 84 | 64 × `10k`, 8 × `10q`, 6 × `8k`, 6 × `Earnings` |
| Question types | 50 × `metrics-generated`, 50 × `domain-relevant`, 50 × `novel-generated` |
| Evidence per question | 1 item: 115 questions · 2 items: 31 · 3 items: 4 |
| Q&A row keys | `financebench_id, company, doc_name, question_type, question_reasoning, domain_question_num, question, answer, justification, dataset_subset_label, evidence` |
| Evidence item keys | `evidence_text, doc_name, evidence_page_num, evidence_text_full_page` |
| Doc-info keys | `doc_name, company, gics_sector, doc_type, doc_period, doc_link` |
| **Page indexing** | `evidence_page_num` is **0-based** and matches PyMuPDF's `doc[i]`. Example: `3M_2018_10K` evidence page 59 = PyMuPDF index 59 = page 60 as printed. |
| License | CC-BY-NC-4.0: non-commercial use only; don't commit the PDFs |
| Secondary dataset (P7) | MultiHop-RAG: `https://github.com/yixuantt/MultiHop-RAG`. 609 articles, 2,556 queries (816 inference, 856 comparison, 583 temporal, 301 null). Check the file names in the repo when you reach P7. |

**Page convention used everywhere in this project:** store `page_idx` as **0-based** (it matches the PDF library and the ground truth). Show `page_idx + 1` only to humans (UI, citations).

---

## 2. Machine setup (one-time)

Target machine: **MacBook with Apple Silicon (M5) and 16 GB RAM**, macOS. `uv` and Docker are already installed.

```bash
# 1. Python 3.12 managed by uv (the system python is 3.14, which is too new for some ML wheels)
uv python install 3.12

# 2. Ollama (local LLM server). Install the app from https://ollama.com/download, or use Homebrew:
brew install ollama
ollama serve &                      # the desktop app starts this for you
ollama pull qwen2.5:7b-instruct     # about 4.7 GB, 4-bit; fits in 16 GB next to everything else
ollama run qwen2.5:7b-instruct "Say hi in 5 words"   # smoke test

# 3. GitHub CLI (optional, for PRs)
brew install gh && gh auth login

# 4. Docker Desktop must be running for Qdrant-in-Docker (P10). In dev we use Qdrant's embedded mode instead.
```

**Memory budget on 16 GB:** Ollama 7B (~5–6 GB) + embedding model (~0.5 GB) + reranker (~1.1 GB) + Qdrant embedded + Python is fine. Close heavy apps during full indexing. If memory is tight, use `qwen2.5:3b-instruct`.

---

## 3. Engineering conventions

### 3.1 Dependencies (`pyproject.toml`)

```toml
[project]
name = "finsight"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "pydantic>=2.7", "python-dotenv>=1.0", "pyyaml>=6.0",
  "typer>=0.12", "rich>=13.7", "httpx>=0.27", "tenacity>=8.3", "tqdm>=4.66",
  "numpy>=1.26", "pandas>=2.2", "pyarrow>=16.0",
  "pymupdf>=1.24",
  "sentence-transformers>=3.0", "torch>=2.3",
  "qdrant-client>=1.10", "bm25s>=0.2.13",
  "anthropic", "fastapi>=0.111", "uvicorn[standard]>=0.30",
]
[project.optional-dependencies]
ui = ["streamlit>=1.36"]
cohere = ["cohere>=5.13"]
[dependency-groups]
dev = ["pytest>=8", "pytest-cov", "ruff>=0.5", "mypy>=1.10", "pre-commit", "types-PyYAML", "ipykernel"]

[project.scripts]
finsight = "finsight.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/finsight"]

[tool.ruff]
line-length = 100
target-version = "py312"
[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]

[tool.mypy]
python_version = "3.12"
strict = false
disallow_untyped_defs = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: needs real models or the full data (deselected in CI)"]
addopts = "-m 'not slow'"
```

### 3.2 Code style

- Type hints on every function. Use `from __future__ import annotations` where helpful.
- **Docstrings explain why, not just what** (the teaching style of the experiment). Each module opens with a 3–8 line docstring covering what concept the module implements, why it exists, and a "More info:" link.
- Use pydantic models for data that crosses a module boundary (§4.1). Use plain dataclasses for internal state.
- Use the `logging` module through `finsight.observability.logging.get_logger(__name__)`. Never `print` in library code; scripts may use `rich` to print.
- Raise specific errors: define `FinsightError` with subclasses `ConfigError`, `DataNotFoundError`, `IndexNotBuiltError`, `ProviderError` in `src/finsight/errors.py`. Error messages say **what to run** to fix the problem (for example `"BM25 index missing. Run: uv run python scripts/05_embed_and_index.py"`).
- Paths always come from `Settings`. Never hard-code paths.

### 3.3 The step-script pattern (cells: terminal, VS Code, or Jupyter)

Every `scripts/NN_name.py` is a **cell-based script** (jupytext "percent" format). The same file runs three ways:

| How | Command | Good for |
|---|---|---|
| Terminal, all at once | `uv run python scripts/NN_name.py --flag value` | quick runs, automation |
| VS Code, cell by cell | open the file and click **Run Cell** above each `# %%` (Interactive Window) | stepping through, inspecting variables |
| Jupyter notebook | `make notebooks` then `make jupyter` (opens `notebooks/NN_name.ipynb`) | exploring with rich DataFrame views |

Rules for every step script:

- **Each cell does one internal step** of the library and shows its result, so you can see what happens in the background.
  Don't hide everything behind one call: expose the stages (e.g. P1 downloads one PDF by hand before the parallel batch).
- **Show data as pandas DataFrames** with `finsight.notebook.show(df, "title")`: HTML in notebooks, plain text in the
  terminal. Use `preview()` to shorten long text and `step("N. title")` to announce each stage.
- **Parameters come from `script_args()`**: flags in the terminal (`limit_docs=Param(None, int)` becomes
  `--limit-docs 3`), the defaults in a notebook (edit the cell to change them).
- **Leave useful variables at the top level** (e.g. `qa_df`, `ev_df`, `report`) and list them in the opening markdown cell.
- **Exit non-zero on failure only in the terminal** (`if not in_notebook(): raise SystemExit(1)`), so a notebook never dies.
- Scripts are thin: all logic lives in `src/finsight/`; `finsight.notebook` is used only by scripts and notebooks.
- `notebooks/*.ipynb` are **generated** (`make notebooks` runs `tools/build_notebooks.py`: no outputs, fixed cell IDs so rebuilds are byte-identical, kernel preset to `finsight-rag` from `make kernel`) and committed for GitHub readers. Never edit them
  by hand; edit the script and regenerate. Ruff lints the scripts, not the notebooks.
- `tests/unit/test_notebook_helpers.py` checks that every step script compiles, opens with a markdown cell, has at
  least 4 cells, and uses `script_args()`.

Template:

```python
# %% [markdown]
# # Step NN — <title>   (Phase PX)
# **What this step does:** ...   **Three ways to run it:** ...   **Variables to explore afterwards:** `df1`, `report`

# %%
import pandas as pd
from finsight.config import load_settings
from finsight.notebook import Param, configure_pandas, in_notebook, preview, script_args, show, step

configure_pandas()

# %% Parameters: in a notebook, edit the defaults here; in the terminal, use the flags.
P = script_args(
    "Step NN: <title>", config=Param(None, str, "profile YAML"), limit=Param(None, int, "...")
)
P

# %% [markdown]
# ## 1. <first internal step>: what happens and why

# %%
step("1. <first internal step>")
settings = load_settings(P.config)
first_df = ...  # call one library function
show(first_df.head(), "first_df.head()")

# %% Summary
step("Summary")
print("Next: scripts/NN+1_....py")
```

The `finsight` CLI (`src/finsight/cli.py`) exposes the same operations as concise production commands (`finsight download`, ...).
**Both call the same library functions.**

### 3.4 Configuration

- `configs/base.yaml` holds the defaults. `configs/local.yaml` and `configs/cloud.yaml` override parts of it. `configs/experiments/*.yaml` define eval matrices.
- `finsight.config.load_settings(path)` deep-merges `base.yaml` ← the given YAML ← environment variables (`FINSIGHT__RETRIEVAL__CANDIDATE_K=100`, with `__` as the nesting separator) ← explicit overrides passed by code.
- `Settings.fingerprint()` returns a short hash of the resolved config. It is stored in every index manifest and every eval result.
- Secrets (`ANTHROPIC_API_KEY`, `COHERE_API_KEY`) are read from the environment or `.env`, **never from YAML**.

### 3.5 Testing

- `tests/unit/`: pure logic (chunkers, RRF, metrics, citation parsing, filter extraction, config merge). Fast, with no models or network.
- `tests/integration/`: the pipeline end to end on `tests/fixtures/` (3 tiny text "documents" and 5 Q&A rows), using **`FakeEmbedder`** (deterministic hashing vectors) and **`FakeLLM`** (returns a canned answer citing `[1]`). These run in CI.
- `@pytest.mark.slow`: needs real models or the full data. Run locally with `uv run pytest -m slow`.

### 3.6 Makefile targets

`make help` lists the real targets; the Makefile is the source of truth. As of P7:

| Target | Runs | Phase |
|---|---|---|
| `make setup` / `make hooks` | `uv sync --all-extras`, pre-commit hooks (installed from the git root) | P0 |
| `make check-env` | `scripts/00_check_env.py` | P0 |
| `make lint` / `make fmt` / `make test` / `make test-slow` | ruff + mypy · auto-fix · fast tests with coverage · real-data tests | P0 |
| `make notebooks` / `make kernel` / `make jupyter` | `tools/build_notebooks.py` · register the `.venv` kernel · JupyterLab | P1 |
| `make data` | `scripts/01_download_data.py` | P1 |
| `make ingest` | `finsight ingest --workers 0`: parse PDFs on all cores, then chunk with the configured strategy | P2 + P3 |
| `make chunk-sweep` | `finsight chunk --sweep`: every strategy × size, report in `results/` | P3 |
| `make index` | `finsight index`: chunk if needed, embed (cached), build Qdrant + BM25; skipped when up to date (`EMBEDDER=<preset> make index`) | P4 |
| `make embedders` / `make compare-embedders E=a,b Q=30` | the preset catalog (keys, cost, time) · dense-retrieval comparison, report in `results/` | P4 |
| `make eval-retrieval` | `finsight eval`: the retrieval ablation (all chunkers × variants, 150 Q), report in `results/` | P5 |
| `uv run finsight query "..."` / `uv run finsight llms` | a cited answer · the LLM presets and what auto picks | P6 |
| `make eval-advanced` / `make multihop` | query-transform + contextual experiments · download+index+evaluate MultiHop-RAG | P7 |
| `uv run finsight sample` | pipelines (baseline vs corrective vs agentic) on a seeded question sample | P7 |
| `make eval`, `make serve`, `make ui` | placeholders until P8, P9 | later |

Learning scripts (`scripts/NN_*.py`) run single-process and print everything; `make` targets call the concise
`finsight` CLI, which can use all cores (see P2 on why processes and cell-based scripts don't mix on macOS).

---

## 4. Core contracts

These are the backbone of the code. Write them in P0 and change them only through the decision log.

### 4.1 Schemas — `src/finsight/schemas.py`

```python
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class DocumentMeta(BaseModel):
    doc_name: str  # "3M_2018_10K" (primary key, also the PDF file stem)
    company: str  # "3M"
    doc_type: str  # "10k" | "10q" | "8k" | "Earnings"
    doc_period: int  # 2018
    gics_sector: str | None = None
    source_url: str | None = None


class Page(BaseModel):
    doc_name: str
    page_idx: int  # 0-based (see §1)
    text: str  # cleaned text, tables rendered as pipe rows
    n_chars: int
    has_table: bool = False


class Chunk(BaseModel):
    chunk_id: str  # f"{doc_name}::p{page_start}-{page_end}::c{seq}"
    doc_name: str
    company: str
    doc_type: str
    doc_period: int
    page_start: int  # 0-based, inclusive
    page_end: int  # 0-based, inclusive
    text: str  # what the LLM and the user see
    embed_text: str  # what gets embedded / BM25-indexed (may have a contextual header)
    n_tokens: int
    strategy: str  # "fixed" | "recursive" | "page_aware" | "parent_child"
    parent_id: str | None = None  # parent_child only


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    rank: int  # 1-based within this stage
    stage: Literal["dense", "sparse", "hybrid", "rerank", "final"]
    scores_by_stage: dict[str, float] = Field(
        default_factory=dict
    )  # for the UI "under the hood" panel


class QueryFilters(BaseModel):
    companies: list[str] = Field(default_factory=list)
    doc_periods: list[int] = Field(default_factory=list)
    doc_types: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    marker: int  # the [n] used in the answer text
    chunk_id: str
    doc_name: str
    page_idx: int
    snippet: str  # first ~300 chars of the chunk


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    abstained: bool
    retrieved: list[RetrievedChunk]
    pipeline: str
    provider: str
    model: str
    timings_ms: dict[
        str, float
    ]  # {"filter": .., "dense": .., "sparse": .., "fuse": .., "rerank": .., "generate": .., "total": ..}
    usage: dict[str, int] = Field(default_factory=dict)  # input_tokens, output_tokens
    request_id: str


class Evidence(BaseModel):
    doc_name: str
    page_idx: int  # 0-based
    text: str


class EvalRecord(BaseModel):
    qid: str  # financebench_id
    question: str
    gold_answer: str | None  # None = unanswerable (MultiHop null queries)
    question_type: str
    company: str | None
    doc_name: str | None
    evidence: list[Evidence]
    dataset: Literal["financebench", "multihop"]
```

### 4.2 Interfaces (ABCs)

```python
# chunking/base.py
class Chunker(ABC):
    name: str

    @abstractmethod
    def chunk(self, pages: list[Page], meta: DocumentMeta) -> list[Chunk]: ...


# embedding/base.py   (P4: slug, provider, embed_queries, preferred_batch added — see §7)
class Embedder(ABC):
    model_name: str  # identity of the vectors: cache keys + index manifest
    slug: str  # file / collection name part, e.g. "bge-small-en-v1-5", "gemini-gemini-embedding-001-768"
    dim: int
    provider: str = "local"

    @abstractmethod
    def embed_documents(
        self, texts: list[str], progress: bool = False
    ) -> np.ndarray: ...  # (n, dim) float32, L2-normalized (document_template applied)
    @abstractmethod
    def embed_query(
        self, text: str
    ) -> np.ndarray: ...  # (dim,), L2-normalized (query_template applied)
    def embed_queries(
        self, texts: list[str]
    ) -> np.ndarray: ...  # batched; online: one request + cache


# retrieval/base.py
class Retriever(ABC):
    name: str

    @abstractmethod
    def retrieve(
        self, query: str, k: int, filters: QueryFilters | None = None
    ) -> list[RetrievedChunk]: ...


class Reranker(ABC):
    name: str

    @abstractmethod
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]: ...


# generation/llm/base.py
class LLMResponse(BaseModel):
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None


class LLM(ABC):
    provider: str
    model: str

    @abstractmethod
    def generate(
        self, system: str, messages: list[dict], max_tokens: int = 1024, json_mode: bool = False
    ) -> LLMResponse: ...
    @abstractmethod
    def stream(
        self, system: str, messages: list[dict], max_tokens: int = 1024
    ) -> Iterator[str]: ...


# pipelines/base.py
class RAGPipeline(ABC):
    name: str

    @abstractmethod
    def answer(self, question: str, filters: QueryFilters | None = None) -> Answer: ...
    def retrieve_only(
        self, question: str, filters: QueryFilters | None = None
    ) -> list[RetrievedChunk]:
        """Used by the retrieval eval (no LLM cost)."""
```

### 4.3 Registries

Each family has a `registry.py` with a decorator:

```python
CHUNKERS: dict[str, type[Chunker]] = {}


def register_chunker(name: str):
    def deco(cls):
        CHUNKERS[name] = cls
        cls.name = name
        return cls

    return deco
```

`finsight.container.Container(settings)` builds objects by name from config, lazily, and caches them (`functools.cached_property`). The API, the CLI, the scripts and the eval runner all get their objects from `Container`. **That is how the scores you report match what the API serves.**

### 4.4 Artifact layout (all paths come from `Settings.paths`)

```
data/raw/financebench/financebench_open_source.jsonl
data/raw/financebench/financebench_document_information.jsonl
data/raw/pdfs/{doc_name}.pdf
data/processed/pages.parquet                    # one row per Page
data/processed/documents.parquet                # one row per DocumentMeta
data/chunks/{strategy}_{chunk_tokens}.parquet   # one row per Chunk
indexes/qdrant/                                 # embedded Qdrant storage
indexes/bm25/{strategy}_{chunk_tokens}/         # bm25s.save() + chunk_ids.txt
indexes/embed_cache/{model_slug}.sqlite         # sha256(model + text) -> vector bytes
indexes/manifest.json                           # see P4
results/{date}_{experiment}/                    # per-question JSONL + report.md + config.yaml
```

---

## 5. Phase-by-phase build

Each phase lists: **Goal → Concepts → Files → Step script → Done when → Pitfalls → References → Docs to update**.

---

### P0 — Scaffold & engineering foundation

**Goal:** an empty but professional project: install, lint, test and CI all work, and a config system is in place.
**Concepts:** src layout, typed config, 12-factor config, dependency injection. See Interview Guide §P0.

**Files**
```
finsight-rag/
  pyproject.toml  Makefile  .env.example  .gitignore  .pre-commit-config.yaml  README.md  CLAUDE.md
  configs/base.yaml  configs/local.yaml  configs/cloud.yaml
  src/finsight/__init__.py  config.py  errors.py  schemas.py  container.py  cli.py
  src/finsight/observability/__init__.py  logging.py  timing.py
  scripts/00_check_env.py
  tests/unit/test_config.py  tests/conftest.py
../.github/workflows/finsight-ci.yml     # at the REPO ROOT (see the pitfalls below)
```

**Implementation notes**
- `config.py`: pydantic `Settings` with nested models `PathsCfg, ChunkingCfg, EmbeddingCfg, VectorStoreCfg, RetrievalCfg, RerankCfg, GenerationCfg, LLMCfg, JudgeCfg, ApiCfg, LoggingCfg` (unknown keys are rejected). `load_settings(path)` implements the merge order in §3.4 with an explicit deep merge plus `python-dotenv` (see §7). `fingerprint()` = `sha256(json.dumps(model_dump(exclude={"secrets"}), sort_keys=True))[:12]`.
- `observability/logging.py`: `configure_logging(level, json: bool)` and `get_logger()`. JSON format includes `ts, level, logger, msg, request_id`, plus any extra fields. A `contextvars` variable holds the request ID.
- `observability/timing.py`: a `Timer` context manager that records `{stage: ms}` into a dict: `with timer.stage("dense"): ...`.
- `.gitignore`: `.env`, `data/`, `indexes/`, `results/**/*.jsonl` (keep `report.md`), `.venv/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`, `*.egg-info`.
- `.env.example`: copy it from the spec (§8 in `PROJECT_SPEC.html`).
- `CLAUDE.md` (project root): 10 lines pointing agents to this guide and its §0 rules.
- **CI** (`hybrid-retrieval/.github/workflows/finsight-ci.yml`):
  ```yaml
  name: finsight-ci
  on:
    push: { paths: ["finsight-rag/**", ".github/workflows/finsight-ci.yml"] }
    pull_request: { paths: ["finsight-rag/**"] }
  defaults: { run: { working-directory: finsight-rag } }
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v7
        - uses: astral-sh/setup-uv@v10.4.0   # full tag: no moving `v10` tag exists
        - run: uv python install 3.12
        - run: uv sync --all-extras
        - run: uv run ruff check . && uv run ruff format --check .
        - run: uv run mypy src
        - run: uv run pytest
  ```

**Step script `scripts/00_check_env.py`.** It prints a table: Python version, torch version, MPS available, resolved config path and fingerprint, data and index folders (exist or missing), Ollama reachable (`GET {OLLAMA_BASE_URL}/api/tags`, listing the model names), `ANTHROPIC_API_KEY` set (masked), `COHERE_API_KEY` set (masked). Exits non-zero only on hard failures (for example, an invalid config).

**Done when**
- [x] `make setup && make lint && make test` pass
- [x] `uv run python scripts/00_check_env.py` prints the table; `uv run finsight --help` lists the commands (stubs are fine)
- [x] CI is green on GitHub after a push

**Pitfalls**
- GitHub only reads workflows from the repo-root `.github/workflows/`. The `working-directory` default handles the subfolder.
- The root `hybrid-retrieval/.gitignore` already ignores `*.parquet`, `indexes/` and `.env`, which is consistent with ours.
- Don't put secrets in YAML. `fingerprint()` must exclude secrets.

**References:** [uv docs](https://docs.astral.sh/uv/) · [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) · [Typer](https://typer.tiangolo.com/) · [12-Factor config](https://12factor.net/config) · [Ruff](https://docs.astral.sh/ruff/) · [GitHub Actions working-directory](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#defaultsrun)

**Docs to update:** README quickstart (setup commands), §6 tracker.

---

### P1 — Data acquisition & exploration

**Goal:** a repeatable download of FinanceBench (Q&A, document metadata, the 84 referenced PDFs) and a first look at the data.
**Concepts:** benchmark anatomy (corpus, queries, qrels), data cards, licensing. See Interview Guide §P1.

**Files**
```
src/finsight/data/__init__.py
src/finsight/data/download.py        # download_financebench(settings, limit_docs=None) -> DownloadReport
src/finsight/data/financebench.py    # load_eval_records(), load_document_meta()
scripts/01_download_data.py
scripts/02_explore_data.py
notebooks/01_explore_financebench.ipynb   # optional, for richer plots
docs/data_card.md
tests/unit/test_financebench_loader.py    # uses a 3-row fixture JSONL
```

**Implementation notes**
- `download.py`:
  1. Use `httpx` with a timeout and `tenacity` retries (3 attempts, exponential backoff). Download both JSONL files to `data/raw/financebench/`.
  2. Read the set of `doc_name`s referenced by the 150 questions, from both `row["doc_name"]` and every `evidence[*]["doc_name"]`.
  3. For each name, download `pdfs/{doc_name}.pdf` to `data/raw/pdfs/` if it isn't there yet. Write to a `.part` file, then rename (atomic). Check that the file starts with `%PDF`.
  4. Return a `DownloadReport(n_questions, n_docs_needed, n_downloaded, n_skipped, total_mb, failures)`.
- `financebench.py`:
  - `load_document_meta() -> dict[str, DocumentMeta]` from the document-information JSONL. Map `doc_link` to `source_url`.
  - `load_eval_records() -> list[EvalRecord]`. Map `evidence_page_num` to `Evidence.page_idx` directly (it is already 0-based, §1).
- `scripts/01_download_data.py` prints the `DownloadReport`, plus a table of doc_type counts. Expect 84 docs and ~165 MB.
- `scripts/02_explore_data.py` prints:
  - question_type counts and the top 10 companies by question count
  - for 3 random questions: the question, gold answer, doc_name, evidence page (shown as 0-based and as printed), and the first 300 chars of the evidence text
  - the distribution of evidence items per question
- `docs/data_card.md` covers: source, license, counts (from the script), schema, page convention, known quirks you find, and how evaluation uses it.

**Done when**
- [x] Re-running `01_download_data.py` skips everything (idempotent)
- [x] 84/84 PDFs are present; the data card has the real numbers
- [x] `test_financebench_loader.py` passes

**Pitfalls:** don't use `datasets.load_dataset` here (a direct JSONL download is simpler and has no extra dependency). Some `doc_name`s contain characters that need URL quoting (use `urllib.parse.quote`).

**References:** [FinanceBench paper (arXiv 2311.11944)](https://arxiv.org/abs/2311.11944) · [FinanceBench GitHub](https://github.com/patronus-ai/financebench) · [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) · [BEIR paper (benchmark format)](https://arxiv.org/abs/2104.08663)

---

### P2 — Ingestion: PDF → clean pages + metadata

**Goal:** turn every PDF into clean per-page text, with tables kept readable, plus document metadata. **Prove the parsing is correct** against the ground truth.
**Concepts:** PDF text extraction, table handling, boilerplate removal, "garbage in, garbage out". See Interview Guide §P2.

**Files (as built)**
```
src/finsight/ingestion/__init__.py
src/finsight/ingestion/pdf_parser.py   # words → rows → lines: extract_words, group_rows, render_row, parse_pdf, plain_text
src/finsight/ingestion/cleaner.py      # find_boilerplate, clean_page, join_hyphenation, clean_document
src/finsight/ingestion/quality.py      # evidence_overlap (space-insensitive), label_number_adjacency, count_*
src/finsight/ingestion/pipeline.py     # ingest_all -> IngestReport; pages/documents parquet + manifest; evidence_check
scripts/03_parse_pdfs.py               # cell-based walkthrough of one page, then the whole corpus
tests/unit/test_ingestion_units.py     # rows, currency, boilerplate, hyphenation, metrics
tests/integration/test_ingest_pipeline.py  # generated PDFs: rows kept, header/footer removed, skip/force
tests/slow/test_evidence_pages.py      # real data: ≥ 95% of evidence on the gold page, no offsets
```

**Implementation notes (as built; see §7 for why this differs from the first draft)**
- Use `import pymupdf` (not `fitz`). Text comes from `page.get_text("words")` (word + bounding box, ~3 ms/page).
- **Row reconstruction instead of `find_tables()`:** group words into visual rows (|Δy| ≤ `row_tolerance_factor` × line height),
  sort each row left → right, and insert ` | ` where the gap exceeds `column_gap_factor` × line height. A currency symbol
  always attaches to the number after it (`| $ 5,363`). A vertical gap > `paragraph_gap_factor` × line height becomes a blank line.
  `find_tables()` was measured at 1.5 s/page (≈5 h for the corpus) with cells that lost their spaces.
- `cleaner.py`: header/footer = a **short** line (≤ 12 words), digits normalised to `#`, found in the top/bottom
  `boilerplate_edge_lines` of ≥ `boilerplate_min_share` of the document's pages. Body repeats are never removed. Hyphenated
  words are re-joined only when the next line starts lowercase and the line isn't a table row. Empty pages are kept.
- `has_table` = at least `min_table_rows` lines with ≥ 2 cells and a number (a heuristic; cover pages with addresses can trigger it).
- Settings live in `configs/base.yaml → ingestion` (part of the fingerprint). `PARSER_VERSION` in `pipeline.py` must be bumped
  whenever parsing or cleaning logic changes, so the manifest forces a rebuild.
- Metadata comes from `load_document_meta()` (P1), not from file names.
- `workers=1` (default) is sequential, ~70 s for the corpus, and safe in notebooks and cell-based scripts. `workers=0` (all cores)
  or `>1` uses processes, ~17 s; use it only from guarded entry points (`finsight ingest`, `make ingest`), because macOS
  starts workers by re-importing the caller's main module.

**Step script `scripts/03_parse_pdfs.py [--doc D --page-idx N] [--limit-docs N] [--force]`.** One page through every stage
(plain baseline → words → rows → boilerplate → cleaned page), then the whole corpus, the outputs, the evidence check, and a
plain-vs-ours label–number adjacency comparison on the evidence pages.

**Done when**
- [x] `pages.parquet` exists, with one row per PDF page for all 84 docs (12,013 rows)
- [x] The evidence check passes for **≥ 95%** of evidence items: measured **100%** (189/189, mean overlap 0.998, 0 page offsets)
- [x] The cleaner's unit tests pass (header removal, hyphenation), plus integration tests on generated PDFs

**Pitfalls:** the 0-based vs. printed page confusion (§1). Two-column prose can merge into one row (rare in SEC filings). Don't drop empty pages. Comparing evidence text space-sensitively (FinanceBench's text lost spaces in places). Using `ProcessPoolExecutor` from a script without a `__main__` guard on macOS.

**References:** [PyMuPDF text extraction](https://pymupdf.readthedocs.io/en/latest/recipes-text.html) · [PyMuPDF tables](https://pymupdf.readthedocs.io/en/latest/page.html#Page.find_tables) · [pypdf (BSD alternative)](https://pypdf.readthedocs.io/) · [SEC EDGAR filing types](https://www.sec.gov/edgar/search-and-access)

---

### P3 — Chunking strategies

**Goal:** four interchangeable chunkers that produce `Chunk`s with complete metadata, plus statistics to compare them.
**Concepts:** chunk size vs. precision/recall, overlap, structure-aware chunking, small-to-big retrieval. See Interview Guide §P3.

**Files (as built)**
```
src/finsight/chunking/tokenizer.py     # HFTokenizer (bge WordPiece, offsets), WhitespaceTokenizer (tests), get_tokenizer
src/finsight/chunking/base.py          # DocText (doc text + page map + token offsets), split_spans, merge_spans, Chunker ABC
src/finsight/chunking/registry.py      # @register_chunker, build_chunker(cfg, tokenizer, reserve_tokens)
src/finsight/chunking/fixed.py         # FixedTokenChunker: token windows, stride = size - overlap
src/finsight/chunking/recursive.py     # RecursiveChunker: paragraph → line → sentence → word, then pack
src/finsight/chunking/page_aware.py    # PageAwareChunker: per page + carry table column headers; is_column_header
src/finsight/chunking/parent_child.py  # ParentChildChunker: page-aware children + page parents (parents())
src/finsight/chunking/contextual.py    # add_context_header → embed_text (evaluated in P7)
src/finsight/chunking/pipeline.py      # chunk_corpus (+ manifest skip), chunk_stats, evidence_coverage, run_sweep, save_sweep_report
scripts/04_chunk.py                    # cell-based: tokens → one page × 4 strategies → overlap → header carry → corpus → sweep
tests/unit/test_chunkers.py  tests/integration/test_chunk_pipeline.py
results/2026-11-24_chunking_sweep/report.md   # the comparison table (committed)
```

**Implementation notes (as built)**
- Tokens are counted with the **embedding model's tokenizer** (`BAAI/bge-small-en-v1.5`, WordPiece, fast). bge accepts 512 tokens
  **including** `[CLS]`/`[SEP]`, so the limit for text is **510** (`config.MAX_EMBED_TOKENS`, enforced on `chunk_tokens`).
- `DocText` tokenizes each document once and stores token start offsets, so the token count of any span is a binary search.
  Chunk sizes are therefore exact, and a whole-corpus run takes ~10 s per strategy.
- Chunk IDs: `{doc}::p{page_start}-{page_end}::c{seq}`; parents: `{doc}::p{p}-{p}::parent`. Deterministic.
- **fixed:** windows over the whole document (ignores pages and words). **recursive:** natural boundaries, still crosses pages.
  **page_aware:** never crosses a page (`page_start == page_end`, exact citations). **parent_child:** page-aware children of
  `child_tokens` (overlap ≤ child/4), parents = the page truncated to `parent_tokens`, in `<name>_parents.parquet`.
- **Table header carry** (`chunking.carry_table_header`, default on): a page-aware chunk that contains table rows but not their
  column header (`(Millions) | 2018 | 2017 | 2016`, `December 31, 2020 | ...`, `Fiscal 2023`, `Q4 2023`) gets the header prepended.
  Pages with a header row are packed with room reserved for it, so it always fits.
- Contextual headers (`chunking.contextual_headers`) reduce the budget by 24 tokens and write `embed_text` only; measured in P7.
- Output names: `<strategy>_<size>[_ctx].parquet` with a `.manifest.json` (inputs hash: ingestion output + chunking settings +
  tokenizer + `CHUNKER_VERSION`). Bump `CHUNKER_VERSION` whenever chunking logic changes.

**Step script `scripts/04_chunk.py [--strategy S --tokens N] [--sweep]`.** Tokens vs words → one page through all 4 strategies →
overlap → header carry on/off → parent/children → corpus chunking with quality stats → (optional) the sweep.

**Done when**
- [x] All 4 chunkers are registered; parquet files are written (12 configurations in the sweep)
- [x] Unit tests: deterministic IDs, overlap respected, page_aware never crosses pages, parent links valid, no chunk over the
  limit (recounted with the real tokenizer: `over_limit_share = 0` for every configuration)
- [x] The stats table is in `docs/evaluation.md` and `results/2026-11-24_chunking_sweep/report.md`

**Pitfalls:** off-by-one in page tracking (tested). Counting with the wrong tokenizer. Forgetting the 2 special tokens. Treating
"evidence in one chunk" as the decision metric: it favours big chunks by construction; P5 decides with retrieval metrics.

**References:** [Pinecone: chunking strategies](https://www.pinecone.io/learn/chunking-strategies/) · [Chroma research: evaluating chunking](https://research.trychroma.com/evaluating-chunking) · [Lost in the Middle (arXiv 2307.03172)](https://arxiv.org/abs/2307.03172) · [HF tokenizers](https://huggingface.co/docs/transformers/main_classes/tokenizer) · [Parent document retrieval](https://python.langchain.com/docs/how_to/parent_document_retriever/)

---

### P4 — Embedding & indexing

**Goal:** dense vectors in Qdrant and a BM25 index, both rebuilt only when their inputs change.
**Concepts:** bi-encoders, normalization and cosine similarity, ANN/HNSW, payload filtering, idempotency. See Interview Guide §P4.

**Files (as built)**
```
src/finsight/embedding/base.py                 # Embedder ABC, build_embedder(cfg) ("fake*" → FakeEmbedder), model_slug, l2_normalize
src/finsight/embedding/sentence_transformer.py # SentenceTransformerEmbedder: lazy model, device auto (mps/cuda/cpu), query instruction
src/finsight/embedding/cache.py                # EmbeddingCache (SQLite), cache_key, embed_with_cache → (vectors, CacheStats)
src/finsight/embedding/fake.py                 # FakeEmbedder(dim=32): hashed word n-grams, deterministic
src/finsight/indexing/manifest.py              # Manifest (indexes/manifest.json), chunks_sha256, needs_rebuild, record
src/finsight/indexing/qdrant_store.py          # open_client, QdrantStore (recreate, upsert_chunks, search, count), build_filter, point_id
src/finsight/indexing/bm25_store.py            # BM25Index (build, save, load, search with mask), tokenize
src/finsight/indexing/pipeline.py              # index_corpus → IndexReport; build_dense, build_sparse; names and paths
scripts/05_embed_and_index.py                  # cell-based: cosine → query instruction → cache → build → BM25 tokens → manifest → sanity → filter → gold rank
tests/unit/test_manifest.py  tests/unit/test_embed_cache.py
tests/integration/test_index_roundtrip.py      # FakeEmbedder + synthetic chunks, embedded Qdrant in a tmp dir
tests/slow/test_index_real.py                  # the real indexes are complete; the 3M gold page is findable
```
The `Container` gained `embedder`, `qdrant_client` (one per process: embedded mode locks its folder), `qdrant`, `collection`,
`bm25` and `close()`. CLI: `finsight index [--strategy S --tokens N --only all|dense|sparse --force]` (chunks first if needed);
`make index`.

**Implementation notes (as built)**
- **Embeddings:** `SentenceTransformer(model, device=...)`, `encode(..., normalize_embeddings=True)`, batch 64. The query
  instruction is passed as `prompt=` to **queries only**. sentence-transformers 6 renamed `get_sentence_embedding_dimension` to
  `get_embedding_dimension` (both handled).
- **Cache:** `key = sha256(model + "\x00" + text)`; texts are de-duplicated before embedding (239 of 31,114 page-aware chunks
  repeat another chunk's text); misses are embedded 1,024 at a time and written after each batch, so an interrupted run resumes.
- **Qdrant:** collection `f"{dataset}__{config_name}__{model_slug}"` (e.g. `financebench__page_aware_384__bge-small-en-v1-5`),
  COSINE distance, point id `uuid5(NAMESPACE_URL, chunk_id)`, payload = the whole `Chunk`, upserts of 256. `build_filter` →
  `FieldCondition(key, MatchAny(any=[...]))` for company / doc_period / doc_type. Payload indexes are created **only on a
  server**: embedded mode ignores them (and warns), and it searches by brute force.
- **BM25:** `bm25s.tokenize(stopwords="en")` (lowercase, ≥ 2-char words, no stemming), `BM25()` defaults (k1 1.5, b 0.75,
  Lucene), saved to `indexes/bm25/<config>/` + `chunk_ids.txt`. Queries are tokenized as strings and looked up in the index's
  vocabulary (unknown words score 0). **Filters use bm25s's `weight_mask`** (applied before ranking) instead of "retrieve 4k,
  then drop": exact, and always k results when k exist.
- **Manifest:** rebuild iff `chunks_sha256` (hash of `chunk_id` + `embed_text`), `embed_model` or `INDEX_VERSION` changed, or
  the index on disk is incomplete (point count ≠ chunk count, `chunk_ids.txt` missing). The config fingerprint is stored but
  doesn't trigger rebuilds (changing the LLM must not re-embed). Writes are atomic (temp file + rename).

**Step script `scripts/05_embed_and_index.py [--strategy S --tokens N] [--force]`.** Embedding settings → cosine similarity of
3 sample texts → query instruction on/off → cache status → build/skip both indexes (IndexReport) → what BM25 tokens look like →
the manifest → sanity query (top 3 dense and BM25, printed pages) → company filter → **gold-page rank for FinanceBench's 3M
capex question** in each index.

**Done when**
- [x] Full indexing of the default strategy succeeds on the Mac: **31,114 chunks in 5 min 6 s** (embedding 293 s at
  106 chunks/s on MPS, Qdrant upsert 11 s, BM25 1.9 s); Qdrant 244 MB, BM25 25 MB, cache 66 MB; peak RSS 1.2 GB
- [x] A re-run is a no-op (0.07 s, "skipped"); `--force` rebuilds, with every vector from the cache
- [x] The integration round-trip test passes with `FakeEmbedder` (search, filters, mask, skip, force, missing collection)

**Pitfalls (met while building):**
- `if cache:` on an object with `__len__` is **False when the cache is empty**, so the first version silently never cached.
  Use `is not None`.
- Embedded Qdrant **locks its folder**: one process at a time. Close notebooks before running the CLI (`Container.close()`).
- Embedded mode warns above 20,000 points; we log the limitation once. The production answer is a Qdrant server (P10).
- Mixing normalized and unnormalized vectors; adding the query instruction to documents.

#### P4b — Online embedders and the embedder comparison (added on request)

**Goal:** any embedding model, local or online, selectable by config, compared on the same chunks; free tiers only, never a
surprise bill. The same provider plumbing is reused by rerankers (P5) and LLMs (P6).

**Files (as built)**
```
configs/embedders.yaml                         # presets: bge-small (default), bge-small-noinstr, bge-base, ollama-nomic,
                                               #   gemini-001, gemini-2, mistral-embed, cohere-v4, jina-v3, voyage-3.5-lite,
                                               #   openai-3-small / -large (PAID, blocked unless allow_paid)
src/finsight/providers/ratelimit.py            # RateLimiter (rpm spacing, tpm window, rpd/rpmonth caps), UsageStore, estimate_tokens
src/finsight/providers/http.py                 # post_json: retry 429/5xx/timeouts with backoff + Retry-After; quota-429 → stop
src/finsight/embedding/remote.py               # ProviderSpec table, RemoteEmbedder + 3 wire formats: OpenAIStyle (OpenAI, Mistral,
                                               #   Jina, Voyage, Ollama), Gemini (batchEmbedContents), Cohere (v2 /embed)
src/finsight/embedding/catalog.py              # catalog_df (key status, free/paid, estimates), estimate(), preset_cfg()
src/finsight/evaluation/retrieval_metrics.py   # hit/recall/ndcg/doc_hit@k, mrr (the P5 contract, built early)
src/finsight/evaluation/embedder_compare.py    # compare_embedders, summarize, by_question_type, disagreements, save_report
scripts/05b_compare_embedders.py               # cell-based: catalog → plan (requests/time/quota) → run → breakdown → report
tests/unit/test_providers.py  test_remote_embedders.py  test_embedder_config.py  test_retrieval_metrics.py
tests/integration/test_embedder_compare.py     # two fake presets + one without a key (skipped, reported)
```
CLI: `finsight embedders` (the catalog), `finsight index --embedder P [--limit-questions N]`,
`finsight compare-embedders --embedders a,b [--limit-questions N]`; `make embedders`, `make compare-embedders E=a,b Q=30`.

**How selection works**
- `embedding.preset` (env `EMBEDDER=gemini-001`, or `--embedder`) fills provider, model, dimensions, templates, role params and
  rate limits from `configs/embedders.yaml`. Anything set explicitly (`FINSIGHT__EMBEDDING__DIMENSIONS=1536`, a profile YAML,
  overrides) wins. `base.yaml` keeps only the preset name and generic knobs (`device`, `batch_size`, `allow_paid`,
  `max_tokens_per_run`).
- The API key is picked by provider (`GEMINI_API_KEY` or `GOOGLE_API_KEY`, `MISTRAL_API_KEY`, `COHERE_API_KEY`, `JINA_API_KEY`,
  `VOYAGE_API_KEY`, `OPENAI_API_KEY`). Missing key → `MissingAPIKeyError` naming the variable and where to get a free key.
- **Zero-spend guard:** providers without a free tier (OpenAI) raise `PaidProviderBlockedError` unless `allow_paid: true`.
  `max_tokens_per_run` is a hard stop checked *before* each request.
- Chunks are sized with `chunking.tokenizer` (bge) whatever the embedder, so every embedder sees identical chunks.

**Free-tier mechanics**
- Client-side limits: requests spaced `60/rpm` s apart, a 60 s token window for `tpm`, `rpd` (UTC day) and `rpmonth` caps
  persisted in `indexes/usage/<provider>.json`, checked before sending. Token counts are estimated (len / 3.5, deliberately high).
- `post_json`: 429 with "per day / per month / insufficient_quota" wording → `QuotaExhaustedError` (stop, don't hammer);
  other 429 → wait `Retry-After` or 2, 4, 8 … ≤ 60 s; 5xx/timeouts → backoff; 401/403 → error naming the key variable.
- Vectors are cached per batch (`preferred_batch` = 4 requests' worth), **query vectors too**: an interrupted or
  quota-stopped run resumes where it stopped, and re-running an evaluation sends zero requests.
- `--limit-questions N` indexes only the filings of the first N questions (index names get `__qN`): 30 questions →
  18 filings, 5,325 chunks (17% of the corpus). Every embedder sees the same subset.

**Identity and names:** local vectors keep the model name (`BAAI/bge-small-en-v1.5`), online ones
`provider/model@dims[#hash of document template/params]` (cache keys + manifest); the slug (`gemini-gemini-embedding-001-768`)
names the collection and the cache file. `bge-small-noinstr` has the same document vectors as `bge-small`, so it reuses the
index and measures only the query instruction.

**Done when**
- [x] Every preset in the catalog validates; `finsight embedders` shows key status and estimates (offline)
- [x] Each wire format's request/response is tested against a mock transport (URL, headers, body, ordering, batching, roles,
  templates, dims, normalization, query cache, budget, dimension mismatch); guards for missing keys and paid providers tested
- [x] The comparison runs end to end with fakes in CI, and for real with local models: see `docs/evaluation.md` (P4b)

**Not verified live:** the online adapters were built from the vendors' API docs (checked 2026-11-24) and tested offline;
no key was available in this session. Run `make compare-embedders E=bge-small,gemini-001 Q=30` once a key is in `.env`,
and record anything that differs in §7.

**Plan for P5/P6 (same pattern):** `configs/rerankers.yaml` (local bge-reranker, Cohere trial, Jina, Voyage) and
`configs/llms.yaml` (Ollama, Claude, Gemini free tier, Mistral free tier, Groq free tier), each provider through
`providers/ratelimit.py` + `providers/http.py`, with the same key lookup, paid guard and response cache.

**References:** [Sentence-BERT (arXiv 1908.10084)](https://arxiv.org/abs/1908.10084) · [BGE / C-Pack (arXiv 2309.07597)](https://arxiv.org/abs/2309.07597) · [bge-small-en-v1.5 model card](https://huggingface.co/BAAI/bge-small-en-v1.5) · [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) · [HNSW (arXiv 1603.09320)](https://arxiv.org/abs/1603.09320) · [Qdrant docs](https://qdrant.tech/documentation/) · [Qdrant filtering](https://qdrant.tech/documentation/concepts/filtering/) · [bm25s](https://github.com/xhluca/bm25s)

---

### P5 — Retrieval + retrieval evaluation

**Goal:** dense, sparse, hybrid (RRF), metadata filters and rerankers (local or online), plus a **retrieval-only eval** across chunkers. This gives the baseline table.
**Concepts:** lexical vs. semantic retrieval, rank fusion, bi-encoder vs. cross-encoder, Recall/MRR/NDCG. See Interview Guide §P5.

**Files (as built)**
```
src/finsight/retrieval/base.py         # Retriever, Reranker ABCs; ChunkStore (chunk_id -> Chunk, parents, filter mask)
src/finsight/retrieval/dense.py        # DenseRetriever(QdrantStore, embedder, collection): filters applied in-search
src/finsight/retrieval/sparse.py       # SparseRetriever(BM25Index, ChunkStore): filters as a pre-ranking document mask
src/finsight/retrieval/hybrid.py       # rrf_fuse (ties by chunk_id, duplicates once, per-stage scores kept), HybridRetriever
src/finsight/retrieval/filters.py      # FilterExtractor (aliases from metadata + tickers/nicknames), apply_fields
src/finsight/retrieval/reranker.py     # NoopReranker, LocalCrossEncoderReranker (bge-reranker-base), APIReranker (cohere/jina/voyage)
src/finsight/retrieval/service.py      # RetrievalService: filters -> dense/BM25 -> RRF -> rerank, every stage + timing recorded
src/finsight/evaluation/runner.py      # run_retrieval_eval(settings, experiment) -> results/<date>_<name>/
src/finsight/evaluation/report.py      # summary_table (mean ± std, p50 ms, n), best_config, retrieval_report
configs/experiments/retrieval_ablation.yaml  (+ retrieval_ablation_extra.yaml: the 2 variants added after the first run)
scripts/06_retrieve.py  scripts/07_eval_retrieval.py
tests/unit/test_rrf.py  test_filters.py  test_rerankers.py  test_retrieval_metrics.py (P4)
tests/integration/conftest.py (tiny indexed corpus)  test_retrieval_service.py
```
CLI: `finsight eval [--experiment ...] [--limit N]`; `make eval-retrieval`. Container: `chunk_store`, `dense_retriever`,
`sparse_retriever`, `reranker`, `filter_extractor`, `retrieval`.

**Implementation notes (as built)**
- Every stage keeps `candidate_k` = 50; the reranker re-orders all 50 and the result is cut to `top_k`. The generator (P6) uses
  the first `rerank.top_n`.
- Rerankers: `rerank.provider` = none | local | cohere | jina | voyage. Online rerankers use the P4 provider plumbing
  (rate limiter, retries, key lookup). `rerank.on_error: fallback` keeps the RRF order and records `reranker_failed`.
- Filters: company aliases are built from the document metadata (name, name without legal suffix) plus a hand map (MMM, J&J,
  AMEX, ...); a company is found in **98%** of the 150 questions. If a filter returns nothing, the service retries unfiltered
  (`filter_returned_nothing`).
- Metrics are the P4 `retrieval_metrics` module; the runner stores one row per chunking × variant × question with extracted
  filters, top-3 and gold pages, per-stage timings.

**Done when**
- [x] The full ablation runs on all 150 questions; reports in `results/2026-11-24_retrieval_ablation/` (+ `_extra`)
- [x] The best configuration is written into `configs/base.yaml`; the reasoning is in `docs/evaluation.md` (P5) and §7
- [x] Unit tests for RRF (ties, duplicates), filters, rerankers and metrics pass

**Pitfalls (met):** hybrid **without** a filter is worse than dense alone here (BM25 is weak on these long, instruction-heavy
questions and RRF gives it equal weight). Reranking with `embed_text` instead of `text`. Counting the same page twice in NDCG.

**References:** [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) · [RRF paper (Cormack et al., 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) · [bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) · [SBERT cross-encoders](https://sbert.net/examples/cross_encoder/applications/README.html) · [DCG/NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) · [ColBERT (late interaction; to read about)](https://arxiv.org/abs/2004.12832)

---

### P6 — Grounded generation

**Goal:** answers with `[n]` citations from any LLM (local or online, chosen by preset or `auto`), citation checks, and abstention.
**Concepts:** grounding prompts, context assembly and token budgets, citation checking, refusal and abstention, provider abstraction. See Interview Guide §P6.

**Files (as built)**
```
configs/llms.yaml                              # presets: auto, ollama-qwen7b, hf-qwen1.5b, claude-opus, claude-haiku,
                                               #   gemini-flash, mistral-small, groq-llama (free tiers), openai-mini (PAID)
src/finsight/generation/llm/base.py            # LLM ABC, LLMResponse (contract §4.2)
src/finsight/generation/llm/ollama.py          # /api/chat with num_ctx on every call; ollama_ready()
src/finsight/generation/llm/anthropic_llm.py   # official SDK; output_config effort; no sampling params; refusal fallbacks
src/finsight/generation/llm/remote_llm.py      # OpenAIChatLLM (Mistral, Groq, OpenAI) + GeminiLLM over httpx + P4 rate limiter
src/finsight/generation/llm/hf_local.py        # transformers on MPS: the always-available offline fallback
src/finsight/generation/llm/fake.py            # FakeLLM (tests)
src/finsight/generation/llm/factory.py         # build_llm, can_run, resolve_auto (the "auto" preset)
src/finsight/generation/prompts/answer_system.md  answer_user.md
src/finsight/generation/prompt_loader.py       # string.Template ($names): filing text is full of braces
src/finsight/generation/context.py             # build_context -> ("[n] company · doc FYyyyy · page p" blocks, marker_map)
src/finsight/generation/citations.py           # parse_citations, auto_cite, is_abstention
src/finsight/pipelines/base.py  registry.py  naive.py  hybrid_rerank.py
scripts/08_generate.py
tests/unit/test_citations.py  test_context.py  test_llm_providers.py  tests/integration/test_pipeline_fake_llm.py
```
CLI: `finsight query "..." [--llm P --pipeline N --json]`, `finsight llms` (presets, can-run-now, what auto picks).
`check-env` gained an **LLM** row (what `auto` resolves to). Container: `llm`, `judge_llm`, `pipeline`.

**Implementation notes (as built)**
- **Selection:** `llm.preset` (`LLM=` in .env, `--llm`). `auto` walks `llm.auto_order`: Ollama (server up + model pulled) →
  Claude (key) → Gemini → Mistral → Groq (free-tier keys) → `hf-qwen1.5b` (always). The judge has its own order, Claude Haiku
  first. Paid-only providers (OpenAI) need `allow_paid`; missing keys raise with the env var and the free-key link.
- **Claude (per the current API):** no `temperature`/`top_p`/`top_k` (400 on current models); depth via
  `output_config={"effort": ...}` (Opus 5.5 thinks adaptively, effort default medium, set explicitly); `max_tokens` ≥ 16,000
  so thinking isn't cut off; Opus calls opt into server-side refusal fallbacks (`betas=["server-side-fallback-2026-07-01"]`,
  `fallbacks="default"`); a final `stop_reason == "refusal"` becomes an abstention with warning `model_refusal`.
- **Local HF model:** Qwen2.5-1.5B-Instruct, fp16 on MPS, greedy; the prompt is cut from the left if it exceeds `num_ctx`.
- **Citations:** invalid markers → `hallucinated_citation`; no valid marker → **`auto_cite`**: every figure in the answer
  (`1,577`, `12.3`; years excluded) is looked up verbatim in the blocks and the first containing block is cited
  (`auto_cited`); none found → `uncited_answer`.
- `Answer` gained `warnings` and `degraded` (contract change, §7).

**Done when**
- [x] The 3M capex question returns a cited answer: local `hf-qwen1.5b` → "**$1,577 million**", cited block = 3M_2018_10K
  p39 (`auto_cited`), 11 s. With `--llm claude-opus` once a key is set (not run: no key in this session)
- [x] An off-topic question (`"Who won the 2018 World Cup?"`) abstains: "Not found in the provided filings."
- [x] Unit tests (citations, context budget, provider request shapes, auto selection, guards) and the FakeLLM integration pass

**Pitfalls (met):** small local models ignore the `[n]` format, and a one-line prompt example flipped the same model into
abstaining: prompts for small models are brittle, so citation repair is deterministic code, not more prompt. Ollama's
`num_ctx` truncation. Mismatched page numbering in the headers (shown as `page_idx + 1`).

**References:** [Original RAG paper (arXiv 2005.11401)](https://arxiv.org/abs/2005.11401) · [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md) · [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) · [Claude API docs](https://platform.claude.com/docs) · [Lost in the Middle](https://arxiv.org/abs/2307.03172)

---

### P7 — Advanced RAG techniques

**Goal:** each advanced technique as a pluggable component, measured against the P5/P6 baseline; plus a second dataset (MultiHop-RAG) that needs cross-document reasoning and abstention on unanswerable questions.
**Concepts:** query rewriting, multi-query, HyDE, decomposition, contextual retrieval, corrective RAG, agentic RAG. See Interview Guide §P7.

**Files (as built)**
```
src/finsight/retrieval/query_transform.py    # multi_query, hyde, decompose (used by hybrid_rerank via retrieval.query_transform)
src/finsight/generation/jsonout.py            # ask_json: LLM -> pydantic, extract first {...}, retry once, else None (fallback)
src/finsight/generation/prompts/{multi_query,hyde,decompose,grade,rewrite,agent_system,agent_json}.md
src/finsight/pipelines/corrective.py          # CorrectiveRAGPipeline: grade -> rewrite+retry -> abstain
src/finsight/pipelines/agentic.py             # AgenticRAGPipeline: search/read/answer tools (Claude native; JSON protocol elsewhere)
src/finsight/data/multihop.py                 # MultiHop-RAG loader (corpus.json + MultiHopRAG.json)
src/finsight/data/datasets.py                 # dispatch: document_meta / pages_df / eval_records by settings.dataset
src/finsight/evaluation/pipeline_sample.py    # run_pipeline_sample: pipelines on a seeded, null-stratified sample
configs/multihop.yaml  configs/experiments/{advanced,contextual}.yaml
scripts/09_advanced_rag.py
tests/integration/test_advanced_rag.py  test_pipeline_sample.py  tests/unit/test_multihop_loader.py
```
CLI: `finsight download --dataset multihop`, `finsight sample --pipelines a,b --n 50`; `make eval-advanced`, `make multihop`.
Config: `retrieval.query_transform` (none|multi_query|hyde|decompose), `crag` (min_score, llm_grader), `agent.max_steps`.

**Implementation notes (as built)**
- **Query transforms** change *what* is searched: multi_query fuses 3 LLM rewrites (+ the original) with RRF; HyDE embeds an LLM-written fake filing passage for the dense side and keeps the question for BM25; decompose splits multi-part questions. Each costs one LLM call; invalid JSON falls back to plain retrieval (recorded in `degraded`).
- **CorrectiveRAG:** grade the top rerank score (and optionally an LLM grader); if weak, rewrite and retry, merge, re-grade; still weak → abstain *without* calling the generator.
- **AgenticRAG:** Claude uses native tool use (`tool_choice: auto` — Claude Opus 5.5 rejects forced tool choice); other LLMs use a JSON-action protocol validated with pydantic, falling back to plain hybrid_rerank after two invalid steps. `agent.max_steps` caps the loop.
- **MultiHop-RAG** (ODC-BY; 609 news articles, 2,556 queries, 301 null): one article = one DocumentMeta + one Page (page 0), so the FinanceBench metrics work unchanged at article level. `settings.dataset` + `datasets.py` route the whole pipeline; `configs/multihop.yaml` keeps its chunk/BM25 folders separate.
- **Performance fix made here (applies to P4/P5):** query embeddings and cross-encoder rerank scores are now cached on disk (keyed by model + text, and model + query + chunk). A re-run embeds and re-scores nothing, so an evaluation that took ~19 s/question cold replays in ~0.1 s **and returns the identical ranking** — fixing both the "embedding happens again and again" slowness and the non-reproducible rerank order (Apple MPS float wobble). Document vectors were already skipped by the index manifest.

**Done when**
- [x] Each technique runs from `scripts/09_advanced_rag.py` and from config
- [x] `results/<date>_advanced_query_transforms/` and `_contextual_headers/` compare each against the baseline (retrieval metrics; generation metrics in P8)
- [x] MultiHop-RAG loads (`finsight download --dataset multihop`) and the agentic pipeline runs on a sample including null queries

**Pitfalls (met):** query transforms multiply LLM calls and latency (reported). A 7B/1.5B model emits JSON wrapped in prose — `ask_json` extracts the first object and retries once. The agent can loop; `agent.max_steps` stops it. Reranking and query embedding recomputed every run until cached (fixed above).

**References:** [HyDE (arXiv 2212.10496)](https://arxiv.org/abs/2212.10496) · [RAG-Fusion (arXiv 2402.03367)](https://arxiv.org/abs/2402.03367) · [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) · [Corrective RAG (arXiv 2401.15884)](https://arxiv.org/abs/2401.15884) · [Self-RAG (arXiv 2310.11511)](https://arxiv.org/abs/2310.11511) · [ReAct (arXiv 2210.03629)](https://arxiv.org/abs/2210.03629) · [MultiHop-RAG (arXiv 2401.15391)](https://arxiv.org/abs/2401.15391) · [Claude tool use docs](https://platform.claude.com/docs)

---

### P8 — End-to-end evaluation harness

**Goal:** score the answers (correctness, faithfulness, abstention), compare pipelines, and gate regressions in CI.
**Concepts:** reference-based vs. reference-free metrics, LLM-as-judge and its biases, statistical noise, regression gates. See Interview Guide §P8.

**Files**
```
src/finsight/evaluation/generation_metrics.py   # numeric_match, contains_match, abstention metrics
src/finsight/evaluation/judge.py                # Judge(llm).correctness(), .faithfulness() with disk cache
src/finsight/generation/prompts/judge_correctness.md  judge_faithfulness.md
src/finsight/evaluation/runner.py               # extend with run_end_to_end_eval()
src/finsight/evaluation/report.py               # extend: generation columns, failure gallery
configs/experiments/end_to_end.yaml
tests/fixtures/regression_set.jsonl  tests/fixtures/regression_thresholds.json
tests/integration/test_regression_gate.py
scripts/10_eval_end_to_end.py
scripts/10b_label_judge_sample.py               # you hand-label 20 answers to validate the judge
```

**Implementation notes**
- `numeric_match(pred, gold, rel_tol=0.01)`: pull the numbers out of both strings (handle `$`, `,`, `%`, `(1,234)` as a negative, `million`/`billion`). The match is true if any predicted number is within tolerance of the gold's main number. Use it for `metrics-generated` questions.
- The judge prompt returns JSON `{"verdict": "correct"|"partially_correct"|"incorrect", "reason": "..."}` for correctness (question, gold, prediction), and `{"supported_claims": n, "total_claims": m, "unsupported": [...]}` for faithfulness (answer, cited chunk texts). Validate with pydantic; on failure, retry once, then mark `judge_error`.
- **Judge cache:** a SQLite store keyed by `sha256(judge_model + prompt)`, so re-running a report costs nothing.
- **Judge choice:** `JUDGE_PROVIDER=anthropic` with `claude-haiku-5-5` and `effort: low` is cheap and stronger than a 7B model. The Ollama judge works offline but is noisier, so report which judge produced each number.
- Abstention metrics: on answerable questions, `false_abstain_rate`; on MultiHop null queries, `correct_abstain_rate`.
- `end_to_end.yaml`: variants = {naive, hybrid_rerank, + multi_query, corrective} × {ollama, anthropic (optional)}, on the best chunker.
- **CI regression gate:** `regression_set.jsonl` holds 10 FinanceBench questions together with their **pre-chunked gold and distractor chunks** (small, committed, built from public filing text; check the license note in the data card). The test builds an in-memory index with `FakeEmbedder` and BM25, runs the hybrid retrieval, and asserts `hit@5 ≥ threshold`. It catches pipeline regressions (fusion, filtering, metric bugs) without models. A heavier "real model" gate runs locally with `-m slow`.

**Step scripts**
- `10_eval_end_to_end.py --experiment configs/experiments/end_to_end.yaml [--limit 30]` prints the summary table (correct %, partial %, faithfulness, abstention, p50/p95 latency, tokens per question, estimated cost) and the path to the failure gallery.
- `10b_label_judge_sample.py` shows 20 random (question, gold, prediction) triples. You type c/p/i for each, and it prints your agreement with the judge (% agreement and a confusion matrix).

**Done when**
- [ ] The full report is in `results/`, and its headline table is copied into the README and `docs/evaluation.md`
- [ ] Judge agreement is measured on 20 hand labels and written down (with honest caveats)
- [ ] The regression gate runs in CI

**Pitfalls:** the judge being lenient towards its own model family (prefer a judge from a different model or provider than the generator). Position and verbosity bias. Reading meaning into 1–2 point differences on 150 questions (report n and the standard deviation).

**References:** [Judging LLM-as-a-Judge (arXiv 2306.05685)](https://arxiv.org/abs/2306.05685) · [RAGAS (arXiv 2309.15217)](https://arxiv.org/abs/2309.15217) · [ARES (arXiv 2311.09476)](https://arxiv.org/abs/2311.09476) · [Anthropic: statistical approach to evals](https://www.anthropic.com/research/statistical-approach-to-model-evals)

---

### P9 — Serving: API + demo UI

**Goal:** a FastAPI service and a Streamlit UI that shows what happens inside the pipeline.
**Concepts:** API design, streaming (SSE), app lifecycle (warm-up), health vs. readiness. See Interview Guide §P9.

**Files**
```
src/finsight/api/__init__.py  main.py  routes.py  models.py  deps.py  middleware.py
ui/streamlit_app.py
scripts/11_smoke_api.py
tests/integration/test_api.py      # FastAPI TestClient + Container with fakes
```

**Implementation notes**
- `main.py`: a `lifespan` context builds the `Container`. If `api.warmup=true`, it loads the models and indexes once at startup. A middleware sets the `X-Request-ID` header (generated if missing) and logs one JSON line per request with the stage timings.
- Endpoints:
  - `GET /health` returns `{"status": "ok"}` (the process is alive)
  - `GET /ready` returns 200 only when the indexes are loaded and the LLM provider is reachable; otherwise 503 with the reasons
  - `POST /v1/query` takes `{question, pipeline?, provider?, filters?, top_k?, include_retrieved?}` and returns an `Answer` (`retrieved` is trimmed unless `include_retrieved`)
  - `POST /v1/query/stream` streams Server-Sent Events: `event: retrieved` (citations preview), `event: token` (each delta), `event: done` (final citations and timings)
  - `GET /v1/config` returns the resolved non-secret config and its fingerprint
- Validation: `question` is 3–1000 characters; `top_k ≤ 20`; reject an unknown `pipeline` or `provider` with a 422 error.
- `ui/streamlit_app.py`: sidebar with pipeline, provider, chunker info and filters. Main area: the answer with clickable citation chips (each expands the page snippet and shows the printed page number), and an **"Under the hood"** expander with four columns (BM25, Dense, RRF, Reranked: top-10 with scores), the timings as a bar chart, and token usage. Include 5 preset example questions from FinanceBench.

**Step script `scripts/11_smoke_api.py`.** It calls `/health`, `/ready` and `/v1/query` with 3 questions, then prints the status codes, latency, a preview of each answer and the request IDs.

**Done when**
- [ ] `make serve` + `scripts/11_smoke_api.py` pass; `make ui` shows the demo
- [ ] `test_api.py` passes (with fakes)
- [ ] A ~60-second demo GIF is recorded for the README

**References:** [FastAPI](https://fastapi.tiangolo.com/) · [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/) · [Server-Sent Events (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) · [Streamlit](https://docs.streamlit.io/)

---

### P10 — Production hardening & deployment

**Goal:** `docker compose up` gives a working system, with fallbacks, caching, basic security and a runbook.
**Concepts:** containers, fallbacks and degradation, caching, prompt injection, observability. See Interview Guide §P10.

**Files**
```
Dockerfile  .dockerignore  docker-compose.yml
src/finsight/cache.py              # query-embedding LRU + answer cache (keyed by question + config fingerprint)
src/finsight/observability/cost.py # token -> $ estimate per provider/model (prices in config, not code)
docs/runbook.md  docs/adr/0001-vector-store.md ... 0004-llm-providers.md
scripts/12_load_test.py            # simple concurrent requests -> p50/p95
```

**Implementation notes**
- Dockerfile: multi-stage. The builder stage uses `uv sync --frozen --no-dev`; the runtime is `python:3.12-slim` with a non-root user and `HF_HOME=/models`. Mount the models and indexes as volumes; don't bake them into the image.
- `docker-compose.yml`: `qdrant` (`qdrant/qdrant`, port 6333, volume), `api` (`QDRANT_URL=http://qdrant:6333`, `OLLAMA_BASE_URL=http://host.docker.internal:11434`, healthcheck on `/health`), and `ui` (depends on `api`). Ollama stays on the host, where it can use the Mac's GPU; Docker on macOS has no GPU passthrough.
- Indexing to the Docker Qdrant: run `scripts/05_embed_and_index.py` with `QDRANT_URL=http://localhost:6333`.
- Fallbacks (configurable): if the Anthropic call fails, fall back to Ollama; if the reranker fails or times out, use the RRF order; if the dense search fails, use BM25 only. Every fallback is logged and shown in `Answer` (e.g. a `degraded: ["reranker_timeout"]` field; add it to the schema through the decision log).
- Security: a request size limit, a simple in-memory rate limiter per IP (note that production would use a gateway), the system-prompt instruction that context is data (P6), no secrets in logs, CORS limited to the UI origin.
- `runbook.md`: how to rebuild the indexes, switch providers, rotate keys, debug a bad answer (find the request ID in the logs, re-run `06_retrieve.py` with the same question, check whether the gold page was retrieved, then check the prompt), and common errors.

**Done when**
- [ ] On a fresh clone: `make setup data ingest index` then `docker compose up` serves answers at `localhost:8000` and the UI at `localhost:8501`
- [ ] `12_load_test.py` reports p50/p95 at 5 concurrent users; the numbers go in the README
- [ ] The ADRs are written

**References:** [Docker multi-stage builds](https://docs.docker.com/build/building/multi-stage/) · [uv in Docker](https://docs.astral.sh/uv/guides/integration/docker/) · [OWASP Top 10 for LLM apps](https://owasp.org/www-project-top-10-for-large-language-model-applications/) · [Qdrant in Docker](https://qdrant.tech/documentation/quickstart/)

---

### P11 — Documentation & showcase

**Goal:** a stranger can understand, run and trust the project within 10 minutes.

**Deliverables**
- `README.md`: one-line pitch → demo GIF → problem → architecture diagram → **results table** → quickstart (5 commands) → project layout → design decisions (links to ADRs) → limitations → next steps → license notes (FinanceBench CC-BY-NC, PyMuPDF AGPL)
- `docs/architecture.md`: the diagrams from the spec plus the sequence of one `/v1/query` request
- `docs/evaluation.md`: methodology, every results table, judge validation, caveats
- `docs/INTERVIEW_GUIDE.md`: fill every "📌 your number" placeholder with real results
- The parent repo README gets a section: "`/` = learning experiment (FiQA retrieval) · `finsight-rag/` = production project"
- Tag `v1.0`

**Done when**
- [ ] A friend or colleague follows the README from a fresh clone without help
- [ ] Every number in the README links to a `results/` report

---

## 6. Progress tracker

> Update this table at the end of every phase (agent or human). "Key numbers" are the facts you'll quote in an interview.

| Phase | Status | Date | Key numbers / notes |
|---|---|---|---|
| P0 Scaffold | ✅ done (CI green) | 2026-11-23 | 29 unit tests, 81% coverage · lint + mypy clean · torch 2.15 with MPS on M5 · env check: 4 ok / 5 info / 1 warn (Ollama not installed yet) |
| P1 Data | ✅ done | 2026-11-23 | 150 Q · 189 evidence items · 84 PDFs (166.5 MB, **12,013 pages**) · 32 companies · download 23 s, re-run 0.1 s · 0 consistency problems · 43 unit tests |
| P2 Ingestion | ✅ done | 2026-11-24 | 12,013 pages (69 empty, 5,098 with tables, 40.4M chars) · **evidence on gold page 189/189 (100%)**, mean overlap 0.998, 0 offsets · label–number adjacency **0.28 → 0.97** vs plain extraction (160 numeric evidence pages) · 70 s sequential / 17 s on 6 processes · 96 header/footer patterns · 75 tests |
| P3 Chunking | ✅ done | 2026-11-24 | 4 strategies × 3 sizes swept · page_aware_384: 31,114 chunks, 0% cross pages, 72% of evidence in one chunk · fixed/recursive cross pages 30–60% · evidence-in-one-chunk 44–48% @256 → 68–72% @384 → 80–87% @480 · table chunks with their column header 64% → 94% (carry) · 0 chunks over the 510-token limit · ~10 s per strategy · 99 tests |
| P4 Indexing | ✅ done | 2026-11-24 | 31,114 page-aware chunks indexed in **5 min 6 s** on M5 (bge-small on MPS, **106 chunks/s**) · Qdrant 244 MB (embedded), BM25 25 MB (27,097 words), cache 66 MB · re-run 0.07 s · 3M capex gold page: dense rank 36 (no instruction) / 47 (with), BM25 121 → the vocabulary gap P5 must close · **online embedders** (Gemini, Mistral, Cohere, Jina, Voyage, Ollama; OpenAI paid-blocked) via presets + rate limits · dense-only baseline (150 Q): hit@10 0.46, MRR 0.23, doc_hit@10 0.89 · q30: bge-base 0.50 vs bge-small 0.43 hit@10 · 139 tests (137 fast + 2 slow) |
| P5 Retrieval | ✅ done | 2026-11-24 | 4 chunkers × 8 variants × 150 Q · **best: recursive_384 + dense + company/year filter: hit@10 0.613**, MRR 0.35, doc_hit 0.99, ~110 ms · BM25 alone 0.15; hybrid (no filter) 0.32 < dense 0.46 · company filter: doc_hit 0.83 → 0.99 · local rerank +0–7 pts at ~2.7 s/q, not adopted · 161 tests |
| P6 Generation | ✅ done | 2026-11-24 | LLM presets + auto (Ollama → Claude → Gemini/Mistral/Groq free tiers → local HF) · 3M capex: local Qwen2.5-1.5B answers **$1,577 million** (correct), 11 s, cited via auto_cite · off-topic question abstains · Claude path tested offline (no key) · 175 tests |
| P7 Advanced | ✅ done | 2026-11-24 | query transforms (multi_query, hyde, decompose), corrective + agentic pipelines, **contextual headers: hit@10 0.613 → 0.673** (150 Q) · query transforms not yet measured (`make eval-advanced`) · MultiHop-RAG loader (609 articles, 2,556 Q, 301 null) via dataset dispatch · **perf fix: query + rerank caches -> re-runs ~19s->0.1s/q and reproducible** · 186 tests |
| P8 Evaluation | 🟨 in progress | 2026-11-24 | drafts in place (see §6.1) · correct %: · faithfulness: |
| P9 Serving | ⬜ | | p50 latency: |
| P10 Production | ⬜ | | |
| P11 Docs | ⬜ | | |

Status legend: ⬜ not started · 🟨 in progress · ✅ done (all "Done when" boxes ticked)

### 6.1 Resume notes (handover between sessions)

> Kept current so a new session (human or agent) can pick up without the chat history. Update it, don't append forever.

**Branches:** P3 merged to `main`. P4 `feat/p4-index`, P5 `feat/p5-retrieval`, P6 `feat/p6-generation`, P7 `feat/p7-advanced`
(fe78351) are pushed; each branches from the previous one. P8 is on `feat/p8-eval` (from P7).

**Environment gotchas**
- VS Code notebook kernels hold the embedded-Qdrant lock on `indexes/qdrant`. For CLI runs while they are open, use a copy:
  `cp -R indexes/qdrant indexes/qdrant_work && rm -f indexes/qdrant_work/.lock` and
  `export FINSIGHT__PATHS__QDRANT_DIR=indexes/qdrant_work` (gitignored). Don't kill the user's kernels.
- No API keys and no Ollama on this machine, so `llm.preset: auto` resolves to the local HF Qwen2.5-1.5B (slow, 1–4 s per call
  on MPS). Long LLM evals: use `--limit`.
- Query vectors and rerank scores are cached in `indexes/embed_cache/` (P7), so re-runs are fast and deterministic.
- Shell heredocs that contain backticks must be quoted (`<<'EOF'`).

**P7 open item:** the query-transform comparison (`make eval-advanced`, 60 Q) hasn't been run to completion yet. When it is,
put its numbers in `docs/evaluation.md` (P7) and the P7 📌 line of the interview guide.

**P8 state (`feat/p8-eval`)**
- In the repo (drafts, untested end to end): `evaluation/generation_metrics.py`, `evaluation/judge.py` (+ prompts
  `judge_correctness.md`, `judge_faithfulness.md`), `evaluation/e2e.py`, `observability/cost.py`,
  `configs/experiments/end_to_end.yaml`, `tools/build_regression_set.py`, scripts `10_eval_end_to_end.py`, `10b_label_judge_sample.py`,
  tests `unit/test_judge.py`, `unit/test_generation_metrics.py`, `integration/test_regression_gate.py`.
- Done: `LLMCfg.price_in/price_out` (USD per 1M tokens, default 0) and `ApiCfg.rate_limit_per_minute` added to `config.py`.
- To do, in order:
  1. ✅ Regression gate: `tests/fixtures/regression_set.jsonl` (10 Q × 10 chunks, built by `tools/build_regression_set.py`) +
     `regression_thresholds.json` (hybrid hit@5 = 0.60 measured, deterministic). 198 tests pass.
  2. Set `price_in`/`price_out` on the paid presets in `configs/llms.yaml` from the providers' current pricing pages. Don't guess:
     a $0 estimate is better than a wrong one. Free tiers and local stay 0.
  3. ✅ Wired: `finsight eval` runs end-to-end when the experiment has `judge:`/`faithfulness:`; `make eval LIMIT=30`.
     3 Q × 3 pipelines + judge = ~3 min on the local LLM. A 30-Q run → `results/2026-11-24_end_to_end/`; record its numbers.
  4. Judge validation (script 10b) needs human labels. Until someone labels, document the judge as **unvalidated**.
  5. `make notebooks`, lint/mypy/tests, update §5 P8 "Done when", this tracker row, §7, `evaluation.md`, the interview guide's P8 📌.

**P9 drafts:** `src/finsight/api/` (models, middleware, deps, routes, main) is on disk **untracked**. Leave it out of the P8
commit. Known issue: `api/main.py:57-58` uses `app` before it is defined (mypy). Still to write: `ui/streamlit_app.py`,
`scripts/11_smoke_api.py`, `tests/integration/test_api.py`, `make serve` / `make ui`.

**P10/P11:** not started (Dockerfile, compose, runbook, ADRs, load test, degraded fallbacks; final README, tag v1.0). Docker
isn't installed on this machine, so mark container steps "not verified locally".

---

## 7. Decision and deviation log

> Every change to a contract in §4, any library API that differed from this guide, and any judgment call. Newest first.

| Date | Phase | Decision / deviation | Why |
|---|---|---|---|
| 2026-11-24 | P7 | **Query + rerank caches** (surfaced by the user): local query embeddings and cross-encoder scores cached on disk (`cached_query_vectors`, `ScoreCache`) | Re-running an eval was re-embedding queries and re-scoring the reranker every time (~19 s/question) and the order drifted on Apple MPS. Now a replay is ~0.1 s and byte-identical. Document vectors were already skipped by the manifest |
| 2026-11-24 | P7 | Datasets behind `datasets.py` dispatch (`settings.dataset`); MultiHop-RAG as one Page per article | The chunking/indexing/eval code stays dataset-agnostic; MultiHop evidence is per-article, which fits the page-range relevance rule |
| 2026-11-24 | P7 | Query transforms live in the `hybrid_rerank` pipeline via `retrieval.query_transform`, not new pipelines | They change the query, not the pipeline shape; keeps the pipeline registry small |
| 2026-11-24 | P7 | Agentic pipeline: Claude native tool use, JSON-action protocol for other LLMs, fallback to hybrid_rerank on repeated invalid JSON | Claude Opus 5.5 rejects forced tool_choice; small local models cannot always emit valid tool JSON |
| 2026-11-24 | P7 | **Contract:** new config `crag`, `agent`; `retrieval.query_transform` now used; added to the fingerprint | Advanced techniques must be traceable in the config hash like everything else |
| 2026-11-24 | P6 | **Contract change:** `LLMCfg`/`JudgeCfg` became presets (`configs/llms.yaml`) with `auto`; env `LLM_PROVIDER`/`OLLAMA_MODEL`/`ANTHROPIC_MODEL`/`JUDGE_PROVIDER` replaced by `LLM`, `LLM_MODEL`, `JUDGE`; `Answer` gained `warnings` and `degraded` | Same model-choice pattern as embedders; any provider (local, free tier, Claude) by one line; no key and no Ollama still works |
| 2026-11-24 | P6 | Local fallback LLM `hf-qwen1.5b` (transformers, MPS) added to `auto` | This Mac has no Ollama/Homebrew and no API key; the project must still answer end to end, offline, at zero cost |
| 2026-11-24 | P6 | Online LLMs over httpx (Gemini, Mistral, Groq, OpenAI-paid) but Claude through the official SDK | Free tiers reuse the P4 limiter/retries; Claude features (effort, refusal fallbacks, typed errors, streaming) come from the SDK |
| 2026-11-24 | P6 | `auto_cite`: deterministic citation repair by verbatim figure match, flagged `auto_cited` | The 1.5B model answers correctly but ignores `[n]`; prompt tweaks flipped it into abstaining. Code is more reliable than prompt for small models, and the flag keeps the metric honest |
| 2026-11-24 | P5 | **Defaults changed by measurement:** chunking `recursive` 384 (was page_aware), `retrieval.mode: dense` (was hybrid), `filter_fields: [company, doc_period]` (was company), `rerank.provider: none` (was local) | Ablation on 150 questions: dense + both filters 0.613 hit@10 vs hybrid + filters + rerank 0.580, at 1/25 of the latency. The year filter helped (+4–14 pts) instead of hurting. Everything stays switchable |
| 2026-11-24 | P5 | Ablation extended with `dense+filter+year` and `hybrid+filter+year+rerank` (`retrieval_ablation_extra.yaml`) | The first grid never combined the year filter with dense-only or with the reranker; the winner was in the gap |
| 2026-11-24 | P5 | Rerankers: local cross-encoder plus Cohere / Jina / Voyage over the P4 provider plumbing; `rerank.on_error: fallback` | Free online rerankers for comparison; a reranker outage must not fail the request |
| 2026-11-24 | P5 | Pipeline name `hybrid_rerank` kept (contract) although the default retrieval is now dense, no rerank | It runs whatever `retrieval`/`rerank` config says; renaming would break the contract for no gain |
| 2026-11-24 | P4 | **Online embedders added** (user request): presets in `configs/embedders.yaml`, 3 wire formats over plain httpx (no vendor SDKs), shared `providers/` (rate limiter, usage store, retrying `post_json`) for P5/P6 too | Compare local vs hosted models with zero spend; one retry/limit path for every provider; nothing extra to install |
| 2026-11-24 | P4 | **Contract change:** `EmbeddingCfg` = preset + provider/model/dimensions/templates/params/rate_limit/allow_paid/max_tokens_per_run; `query_instruction` replaced by `query_template`; `Embedder` gains `slug`, `provider`, `embed_queries`, `preferred_batch`; `Secrets` gains gemini/openai/mistral/jina/voyage keys; new `chunking.tokenizer` | Templates cover bge, Gemini 2, nomic uniformly; chunk sizes must not depend on the embedder; bge identities unchanged so the built index and cache stay valid |
| 2026-11-24 | P4 | Paid-only providers refused unless `allow_paid: true`; `max_tokens_per_run` checked before every request | The user runs free tiers only: a typo in .env must never spend money |
| 2026-11-24 | P4 | Free-tier limits in presets are conservative defaults, not vendor facts | Vendor docs and third-party pages disagree and limits differ per account; the dashboard is the truth, the preset is overridable |
| 2026-11-24 | P4 | Only explicit per-day/per-month wording makes a 429 a hard stop | Google's per-MINUTE 429 also says "exceeded your current quota … billing"; stopping on it would abort runs that only need to wait |
| 2026-11-24 | P4 | Query vectors cached in the embedder's SQLite file; `--limit-questions N` subset indexes (`__qN`) | Re-running an eval costs 0 requests; a 30-question subset (5,325 chunks) makes rate-limited APIs finish in minutes |
| 2026-11-24 | P4 | `retrieval_metrics.py` built in P4 instead of P5 | The embedder comparison needs it; P5 reuses it unchanged |
| 2026-11-24 | P4 | BM25 filters via bm25s `weight_mask` (pre-ranking) instead of "retrieve k×4, then drop" | Exact, and returns k matches whenever k exist; post-filtering can return fewer than k for a small company |
| 2026-11-24 | P4 | Payload indexes created only on a Qdrant **server**; embedded mode warnings (> 20k points, payload indexes) silenced and logged once | Embedded mode ignores payload indexes and searches by brute force (~31k points is fine). Production uses a server (P10) |
| 2026-11-24 | P4 | Manifest rebuild keys = chunk content hash + embed model + `INDEX_VERSION` (+ index completeness); fingerprint stored only | Changing the LLM or top_k changes the fingerprint but must not re-embed 31k chunks |
| 2026-11-24 | P4 | Collection and BM25 names include the chunking config (`page_aware_384`), so all configs coexist | P5 compares chunkers on retrieval metrics without rebuilding |
| 2026-11-24 | P4 | The sanity script also reports the **gold-page rank** for FinanceBench's 3M capex question; the slow test checks findability with the filing's wording | The question says "capital expenditure", the filing "Purchases of property, plant and equipment": dense rank 36–47, BM25 121. Retrieval quality is a P5 metric, not a P4 assertion |
| 2026-11-24 | P3 | **Contract change:** `chunk_tokens`/`child_tokens` capped at **510** (`MAX_EMBED_TOKENS`), not 512; new `chunking.carry_table_header` | bge's 512 includes `[CLS]`/`[SEP]`. Carrying the column header raised table chunks that contain it from 64% to 94% |
| 2026-11-24 | P3 | Token counts via one tokenization per document + binary search over token offsets (`DocText`) | Exact sizes with the model's tokenizer at ~10 s per corpus pass, instead of re-tokenizing every candidate piece |
| 2026-11-24 | P3 | Default stays `page_aware` 384; **final chunk choice deferred to P5** | "Evidence in one chunk" favours big chunks by construction (fixed_480 scores highest but 60% of its chunks span two pages); retrieval metrics must decide |
| 2026-11-24 | P3 | `results/` reports use the local date; `sweep.csv` is ignored by the parent repo's `*.csv` rule, `report.md` holds the same table | Keep the committed artifact readable on GitHub |
| 2026-11-24 | P2 | **Row reconstruction from word boxes instead of `page.find_tables()`**; no `tables.py` (logic lives in `pdf_parser.py`) | Measured `find_tables()` at 1.5 s/page (≈5 h for 12k pages) with cells that lost spaces and split numbers. Rows from `get_text("words")` take ~3 ms/page and raise label–number adjacency from 0.28 to 0.97 |
| 2026-11-24 | P2 | **Contract change:** new `IngestionCfg` section (`ingestion:` in base.yaml), included in the fingerprint | Parsing settings change results, so they must be traceable |
| 2026-11-24 | P2 | Evidence overlap compares **space-insensitively** (substring match on letters+digits) | FinanceBench's evidence text came from an older extractor that dropped spaces ("Netsales"); PyMuPDF 2.1 keeps them. A space-sensitive metric would penalise correct parsing |
| 2026-11-24 | P2 | Header/footer candidates must be short (≤ 12 words) and at a page edge | A synthetic test with repeated body lines showed edge-only detection could empty pages; real headers are short |
| 2026-11-24 | P2 | Sequential parsing by default (`workers=1`); processes only via the CLI | macOS starts workers by re-importing `__main__`; cell-based scripts have no main guard, so workers would re-run the script |
| 2026-11-24 | P2 | Label–number adjacency reported only on pages with ≥ 5 numbers | On prose pages a single stray number gives a meaningless 0.0 |
| 2026-11-23 | P1 | **Step scripts became cell-based** (`# %%`) and run three ways: terminal, VS Code cells, Jupyter (`notebooks/` generated by jupytext). New `finsight.notebook` helpers (`script_args`, `show`, `preview`, `step`). Added dev deps `jupytext`, `jupyterlab`. `tools/build_notebooks.py` gives cells fixed IDs (jupytext's CLI assigns random IDs, so every rebuild showed as a git diff) and presets the kernel; `make kernel` registers `.venv` as "Python 3.12 (finsight-rag)" | Requested by the user: run each step individually and see what happens inside, with pandas DataFrames. Typer was dropped from the scripts (the `finsight` CLI keeps it) |
| 2026-11-23 | P1 | Data-card correction: the 50 unlabelled `question_reasoning` rows are the **novel-generated** questions | Found with the new `pd.crosstab` cell in step 02; the first draft wrongly said domain-relevant |
| 2026-11-23 | P1 | `.gitignore` rules anchored to the project root (`/data/`, `/indexes/`, `/models/`) | The unanchored `data/` also matched the source package `src/finsight/data/`, which silently left the P1 code untracked. Watch for this with any new folder name: the parent repo's `.gitignore` has an unanchored `indexes/` (we use `indexing/` for code) and ignores `*.parquet` and `*.csv` anywhere, so test fixtures must be JSONL |
| 2026-11-23 | P1 | **Contract change:** added `EvalRecord.question_reasoning: str | None = None` | Lets P5/P8 break results down by reasoning type (numerical vs. extraction) |
| 2026-11-23 | P1 | Added `validate_dataset()` and a download manifest (size + SHA-256 per file) | Consistency checks between questions, metadata and PDFs; proof of exactly which data produced the results |
| 2026-11-23 | P1 | The duplicate `FOOTLOCKER_2023_annualreport` metadata row: keep the first, log at INFO | Known upstream quirk; no question uses that doc (see data_card.md) |
| 2026-11-23 | P1 | Skipped the optional `notebooks/01_explore_financebench.ipynb` | `scripts/02_explore_data.py` covers the same exploration and runs in CI-friendly form |
| 2026-11-23 | P0 | **Installed versions are newer than this guide assumed:** torch 2.15, sentence-transformers 6.3, transformers 5.24, pymupdf 2.1, qdrant-client 1.20, anthropic 2.12, fastapi 0.165 | Locked in `uv.lock`. Re-check the library calls in P2 (pymupdf), P4 (sentence-transformers, qdrant `query_points`) and P6 (anthropic) against the current docs before relying on them |
| 2026-11-23 | P0 | Config uses pydantic models + an explicit deep merge + `python-dotenv`, not `pydantic-settings` sources | The YAML → profile → env-alias → `FINSIGHT__` → override chain is easier to read and test as one explicit function. Secrets are a separate `Secrets` model (SecretStr), excluded from dumps and the fingerprint |
| 2026-11-23 | P0 | Fingerprint covers only result-affecting sections (dataset, chunking, embedding, retrieval, rerank, pipeline, generation, llm, judge) | Paths, API and logging settings are machine-specific and must not change the result hash |
| 2026-11-23 | P0 | Added `diagnostics.py` (shared by the script and `finsight check-env`) and a `finsight show-config` command | Keeps the step script thin; lets you print the resolved config |
| 2026-11-23 | P0 | Linux installs use the CPU-only torch index (`[tool.uv.sources]`) | CI and Docker don't need multi-GB CUDA wheels; macOS keeps MPS |
| 2026-11-23 | P0 | CI uses `actions/checkout@v7` and `astral-sh/setup-uv@v10.4.0` (latest releases) | The draft's versions were outdated. setup-uv publishes no moving `v10` tag (the first CI run failed on it), so it is pinned to the full tag |
| 2026-11-23 | P0 | pre-commit config lives in `finsight-rag/` and is installed with `-c` from the git root; ruff `known-first-party = ["finsight"]` | The git root is the parent folder; without the setting, ruff sorted imports differently depending on the directory it ran from |
| 2026-11-23 | spec | Page indexes stored 0-based (`page_idx`), shown +1 | FinanceBench `evidence_page_num` is 0-based (verified against PyMuPDF) |
| 2026-11-23 | spec | Download PDFs directly from GitHub raw, not via HF `datasets` | Simpler, no extra dependency; the HF repo holds only the JSONL |
| 2026-11-23 | spec | Default metadata filter = company only | A year filter can drop the prior-year data that 10-Ks contain; tested as an ablation |

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `torch` wheel install fails | Python 3.13/3.14 in use | `uv python pin 3.12 && uv sync` |
| `warning: VIRTUAL_ENV=.../hybrid-retrieval/venv does not match ... .venv` | The parent experiment's venv is still activated in your shell | Harmless (uv ignores it). Run `deactivate`, or open a new terminal |
| New source folder missing from `git status` | A `.gitignore` pattern matches its name | `git check-ignore -v <path>` shows which rule; anchor the rule with a leading `/` |
| VS Code says packages are "not installed" | The editor is using the parent `venv` interpreter | Cmd+Shift+P → "Python: Select Interpreter" → `finsight-rag/.venv/bin/python` |
| Embedding runs on CPU | MPS not detected | Check `torch.backends.mps.is_available()`; update macOS or torch |
| `RuntimeError: Storage folder ... is already accessed by another instance of Qdrant client` | Embedded Qdrant lock (a notebook or the API still has it open) | Run the last cell (`c.close()`) or restart that kernel; or run Qdrant in Docker and set `QDRANT_URL` |
| `make index` re-embeds everything after re-chunking | Chunk text changed (expected); only new texts miss the cache | Check `cache_hits` in the IndexReport; delete `indexes/embed_cache/` only to reclaim space |
| Answers ignore most of the context | Ollama `num_ctx` too small | Set `options.num_ctx` (8192) in `ollama.py` |
| Anthropic 400 "temperature ..." | A sampling parameter was sent | Remove `temperature`, `top_p` and `top_k` |
| `ConnectError` to Ollama | The server isn't running | `ollama serve`, or open the Ollama app |
| Evidence-page check is low | Page offset or parsing issue | Check the 0-based convention; inspect the worst 5 in `03_parse_pdfs.py` |
| `MissingAPIKeyError: Embedder 'gemini-001' needs GEMINI_API_KEY` | Online preset selected, key not in `.env` | Add the key (free link in the message), or `EMBEDDER=bge-small` |
| `PaidProviderBlockedError` | An OpenAI preset with `allow_paid: false` | Intended: pick a free preset. Only set `FINSIGHT__EMBEDDING__ALLOW_PAID=true` if you mean to pay |
| `QuotaExhaustedError: daily request cap reached` | The preset's `rpd`/`rpmonth` (or the provider's) is used up | Re-run after the reset (UTC day): cached vectors are not re-sent. Or use `--limit-questions 30` |
| Repeated "provider request failed, retrying" (429) | Client limits above your account's real limits | Lower `rate_limit.rpm`/`tpm` for that preset (`FINSIGHT__EMBEDDING__RATE_LIMIT__RPM=10`) |
| Swap or memory pressure during indexing | Too much in RAM | Lower `embedding.batch_size`, close apps, index one strategy at a time |

---

## 9. Reference index

**Core RAG and retrieval:** [RAG (Lewis et al.)](https://arxiv.org/abs/2005.11401) · [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) · [RRF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) · [Sentence-BERT](https://arxiv.org/abs/1908.10084) · [BGE / C-Pack](https://arxiv.org/abs/2309.07597) · [HNSW](https://arxiv.org/abs/1603.09320) · [ColBERT](https://arxiv.org/abs/2004.12832) · [BEIR](https://arxiv.org/abs/2104.08663) · [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

**Advanced RAG:** [HyDE](https://arxiv.org/abs/2212.10496) · [RAG-Fusion](https://arxiv.org/abs/2402.03367) · [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) · [CRAG](https://arxiv.org/abs/2401.15884) · [Self-RAG](https://arxiv.org/abs/2310.11511) · [ReAct](https://arxiv.org/abs/2210.03629) · [Lost in the Middle](https://arxiv.org/abs/2307.03172) · [GraphRAG (Microsoft)](https://arxiv.org/abs/2404.16130)

**Evaluation:** [LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) · [RAGAS](https://arxiv.org/abs/2309.15217) · [ARES](https://arxiv.org/abs/2311.09476) · [NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) · [FinanceBench](https://arxiv.org/abs/2311.11944) · [MultiHop-RAG](https://arxiv.org/abs/2401.15391)

**Libraries:** [uv](https://docs.astral.sh/uv/) · [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) · [Typer](https://typer.tiangolo.com/) · [PyMuPDF](https://pymupdf.readthedocs.io/) · [sentence-transformers](https://sbert.net/) · [Qdrant](https://qdrant.tech/documentation/) · [bm25s](https://github.com/xhluca/bm25s) · [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md) · [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) · [Claude docs](https://platform.claude.com/docs) · [FastAPI](https://fastapi.tiangolo.com/) · [Streamlit](https://docs.streamlit.io/)

**Production:** [12-Factor App](https://12factor.net/) · [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) · [Docker multi-stage](https://docs.docker.com/build/building/multi-stage/) · [GitHub Actions syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
