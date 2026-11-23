# FinSight RAG: common tasks. Run `make help` to list them.
# Every target runs inside the uv-managed environment (.venv, Python 3.12).

.DEFAULT_GOAL := help
.PHONY: help setup hooks check-env lint fmt test test-slow notebooks kernel jupyter data ingest index eval-retrieval eval serve ui clean-cache

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## install all dependencies (incl. dev + extras) and git hooks
	uv sync --all-extras
	$(MAKE) hooks

hooks: ## install pre-commit hooks (config lives in this subfolder; git root is the parent)
	cd .. && uv run --project finsight-rag pre-commit install -c finsight-rag/.pre-commit-config.yaml

check-env: ## P0: check Python, MPS, packages, config, Ollama, keys
	uv run python scripts/00_check_env.py

lint: ## ruff lint + format check + mypy
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src

fmt: ## auto-format and auto-fix lint issues
	uv run ruff format .
	uv run ruff check --fix .

test: ## fast tests (no real models / data) with coverage
	uv run pytest --cov=finsight --cov-report=term-missing

test-slow: ## tests that need real models or the full dataset
	uv run pytest -m slow

notebooks: ## regenerate notebooks/*.ipynb from the cell-based scripts/*.py (deterministic, no outputs)
	uv run python tools/build_notebooks.py

kernel: ## register .venv as the Jupyter kernel "Python 3.12 (finsight-rag)" (for VS Code / Jupyter)
	uv run python -m ipykernel install --user --name finsight-rag --display-name "Python 3.12 (finsight-rag)"

jupyter: ## open JupyterLab in notebooks/ (uses this project's .venv kernel)
	uv run jupyter lab notebooks/

data: ## P1: download FinanceBench
	uv run python scripts/01_download_data.py

ingest: ## P2: parse PDFs into clean pages (all cores; chunking is added in P3)
	uv run finsight ingest --workers 0

index: ## P4: embed + build Qdrant and BM25 indexes
	uv run python scripts/05_embed_and_index.py

eval-retrieval: ## P5: retrieval ablation
	uv run python scripts/07_eval_retrieval.py --experiment configs/experiments/retrieval_ablation.yaml

eval: ## P8: end-to-end evaluation
	uv run python scripts/10_eval_end_to_end.py --experiment configs/experiments/end_to_end.yaml

serve: ## P9: FastAPI dev server on :8000
	uv run uvicorn finsight.api.main:app --reload --port 8000

ui: ## P9: Streamlit demo UI on :8501
	uv run streamlit run ui/streamlit_app.py

clean-cache: ## remove tool caches (keeps data/ and indexes/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
