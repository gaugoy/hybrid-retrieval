# FinSight RAG: API + UI image (multi-stage, non-root, models and indexes mounted, not baked in).
#   docker compose up            (see docker-compose.yml and docs/runbook.md)
# Linux wheels: CPU-only torch (pyproject [tool.uv.sources]); Docker on macOS has no GPU passthrough,
# so the fast path for the LLM is Ollama on the host (OLLAMA_BASE_URL=http://host.docker.internal:11434).

# ---------------------------------------------------------------- builder
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# dependencies first (cached layer), then the project itself
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ui --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ui

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src ./src
COPY --chown=app:app configs ./configs
COPY --chown=app:app ui ./ui
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    FINSIGHT_ROOT=/app \
    FINSIGHT__LOGGING__JSON_FORMAT=true
USER app
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["uvicorn", "finsight.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
