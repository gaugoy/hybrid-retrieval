"""
Typed configuration: one ``Settings`` object built from layered sources.

WHY
  A RAG system has dozens of knobs: chunk size, k values, model names, providers. When those are
  scattered as constants across files, you can't tell which settings produced a given result.
  Here every knob lives in one validated object, and ``Settings.fingerprint()`` hashes the
  settings that affect results. Every index manifest and every eval report stores that hash,
  so any number can be traced back to the exact config that produced it.

LAYERS (lowest → highest precedence)
  1. configs/base.yaml                 defaults for everything
  2. the chosen YAML (local/cloud/...) deep-merged on top
  3. environment variables              FINSIGHT__SECTION__KEY=value, plus a few friendly aliases
                                        (LLM_PROVIDER, OLLAMA_MODEL, QDRANT_URL, ...)
  4. ``overrides`` passed in code       e.g. by the eval runner for each experiment variant

Secrets (API keys) are kept apart in ``Secrets``. They never come from YAML and are never part
of the fingerprint or of ``model_dump()``.

More info: https://12factor.net/config
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator

from finsight.errors import ConfigError

DEFAULT_PROFILE = "configs/local.yaml"
ENV_PREFIX = "FINSIGHT__"

# Friendly env-var names (used in .env.example) → dotted config path.
ENV_ALIASES: dict[str, str] = {
    "LLM_PROVIDER": "llm.provider",
    "OLLAMA_BASE_URL": "llm.ollama_base_url",
    "OLLAMA_MODEL": "llm.ollama_model",
    "ANTHROPIC_MODEL": "llm.anthropic_model",
    "JUDGE_PROVIDER": "judge.provider",
    "QDRANT_URL": "vector_store.qdrant_url",
    "LOG_LEVEL": "logging.level",
}

# Only these sections change results, so only they go into the fingerprint.
# Paths, API and logging settings are machine- or deployment-specific.
FINGERPRINT_FIELDS = {
    "dataset",
    "ingestion",
    "chunking",
    "embedding",
    "retrieval",
    "rerank",
    "pipeline",
    "generation",
    "llm",
    "judge",
}


class _Strict(BaseModel):
    """Base for config sections: unknown keys are errors, so typos in YAML don't pass silently."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------------------------
# Config sections
# --------------------------------------------------------------------------------------------


class PathsCfg(_Strict):
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    pdf_dir: Path = Path("data/raw/pdfs")
    processed_dir: Path = Path("data/processed")
    chunks_dir: Path = Path("data/chunks")
    index_dir: Path = Path("indexes")
    qdrant_dir: Path = Path("indexes/qdrant")
    bm25_dir: Path = Path("indexes/bm25")
    embed_cache_dir: Path = Path("indexes/embed_cache")
    results_dir: Path = Path("results")

    def resolved(self, root: Path) -> PathsCfg:
        """Return a copy with every relative path made absolute against ``root``."""
        values = {
            name: (p if p.is_absolute() else (root / p).resolve())
            for name, p in self.model_dump().items()
        }
        return PathsCfg(**values)


class IngestionCfg(_Strict):
    """How PDF pages become text (P2). See finsight.ingestion for what each knob does."""

    row_tolerance_factor: float = Field(0.5, gt=0, le=2)  # same row if |Δy| ≤ factor × line height
    column_gap_factor: float = Field(
        1.5, gt=0
    )  # gap > factor × line height starts a new " | " cell
    paragraph_gap_factor: float = Field(
        1.2, gt=0
    )  # vertical gap > factor × line height → blank line
    boilerplate_edge_lines: int = Field(3, ge=0)  # only the top/bottom N lines can be header/footer
    boilerplate_min_share: float = Field(0.5, gt=0, le=1)  # repeated on ≥ this share of pages
    min_table_rows: int = Field(3, ge=1)  # rows with ≥2 cells incl. a number → page has_table


class ChunkingCfg(_Strict):
    strategy: Literal["fixed", "recursive", "page_aware", "parent_child"] = "page_aware"
    chunk_tokens: int = Field(384, gt=0, le=512)
    overlap_tokens: int = Field(64, ge=0)
    child_tokens: int = Field(200, gt=0, le=512)
    parent_tokens: int = Field(1500, gt=0)
    contextual_headers: bool = False

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> ChunkingCfg:
        if self.overlap_tokens >= self.chunk_tokens:
            raise ValueError("chunking.overlap_tokens must be smaller than chunking.chunk_tokens")
        return self


class EmbeddingCfg(_Strict):
    model: str = "BAAI/bge-small-en-v1.5"
    device: Literal["auto", "mps", "cuda", "cpu"] = "auto"
    batch_size: int = Field(64, gt=0)
    query_instruction: str = "Represent this sentence for searching relevant passages: "


class VectorStoreCfg(_Strict):
    qdrant_url: str | None = None  # None → embedded mode at paths.qdrant_dir


class RetrievalCfg(_Strict):
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    candidate_k: int = Field(50, gt=0)
    top_k: int = Field(10, gt=0)
    rrf_k: int = Field(60, gt=0)
    use_metadata_filter: bool = True
    filter_fields: list[Literal["company", "doc_period", "doc_type"]] = Field(
        default_factory=lambda: ["company"]  # type: ignore[arg-type]
    )
    query_transform: Literal["none", "multi_query", "hyde", "decompose"] = "none"


class RerankCfg(_Strict):
    provider: Literal["none", "local", "cohere"] = "local"
    model: str = "BAAI/bge-reranker-base"
    top_n: int = Field(8, gt=0)
    cohere_model: str = "rerank-v4.0-fast"


class GenerationCfg(_Strict):
    max_context_tokens: int = Field(6000, gt=0)
    max_answer_tokens: int = Field(1024, gt=0)
    abstain_phrase: str = "Not found in the provided filings."


class LLMCfg(_Strict):
    provider: Literal["ollama", "anthropic"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_num_ctx: int = Field(8192, gt=0)
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: Literal["low", "medium", "high"] = "medium"
    timeout_s: float = Field(120, gt=0)


class JudgeCfg(_Strict):
    provider: Literal["ollama", "anthropic"] = "ollama"
    ollama_model: str = "qwen2.5:7b-instruct"
    anthropic_model: str = "claude-haiku-5-5"
    anthropic_effort: Literal["low", "medium", "high"] = "low"


class ApiCfg(_Strict):
    host: str = "127.0.0.1"
    port: int = Field(8000, gt=0, lt=65536)
    warmup: bool = False
    max_question_chars: int = Field(1000, gt=0)
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8501"])


class LoggingCfg(_Strict):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_format: bool = False


class Secrets(BaseModel):
    """API keys. SecretStr hides them in repr/logs; call ``.get_secret_value()`` to use one."""

    anthropic_api_key: SecretStr | None = None
    cohere_api_key: SecretStr | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Secrets:
        def pick(name: str) -> SecretStr | None:
            value = (env.get(name) or "").strip()
            return SecretStr(value) if value else None

        return cls(
            anthropic_api_key=pick("ANTHROPIC_API_KEY"),
            cohere_api_key=pick("COHERE_API_KEY"),
        )


class Settings(_Strict):
    dataset: Literal["financebench", "multihop"] = "financebench"
    paths: PathsCfg = Field(default_factory=PathsCfg)
    ingestion: IngestionCfg = Field(default_factory=IngestionCfg)
    chunking: ChunkingCfg = Field(default_factory=ChunkingCfg)
    embedding: EmbeddingCfg = Field(default_factory=EmbeddingCfg)
    vector_store: VectorStoreCfg = Field(default_factory=VectorStoreCfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)
    rerank: RerankCfg = Field(default_factory=RerankCfg)
    pipeline: Literal["naive", "hybrid_rerank", "corrective", "agentic"] = "hybrid_rerank"
    generation: GenerationCfg = Field(default_factory=GenerationCfg)
    llm: LLMCfg = Field(default_factory=LLMCfg)
    judge: JudgeCfg = Field(default_factory=JudgeCfg)
    api: ApiCfg = Field(default_factory=ApiCfg)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)

    # Runtime-only fields, excluded from model_dump() and therefore from the fingerprint.
    root: Path = Field(default=Path("."), exclude=True)
    config_path: Path | None = Field(default=None, exclude=True)
    secrets: Secrets = Field(default_factory=Secrets, exclude=True)

    def fingerprint(self) -> str:
        """Short, stable hash of every setting that can change a result."""
        payload = self.model_dump(mode="json", include=FINGERPRINT_FIELDS)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------


def find_project_root(start: Path | None = None) -> Path:
    """Locate the project root: the folder that contains ``configs/base.yaml``.

    Order: $FINSIGHT_ROOT → walk up from ``start``/cwd → the source checkout this file lives in.
    """
    env_root = os.environ.get("FINSIGHT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "configs" / "base.yaml").is_file():
            return candidate
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "configs" / "base.yaml").is_file():
        return source_root
    raise ConfigError(
        "Could not find the project root (a folder containing configs/base.yaml). "
        "Run from inside finsight-rag/ or set FINSIGHT_ROOT."
    )


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``. Dicts merge; other values replace."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _set_dotted(target: dict[str, Any], dotted: str, value: Any) -> None:
    node = target
    *parents, leaf = dotted.split(".")
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value


def _parse_env_value(raw: str) -> Any:
    """Env values are strings; let JSON/YAML scalars through so lists, ints and bools work."""
    raw = raw.strip()
    if raw.lower() in {"null", "none"}:
        return None
    if raw[:1] in "[{":
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw  # pydantic coerces "64" → 64, "true" → True


def env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    """Turn environment variables into a nested override dict. Empty values are ignored."""
    out: dict[str, Any] = {}
    for name, dotted in ENV_ALIASES.items():
        value = env.get(name)
        if value is not None and value.strip():
            _set_dotted(out, dotted, _parse_env_value(value))
    for name, value in env.items():
        if name.startswith(ENV_PREFIX) and value is not None and value.strip():
            dotted = name[len(ENV_PREFIX) :].lower().replace("__", ".")
            _set_dotted(out, dotted, _parse_env_value(value))
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping, got {type(data).__name__}")
    return data


def load_settings(
    config: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    root: Path | None = None,
) -> Settings:
    """Build the resolved ``Settings``.

    Args:
        config: profile YAML (relative to the project root or absolute). Defaults to
            $FINSIGHT_CONFIG, then ``configs/local.yaml``.
        overrides: nested dict applied last (highest precedence).
        env: environment mapping. Defaults to ``.env`` values overlaid by the real environment
            (real environment variables win). Tests pass their own dict.
        root: project root. Defaults to ``find_project_root()``.
    """
    root = (root or find_project_root()).resolve()
    if env is None:
        dotenv = {k: v for k, v in dotenv_values(root / ".env").items() if v is not None}
        env = {**dotenv, **os.environ}

    config = config or env.get("FINSIGHT_CONFIG") or DEFAULT_PROFILE
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = root / config_path

    merged = _read_yaml(root / "configs" / "base.yaml")
    if config_path.resolve() != (root / "configs" / "base.yaml").resolve():
        merged = deep_merge(merged, _read_yaml(config_path))
    merged = deep_merge(merged, env_overrides(env))
    if overrides:
        merged = deep_merge(merged, overrides)

    try:
        settings = Settings.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(
            f"Invalid configuration ({config_path.name} + env/overrides):\n{exc}"
        ) from exc

    settings.root = root
    settings.config_path = config_path
    settings.paths = settings.paths.resolved(root)
    settings.secrets = Secrets.from_env(env)
    return settings


def mask_secret(value: SecretStr | str | None) -> str:
    """Show just enough of a key to recognise it: 'sk-...abcd'. Never log full keys."""
    if value is None:
        return "not set"
    raw = value.get_secret_value() if isinstance(value, SecretStr) else value
    if len(raw) <= 12:
        return "***"
    return f"{raw[:3]}...{raw[-4:]}"
