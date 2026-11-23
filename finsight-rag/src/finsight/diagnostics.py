"""
Environment checks: "is this machine ready to run FinSight?"

WHY
  Most "it doesn't work" problems in ML projects are environmental: wrong Python, no GPU
  backend, a local server not running, a missing key. One command that checks them all saves
  hours. The step script ``scripts/00_check_env.py`` and ``finsight check-env`` both render
  these results.

STATUS LEVELS
  ok    working as expected
  info  not needed yet (e.g. data folder before P1)
  warn  optional feature unavailable (e.g. no Anthropic key, Ollama not running)
  fail  the project cannot run (the script exits with code 1)
"""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from rich.console import Console
from rich.table import Table

from finsight.config import Settings, mask_secret

Status = Literal["ok", "info", "warn", "fail"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    detail: str
    hint: str = ""


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _rel(path: Path | None, root: Path) -> str:
    """Show paths relative to the project root when possible: shorter and easier to read."""
    if path is None:
        return "-"
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def check_python() -> CheckResult:
    v = sys.version_info
    detail = f"{platform.python_version()} ({platform.machine()})"
    if (v.major, v.minor) == (3, 12):
        return CheckResult("Python", "ok", detail)
    return CheckResult("Python", "fail", detail, "Run: uv python pin 3.12 && uv sync")


def check_torch() -> CheckResult:
    """Imports torch (about 1–2 s); only diagnostics do this eagerly."""
    try:
        import torch
    except ImportError:
        return CheckResult("PyTorch", "fail", "not installed", "Run: uv sync")
    if torch.backends.mps.is_available():
        return CheckResult("PyTorch", "ok", f"{torch.__version__} · Apple GPU (MPS) available")
    if torch.cuda.is_available():
        return CheckResult("PyTorch", "ok", f"{torch.__version__} · CUDA available")
    return CheckResult(
        "PyTorch", "warn", f"{torch.__version__} · CPU only", "Embedding/reranking will be slower"
    )


def check_key_packages() -> CheckResult:
    names = ["sentence-transformers", "qdrant-client", "bm25s", "pymupdf", "anthropic", "fastapi"]
    found = {n: _pkg_version(n) for n in names}
    missing = [n for n, v in found.items() if v is None]
    detail = ", ".join(f"{n} {v}" for n, v in found.items() if v)
    if missing:
        return CheckResult("Packages", "fail", f"missing: {', '.join(missing)}", "Run: uv sync")
    return CheckResult("Packages", "ok", detail)


def check_config(settings: Settings) -> CheckResult:
    return CheckResult(
        "Config",
        "ok",
        f"{_rel(settings.config_path, settings.root)} · fingerprint {settings.fingerprint()} · "
        f"llm={settings.llm.provider} judge={settings.judge.provider} "
        f"rerank={settings.rerank.provider}",
    )


def check_folders(settings: Settings) -> list[CheckResult]:
    p = settings.paths
    items = [
        ("Raw data", p.raw_dir, "P1: uv run python scripts/01_download_data.py"),
        ("Processed data", p.processed_dir, "P2: uv run python scripts/03_parse_pdfs.py"),
        ("Indexes", p.index_dir, "P4: uv run python scripts/05_embed_and_index.py"),
    ]
    out = []
    for name, path, hint in items:
        if path.exists() and any(path.iterdir()):
            out.append(CheckResult(name, "ok", _rel(path, settings.root)))
        else:
            out.append(
                CheckResult(name, "info", f"{_rel(path, settings.root)}/ (not created yet)", hint)
            )
    return out


def check_ollama(settings: Settings, timeout_s: float = 2.0) -> CheckResult:
    base = settings.llm.ollama_base_url.rstrip("/")
    wanted = settings.llm.ollama_model
    needed = settings.llm.provider == "ollama" or settings.judge.provider == "ollama"
    level: Status = "warn" if needed else "info"
    try:
        resp = httpx.get(f"{base}/api/tags", timeout=timeout_s)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
    except (httpx.HTTPError, ValueError) as exc:
        return CheckResult(
            "Ollama",
            level,
            f"not reachable at {base} ({type(exc).__name__})",
            "Install from https://ollama.com/download (or brew install ollama), then: ollama serve",
        )
    if wanted in models or f"{wanted}:latest" in models:
        return CheckResult("Ollama", "ok", f"{base} · model '{wanted}' ready")
    return CheckResult(
        "Ollama",
        level,
        f"{base} reachable, but model '{wanted}' not pulled (have: {models or 'none'})",
        f"Run: ollama pull {wanted}",
    )


def check_secrets(settings: Settings) -> list[CheckResult]:
    s = settings.secrets
    needs_anthropic = "anthropic" in (settings.llm.provider, settings.judge.provider)
    anthropic_status: Status = (
        "ok" if s.anthropic_api_key else ("fail" if needs_anthropic else "info")
    )
    cohere_status: Status = (
        "ok" if s.cohere_api_key else ("fail" if settings.rerank.provider == "cohere" else "info")
    )
    return [
        CheckResult(
            "ANTHROPIC_API_KEY",
            anthropic_status,
            mask_secret(s.anthropic_api_key),
            "" if s.anthropic_api_key else "Optional: add to .env to use configs/cloud.yaml",
        ),
        CheckResult(
            "COHERE_API_KEY",
            cohere_status,
            mask_secret(s.cohere_api_key),
            "" if s.cohere_api_key else "Optional: only for the Cohere reranker comparison",
        ),
    ]


_ICONS = {
    "ok": "[green]✔ ok[/]",
    "info": "[blue]ℹ info[/]",
    "warn": "[yellow]⚠ warn[/]",
    "fail": "[red]✘ fail[/]",
}


def render_checks(results: list[CheckResult], console: Console) -> None:
    """Print the results as a table (used by scripts/00_check_env.py and `finsight check-env`)."""
    table = Table(show_lines=False, header_style="bold")
    table.add_column("Check", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Detail")
    table.add_column("How to fix / next step", style="dim")
    for r in results:
        table.add_row(r.name, _ICONS[r.status], r.detail, r.hint)
    console.print(table)


def run_all_checks(settings: Settings) -> list[CheckResult]:
    return [
        check_python(),
        check_torch(),
        check_key_packages(),
        check_config(settings),
        *check_folders(settings),
        check_ollama(settings),
        *check_secrets(settings),
    ]
