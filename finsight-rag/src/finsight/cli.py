"""
The ``finsight`` command-line interface: the production entry point.

WHY TWO ENTRY POINTS?
  scripts/NN_*.py  learning steps: verbose, print samples, explain what happened.
  finsight <cmd>   automation: concise, scriptable, used by the Makefile, Docker and CI.
  Both call the same library functions, so they can never drift apart.

Commands not built yet exit with code 2 and name the phase that adds them.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from finsight.config import load_settings
from finsight.errors import FinsightError

app = typer.Typer(add_completion=False, no_args_is_help=True, help="FinSight RAG command line.")
console = Console()

ConfigOpt = typer.Option(
    None, "--config", "-c", help="Profile YAML (default: $FINSIGHT_CONFIG or configs/local.yaml)"
)


@app.command("check-env")
def check_env(config: Path | None = ConfigOpt) -> None:
    """Check Python, PyTorch/MPS, packages, config, folders, Ollama and API keys."""
    from finsight.diagnostics import render_checks, run_all_checks

    try:
        settings = load_settings(config)
    except FinsightError as exc:
        console.print(f"[red]✘ {exc}[/]")
        raise typer.Exit(1) from exc
    results = run_all_checks(settings)
    render_checks(results, console)
    if any(r.status == "fail" for r in results):
        raise typer.Exit(1)


@app.command("show-config")
def show_config(config: Path | None = ConfigOpt) -> None:
    """Print the fully resolved configuration (secrets excluded) and its fingerprint."""
    settings = load_settings(config)
    console.print_json(settings.model_dump_json(exclude={"paths"}))
    console.print(f"fingerprint: [bold]{settings.fingerprint()}[/]")


def _not_yet(phase: str) -> None:
    console.print(f"[yellow]Not implemented yet. This command arrives in phase {phase}.[/]")
    raise typer.Exit(2)


@app.command()
def download() -> None:
    """Download FinanceBench (Q&A, doc metadata, referenced PDFs). [P1]"""
    _not_yet("P1")


@app.command()
def ingest() -> None:
    """Parse PDFs into pages and chunk them. [P2–P3]"""
    _not_yet("P2/P3")


@app.command()
def index() -> None:
    """Embed chunks and build the Qdrant + BM25 indexes. [P4]"""
    _not_yet("P4")


@app.command()
def query(question: str) -> None:
    """Answer a question with citations. [P6]"""
    _not_yet("P6")


@app.command("eval")
def eval_cmd() -> None:
    """Run an evaluation experiment. [P5/P8]"""
    _not_yet("P5/P8")


@app.command()
def serve() -> None:
    """Start the FastAPI server. [P9]"""
    _not_yet("P9")


if __name__ == "__main__":
    app()
