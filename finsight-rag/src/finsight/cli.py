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

from finsight.config import Settings, load_settings
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
def download(
    config: Path | None = ConfigOpt,
    limit_docs: int | None = typer.Option(None, help="Only the first N PDFs (trial run)"),
    force: bool = typer.Option(False, help="Re-download even if files exist"),
) -> None:
    """Download FinanceBench (Q&A, doc metadata, referenced PDFs). [P1]"""
    from finsight.data.download import download_financebench
    from finsight.observability.logging import configure_logging

    settings = load_settings(config)
    configure_logging(settings.logging.level, settings.logging.json_format)
    report = download_financebench(settings, limit_docs=limit_docs, force=force)
    console.print(
        f"questions={report.n_questions} docs={report.n_docs_needed} "
        f"downloaded={report.n_downloaded} skipped={report.n_skipped} "
        f"on_disk={report.bytes_on_disk / 1e6:.1f}MB time={report.elapsed_s}s"
    )
    if not report.ok:
        for name, err in report.failures.items():
            console.print(f"[red]✘ {name}: {err}[/]")
        raise typer.Exit(1)


@app.command()
def ingest(
    config: Path | None = ConfigOpt,
    limit_docs: int | None = typer.Option(None, help="Only the first N documents (trial run)"),
    workers: int = typer.Option(0, help="Processes: 0 = all cores, 1 = sequential"),
    force: bool = typer.Option(False, help="Re-parse even if outputs are up to date"),
) -> None:
    """Parse PDFs into clean pages, then chunk them (configured strategy). [P2+P3]"""
    from finsight.ingestion.pipeline import ingest_all
    from finsight.observability.logging import configure_logging

    settings = load_settings(config)
    configure_logging(settings.logging.level, settings.logging.json_format)
    report = ingest_all(settings, limit_docs=limit_docs, workers=workers, force=force)
    console.print(
        f"parse: docs={report.n_docs} pages={report.n_pages} empty={report.n_empty_pages} "
        f"tables={report.n_table_pages} skipped={report.skipped} time={report.elapsed_s}s"
    )
    if report.failures:
        for name, err in report.failures.items():
            console.print(f"[red]✘ {name}: {err}[/]")
        raise typer.Exit(1)
    _chunk(settings, force=force)


def _chunk(settings: Settings, force: bool = False) -> None:
    from finsight.chunking.pipeline import chunk_corpus

    r = chunk_corpus(settings, force=force)
    console.print(
        f"chunk: {r.name} chunks={r.n_chunks} parents={r.n_parents} "
        f"skipped={r.skipped} time={r.elapsed_s}s"
    )


@app.command()
def chunk(
    config: Path | None = ConfigOpt,
    strategy: str | None = typer.Option(None, help="fixed | recursive | page_aware | parent_child"),
    tokens: int | None = typer.Option(None, help="chunk_tokens (child_tokens for parent_child)"),
    sweep: bool = typer.Option(
        False, help="Run every strategy × size and save a report in results/"
    ),
    force: bool = typer.Option(False, help="Re-chunk even if up to date"),
) -> None:
    """Chunk pages.parquet with the configured (or given) strategy. [P3]"""
    from finsight.chunking.pipeline import run_sweep, save_sweep_report, to_markdown

    overrides: dict[str, object] = {}
    if strategy:
        overrides["strategy"] = strategy
    if tokens:
        overrides["child_tokens" if strategy == "parent_child" else "chunk_tokens"] = tokens
    settings = load_settings(config, {"chunking": overrides} if overrides else None)
    if sweep:
        df = run_sweep(settings, force=force)
        out = save_sweep_report(settings, df)
        console.print(to_markdown(df))
        console.print(f"report: {out}")
        return
    _chunk(settings, force=force)


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
