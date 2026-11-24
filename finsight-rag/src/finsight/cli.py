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

    overrides = _chunking_overrides(strategy, tokens)
    settings = load_settings(config, {"chunking": overrides} if overrides else None)
    if sweep:
        df = run_sweep(settings, force=force)
        out = save_sweep_report(settings, df)
        console.print(to_markdown(df))
        console.print(f"report: {out}")
        return
    _chunk(settings, force=force)


def _chunking_overrides(strategy: str | None, tokens: int | None) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if strategy:
        overrides["strategy"] = strategy
    if tokens:
        overrides["child_tokens" if strategy == "parent_child" else "chunk_tokens"] = tokens
    return overrides


EmbedderOpt = typer.Option(
    None,
    "--embedder",
    "-e",
    help="Embedding preset (configs/embedders.yaml; list: finsight embedders)",
)


def _settings(
    config: Path | None, strategy: str | None, tokens: int | None, embedder: str | None
) -> Settings:
    overrides: dict[str, object] = {}
    chunking = _chunking_overrides(strategy, tokens)
    if chunking:
        overrides["chunking"] = chunking
    if embedder:
        overrides["embedding"] = {"preset": embedder}
    return load_settings(config, overrides or None)


@app.command()
def embedders(
    config: Path | None = ConfigOpt,
    estimate: bool = typer.Option(True, help="Estimate requests/time/cost for the current chunks"),
) -> None:
    """List embedding presets: provider, free/paid, API-key status, cost and time estimates. [P4]"""
    from finsight.chunking.pipeline import load_chunks_df
    from finsight.embedding.catalog import catalog_df
    from finsight.errors import DataNotFoundError

    settings = load_settings(config)
    texts = None
    if estimate:
        try:
            texts = load_chunks_df(settings)["embed_text"].tolist()
        except DataNotFoundError:
            console.print("[dim](no chunks yet: run make ingest for estimates)[/]")
    from rich.table import Table

    df = catalog_df(settings, texts)
    table = Table(header_style="bold", title="Embedding presets (configs/embedders.yaml)")
    cols = ["preset", "provider", "model", "dims", "cost", "key"]
    if texts is not None:
        cols += ["requests", "minutes_at_limits", "quota", "usd"]
    for c in cols:
        table.add_column(c.replace("minutes_at_limits", "min @ limits"), overflow="fold")
    for row in df[cols].itertuples(index=False):
        cells = ["?" if isinstance(v, float) and v != v else str(v) for v in row]
        ready = row.key in ("set", "not needed")
        table.add_row(*cells, style="red" if row.cost == "PAID" else ("green" if ready else "dim"))
    console.print(table)
    console.print(
        f"\ncurrent: embedding.preset = {settings.embedding.preset}  "
        "(set EMBEDDER=<preset> in .env, or pass --embedder)"
    )
    if texts is not None:
        console.print(
            f"estimates for {len(texts):,} chunks of {settings.chunking.strategy}_"
            f"{settings.chunking.chunk_tokens}; notes: configs/embedders.yaml"
        )


@app.command()
def index(
    config: Path | None = ConfigOpt,
    strategy: str | None = typer.Option(None, help="fixed | recursive | page_aware | parent_child"),
    tokens: int | None = typer.Option(None, help="chunk_tokens (child_tokens for parent_child)"),
    embedder: str | None = EmbedderOpt,
    limit_questions: int | None = typer.Option(
        None, help="Only index the filings the first N questions need (free-tier trials)"
    ),
    only: str = typer.Option("all", help="all | dense | sparse"),
    force: bool = typer.Option(False, help="Rebuild even if the manifest says up to date"),
) -> None:
    """Embed chunks and build the Qdrant + BM25 indexes (chunks first if needed). [P4]"""
    from finsight.indexing.pipeline import index_corpus, subset_for_questions
    from finsight.observability.logging import configure_logging

    if only not in ("all", "dense", "sparse"):
        raise typer.BadParameter("--only must be all, dense or sparse")
    settings = _settings(config, strategy, tokens, embedder)
    configure_logging(settings.logging.level, settings.logging.json_format)
    _chunk(settings)
    docs, tag = subset_for_questions(settings, limit_questions) if limit_questions else (None, None)
    r = index_corpus(settings, force=force, only=only, doc_names=docs, tag=tag)  # type: ignore[arg-type]
    console.print(
        f"dense: {r.collection} points={r.qdrant_points} skipped={r.dense_skipped} "
        f"device={r.device} cache hits/misses={r.cache_hits}/{r.cache_misses} "
        f"chunks/s={r.chunks_per_s} size={r.qdrant_mb}MB"
    )
    console.print(
        f"sparse: {r.bm25_dir} docs={r.bm25_docs} vocab={r.bm25_vocab} "
        f"skipped={r.bm25_skipped} size={r.bm25_mb}MB · total {r.elapsed_s}s"
    )
    if r.api_requests:
        console.print(f"api: {r.api_requests} requests, ~{r.api_est_tokens:,} tokens")


@app.command("compare-embedders")
def compare_embedders_cmd(
    config: Path | None = ConfigOpt,
    embedders: str = typer.Option(
        "bge-small,bge-small-noinstr", help="Comma-separated presets (finsight embedders)"
    ),
    limit_questions: int | None = typer.Option(
        None, help="Use the first N questions and only their filings (recommended for free tiers)"
    ),
    force: bool = typer.Option(False, help="Rebuild the indexes"),
) -> None:
    """Dense-retrieval comparison of embedders (local vs online) on the same chunks. [P4]"""
    from finsight.evaluation.embedder_compare import compare_embedders, save_report
    from finsight.observability.logging import configure_logging

    settings = load_settings(config)
    configure_logging(settings.logging.level, settings.logging.json_format)
    _chunk(settings)
    presets = [p.strip() for p in embedders.split(",") if p.strip()]
    comp = compare_embedders(settings, presets, limit_questions=limit_questions, force=force)
    if not comp.summary.empty:
        console.print(comp.summary.to_string())
    for preset, why in comp.skipped.items():
        console.print(f"[yellow]skipped {preset}: {why}[/]")
    console.print(f"report: {save_report(settings, comp)}")


@app.command()
def query(question: str) -> None:
    """Answer a question with citations. [P6]"""
    _not_yet("P6")


@app.command("eval")
def eval_cmd(
    config: Path | None = ConfigOpt,
    experiment: Path = typer.Option(
        Path("configs/experiments/retrieval_ablation.yaml"), help="Experiment YAML"
    ),
    limit: int | None = typer.Option(None, help="First N questions only"),
) -> None:
    """Run a retrieval evaluation experiment; report in results/. [P5]"""
    from finsight.evaluation.runner import load_experiment, run_retrieval_eval
    from finsight.observability.logging import configure_logging

    settings = load_settings(config)
    configure_logging(settings.logging.level, settings.logging.json_format)
    exp_path = experiment if experiment.is_absolute() else settings.root / experiment
    out = run_retrieval_eval(settings, load_experiment(exp_path), limit=limit)
    console.print((out / "report.md").read_text())
    console.print(f"report: {out}")


@app.command()
def serve() -> None:
    """Start the FastAPI server. [P9]"""
    _not_yet("P9")


if __name__ == "__main__":
    app()
