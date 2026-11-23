"""
Step 00 — Check the environment   (Phase P0)

WHAT THIS STEP DOES
  Loads the layered config (base.yaml + profile + .env + environment) and checks everything
  FinSight needs: Python 3.12, PyTorch with the Apple GPU (MPS), key packages, data/index
  folders, the local Ollama server and model, and (optionally) API keys.

WHAT TO LOOK AT AFTER RUNNING
  - The table: "fail" rows must be fixed before continuing. "warn" means an optional feature
    is off (for example Ollama, which you need from P6). "info" means "not needed yet".
  - The config fingerprint: change a value (e.g. FINSIGHT__CHUNKING__CHUNK_TOKENS=256) and
    run again. The fingerprint changes, which is how results stay traceable to their config.
  - The resolved settings summary: what the pipeline *would* build with this config.

RUN
  uv run python scripts/00_check_env.py
  uv run python scripts/00_check_env.py --config configs/cloud.yaml
  FINSIGHT__CHUNKING__CHUNK_TOKENS=256 uv run python scripts/00_check_env.py
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from finsight.config import ENV_ALIASES, load_settings
from finsight.container import Container
from finsight.diagnostics import render_checks, run_all_checks

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    config: Path | None = typer.Option(None, help="Profile YAML (default configs/local.yaml)"),
) -> None:
    console.rule("[bold]Step 00 — Check the environment")

    # 1. Load the layered configuration.
    settings = load_settings(config)

    # 2. Run every check and print the table.
    results = run_all_checks(settings)
    render_checks(results, console)

    # 3. WHAT JUST HAPPENED: explain the config that was resolved.
    console.rule("[bold]What just happened")
    console.print(
        f"Project root     : {settings.root}\n"
        f"Profile          : {settings.config_path}  (merged on top of configs/base.yaml)\n"
        f"Fingerprint      : [bold]{settings.fingerprint()}[/]"
        "  (hash of every result-affecting setting)\n"
        f"Env aliases read : {', '.join(ENV_ALIASES)}, plus any FINSIGHT__SECTION__KEY"
    )
    summary = Table(title="What the pipeline would build with this config", header_style="bold")
    summary.add_column("Component")
    summary.add_column("Setting")
    for key, value in Container(settings).describe().items():
        summary.add_row(key, value)
    console.print(summary)

    counts = {s: sum(r.status == s for r in results) for s in ("ok", "info", "warn", "fail")}
    console.print(
        f"Result: [green]{counts['ok']} ok[/] · [blue]{counts['info']} info[/] · "
        f"[yellow]{counts['warn']} warn[/] · [red]{counts['fail']} fail[/]"
    )
    if counts["fail"]:
        console.rule("[red]Fix the failing checks above, then run again")
        raise typer.Exit(1)
    console.rule("[green]Done: ready for P1 (uv run python scripts/01_download_data.py)")


if __name__ == "__main__":
    app()
