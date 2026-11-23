"""
Helpers that let one step script run three ways: whole from the terminal, cell by cell in
VS Code (the `# %%` markers), or as a Jupyter notebook generated from it.

WHY
  When you're learning, you want to stop after each step and poke at the variables:
  `qa_df.head()`, `needed[:5]`, `report`. When you're automating, you want one command with
  flags. These helpers keep a single source file for both:

    P = script_args(limit_docs=Param(None, int, "only the first N PDFs"))
        terminal → parsed from --limit-docs 3
        notebook → the default you typed (edit the cell to change it)

    show(df, "Questions")
        terminal → a plain-text table
        notebook → the rich HTML DataFrame view

    step("Step 2 — Load the raw files")
        terminal → a section rule
        notebook → a heading

Library code never imports this module. Only scripts/ and notebooks/ use it.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from rich.console import Console

console = Console()


@dataclass(frozen=True)
class Param:
    """A script parameter: its default (used in notebooks), type, and CLI help text."""

    default: Any
    type: type = str
    help: str = ""


def in_notebook() -> bool:
    """True inside Jupyter, JupyterLab or the VS Code Interactive Window."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"


def script_args(
    description: str = "", argv: list[str] | None = None, **params: Param | Any
) -> SimpleNamespace:
    """Return parameters as attributes. Terminal: parsed from flags. Notebook: the defaults.

    ``limit_docs=Param(None, int)`` becomes ``--limit-docs N``; a bool becomes
    ``--force / --no-force``. Plain values are allowed: their type is inferred.
    """
    specs = {k: v if isinstance(v, Param) else Param(v, type(v)) for k, v in params.items()}
    if in_notebook():
        return SimpleNamespace(**{k: p.default for k, p in specs.items()})

    parser = argparse.ArgumentParser(description=description)
    for name, p in specs.items():
        flag = "--" + name.replace("_", "-")
        if p.type is bool:
            parser.add_argument(
                flag,
                dest=name,
                default=p.default,
                help=p.help,
                action=argparse.BooleanOptionalAction,
            )
        else:
            parser.add_argument(flag, dest=name, default=p.default, type=p.type, help=p.help)
    return parser.parse_args(sys.argv[1:] if argv is None else argv)  # type: ignore[return-value]


def configure_pandas() -> None:
    """Readable DataFrames in both terminal and notebook."""
    import pandas as pd

    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 70)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 60)


def preview(value: Any, max_chars: int = 200) -> Any:
    """Shorten long strings (recursively inside dicts/lists) so raw records stay readable."""
    if isinstance(value, str):
        flat = " ".join(value.split())
        return (
            flat
            if len(flat) <= max_chars
            else flat[:max_chars] + f"… (+{len(flat) - max_chars} chars)"
        )
    if isinstance(value, dict):
        return {k: preview(v, max_chars) for k, v in value.items()}
    if isinstance(value, list):
        return [preview(v, max_chars) for v in value]
    return value


def step(title: str) -> None:
    """Announce a step: a heading in notebooks, a rule in the terminal."""
    if in_notebook():
        from IPython.display import Markdown, display

        display(Markdown(f"### {title}"))
    else:
        console.rule(f"[bold]{title}")


def show(obj: Any, title: str | None = None) -> None:
    """Display a DataFrame/Series/object nicely wherever we're running."""
    if in_notebook():
        from IPython.display import Markdown, display

        if title:
            display(Markdown(f"**{title}**"))
        display(obj)
        return
    if title:
        console.print(f"[bold cyan]{title}[/]")
    try:
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            # to_string() ignores display.max_colwidth unless it is passed explicitly.
            print(obj.to_string(max_colwidth=pd.get_option("display.max_colwidth")))
            print()
            return
        if isinstance(obj, pd.Series):
            print(obj.to_string())
            print()
            return
    except ImportError:
        pass
    console.print(obj)
    print()
