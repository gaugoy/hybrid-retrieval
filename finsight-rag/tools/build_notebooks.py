"""
Build notebooks/*.ipynb from the cell-based step scripts in scripts/ (run: make notebooks).

WHY A SCRIPT INSTEAD OF THE jupytext CLI
  - Deterministic output: jupytext assigns random cell IDs on every conversion, so each rebuild
    would show up as a git diff. Here IDs are fixed (cell-00, cell-01, ...), so rebuilding an
    unchanged script produces a byte-identical notebook.
  - The kernel is preset to "Python 3.12 (finsight-rag)" (registered by `make kernel`), so
    VS Code and Jupyter pick this project's .venv without asking.
  - No outputs and no jupytext metadata: the notebooks stay small and clean for GitHub.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jupytext
import nbformat

ROOT = Path(__file__).resolve().parents[1]
KERNELSPEC = {
    "name": "finsight-rag",
    "display_name": "Python 3.12 (finsight-rag)",
    "language": "python",
}


def build_notebook(script: Path) -> nbformat.NotebookNode:
    nb = jupytext.read(script, fmt="py:percent")
    nb.metadata = {
        "kernelspec": KERNELSPEC,
        "language_info": {"name": "python"},
    }
    for i, cell in enumerate(nb.cells):
        cell["id"] = f"cell-{i:02d}"
        cell["metadata"] = {}
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return nb


def build_all(scripts_dir: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for script in sorted(scripts_dir.glob("[0-9]*.py")):
        out = out_dir / f"{script.stem}.ipynb"
        nbformat.write(build_notebook(script), out)
        written.append(out)
    return written


if __name__ == "__main__":
    for path in build_all(ROOT / "scripts", ROOT / "notebooks"):
        print(f"  scripts/{path.stem}.py -> {path.relative_to(ROOT)}")
    sys.exit(0)
