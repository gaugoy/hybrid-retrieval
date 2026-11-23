"""Notebook/terminal helpers used by the step scripts, and the scripts' cell structure."""

from __future__ import annotations

import py_compile
from pathlib import Path

import pandas as pd
import pytest

from finsight import notebook as nb
from finsight.notebook import Param, script_args

SCRIPTS = sorted((Path(__file__).resolve().parents[2] / "scripts").glob("[0-9]*.py"))


def test_script_args_parses_flags_in_terminal() -> None:
    p = script_args(
        "demo",
        argv=["--limit-docs", "3", "--force", "--config", "configs/cloud.yaml"],
        limit_docs=Param(None, int),
        force=Param(False, bool),
        config=Param(None, str),
        workers=4,  # plain value: type inferred
    )
    assert (p.limit_docs, p.force, p.config, p.workers) == (3, True, "configs/cloud.yaml", 4)


def test_script_args_defaults_and_no_flag(capsys: pytest.CaptureFixture[str]) -> None:
    p = script_args(argv=["--no-force"], limit_docs=Param(None, int), force=Param(True, bool))
    assert p.limit_docs is None
    assert p.force is False


def test_script_args_returns_defaults_in_notebook(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nb, "in_notebook", lambda: True)
    p = script_args(argv=["--limit-docs", "9"], limit_docs=Param(2, int))
    assert p.limit_docs == 2  # notebooks ignore argv: you edit the cell instead


def test_in_notebook_false_under_pytest() -> None:
    assert nb.in_notebook() is False


def test_show_prints_dataframe_as_text(capsys: pytest.CaptureFixture[str]) -> None:
    nb.show(pd.DataFrame({"doc": ["3M_2018_10K"], "pages": [160]}), "Docs")
    out = capsys.readouterr().out
    assert "Docs" in out
    assert "3M_2018_10K" in out
    assert "160" in out


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_step_scripts_compile_and_have_cells(script: Path) -> None:
    py_compile.compile(str(script), doraise=True)
    text = script.read_text()
    assert text.startswith("# %% [markdown]"), "step scripts open with a markdown cell"
    assert text.count("# %%") >= 4, "step scripts are split into cells"
    assert "script_args(" in text, "parameters come from script_args() (flags or notebook defaults)"


def test_notebook_build_is_deterministic(tmp_path: Path) -> None:
    import importlib.util
    import json

    tool = Path(__file__).resolve().parents[2] / "tools" / "build_notebooks.py"
    spec = importlib.util.spec_from_file_location("build_notebooks", tool)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    scripts_dir = tool.parents[1] / "scripts"
    first = [p.read_bytes() for p in mod.build_all(scripts_dir, tmp_path / "a")]
    second = [p.read_bytes() for p in mod.build_all(scripts_dir, tmp_path / "b")]
    assert first == second  # byte-identical: rebuilding never creates a git diff

    nb = json.loads(first[0])
    assert nb["metadata"]["kernelspec"]["name"] == "finsight-rag"
    assert nb["cells"][0]["id"] == "cell-00"
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")
