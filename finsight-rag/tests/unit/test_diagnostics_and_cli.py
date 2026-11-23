"""Environment checks and CLI wiring (no network: Ollama is pointed at a closed port)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from finsight.cli import app
from finsight.config import load_settings
from finsight.diagnostics import check_folders, check_ollama, check_secrets
from finsight.schemas import Chunk, QueryFilters


def test_ollama_unreachable_is_warning_not_failure(project_root: Path) -> None:
    s = load_settings(root=project_root, env={"OLLAMA_BASE_URL": "http://127.0.0.1:9"})
    result = check_ollama(s, timeout_s=0.5)
    assert result.status == "warn"
    assert "ollama" in result.hint.lower()


def test_missing_anthropic_key_fails_only_when_cloud_profile(project_root: Path) -> None:
    local = load_settings(root=project_root, env={})
    assert check_secrets(local)[0].status == "info"
    cloud = load_settings("configs/cloud.yaml", root=project_root, env={})
    assert check_secrets(cloud)[0].status == "fail"


def test_folders_report_info_before_data_exists(project_root: Path) -> None:
    s = load_settings(root=project_root, env={})
    assert all(r.status == "info" for r in check_folders(s))


def test_cli_help_lists_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in (
        "check-env",
        "show-config",
        "download",
        "ingest",
        "index",
        "query",
        "eval",
        "serve",
    ):
        assert cmd in result.output


def test_cli_unbuilt_command_names_its_phase() -> None:
    result = CliRunner().invoke(app, ["index"])
    assert result.exit_code == 2
    assert "P4" in result.output


def test_schema_contracts_basic() -> None:
    chunk = Chunk(
        chunk_id="3M_2018_10K::p59-59::c0",
        doc_name="3M_2018_10K",
        company="3M",
        doc_type="10k",
        doc_period=2018,
        page_start=59,
        page_end=59,
        text="t",
        embed_text="t",
        n_tokens=1,
        strategy="page_aware",
    )
    assert chunk.page_start == 59
    assert QueryFilters().is_empty()
    assert not QueryFilters(companies=["3M"]).is_empty()
