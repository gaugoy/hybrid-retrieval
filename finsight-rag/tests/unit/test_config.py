"""Config layering, validation, fingerprinting and secret handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from finsight.config import deep_merge, env_overrides, load_settings, mask_secret
from finsight.errors import ConfigError


def test_base_and_local_profile_load(project_root: Path) -> None:
    s = load_settings(root=project_root, env={})
    assert (s.llm.preset, s.llm.provider) == ("auto", "auto")
    assert s.judge.auto_order[0] == "claude-haiku"
    assert s.chunking.strategy == "recursive"
    assert s.config_path == project_root / "configs" / "local.yaml"


def test_paths_are_resolved_against_root(project_root: Path) -> None:
    s = load_settings(root=project_root, env={})
    assert s.paths.pdf_dir == (project_root / "data" / "raw" / "pdfs").resolve()
    assert s.paths.pdf_dir.is_absolute()


def test_cloud_profile_switches_providers(project_root: Path) -> None:
    s = load_settings("configs/cloud.yaml", root=project_root, env={})
    assert (s.llm.provider, s.llm.model, s.llm.effort) == ("anthropic", "claude-opus-5-5", "medium")
    assert (s.judge.provider, s.judge.model, s.judge.effort) == (
        "anthropic",
        "claude-haiku-5-5",
        "low",
    )


def test_precedence_yaml_then_env_then_overrides(project_root: Path) -> None:
    profile = project_root / "configs" / "p.yaml"
    profile.write_text("retrieval:\n  candidate_k: 70\n  top_k: 7\n")
    env = {"FINSIGHT__RETRIEVAL__CANDIDATE_K": "80"}

    s = load_settings(profile, root=project_root, env=env)
    assert s.retrieval.top_k == 7  # from YAML
    assert s.retrieval.candidate_k == 80  # env beats YAML

    s = load_settings(profile, {"retrieval": {"candidate_k": 90}}, root=project_root, env=env)
    assert s.retrieval.candidate_k == 90  # code override beats env
    assert s.retrieval.rrf_k == 60  # untouched values keep the base default


def test_env_aliases_and_empty_values(project_root: Path) -> None:
    env = {"LLM": "ollama-qwen7b", "LLM_MODEL": "llama3.1:8b", "QDRANT_URL": ""}
    s = load_settings(root=project_root, env=env)
    assert s.llm.provider == "ollama"
    assert s.llm.model == "llama3.1:8b"  # explicit model beats the preset's
    assert s.vector_store.qdrant_url is None  # empty string means "not set"


def test_env_json_list_value() -> None:
    out = env_overrides({"FINSIGHT__RETRIEVAL__FILTER_FIELDS": '["company", "doc_period"]'})
    assert out == {"retrieval": {"filter_fields": ["company", "doc_period"]}}


def test_dotenv_file_is_read_but_real_env_wins(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (project_root / ".env").write_text(
        "LLM_MODEL=from-dotenv\nLLM=ollama-qwen7b\nANTHROPIC_API_KEY=sk-ant-test-1234567890\n"
    )
    monkeypatch.setenv("LLM_MODEL", "from-real-env")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = load_settings(root=project_root)
    assert s.llm.model == "from-real-env"
    assert s.secrets.anthropic_api_key is not None
    assert s.secrets.anthropic_api_key.get_secret_value() == "sk-ant-test-1234567890"


@pytest.mark.parametrize(
    "overrides",
    [
        {"chunking": {"overlap_tokens": 400, "chunk_tokens": 384}},  # overlap >= chunk size
        {"chunking": {"chunk_tokens": 1024}},  # beyond the embedding model's 512-token limit
        {"llm": {"provider": "openai"}},  # unknown provider
        {"retrieval": {"candiate_k": 5}},  # typo: unknown keys are rejected
    ],
)
def test_invalid_values_raise_config_error(project_root: Path, overrides: dict) -> None:
    with pytest.raises(ConfigError):
        load_settings(overrides=overrides, root=project_root, env={})


def test_missing_profile_raises(project_root: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_settings("configs/nope.yaml", root=project_root, env={})


def test_fingerprint_tracks_result_affecting_settings_only(
    project_root: Path, tmp_path: Path
) -> None:
    a = load_settings(root=project_root, env={})
    b = load_settings(root=project_root, env={})
    assert a.fingerprint() == b.fingerprint()
    assert len(a.fingerprint()) == 12

    changed = load_settings(
        overrides={"chunking": {"chunk_tokens": 256}}, root=project_root, env={}
    )
    assert changed.fingerprint() != a.fingerprint()

    # Secrets, logging, API and paths don't affect results, so they don't change the fingerprint.
    same = load_settings(
        overrides={
            "logging": {"level": "DEBUG"},
            "api": {"port": 9000},
            "paths": {"data_dir": str(tmp_path)},
        },
        root=project_root,
        env={"ANTHROPIC_API_KEY": "sk-ant-secret-value-xyz"},
    )
    assert same.fingerprint() == a.fingerprint()


def test_secrets_never_dumped(project_root: Path) -> None:
    s = load_settings(root=project_root, env={"ANTHROPIC_API_KEY": "sk-ant-super-secret-0000"})
    assert "super-secret" not in s.model_dump_json()
    assert "super-secret" not in repr(s.secrets)


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"b": 1, "c": 2}, "x": [1]}
    merged = deep_merge(base, {"a": {"b": 10}, "x": [2]})
    assert merged == {"a": {"b": 10, "c": 2}, "x": [2]}
    assert base == {"a": {"b": 1, "c": 2}, "x": [1]}


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "not set"), ("short", "***"), ("sk-ant-api03-abcdefgh1234", "sk-...1234")],
)
def test_mask_secret(value: str | None, expected: str) -> None:
    assert mask_secret(value) == expected
