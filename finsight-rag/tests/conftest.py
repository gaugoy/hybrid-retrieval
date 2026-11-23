"""Shared pytest fixtures."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """A throwaway project root containing a copy of the real configs/ folder.

    Tests that write YAML or .env files do it here, never in the real project.
    """
    shutil.copytree(PROJECT_ROOT / "configs", tmp_path / "configs")
    return tmp_path
