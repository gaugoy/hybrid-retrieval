"""Index manifest: rebuild only when what the index was built FROM changes."""

from __future__ import annotations

from pathlib import Path

from finsight.indexing.manifest import Manifest, chunks_sha256


def test_needs_rebuild_tracks_content_model_and_version(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.json")
    sha = chunks_sha256(["a", "b"], ["text a", "text b"])
    expected = {"chunks_sha256": sha, "embed_model": "bge", "version": "v1"}
    assert m.needs_rebuild("col", **expected)  # never built

    m.record("col", n_chunks=2, config_fingerprint="abc", **expected)
    assert not m.needs_rebuild("col", **expected)
    assert Manifest(tmp_path / "manifest.json").entry("col")["n_chunks"] == 2  # persisted
    assert "built_at" in m.entry("col")  # type: ignore[operator]

    assert m.needs_rebuild("col", **{**expected, "embed_model": "other"})
    assert m.needs_rebuild("col", **{**expected, "version": "v2"})
    changed = chunks_sha256(["a", "b"], ["text a", "text B"])
    assert m.needs_rebuild("col", **{**expected, "chunks_sha256": changed})


def test_fingerprint_is_informational_only(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.json")
    m.record("col", chunks_sha256="x", version="v1", config_fingerprint="llm-A")
    # a different config fingerprint (e.g. another LLM) is not a REBUILD_KEY
    assert not m.needs_rebuild("col", chunks_sha256="x", version="v1", config_fingerprint="llm-B")


def test_entries_are_independent_and_removable(tmp_path: Path) -> None:
    m = Manifest(tmp_path / "manifest.json")
    m.record("dense", chunks_sha256="x", version="v1")
    m.record("bm25", chunks_sha256="y", version="v1")
    assert set(m.load()) == {"dense", "bm25"}
    m.remove("dense")
    assert set(m.load()) == {"bm25"}


def test_chunks_sha_depends_on_ids_and_order() -> None:
    a = chunks_sha256(["1", "2"], ["x", "y"])
    assert a == chunks_sha256(["1", "2"], ["x", "y"])
    assert a != chunks_sha256(["2", "1"], ["y", "x"])
    assert a != chunks_sha256(["1", "3"], ["x", "y"])
    assert chunks_sha256(["ab"], ["c"]) != chunks_sha256(["a"], ["bc"])  # separators matter
