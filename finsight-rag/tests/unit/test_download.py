"""Downloader behaviour against a fake HTTP server (httpx.MockTransport): no real network."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import httpx
import pytest
from tenacity import wait_none

from finsight.config import Settings, load_settings
from finsight.data import download as dl
from finsight.data import financebench as fb

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "financebench"
FAKE_PDF = b"%PDF-1.7\n" + b"0" * 2048


class FakeServer:
    """Serves the fixture JSONL files and fake PDFs; can inject failures per document."""

    def __init__(
        self,
        broken: dict[str, int] | None = None,
        flaky: dict[str, int] | None = None,
        html_instead_of_pdf: set[str] | None = None,
    ) -> None:
        self.broken = broken or {}  # doc -> permanent HTTP status
        self.flaky = dict(flaky or {})  # doc -> number of 503s before success
        self.html = html_instead_of_pdf or set()
        self.hits: Counter[str] = Counter()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.hits[url] += 1
        if url == fb.QA_URL:
            return httpx.Response(200, content=(FIXTURES / fb.QA_FILE).read_bytes())
        if url == fb.DOCS_URL:
            return httpx.Response(200, content=(FIXTURES / fb.DOCS_FILE).read_bytes())
        doc = url.rsplit("/", 1)[-1].removesuffix(".pdf")
        if doc in self.broken:
            return httpx.Response(self.broken[doc])
        if self.flaky.get(doc, 0) > 0:
            self.flaky[doc] -= 1
            return httpx.Response(503)
        if doc in self.html:
            return httpx.Response(200, content=b"<html>rate limited</html>" + b" " * 2048)
        return httpx.Response(200, content=FAKE_PDF)


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dl.download_file.retry, "wait", wait_none())  # type: ignore[attr-defined]


@pytest.fixture
def settings(project_root: Path) -> Settings:
    return load_settings(root=project_root, env={})


def _client(server: FakeServer) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(server))


def test_downloads_only_referenced_pdfs_and_writes_manifest(settings: Settings) -> None:
    server = FakeServer()
    report = dl.download_financebench(settings, client=_client(server), progress=False)

    assert report.ok
    assert (report.n_questions, report.n_docs_needed, report.n_downloaded) == (3, 3, 3)
    pdfs = sorted(p.name for p in settings.paths.pdf_dir.glob("*.pdf"))
    assert pdfs == ["ACME_2020_10K.pdf", "ACME_2021_10K.pdf", "BETA_2019_10Q.pdf"]  # not UNUSED
    assert not list(settings.paths.pdf_dir.glob("*.part"))  # no leftovers

    manifest = json.loads(report.manifest_path.read_text())  # type: ignore[union-attr]
    assert manifest["n_files"] == 5  # 2 JSONL + 3 PDFs
    assert all(len(f["sha256"]) == 64 for f in manifest["files"].values())


def test_rerun_is_idempotent(settings: Settings) -> None:
    server = FakeServer()
    dl.download_financebench(settings, client=_client(server), progress=False)
    hits_after_first = sum(server.hits.values())

    report = dl.download_financebench(settings, client=_client(server), progress=False)
    assert (report.n_downloaded, report.n_skipped) == (0, 3)
    assert sum(server.hits.values()) == hits_after_first  # nothing fetched again


def test_transient_503_is_retried(settings: Settings) -> None:
    server = FakeServer(flaky={"ACME_2020_10K": 2})
    report = dl.download_financebench(settings, client=_client(server), progress=False)
    assert report.ok
    assert server.hits[dl.pdf_url("ACME_2020_10K")] == 3  # 2 failures + 1 success


def test_404_is_not_retried_and_is_reported(settings: Settings) -> None:
    server = FakeServer(broken={"BETA_2019_10Q": 404})
    report = dl.download_financebench(settings, client=_client(server), progress=False)
    assert not report.ok
    assert "BETA_2019_10Q" in report.failures
    assert server.hits[dl.pdf_url("BETA_2019_10Q")] == 1  # permanent error: one attempt only
    assert report.n_downloaded == 2  # the other documents still succeed


def test_non_pdf_response_is_rejected_and_not_saved(settings: Settings) -> None:
    server = FakeServer(html_instead_of_pdf={"ACME_2021_10K"})
    report = dl.download_financebench(settings, client=_client(server), progress=False)
    assert "ACME_2021_10K" in report.failures
    assert not (settings.paths.pdf_dir / "ACME_2021_10K.pdf").exists()


def test_limit_docs(settings: Settings) -> None:
    report = dl.download_financebench(
        settings, client=_client(FakeServer()), limit_docs=1, progress=False
    )
    assert (report.n_docs_needed, report.n_downloaded) == (1, 1)


def test_pdf_url_is_quoted() -> None:
    assert dl.pdf_url("A B&C").endswith("/pdfs/A%20B%26C.pdf")
