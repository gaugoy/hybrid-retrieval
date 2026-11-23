"""
Reproducible, idempotent download of FinanceBench.

WHAT IT DOES
  1. Downloads the two JSONL files (Q&A + document metadata).
  2. Works out which PDFs the 150 questions actually reference (84 of the repo's 368).
  3. Downloads only those, in parallel, skipping files already on disk.
  4. Writes a manifest with the size and SHA-256 of every file, so you can prove later
     exactly which data your results came from.

PRODUCTION HABITS ON SHOW
  - Retries with exponential backoff, for network errors and HTTP 429/5xx only (a 404 is not
    retried: it will never succeed).
  - Atomic writes: download to `name.part`, check it, then rename. A crash never leaves a
    half-written file that looks complete.
  - Validation: a PDF must start with the `%PDF` magic bytes (an HTML error page would not).
  - Idempotency: re-running skips valid files, so it is safe to run as often as you like.

More info: https://www.python-httpx.org/ · https://tenacity.readthedocs.io/
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from finsight.config import Settings
from finsight.data import financebench as fb
from finsight.observability.logging import get_logger

log = get_logger(__name__)

USER_AGENT = "finsight-rag/0.1 (research benchmark download)"
PDF_MAGIC = b"%PDF"
MANIFEST_NAME = "download_manifest.json"


@dataclass
class DownloadReport:
    n_questions: int = 0
    n_docs_needed: int = 0
    n_downloaded: int = 0
    n_skipped: int = 0
    bytes_downloaded: int = 0
    bytes_on_disk: int = 0
    elapsed_s: float = 0.0
    failures: dict[str, str] = field(default_factory=dict)
    manifest_path: Path | None = None

    @property
    def ok(self) -> bool:
        return not self.failures


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    reraise=True,
)
def download_file(client: httpx.Client, url: str, dest: Path, expect_pdf: bool = False) -> int:
    """Stream ``url`` to ``dest`` atomically. Returns the number of bytes written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    written = 0
    try:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1 << 16):
                    f.write(chunk)
                    written += len(chunk)
        if expect_pdf and not is_valid_pdf(tmp):
            raise ValueError(f"{url} did not return a PDF (missing %PDF header)")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return written


def is_valid_pdf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as f:
        return f.read(4) == PDF_MAGIC


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pdf_url(doc_name: str) -> str:
    return fb.PDF_URL_TEMPLATE.format(doc_name=quote(doc_name))


def make_client(timeout_s: float = 60.0) -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(timeout_s, connect=10.0),
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )


def download_financebench(
    settings: Settings,
    *,
    limit_docs: int | None = None,
    force: bool = False,
    workers: int = 4,
    client: httpx.Client | None = None,
    progress: bool = True,
) -> DownloadReport:
    """Download the Q&A, document metadata and the referenced PDFs. Safe to re-run."""
    start = time.perf_counter()
    report = DownloadReport()
    own_client = client is None
    client = client or make_client()
    raw = fb.raw_dir(settings)
    pdf_dir = settings.paths.pdf_dir

    try:
        # 1. The two JSONL files are small (~1 MB): always refresh them unless present.
        for url, name in ((fb.QA_URL, fb.QA_FILE), (fb.DOCS_URL, fb.DOCS_FILE)):
            dest = raw / name
            if force or not dest.is_file():
                report.bytes_downloaded += download_file(client, url, dest)
                log.info("downloaded", extra={"file": name})

        # 2. Which PDFs do the questions need?
        qa_rows = fb.read_jsonl(raw / fb.QA_FILE)
        report.n_questions = len(qa_rows)
        needed = fb.referenced_doc_names(qa_rows)
        if limit_docs is not None:
            needed = needed[:limit_docs]
        report.n_docs_needed = len(needed)

        # 3. Download missing PDFs in parallel (I/O-bound, so threads are the right tool).
        todo = [n for n in needed if force or not is_valid_pdf(pdf_dir / f"{n}.pdf")]
        report.n_skipped = len(needed) - len(todo)
        bar = tqdm(total=len(todo), desc="PDFs", unit="file", disable=not progress or not todo)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(download_file, client, pdf_url(n), pdf_dir / f"{n}.pdf", True): n
                for n in todo
            }
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    report.bytes_downloaded += fut.result()
                    report.n_downloaded += 1
                except Exception as exc:  # report every failure, don't stop the batch
                    report.failures[name] = f"{type(exc).__name__}: {exc}"
                    log.warning("pdf download failed", extra={"doc": name, "error": str(exc)})
                bar.update(1)
        bar.close()

        # 4. Manifest: size + checksum of every file we rely on.
        report.manifest_path = write_manifest(settings, needed)
        report.bytes_on_disk = sum(
            p.stat().st_size for p in [raw / fb.QA_FILE, raw / fb.DOCS_FILE]
        ) + sum(
            (pdf_dir / f"{n}.pdf").stat().st_size
            for n in needed
            if (pdf_dir / f"{n}.pdf").is_file()
        )
    finally:
        if own_client:
            client.close()

    report.elapsed_s = round(time.perf_counter() - start, 2)
    return report


def write_manifest(settings: Settings, doc_names: list[str]) -> Path:
    raw = fb.raw_dir(settings)
    files: dict[str, dict[str, object]] = {}
    paths = [raw / fb.QA_FILE, raw / fb.DOCS_FILE] + [
        settings.paths.pdf_dir / f"{n}.pdf" for n in doc_names
    ]
    for p in paths:
        if p.is_file():
            rel = str(p.relative_to(settings.paths.data_dir))
            files[rel] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    manifest = {
        "dataset": "financebench",
        "source": fb.REPO_RAW,
        "license": "CC-BY-NC-4.0",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_files": len(files),
        "files": files,
    }
    out = raw / MANIFEST_NAME
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return out
