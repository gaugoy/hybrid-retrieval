# %% [markdown]
# # Step 01 — Download FinanceBench   (Phase P1)
#
# **What this step does:** downloads the FinanceBench Q&A file (150 questions), the document
# metadata file, and **only** the PDFs those questions reference (84 filings, ~166 MB, out of 368
# in the repo). Each cell below does one internal step of `finsight.data.download`, so you can see
# what happens in the background and inspect the variables.
#
# **Three ways to run it**
# - Terminal, all at once: `uv run python scripts/01_download_data.py [--limit-docs 3] [--force]`
# - VS Code, cell by cell: click **Run Cell** above each `# %%`
# - Jupyter: `make notebooks`, then open `notebooks/01_download_data.ipynb`
#
# **Variables to explore afterwards:** `qa_df`, `docs_df`, `needed`, `needed_df`, `report`,
# `manifest_df`.

# %%
import json

import pandas as pd

from finsight.config import load_settings
from finsight.data import download as dl
from finsight.data import financebench as fb
from finsight.notebook import (
    Param,
    configure_pandas,
    in_notebook,
    preview,
    script_args,
    show,
    step,
)
from finsight.observability.logging import configure_logging

configure_pandas()

# %% Parameters: in a notebook, edit the defaults here; in the terminal, use the flags.
P = script_args(
    "Step 01: download FinanceBench",
    config=Param(None, str, "profile YAML (default: configs/local.yaml)"),
    limit_docs=Param(None, int, "only download the first N referenced PDFs (quick trial)"),
    force=Param(False, bool, "re-download files even if they already exist"),
    workers=Param(4, int, "parallel PDF downloads"),
)
P

# %%
settings = load_settings(P.config)
configure_logging(settings.logging.level, settings.logging.json_format)
raw_dir = fb.raw_dir(settings)
pdf_dir = settings.paths.pdf_dir
show(
    pd.Series(
        {
            "JSONL folder": str(raw_dir.relative_to(settings.root)),
            "PDF folder": str(pdf_dir.relative_to(settings.root)),
            "source": fb.REPO_RAW,
        }
    ),
    "Where files go",
)

# %% [markdown]
# ## 1. Download the two small JSONL files
# `download_file()` streams the URL into `name.part`, then renames it to `name`. A crash
# mid-download never leaves a half-written file that looks complete. Network errors and HTTP
# 429/5xx are retried with exponential backoff; a 404 is not (it would never succeed).

# %%
step("1. Download the Q&A and document-metadata JSONL files")
client = dl.make_client()
jsonl_rows = []
for url, name in ((fb.QA_URL, fb.QA_FILE), (fb.DOCS_URL, fb.DOCS_FILE)):
    dest = raw_dir / name
    action = "skipped (exists)"
    if P.force or not dest.is_file():
        dl.download_file(client, url, dest)
        action = "downloaded"
    jsonl_rows.append({"file": name, "action": action, "KB": round(dest.stat().st_size / 1e3, 1)})
show(pd.DataFrame(jsonl_rows), "JSONL files")

# %% [markdown]
# ## 2. Look at the raw files with pandas
# One JSON object per line. Q&A: one row per question, with a nested `evidence` list.
# Document info: one row per filing (company, type, fiscal period, original link).

# %%
step("2. Look at the raw files")
qa_df = pd.read_json(raw_dir / fb.QA_FILE, lines=True)
docs_df = pd.read_json(raw_dir / fb.DOCS_FILE, lines=True)
print(f"qa_df: {qa_df.shape[0]} rows × {qa_df.shape[1]} columns")
print(f"docs_df: {docs_df.shape[0]} rows, {docs_df['doc_name'].nunique()} unique doc names\n")
show(
    qa_df[["financebench_id", "company", "doc_name", "question_type", "question", "answer"]].head(),
    "qa_df.head()",
)
show(
    preview(qa_df.loc[0, "evidence"][0]), "One evidence item (nested in a Q&A row, text shortened)"
)
show(docs_df.head(), "docs_df.head()")

# %% [markdown]
# ### A data quirk: one document is listed twice
# The loader (`fb.parse_document_meta`) keeps the first row. No question uses this document.

# %%
show(docs_df[docs_df["doc_name"].duplicated(keep=False)], "Duplicate doc_name rows")

# %% [markdown]
# ## 3. Which PDFs do we actually need?
# The repo has 368 PDFs (~705 MB). The questions reference only some of them, via `doc_name` and
# each evidence item's `doc_name`. We download just those.

# %%
step("3. Work out which PDFs the questions need")
needed = fb.referenced_doc_names(qa_df.to_dict("records"))
if P.limit_docs is not None:
    needed = needed[: P.limit_docs]
needed_df = docs_df.drop_duplicates("doc_name").set_index("doc_name").loc[needed].reset_index()
needed_df["on_disk"] = [dl.is_valid_pdf(pdf_dir / f"{n}.pdf") for n in needed]
print(
    f"{len(needed)} PDFs in scope ({docs_df['doc_name'].nunique()} documents listed in metadata)\n"
)
show(needed_df["doc_type"].value_counts(), "Needed PDFs by doc_type")
show(
    needed_df[["doc_name", "company", "doc_type", "doc_period", "on_disk"]].head(10),
    "needed_df (first 10)",
)

# %% [markdown]
# ## 4. Download ONE PDF by hand, to see the mechanics
# The URL comes from a template; then a check: a real PDF starts with the 4 magic bytes `%PDF`
# (an HTML error page would not, so it's rejected instead of saved).

# %%
step("4. One PDF by hand")
example = needed[0]
example_url = dl.pdf_url(example)
example_path = pdf_dir / f"{example}.pdf"
if P.force or not dl.is_valid_pdf(example_path):
    n_bytes = dl.download_file(client, example_url, example_path, expect_pdf=True)
    print(f"downloaded {n_bytes / 1e6:.1f} MB")
else:
    print("already on disk (valid), skipped")
with example_path.open("rb") as f:
    first_bytes = f.read(8)
show(
    pd.Series(
        {
            "doc": example,
            "url": example_url,
            "first 8 bytes": repr(first_bytes),
            "valid PDF": dl.is_valid_pdf(example_path),
            "MB": round(example_path.stat().st_size / 1e6, 2),
        }
    ),
    "The example PDF",
)

# %% [markdown]
# ## 5. Download the rest in parallel
# The same steps for every PDF, 4 at a time (`ThreadPoolExecutor`; downloads wait on the network,
# so threads are the right tool). Valid files are skipped, which makes re-runs instant.
# A failure is recorded and the batch continues.

# %%
step("5. Download all needed PDFs (idempotent)")
client.close()
report = dl.download_financebench(
    settings, limit_docs=P.limit_docs, force=P.force, workers=P.workers, progress=not in_notebook()
)
show(
    pd.Series(
        {
            "questions": report.n_questions,
            "PDFs in scope": report.n_docs_needed,
            "downloaded now": report.n_downloaded,
            "skipped (already valid)": report.n_skipped,
            "downloaded this run (MB)": round(report.bytes_downloaded / 1e6, 1),
            "total on disk (MB)": round(report.bytes_on_disk / 1e6, 1),
            "time (s)": report.elapsed_s,
            "failures": len(report.failures),
        },
        dtype=object,
    ),
    "DownloadReport",
)
if report.failures:
    show(pd.Series(report.failures, name="error"), "Failures: re-run to retry just these")

# %% [markdown]
# ## 6. The manifest: proof of exactly which data you used
# Size and SHA-256 of every file. If someone re-downloads and a checksum differs, the data changed.

# %%
step("6. The download manifest")
manifest = json.loads(report.manifest_path.read_text())  # type: ignore[union-attr]
manifest_df = (
    pd.DataFrame.from_dict(manifest["files"], orient="index")
    .rename_axis("file")
    .assign(MB=lambda d: (d["bytes"] / 1e6).round(2))
    .sort_values("bytes", ascending=False)
)
print(
    f"{manifest['n_files']} files · license {manifest['license']} · "
    f"created {manifest['created_at']}\n"
)
show(manifest_df[["MB", "sha256"]].head(10), "Largest files")

# %% Summary
step("Summary")
if report.failures:
    print("Some downloads failed: re-run this script (finished files are skipped).")
    if not in_notebook():
        raise SystemExit(1)
else:
    print("Done. Run it again: 'downloaded now' should be 0. Next: scripts/02_explore_data.py")
