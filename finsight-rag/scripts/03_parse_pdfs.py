# %% [markdown]
# # Step 03 — Parse PDFs into clean pages   (Phase P2)
#
# **What this step does:** turns every page of the 84 filings into clean text that keeps
# **table rows together** (`Net sales | $ 11,303 | $ 11,503`), removes repeated headers/footers,
# re-joins hyphenated words, and then **proves** the parsing is right: FinanceBench's evidence
# text must be found on our parsed page at the gold page index.
#
# The first cells take ONE page (3M's 2018 cash-flow statement, the evidence for "What is 3M's
# FY2018 capex?") through each internal stage. Then the whole corpus is processed.
#
# **Three ways to run it**
# - Terminal: `uv run python scripts/03_parse_pdfs.py [--doc NIKE_2021_10K --page-idx 58] [--force]`
# - VS Code, cell by cell: click **Run Cell** above each `# %%`
# - Jupyter: `make notebooks`, then open `notebooks/03_parse_pdfs.ipynb`
#
# **Variables to explore afterwards:** `words_df`, `rows_df`, `boilerplate_df`, `pages_df`,
# `documents_df`, `evidence_df`, `compare_df`.

# %%
import itertools
import statistics

import pandas as pd
import pymupdf

from finsight.config import load_settings
from finsight.data import financebench as fb
from finsight.ingestion import pipeline as ip
from finsight.ingestion.cleaner import clean_document, find_boilerplate, normalize_line
from finsight.ingestion.pdf_parser import (
    extract_words,
    group_rows,
    parse_pdf,
    plain_text,
    render_row,
)
from finsight.ingestion.quality import count_numbers, label_number_adjacency
from finsight.notebook import Param, configure_pandas, in_notebook, script_args, show, step
from finsight.observability.logging import configure_logging

configure_pandas()

# %% Parameters: in a notebook, edit the defaults here; in the terminal, use the flags.
P = script_args(
    "Step 03: parse PDFs into clean pages",
    config=Param(None, str, "profile YAML (default: configs/local.yaml)"),
    doc=Param("3M_2018_10K", str, "document to walk through in cells 1-5"),
    page_idx=Param(59, int, "0-based page of that document (59 = printed page 60)"),
    limit_docs=Param(None, int, "only parse the first N documents (quick trial)"),
    force=Param(False, bool, "re-parse even if the outputs are up to date"),
    workers=Param(1, int, "processes; keep 1 here (see the note in cell 6)"),
)
P

# %%
settings = load_settings(P.config)
configure_logging(settings.logging.level, settings.logging.json_format)
cfg = settings.ingestion
pdf = fb.pdf_path(settings, P.doc)
show(
    pd.Series(cfg.model_dump(), dtype=object), "Ingestion settings (configs/base.yaml → ingestion)"
)

# %% [markdown]
# ## 1. The baseline: PyMuPDF's plain text extraction
# `page.get_text("text")` emits text block by block. In a financial statement every number is a
# block of its own, so numbers land on separate lines, away from their labels. The
# **label–number adjacency** score is the share of numbers that share a line with a word.

# %%
step(f"1. Plain extraction of {P.doc}, page_idx {P.page_idx} (printed page {P.page_idx + 1})")
plain = plain_text(pdf, P.page_idx)
print("\n".join([ln for ln in plain.splitlines() if ln.strip()][:22]))
print(f"\n… label–number adjacency: {label_number_adjacency(plain):.2f}")

# %% [markdown]
# ## 2. Words with coordinates
# `page.get_text("words")` returns every word with its bounding box (x0, y0, x1, y1), in points
# (1/72 inch, origin top-left). This is the raw material for rebuilding rows.

# %%
step("2. Words with coordinates")
with pymupdf.open(pdf) as d:
    words = extract_words(d[P.page_idx])
words_df = pd.DataFrame([w.__dict__ for w in words]).round(1)
line_height = statistics.median(w.height for w in words)
print(f"{len(words)} words · median line height {line_height:.1f} pt")
print(
    f"→ same row if |Δy| ≤ {cfg.row_tolerance_factor * line_height:.1f} pt · new column if gap > "
    f"{cfg.column_gap_factor * line_height:.1f} pt\n"
)
show(words_df.head(15), "words_df.head(15)")

# %% [markdown]
# ## 3. Words → rows → text lines
# Words whose vertical centres are close form one row; sorted left → right, a wide horizontal gap
# becomes " | ". A `$` always attaches to the number after it.

# %%
step("3. Rebuild rows")
rows = group_rows(words, cfg.row_tolerance_factor)
gap = cfg.column_gap_factor * line_height
rows_df = pd.DataFrame(
    [
        {
            "y": round(statistics.mean(w.y_center for w in r), 1),
            "n_words": len(r),
            "widest_gap_pt": round(
                max((b.x0 - a.x1 for a, b in itertools.pairwise(r)), default=0), 1
            ),
            "text": render_row(r, gap),
        }
        for r in rows
    ]
)
show(rows_df.head(20), f"rows_df: {len(rows_df)} rows (first 20)")

# %% [markdown]
# ## 4. Boilerplate: what repeats at the page edges across the whole document?
# Digits become `#`, so "60" and "61" both read `##`. A short line in the top/bottom
# `boilerplate_edge_lines` of ≥ `boilerplate_min_share` of the pages is a header/footer.

# %%
step(f"4. Boilerplate in {P.doc}")
raw_pages = parse_pdf(pdf, cfg)
all_lines = [p.lines for p in raw_pages]
boilerplate = find_boilerplate(all_lines, cfg.boilerplate_edge_lines, cfg.boilerplate_min_share)
normalised_pages = [{normalize_line(ln) for ln in lines} for lines in all_lines]
boilerplate_df = pd.DataFrame(
    [
        {
            "boilerplate (normalised)": b,
            "pages containing it": sum(b in page for page in normalised_pages),
        }
        for b in sorted(boilerplate)
    ]
)
print(f"{len(raw_pages)} pages parsed")
show(boilerplate_df, "Detected header/footer lines")

# %% [markdown]
# ## 5. The cleaned page: before vs after
# Boilerplate removed, hyphenation joined, blank lines collapsed. Compare the adjacency score
# with cell 1.

# %%
step("5. Cleaned page")
cleaned_texts, _ = clean_document(all_lines, cfg)
cleaned = cleaned_texts[P.page_idx]
print("\n".join(cleaned.splitlines()[:22]))
print(
    f"\n… label–number adjacency: plain {label_number_adjacency(plain):.2f} → "
    f"ours {label_number_adjacency(cleaned):.2f}"
)

# %% [markdown]
# ## 6. Parse the whole corpus
# The same steps for all 84 documents (~12k pages, about a minute with `workers=1`). The output is
# skipped when nothing changed (PDF checksums + ingestion settings + parser version, in
# `data/processed/ingest_manifest.json`).
#
# *Note:* `workers > 1` uses separate processes. On macOS those re-import the calling script, and
# a cell-based script like this one would re-run itself in each worker. For parallel parsing use
# the CLI instead: `uv run finsight ingest --workers 0` (0 = all cores, ~17 s).

# %%
step("6. Parse all documents")
report = ip.ingest_all(
    settings, limit_docs=P.limit_docs, force=P.force, workers=P.workers, progress=not in_notebook()
)
show(
    pd.Series(
        {
            "documents": report.n_docs,
            "pages": report.n_pages,
            "empty pages (no text layer)": report.n_empty_pages,
            "pages with tables": report.n_table_pages,
            "characters": f"{report.total_chars:,}",
            "boilerplate patterns removed": report.boilerplate_lines,
            "skipped (up to date)": report.skipped,
            "time (s)": report.elapsed_s,
            "failures": len(report.failures),
        },
        dtype=object,
    ),
    "IngestReport",
)

# %% [markdown]
# ## 7. What we produced
# `pages.parquet`: one row per page. `documents.parquet`: one row per filing, with statistics.

# %%
step("7. Outputs")
pages_df = ip.load_pages_df(settings)
documents_df = ip.load_documents_df(settings)
show(pages_df.head(3), "pages_df.head(3)")
show(pages_df["n_chars"].describe().round(0), "Characters per page")
show(
    documents_df.sort_values("n_pages", ascending=False)[
        ["doc_name", "doc_type", "n_pages", "n_table_pages", "n_empty_pages", "parse_s"]
    ].head(8),
    "Largest documents",
)
show(
    documents_df[documents_df["n_empty_pages"] > 0][["doc_name", "n_pages", "n_empty_pages"]],
    "Documents with empty pages (scanned / image-only)",
)

# %% [markdown]
# ## 8. Proof: is every evidence passage on the page we parsed?
# For all 189 evidence items, the share of evidence tokens found on our page (space-insensitive,
# because FinanceBench's text sometimes lost spaces: "Netsales"). `best_page_idx` is the best
# match in the whole document: if it differed from `page_idx`, we'd have a page-offset bug.

# %%
step("8. Evidence check")
evidence_df = ip.evidence_check(settings, pages_df)
share = (evidence_df["overlap"] >= 0.8).mean()
offsets = int((evidence_df["best_page_idx"] != evidence_df["page_idx"]).sum())
print(
    f"{len(evidence_df)} evidence items · {share:.1%} with overlap ≥ 0.8 · mean "
    f"{evidence_df['overlap'].mean():.3f} · page offsets: {offsets}\n"
)
show(evidence_df.sort_values("overlap").head(5), "Lowest-scoring evidence items")

# %% [markdown]
# ### Plain extraction vs ours, on the evidence pages
# Both contain the evidence words. The difference is whether the numbers stay next to their labels.
# Adjacency is only meaningful on pages with several numbers, so prose pages (< 5 numbers) are
# excluded from the mean.

# %%
pages_idx = pages_df.set_index(["doc_name", "page_idx"])["text"]
compare_rows = []
for ev in evidence_df.drop_duplicates(["doc_name", "page_idx"]).itertuples():
    base = plain_text(fb.pdf_path(settings, ev.doc_name), ev.page_idx)
    ours = pages_idx[(ev.doc_name, ev.page_idx)]
    compare_rows.append(
        {
            "doc_name": ev.doc_name,
            "page_idx": ev.page_idx,
            "adjacency_plain": round(label_number_adjacency(base), 2),
            "adjacency_ours": round(label_number_adjacency(ours), 2),
            "n_numbers": count_numbers(ours),
        }
    )
compare_df = pd.DataFrame(compare_rows)
numeric = compare_df[compare_df["n_numbers"] >= 5]
print(f"{len(numeric)} of {len(compare_df)} evidence pages have ≥ 5 numbers\n")
show(numeric[["adjacency_plain", "adjacency_ours"]].mean().round(2), "Mean label–number adjacency")
show(numeric.sort_values("adjacency_ours").head(5), "Numeric evidence pages where ours is lowest")

# %% Summary
step("Summary")
if share < 0.95:
    print(f"Evidence check below 95% ({share:.1%}): inspect the lowest-scoring items above.")
    if not in_notebook():
        raise SystemExit(1)
else:
    print("P2 done: pages.parquet is ready. Next: scripts/04_chunk.py (P3)")
