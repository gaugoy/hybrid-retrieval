"""
Dataset dispatch: the shared code (chunking, indexing, evaluation, filters) asks this module for
documents, pages and questions, and gets FinanceBench or MultiHop-RAG according to
`settings.dataset`. FinanceBench pages come from the PDF parser (P2); MultiHop-RAG articles are
already text (one page each).
"""

from __future__ import annotations

import pandas as pd

from finsight.config import Settings
from finsight.schemas import DocumentMeta, EvalRecord


def document_meta(settings: Settings) -> dict[str, DocumentMeta]:
    if settings.dataset == "multihop":
        from finsight.data import multihop

        return multihop.load_document_meta(settings)
    from finsight.data import financebench as fb

    return fb.load_document_meta(settings)


def eval_records(settings: Settings) -> list[EvalRecord]:
    if settings.dataset == "multihop":
        from finsight.data import multihop

        return multihop.load_eval_records(settings)
    from finsight.data import financebench as fb

    return fb.load_eval_records(settings)


def pages_df(settings: Settings) -> pd.DataFrame:
    if settings.dataset == "multihop":
        from finsight.data import multihop

        return multihop.load_pages_df(settings)
    from finsight.ingestion import pipeline as ingest

    return ingest.load_pages_df(settings)
