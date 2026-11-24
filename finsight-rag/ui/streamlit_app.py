"""
FinSight RAG demo UI: ask a question, see the cited answer AND what happened inside the pipeline.

    make serve   # terminal 1: the API on :8000
    make ui      # terminal 2: this app on :8501

The UI is a thin client of the HTTP API (it never loads models or indexes itself), so what you
see is exactly what an API consumer gets.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("FINSIGHT_API_URL", "http://127.0.0.1:8000")
EXAMPLES = [
    "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to "
    "the question by relying on the details shown in the cash flow statement.",
    "Is 3M a capital-intensive business based on FY2022 data?",
    "Does Adobe have an improving Free cashflow conversion as of FY2022?",
    "What was the key agenda of the AMCOR's 8k filing dated 1st July 2022?",
    "What are the major products and services that AMD sells as of FY22?",
]
STAGES = [("sparse", "BM25"), ("dense", "Dense"), ("rrf", "RRF"), ("rerank", "Reranked")]

st.set_page_config(page_title="FinSight RAG", page_icon="📄", layout="wide")


@st.cache_data(ttl=30)
def api_config() -> dict[str, Any]:
    try:
        return dict(httpx.get(f"{API}/v1/config", timeout=5).json())
    except httpx.HTTPError as exc:
        return {"error": str(exc)}


# ------------------------------------------------------------------ sidebar
cfg = api_config()
with st.sidebar:
    st.header("Settings")
    if "error" in cfg:
        st.error(f"API not reachable at {API}. Start it with `make serve`.\n\n{cfg['error']}")
    else:
        st.caption(f"config `{cfg.get('config')}` · fingerprint `{cfg.get('fingerprint')}`")
        st.markdown(
            f"**Chunking** {cfg.get('chunking')}  \n**Embedding** {cfg.get('embedding')}  \n"
            f"**Retrieval** {cfg.get('retrieval')}  \n**Rerank** {cfg.get('rerank')}  \n"
            f"**LLM** {cfg.get('llm')}"
        )
    pipeline = st.selectbox(
        "Pipeline", ["(config default)", "hybrid_rerank", "naive", "corrective", "agentic"]
    )
    llm = st.text_input("LLM preset (blank = config)", "", help="A name from configs/llms.yaml")
    top_k = st.slider("Chunks given to the LLM (top_k)", 1, 20, 10)
    st.subheader("Filters (blank = extracted from the question)")
    companies = st.text_input("Companies (comma-separated)", "")
    years = st.text_input("Fiscal years (comma-separated)", "")

# ------------------------------------------------------------------ main
st.title("FinSight RAG")
st.caption("Grounded, cited answers over SEC filings (FinanceBench). Every claim links to a page.")

example = st.selectbox("Example questions", ["(type your own)", *EXAMPLES])
question = st.text_area(
    "Question", "" if example == "(type your own)" else example, height=80, max_chars=1000
)

if st.button("Ask", type="primary", disabled=len(question.strip()) < 3):
    payload: dict[str, Any] = {
        "question": question.strip(),
        "top_k": top_k,
        "include_retrieved": True,
    }
    if pipeline != "(config default)":
        payload["pipeline"] = pipeline
    if llm.strip():
        payload["llm"] = llm.strip()
    filters = {
        "companies": [c.strip() for c in companies.split(",") if c.strip()],
        "doc_periods": [int(y) for y in years.replace(" ", "").split(",") if y.isdigit()],
    }
    if filters["companies"] or filters["doc_periods"]:
        payload["filters"] = filters
    with st.spinner("Retrieving and generating (the local LLM takes a few seconds)..."):
        try:
            r = httpx.post(f"{API}/v1/query", json=payload, timeout=600)
        except httpx.HTTPError as exc:
            st.error(f"Request failed: {exc}")
            st.stop()
    if r.status_code != 200:
        st.error(f"{r.status_code}: {r.json().get('detail', r.text)}")
        st.stop()
    ans = r.json()

    # ---- the answer
    if ans["abstained"]:
        st.warning(ans["answer"])
    else:
        st.markdown(ans["answer"])
    for w in ans.get("warnings", []) + ans.get("degraded", []):
        st.caption(f"⚠ {w}")

    # ---- citation chips
    if ans["citations"]:
        st.subheader("Sources")
        cols = st.columns(min(len(ans["citations"]), 4))
        for i, c in enumerate(ans["citations"]):
            with cols[i % len(cols)].expander(
                f"[{c['marker']}] {c['doc_name']} · p.{c['page_idx'] + 1}"
            ):
                st.write(c["snippet"])
                st.caption(f"PDF page {c['page_idx'] + 1} (0-based index {c['page_idx']})")

    st.caption(
        f"{ans['pipeline']} · {ans['provider']}:{ans['model']} · request `{ans['request_id']}` · "
        f"{ans['timings_ms'].get('total', 0) / 1000:.1f} s"
    )

    # ---- under the hood
    with st.expander("Under the hood", expanded=False):
        retrieved = ans.get("retrieved", [])
        cols = st.columns(len(STAGES))
        for col, (key, label) in zip(cols, STAGES, strict=True):
            rows = [
                {
                    "doc": rc["chunk"]["doc_name"],
                    "page": rc["chunk"]["page_start"] + 1,
                    "score": round(rc["scores_by_stage"][key], 4),
                }
                for rc in retrieved
                if key in rc.get("scores_by_stage", {})
            ]
            col.markdown(f"**{label}**")
            if rows:
                col.dataframe(
                    pd.DataFrame(rows).sort_values("score", ascending=False).head(10),
                    hide_index=True,
                )
            else:
                col.caption("not used by this config")
        st.caption("Each column ranks the chunks the LLM received by that stage's score.")
        t1, t2 = st.columns(2)
        timings = {k: v for k, v in ans["timings_ms"].items() if k != "total"}
        if timings:
            t1.markdown("**Timings (ms)**")
            t1.bar_chart(pd.Series(timings, name="ms"))
        t2.markdown("**Token usage**")
        t2.json(ans.get("usage", {}))
