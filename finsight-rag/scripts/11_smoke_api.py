# %% [markdown]
# # Step 11 — Smoke-test the running API   (Phase P9)
#
# **What this step does:** calls a running FinSight API the way a client would: `/health`
# (alive), `/ready` (indexes + LLM loaded), then `/v1/query` with 3 questions and one streamed
# query. It prints status codes, latency, an answer preview and the request ids that tie each
# call to its log line.
#
# Start the server first, in another terminal:  `make serve`   (http://127.0.0.1:8000/docs)

# %%
import json
import time

import httpx
import pandas as pd

from finsight.notebook import Param, configure_pandas, script_args, show, step

configure_pandas()

# %% Parameters
P = script_args(
    "Step 11: smoke-test the API",
    url=Param("http://127.0.0.1:8000", str, "base URL of the running API"),
    timeout=Param(300.0, float, "seconds per request (the local LLM is slow)"),
)
P

QUESTIONS = [
    "What is the FY2018 capital expenditure amount (in USD millions) for 3M?",
    "What was Adobe's operating margin in FY2022?",
    "Who won the 2018 FIFA World Cup?",  # off-topic: should abstain
]

# %%
step("1. Liveness and readiness")
client = httpx.Client(base_url=P.url, timeout=P.timeout)
rows = []
for path in ("/health", "/ready"):
    t0 = time.perf_counter()
    r = client.get(path)
    rows.append(
        {
            "call": f"GET {path}",
            "status": r.status_code,
            "ms": round((time.perf_counter() - t0) * 1000),
            "body": r.text[:120],
        }
    )
show(pd.DataFrame(rows), "Probes (ready = 503 means: read the reasons in body)")

# %%
step("2. Three questions through /v1/query")
rows = []
for q in QUESTIONS:
    t0 = time.perf_counter()
    r = client.post("/v1/query", json={"question": q})
    ms = round((time.perf_counter() - t0) * 1000)
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    rows.append(
        {
            "question": q[:50],
            "status": r.status_code,
            "ms": ms,
            "request_id": r.headers.get("X-Request-ID"),
            "abstained": body.get("abstained"),
            "citations": [
                f"{c['doc_name']} p{c['page_idx'] + 1}" for c in body.get("citations", [])
            ][:3],
            "answer": (body.get("answer") or body.get("detail") or "")[:100],
        }
    )
answers_df = pd.DataFrame(rows)
show(answers_df, "Answers")

# %%
step("3. Streaming (Server-Sent Events)")
events: list[str] = []
t0 = time.perf_counter()
first_token_ms = None
with client.stream("POST", "/v1/query/stream", json={"question": QUESTIONS[0]}) as r:
    for block in r.iter_text():
        for ev in block.strip().split("\n\n"):
            if not ev.startswith("event:"):
                continue
            name = ev.split("\n")[0].removeprefix("event: ")
            if name == "token" and first_token_ms is None:
                first_token_ms = round((time.perf_counter() - t0) * 1000)
            events.append(name)
            if name == "done":
                done = json.loads(ev.split("data: ", 1)[1])
show(
    pd.DataFrame(
        [
            {
                "events": len(events),
                "first": events[0] if events else None,
                "last": events[-1] if events else None,
                "time_to_first_token_ms": first_token_ms,
                "total_ms": round((time.perf_counter() - t0) * 1000),
            }
        ]
    ),
    "Stream",
)

# %% Summary
ok = all(s == 200 for s in answers_df["status"])
print("PASS" if ok and events and events[-1] == "done" else "FAIL: see the tables above")
