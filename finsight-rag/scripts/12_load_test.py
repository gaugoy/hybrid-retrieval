# %% [markdown]
# # Step 12 — Load test the running API   (Phase P10)
#
# **What this step does:** sends N real FinanceBench questions to `/v1/query` with 1 and then
# `users` concurrent clients and reports p50/p95 latency, throughput and errors. A last pass
# repeats the same questions to show the answer cache (`X-Cache: hit`).
#
# Start the server first:  `make serve`  (no Docker needed)

# %%
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pandas as pd

from finsight.config import load_settings
from finsight.data import datasets as ds
from finsight.notebook import Param, configure_pandas, script_args, show, step

configure_pandas()

# %% Parameters
P = script_args(
    "Step 12: load-test the API",
    url=Param("http://127.0.0.1:8000", str, "base URL of the running API"),
    users=Param(5, int, "concurrent clients in the load pass"),
    n=Param(10, int, "requests per pass (distinct questions)"),
    offset=Param(20, int, "skip the first questions (so earlier runs' cached answers don't count)"),
    timeout=Param(900.0, float, "seconds per request"),
)
P

# %%
step("1. Questions and warm-up")
records = ds.eval_records(load_settings())
questions = [r.question for r in records[P.offset : P.offset + 2 * P.n]]
client = httpx.Client(
    base_url=P.url, timeout=P.timeout, limits=httpx.Limits(max_connections=P.users * 2)
)
print("ready:", client.get("/ready").json())
client.post("/v1/query", json={"question": "warm-up: what does 3M sell?"})


def call(q: str) -> dict:
    t0 = time.perf_counter()
    try:
        r = client.post("/v1/query", json={"question": q})
        status, cache = r.status_code, r.headers.get("X-Cache", "-")
    except httpx.HTTPError as exc:
        status, cache = type(exc).__name__, "-"
    return {"status": status, "s": time.perf_counter() - t0, "cache": cache}


def run_pass(name: str, qs: list[str], users: int) -> dict:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(users) as ex:
        res = list(ex.map(call, qs))
    wall = time.perf_counter() - t0
    lat = sorted(r["s"] for r in res)
    return {
        "pass": name,
        "users": users,
        "requests": len(res),
        "errors": sum(r["status"] != 200 for r in res),
        "cache_hits": sum(r["cache"] == "hit" for r in res),
        "p50_s": round(statistics.median(lat), 2),
        "p95_s": round(lat[max(0, round(0.95 * len(lat)) - 1)], 2),
        "throughput_rpm": round(len(res) / wall * 60, 1),
    }


# %%
step("2. One user, then concurrent users (distinct questions: cache misses)")
rows = [run_pass("sequential", questions[: P.n], 1)]
rows.append(run_pass("concurrent", questions[P.n : 2 * P.n], P.users))

# %%
step("3. Same questions again (answer cache)")
rows.append(run_pass("cached", questions[P.n : 2 * P.n], P.users))
load_df = pd.DataFrame(rows)
show(load_df, "Load test")

# %% Summary
ok = load_df["errors"].sum() == 0
print("PASS" if ok else "FAIL: requests failed, see the errors column and the API log")
