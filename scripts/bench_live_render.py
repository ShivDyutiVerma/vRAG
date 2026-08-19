"""Measurement-only live latency benchmark against the real, public Render deployment
(https://vrag-voice.onrender.com) -- no production code, deployment config, FAISS, corpus,
embedding model, tokenizer, or Render plan changed by this script. Written to answer one
question: local Docker retrieve is ~12-47ms, live Render retrieve was observed at 190ms-1.1s
across a 9-query spot check -- is that retrieval computation, Track B waiting, network/LLM
latency, Render CPU contention, or a mix?

Real queries only, sampled from the same 500-query held-out set every other eval in this project
uses (eval/heldout_queries.json), fixed seed matching project convention (R-027 etc.). Requests
are sent strictly SEQUENTIALLY, one at a time -- concurrent requests would introduce server-side
queueing that convolves with the exact contention question this script exists to isolate
(CLAUDE.md's "the latency pass stays strictly sequential" principle, applied here to a live
measurement, not just the local bench_latency.py path).

t_pipeline, per AGENT_BUILD_SPEC.md §3.2, is measured server-side and explicitly excludes
client->server network transit -- so it's computed here as the sum of every stage duration in
the response's own `timings_ms` (the same "true stage-cost" quantity docs/LATENCY_BUDGET.md
already reports for the local campaign), not the wall-clock round trip. The wall-clock round trip
is reported separately, labeled "client-visible", specifically so the gap between the two is
itself evidence -- a large, variable gap points at network/queueing/CPU-scheduling delay that
happens before the server's own stage clock even starts, which t_pipeline structurally cannot see.

/ask is not a streaming endpoint, so Track B's `generate` timing here is the stage's total
duration once it completes (success, repair, or timeout) -- not true token-by-token TTFT. Stated
plainly in the output, not presented as something it isn't.

Usage: python scripts/bench_live_render.py
"""

from __future__ import annotations

import json
import random
import statistics
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
HELDOUT_PATH = REPO_ROOT / "eval" / "heldout_queries.json"
LIVE_URL = "https://vrag-voice.onrender.com/ask"
SEED = 42
N_QUERIES = 40
STAGE_ORDER_TO_TRACK_A = ["input_guard", "scope_guard", "retrieve", "ground_gate", "extract_answer"]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(len(s) * pct / 100), len(s) - 1)]


def run_benchmark() -> list[dict]:
    heldout = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    sample = rng.sample(heldout, N_QUERIES)

    records: list[dict] = []
    client = httpx.Client(timeout=30.0)

    print(f"benchmark_start_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    for i, row in enumerate(sample):
        query = row["query"]
        t0_wall = time.time()
        t0 = time.perf_counter()
        error = None
        status_code = None
        data: dict = {}
        try:
            resp = client.post(LIVE_URL, json={"query": query, "k": 5})
            status_code = resp.status_code
            data = resp.json()
        except Exception as e:  # noqa: BLE001 -- record and continue, don't abort the run
            error = str(e)
        t1 = time.perf_counter()
        client_ms = (t1 - t0) * 1000

        timings = data.get("timings_ms", {}) if data else {}
        stage_sum = round(sum(timings.values()), 3) if timings else None

        cum = 0.0
        have_all_track_a_stages = True
        for s in STAGE_ORDER_TO_TRACK_A:
            if s in timings:
                cum += timings[s]
            else:
                have_all_track_a_stages = False
        time_to_track_a_ms = round(cum, 3) if have_all_track_a_stages else None

        record = {
            "query_id": row.get("query_id", i),
            "request_start_wall_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t0_wall)),
            "http_status": status_code,
            "error": error,
            "client_visible_ms": round(client_ms, 2),
            "t_pipeline_stage_sum_ms": stage_sum,
            "client_minus_stagesum_ms": (
                round(client_ms - stage_sum, 2) if stage_sum is not None else None
            ),
            "retrieve_ms": timings.get("retrieve"),
            "time_to_track_a_answer_ms": time_to_track_a_ms,
            "extract_answer_ms": timings.get("extract_answer"),
            "generate_stage_ms": timings.get("generate"),  # Track B stage duration, not true TTFT
            "status": data.get("status"),
            "track": data.get("track"),
            "stages_skipped": data.get("stages_skipped"),
            "timings_ms": timings,
        }
        records.append(record)
        print(
            f"[{i + 1}/{N_QUERIES}] status={record['status']} track={record['track']} "
            f"client={record['client_visible_ms']:.1f}ms stage_sum={stage_sum} "
            f"retrieve={record['retrieve_ms']} skipped={record['stages_skipped']}"
        )

    print(f"benchmark_end_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    return records


def summarize(records: list[dict]) -> None:
    client_vals = [r["client_visible_ms"] for r in records if r["client_visible_ms"] is not None]
    stage_vals = [
        r["t_pipeline_stage_sum_ms"] for r in records if r["t_pipeline_stage_sum_ms"] is not None
    ]
    retrieve_vals = [r["retrieve_ms"] for r in records if r["retrieve_ms"] is not None]

    def report(name: str, vals: list[float]) -> None:
        if not vals:
            print(f"{name}: no data")
            return
        print(
            f"{name}: n={len(vals)} min={min(vals):.1f} P50={percentile(vals, 50):.1f} "
            f"P70={percentile(vals, 70):.1f} P95={percentile(vals, 95):.1f} "
            f"P100={max(vals):.1f} max={max(vals):.1f}"
        )

    print("\n===== SUMMARY (all times in ms) =====")
    report("retrieve (server-reported stage time)", retrieve_vals)
    report("t_pipeline (server-side stage-sum, per spec definition)", stage_vals)
    report("client-visible (wall-clock round trip)", client_vals)

    for label, threshold in [(">200ms", 200), (">250ms", 250), (">500ms", 500), (">1000ms", 1000)]:
        n_stage = sum(1 for v in stage_vals if v > threshold)
        n_client = sum(1 for v in client_vals if v > threshold)
        print(
            f"queries {label}: t_pipeline(stage-sum)={n_stage}/{len(stage_vals)}  "
            f"client-visible={n_client}/{len(client_vals)}"
        )

    n_generate_attempted = sum(1 for r in records if "generate" not in (r["stages_skipped"] or []))
    n_generate_succeeded = sum(1 for r in records if r["track"] == "generative")
    print(
        f"\nTrack B attempted: {n_generate_attempted}/{len(records)}  "
        f"succeeded (track=generative): {n_generate_succeeded}/{len(records)}"
    )

    gaps = [
        r["client_minus_stagesum_ms"]
        for r in records
        if r["client_minus_stagesum_ms"] is not None
    ]
    if gaps:
        print(
            f"\nclient_visible - t_pipeline(stage-sum) gap: mean={statistics.mean(gaps):.1f}ms "
            f"stdev={statistics.pstdev(gaps):.1f}ms min={min(gaps):.1f}ms max={max(gaps):.1f}ms"
        )
        print(
            "(this gap is network transit + anything server-side that happens before the first "
            "stage's clock starts -- e.g. queueing -- t_pipeline's own definition excludes it)"
        )


if __name__ == "__main__":
    records = run_benchmark()
    out_path = REPO_ROOT / "eval" / "live_render_latency_results.json"
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")
    summarize(records)
