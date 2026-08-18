"""P6 latency campaign, task 3 (docs/BUILD_PLAN.md) -- the real, honest P50/P70/P100 numbers.

CLAUDE.md hard rules this script exists to honor:
  - Never report a latency number not produced by this script.
  - Never enable caching during a latency benchmark -- this project has no response/embedding
    cache anywhere in the request path (verified: no cache module exists under src/vrag/), so
    there's nothing to disable, which is itself worth stating rather than assuming.
  - Never run more than one config at a time -- this script assumes it is the ONLY thing running
    against the target server, on an otherwise-idle machine, and measures ONE fixed config
    end to end. Re-run it fresh for a different config; never interleave.

Methodology: 100 queries (eval/test_queries.json) x 5 repetitions = 500 timed samples, after one
full untimed warm-up pass through all 100 queries (discarded, not counted -- named explicitly in
the report so nobody mistakes 5 reps for 6). Total latency (t_pipeline) is measured client-side
with wall-clock time.perf_counter() around the full HTTP round trip -- the honest number, including
serialization/network overhead, not just the sum of server-reported per-stage timings. Per-stage
breakdown comes from the server's own `timings_ms` field on every response (already instrumented
in the harness, src/vrag/harness/stages.py), so this script doesn't re-invent stage timing, only
aggregates what the server already measured internally with `perf_counter_ns()`.

Usage:
  1. Start the server fresh, alone, on an idle machine:
     uvicorn vrag.api.main:app --app-dir src --host 127.0.0.1 --port 8000
  2. python scripts/bench_latency.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
QUERIES_PATH = REPO_ROOT / "eval" / "test_queries.json"
RESULTS_PATH = REPO_ROOT / "eval" / "latency_results.json"
TRACK_B_RESULTS_PATH = REPO_ROOT / "eval" / "latency_track_b_results.json"

N_REPS = 5
N_TRACK_B_QUERIES = 10
N_TRACK_B_RUNS_PER_QUERY = 3  # CLAUDE.md ablation discipline: same config 3x, report the spread


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(len(s) * pct / 100), len(s) - 1)]


def run_one(client: httpx.Client, base_url: str, query: str, k: int = 5) -> dict:
    t0 = time.perf_counter()
    resp = client.post(f"{base_url}/ask", json={"query": query, "k": k}, timeout=30.0)
    wall_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    body = resp.json()
    return {
        "wall_ms": wall_ms,
        "status": body.get("status"),
        "track": body.get("track"),
        "timings_ms": body.get("timings_ms", {}),
        "stages_skipped": body.get("stages_skipped", []),
    }


def bench(base_url: str, queries: list[dict], reps: int = N_REPS) -> list[dict]:
    samples = []
    with httpx.Client() as client:
        print(f"Warm-up pass: {len(queries)} queries, results discarded...")
        for row in queries:
            try:
                run_one(client, base_url, row["query"])
            except httpx.HTTPError as e:
                print(f"  warm-up call failed (query={row['query']!r}): {e}")

        print(f"Timed passes: {reps} x {len(queries)} queries = {reps * len(queries)} samples...")
        for rep in range(reps):
            for i, row in enumerate(queries):
                try:
                    result = run_one(client, base_url, row["query"])
                except httpx.HTTPError as e:
                    print(f"  rep {rep} query {i} failed: {e}")
                    continue
                result["category"] = row["category"]
                result["rep"] = rep
                samples.append(result)
            print(f"  rep {rep + 1}/{reps} done ({len(samples)} samples so far)")
    return samples


def summarize(samples: list[dict]) -> dict:
    def pct_block(values: list[float]) -> dict:
        return {
            "p50": _percentile(values, 50),
            "p70": _percentile(values, 70),
            "p95": _percentile(values, 95),
            "p100": _percentile(values, 100),
            "mean": statistics.mean(values) if values else 0.0,
            "n": len(values),
        }

    overall = pct_block([s["wall_ms"] for s in samples])

    by_track: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        track = s.get("track") or "none"
        by_track[track].append(s["wall_ms"])
    track_summary = {track: pct_block(vals) for track, vals in by_track.items()}

    by_status: dict[str, int] = defaultdict(int)
    for s in samples:
        by_status[s.get("status", "unknown")] += 1

    stage_names: set[str] = set()
    for s in samples:
        stage_names |= set(s["timings_ms"].keys())
    per_stage = {
        stage: pct_block([s["timings_ms"][stage] for s in samples if stage in s["timings_ms"]])
        for stage in sorted(stage_names)
    }

    return {
        "overall_wall_ms": overall,
        "by_track_wall_ms": track_summary,
        "by_status_count": dict(by_status),
        "per_stage_server_ms": per_stage,
    }


async def bench_track_b(queries: list[dict]) -> list[dict]:
    """Track B's real, standalone completion latency -- calls generation.sarvam_llm.generate()
    directly with real retrieved chunks, bypassing GenerateStage's budget gate entirely (that
    gate always shows ~0ms of "success" for Track B under the 200ms default budget, since it
    only ever gets a fair chance to finish -- see the P50=246ms vs true retrieve+extract ~10ms
    finding this script surfaced). This measures full round-trip completion time via generate()'s
    own (streaming-internally, complete-on-return) implementation -- NOT first-token TTFT, which
    would need lower-level access to the streaming deltas this wrapper doesn't expose. Reported
    honestly as "completion latency", not mislabeled as TTFT.
    """
    from vrag.generation.sarvam_llm import generate as generate_track_b
    from vrag.retrieval.interface import retrieve

    in_domain = [q for q in queries if q["category"] == "in_domain"][:N_TRACK_B_QUERIES]
    results = []
    for row in in_domain:
        chunks = await retrieve(row["query"], k=5)
        if not chunks:
            continue
        for run in range(N_TRACK_B_RUNS_PER_QUERY):
            t0 = time.perf_counter()
            result = await generate_track_b(row["query"], chunks, timeout_s=15.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            results.append(
                {
                    "query": row["query"],
                    "run": run,
                    "elapsed_ms": elapsed_ms,
                    "succeeded": result is not None,
                }
            )
            print(f"  Track B: query={row['query'][:30]!r} run={run} "
                  f"succeeded={result is not None} elapsed={elapsed_ms:.1f}ms")
    return results


def summarize_track_b(results: list[dict]) -> dict:
    succeeded = [r["elapsed_ms"] for r in results if r["succeeded"]]
    failed = [r["elapsed_ms"] for r in results if not r["succeeded"]]
    return {
        "n_total": len(results),
        "n_succeeded": len(succeeded),
        "n_failed": len(failed),
        "succeeded_ms": {
            "p50": _percentile(succeeded, 50),
            "p70": _percentile(succeeded, 70),
            "p100": _percentile(succeeded, 100),
            "mean": statistics.mean(succeeded) if succeeded else 0.0,
        }
        if succeeded
        else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--reps", type=int, default=N_REPS)
    parser.add_argument(
        "--skip-track-b", action="store_true",
        help="skip the standalone Track B measurement (needs a real Sarvam call per run)",
    )
    args = parser.parse_args()

    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(queries)} queries from {QUERIES_PATH}")

    samples = bench(args.base_url, queries, reps=args.reps)
    summary = summarize(samples)

    RESULTS_PATH.write_text(
        json.dumps({"samples": samples, "summary": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n=== Overall t_pipeline (wall-clock, {len(samples)} samples) ===")
    o = summary["overall_wall_ms"]
    print(f"P50={o['p50']:.1f}ms  P70={o['p70']:.1f}ms  P95={o['p95']:.1f}ms  "
          f"P100={o['p100']:.1f}ms  mean={o['mean']:.1f}ms")

    print("\n=== By track ===")
    for track, t in summary["by_track_wall_ms"].items():
        print(f"{track:12s} n={t['n']:4d}  P50={t['p50']:.1f}ms  P70={t['p70']:.1f}ms  "
              f"P100={t['p100']:.1f}ms")

    print("\n=== By status ===")
    print(summary["by_status_count"])

    print("\n=== Per-stage server-reported ms (P50 / P100) ===")
    for stage, s in summary["per_stage_server_ms"].items():
        print(f"{stage:16s} P50={s['p50']:.3f}ms  P100={s['p100']:.3f}ms  n={s['n']}")

    print(f"\nWrote full results to {RESULTS_PATH}")

    if not args.skip_track_b:
        print(f"\n=== Standalone Track B: {N_TRACK_B_QUERIES} queries x "
              f"{N_TRACK_B_RUNS_PER_QUERY} runs (real Sarvam calls, bypassing the budget gate) ===")
        tb_results = asyncio.run(bench_track_b(queries))
        tb_summary = summarize_track_b(tb_results)
        TRACK_B_RESULTS_PATH.write_text(
            json.dumps(
                {"samples": tb_results, "summary": tb_summary}, indent=2, ensure_ascii=False
            ),
            encoding="utf-8",
        )
        print(f"\nTrack B: {tb_summary['n_succeeded']}/{tb_summary['n_total']} succeeded")
        if tb_summary["succeeded_ms"]:
            s = tb_summary["succeeded_ms"]
            print(f"Completion latency (succeeded only): P50={s['p50']:.1f}ms "
                  f"P70={s['p70']:.1f}ms P100={s['p100']:.1f}ms mean={s['mean']:.1f}ms")
        print(f"Wrote Track B results to {TRACK_B_RESULTS_PATH}")
