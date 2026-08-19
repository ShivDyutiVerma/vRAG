# LATENCY_BUDGET.md

Target vs. measured, ms. Targets are hypotheses from `docs/AGENT_BUILD_SPEC.md` §4. **Every number
below was produced by `scripts/bench_latency.py`** (`CLAUDE.md` hard rule) run against a real,
freshly-started local server, or by `docs/DECISIONS_R.md` R-036's live-Render investigation script —
never estimated. No caching exists anywhere in this codebase to disable (verified: no cache module
under `src/vrag/`). Full raw data: `eval/latency_results.json`, `eval/latency_track_b_results.json`,
`eval/live_render_latency_results.json`.

**This file mixes two genuinely different environments — read the label on every number.**
**LOCAL/DOCKER** = this dev machine or a Docker container on it, full CPU, no network hop between
client and server. **LIVE RENDER** = the actual deployed free-tier instance
(`https://vrag-voice.onrender.com`), shared 0.1 vCPU, real network. They are not interchangeable,
and the gap between them is itself the headline finding of this document.

## The headline finding: `t_pipeline` meets budget locally; Render's free-tier CPU is the real constraint

| Measurement | Environment | P50 | P70 | P95 | P100 | n |
|---|---|---|---|---|---|---|
| Track A, true stage cost (server `timings_ms` sum: G1+G2+retrieve+G3+extract+G5) | LOCAL | **5.2ms** | 5.7ms | 6.9ms | 8.9ms | 275 |
| End-to-end wall-clock, answered query — **current** | LOCAL | **10.1ms** | 10.6ms | 11.8ms | 16.2ms | 275 |
| Abstained query (G3 gate), end-to-end wall-clock | LOCAL | 8.0ms | — | — | 13.2ms | 165 |
| Refused query (G1/G2), end-to-end wall-clock | LOCAL | 1.2-1.5ms | — | — | 4.1ms | 60 |
| `retrieve` / `t_pipeline` stage-sum, 40 real sequential queries | **LIVE RENDER** | **594.9ms** | — | — | **1105.7ms** | 40 |

**Locally, both the stage-cost budget and the end-to-end wall-clock budget are comfortably met.**
On the live, free-tier-deployed URL, `t_pipeline` does **not** meet 200ms — stated plainly, see
below for why, not fudged into looking closer than it is.

## Local wall-clock: what changed, and why (read this before trusting the 10.1ms number)

**Before `docs/DECISIONS_R.md` R-036 (2026-08-19, superseded — kept as a real, dated record, not
deleted):** the end-to-end wall-clock for an answered query measured **P50 ≈ 213-246ms, P100
236-383ms** (3 runs) — 20-40x the true stage cost. `GenerateStage` attempted a real Track B
(Sarvam) call on *every* answerable query with whatever budget remained (~190ms of the 200ms
default), but Sarvam's real completion time is ~2s (§4/Generation) — so the attempt was
structurally guaranteed to time out, and the circuit breaker never tripped to stop it (it only
counts failures at ≥2.0s, "fair chance," as a health signal; a 190ms timeout never reaches that
threshold). The client waited out that entire doomed attempt before Track A's already-ready answer
(computed in ~5ms) was finally returned. Verified as a real effect, not a measurement artifact, by
two independent facts: refused/abstained queries (which skip `GenerateStage` entirely) stayed fast
throughout; only answered queries were affected, by almost exactly the size of the remaining
budget window.

**R-036's fix:** `GenerateStage.min_viable_ms` (`src/vrag/harness/stages.py`) changed from the
spec's aspirational 110ms TTFT target to `circuit_breaker.MIN_FAIR_TIMEOUT_S * 1000` (2000ms) —
reusing an existing constant, not inventing a new one. `run_pipeline`'s existing generic
`can_afford(min_viable_ms)` pre-flight check now sheds Track B *before it starts* whenever
remaining budget can't realistically fit a fair Sarvam attempt, instead of starting it and paying
up to a ~2s doomed wait. Nothing inside `GenerateStage.run()` itself changed — timeout, circuit
breaker, G4 gate, structured-output repair, and fallback are all identical; the pre-existing
circuit-breaker test suite passes unmodified, direct proof of that.

**Re-measured 2026-08-20, after the redeploy that shipped this fix, using the same
`scripts/bench_latency.py` methodology (100 queries × 5 reps, one discarded warm-up pass):
P50=10.1ms, P70=10.6ms, P95=11.8ms, P100=16.2ms** — the doomed-attempt gap is gone locally. Every
one of the 275 answered samples in this run used Track A only (`{'extractive': 500}` — 0 samples
used Track B's stream), which is the **expected, by-design consequence** of the gate under a
200ms total budget: Sarvam realistically needs ~2s minimum, so a 200ms budget structurally never
has room to attempt it fairly, and the gate correctly skips straight to Track A every time rather
than trying and failing. This does not mean Track B is broken or unused — see §4/Generation for
its own real, positive, standalone measurement when given a fair budget.

## Per-stage table (LOCAL)

| # | Stage | Owner | Target p50 | Measured P50 | Measured P100 | Notes |
|---|-------|-------|-----------|--------------|----------------|-------|
| 1 | Input guardrail (G1) | P | 2ms | 0.018ms | 0.040ms | |
| 2 | Scope/language guardrail (G2) | P | — | 0.010ms | 0.028ms | |
| 3 | Query embed + dense search (retrieve) | R | 8ms + 3ms | 8.4ms | 14.4ms | Dense-only (A3 winner, R-010) — no sparse/rerank on the shipped path |
| 4 | Sparse search (BM25) | R | 5ms | N/A | N/A | **Not on the shipped path** — A3 found dense-only beats hybrid (R-010); architecturally absent, not un-measured |
| 5 | RRF fusion | R | <1ms | N/A | N/A | Same — not on the shipped path |
| 6 | Cross-encoder rerank | R | 25ms | N/A | N/A | **Not on the shipped path** — A4 (R-012) found `none` wins outright; a targeted follow-up (R-038) confirmed this holds even on the failure case reranking was hypothesized to fix. Architecturally absent |
| 7 | Grounding gate (G3) | Joint | 1ms | 0.016ms | 0.033ms | |
| 8a | Track A extractive span select | P | 10ms | 0.005ms | 0.018ms | |
| 8b | Track B generation (full completion, not TTFT) | P | 110ms (TTFT) | 1976.5ms | 6429.8ms | Standalone measurement, bypasses the harness's budget gate — see §4/Generation. Under the default 200ms budget this stage does not run (shed by the pre-flight gate, see above) |
| 9 | Output guardrail (G5) | Joint | 5ms | 0.076ms | 0.794ms | |
| | **Total to Track A answer (true stage cost)** | | ~30ms | **5.2ms** | 8.9ms | Comfortably under 200ms |
| | **Total to Track A answer (end-to-end wall-clock, current)** | | — | **10.1ms** | 16.2ms | Now tracks stage cost closely — the pre-R-036 gap is closed |

## Track B, standalone (bypasses the harness's budget gate entirely)

Measured by calling `generation.sarvam_llm.generate()` directly with real retrieved chunks and a
generous 15s timeout — the number Track B actually delivers when given a fair chance, distinct
from what the 200ms-budget harness measures (which now sheds it cleanly, per above).

- **19/30 calls succeeded (63.3%).** Completion latency (succeeded only): P50=1976.5ms,
  P70=2557.3ms, P100=6429.8ms, mean=2483.1ms.
- **A real bug was found and fixed while building this measurement**, not a pre-existing known
  issue: before the fix, **0/30 calls succeeded**. `src/vrag/generation/sarvam_llm.py`'s streaming
  handler used `choices[0]["delta"].get("content", "")`, whose default only applies when the key is
  *absent* — Sarvam's SSE stream sometimes sends an explicit `"content": null`, which `.get()`
  returns as `None`, then crashed on `accumulated += delta`. One-line fix: `.get("content") or ""`.
  Verified via `pytest tests/generation` (27/27 still green) and by direct re-testing — this bug
  alone explains the jump from 0% to 63% Track B success.
- **The remaining 11/30 failures (37%) are the real, already-documented Sarvam-side reliability
  issue** (P-R20 in `docs/RISKS.md`, `docs/DECISIONS_P.md` P-017/P-019): the model sometimes stalls
  mid-completion and pads whitespace instead of finishing. This is provider-side, already mitigated
  (stall-detection abort inside the real pipeline), and worth reporting to Sarvam — not something
  this project can fix.

## LIVE RENDER: why the free tier doesn't meet 200ms, and what was actually measured

**`docs/DECISIONS_R.md` R-036** ran `scripts/bench_live_render.py` — a real 40-query sequential
benchmark against the live deployed URL, not a local proxy — and found `retrieve`/`t_pipeline`
stage-sum **P50=594.9ms, P100=1105.7ms even on a warm instance**, 10-40x the identical local/Docker
code path (12-47ms). Two real, checked (not inferred) findings narrow the cause:

- **Not memory pressure.** Render's own CPU/memory metrics API showed steady memory (~409-430MB)
  throughout the run.
- **Root cause: CPU contention (Render free tier is 0.1 shared vCPU vs. Starter's 0.5, per
  Render's own published specs), compounded by a directly-observed free-tier instance
  spin-up/replacement mid-benchmark** (a new instance ID appeared mid-run with memory still
  climbing, matching Render's documented "spins down after 15 min idle" free-tier policy exactly).
  Track B was attempted on only 1/40 queries in this run (most queries in the sample were
  abstained/refused), ruling that out as a confound; a secondary, smaller (~230ms) consistent
  network gap was also present once warm.

**This is a hosting-tier limitation, not a code regression.** The exact same request path that
runs in single-digit milliseconds locally is genuinely, measurably CPU-starved on Render's free
instance. **`t_pipeline` does not hold under 200ms on the live deployed URL for a retrieval-
touching request, and this document does not claim otherwise.** A paid tier (0.5+ dedicated vCPU)
would very likely close most of this gap, since the code itself is not the bottleneck — untested,
would need its own real measurement, not assumed from the free-tier numbers.

## Response-status mix (LOCAL, 500 samples, real query distribution: 60% in-domain / 20% off-topic
/ 10% unsafe / 10% degenerate, `eval/test_queries.json`)

| Status | Count | % |
|---|---|---|
| answered | 275 | 55% |
| abstained (G3 confidence gate) | 165 | 33% |
| refused (G1/G2) | 60 | 12% |

## Hot-path rules that keep this budget honest

- No network calls on the hot path except the LLM (embeddings, BM25, reranker, guardrails all
  in-process — this is why a hosted vector DB was rejected outright, see `docs/TECH_MENU.md` S6).
- No cold starts in steady state — eager warmup at boot (`docs/DECISIONS_R.md` R-035) means
  `/healthz` never returns `200` until the retriever has already loaded *and* run one real warmup
  embedding; verified directly in Docker: the port refused connections until 5.42s after container
  start, at which point the first successful response was already `{"retrieval":"real"}`. First
  real query after that: 42.2ms (was 1125.8ms before R-035's fix).
- No per-request disk I/O on the hot path — the index is memory-resident (`SQLiteChunkLookup` reads
  are lazy per-chunk, not per-request-bulk; FAISS is fully in memory).

## What's still open

- **`t_e2e_voice` (STT round trip) is not yet measured by a dedicated script** — `eval/audio/` has
  95 real Sarvam-TTS-synthesized 16kHz WAV files ready, matching `WS /voice`'s expected input
  format, but the WebSocket benchmark client itself isn't built.
- `tests/test_latency_regression.py` **exists and passes** (`test_track_a_stage_cost_p50_under_200ms`)
  — CI-gates Track A's stage cost, not yet extended to gate the end-to-end wall-clock number.
- The LIVE RENDER number is a single 40-query run (R-036), not repeated 3x per `CLAUDE.md`'s own
  "run the same config 3x, report the spread" discipline — a real gap, not silently glossed over.
  A second live-Render run would strengthen this number but wasn't repeated before the redeploy
  that shipped the STT/frontend fixes (2026-08-20), since retrieval/latency architecture was
  explicitly frozen for that redeploy.

## Provider RTT probe

Not run as a separate probe — Track B's standalone measurement above (real Sarvam calls, N=30)
supersedes the originally-planned `scripts/probe_latency.py` TCP/TLS/TTFT breakdown for this
project's purposes. ADR-003 was never separately recorded as its own probe; see
`docs/DECISIONS_P.md` P-012 for the earlier probe data gathered during Track B development.
