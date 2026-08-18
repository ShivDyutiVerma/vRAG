# LATENCY_BUDGET.md

Target vs. measured, ms. Targets are hypotheses from `AGENT_BUILD_SPEC.md` §4. **Every number below
was produced by `scripts/bench_latency.py`** (`CLAUDE.md` hard rule) — run locally (deployment is
parked, see `docs/RISKS.md` R4/R-R21), 100-query test set (`eval/test_queries.json`), 5 timed
repetitions after one discarded warm-up pass = 500 samples for the main run; the standalone Track B
measurement is 10 queries x 3 runs = 30 real Sarvam calls, per CLAUDE.md's "run the same config 3x,
report the spread" ablation discipline. No caching exists anywhere in this codebase to disable
(verified: no cache module under `src/vrag/`). Full raw data: `eval/latency_results.json`,
`eval/latency_track_b_results.json`.

## The headline finding: Track A is fast, but the wire never lets you see that

| Measurement | P50 | P70 | P95 | P100 | n |
|---|---|---|---|---|---|
| **Track A, true stage cost** (server-internal `timings_ms` sum: G1+G2+retrieve+G3+extract+G5) | **5.2ms** | 5.7ms | 6.9ms | 8.9ms | 275 |
| **"Answered" query, end-to-end wall-clock** (what a client actually experiences today) | **~213-246ms** (3 runs) | ~218-258ms | ~227-293ms | 236-383ms | 275 x 3 runs |
| Abstained query (G3 gate), end-to-end wall-clock | 8.0ms | — | — | 13.2ms | 165 |
| Refused query (G1/G2), end-to-end wall-clock | 1.2-1.5ms | — | — | 4.1ms | 60 |

**Why the huge gap on answered queries:** `GenerateStage` (`src/vrag/harness/stages.py`) attempts a
real Track B (Sarvam) call on *every* answerable query with whatever budget remains (~190ms after
retrieval, out of the 200ms default). Sarvam's real completion time is ~2s (measured below) — so
that attempt is **structurally guaranteed to time out** under the default budget. The circuit
breaker that's supposed to detect "Track B keeps failing, stop trying" never trips, because it only
counts failures at >=2.0s ("fair chance", `circuit_breaker.MIN_FAIR_TIMEOUT_S`) as a health signal,
and a 190ms timeout never reaches that threshold — so it's never counted as a health signal in
either direction, and the attempt repeats every single request. The client waits out that entire
doomed attempt before Track A's already-ready answer (computed in ~5ms) is finally returned.

This is not a measurement bug — confirmed directly by two independent facts: (1) refused/abstained
queries, which skip `GenerateStage` entirely via `_upstream_refused`, are genuinely fast (1-13ms,
matching their stage-sum almost exactly); (2) answered queries are the only ones affected, and by
almost exactly the size of the remaining budget window. **Decision (per the user, 2026-08-19): report
both numbers honestly rather than change the harness's behavior** — `GenerateStage`'s design
(give Track B a fair shot within budget, shed cleanly if it can't make it) is deliberate, tested,
and correct in spirit; it's the fact that Sarvam is currently ~10x slower than the total budget that
makes every attempt futile in practice, not a flaw in the shedding logic itself. This matches
`docs/BUILD_PLAN.md` P6's own guidance almost exactly: *"If Track B can't clear 200ms, say so
plainly and show the layered design that does."*

## Per-stage table

| # | Stage | Owner | Target p50 | Measured P50 | Measured P100 | Notes |
|---|-------|-------|-----------|--------------|----------------|-------|
| 1 | Input guardrail (G1) | P | 2ms | 0.017ms | 0.136ms | |
| 2 | Scope/language guardrail (G2) | P | — | 0.010ms | 0.028ms | |
| 3 | Query embed + dense search (retrieve) | R | 8ms + 3ms | 5.1ms | 8.9ms | Dense-only (A3 winner, R-010) — no sparse/rerank on the shipped path |
| 4 | Sparse search (BM25) | R | 5ms | N/A | N/A | **Not on the shipped path** — A3 found dense-only beats hybrid (R-010); this row is architecturally absent, not un-measured |
| 5 | RRF fusion | R | <1ms | N/A | N/A | Same — not on the shipped path |
| 6 | Cross-encoder rerank | R | 25ms | N/A | N/A | **Not on the shipped path** — A4 found `none` wins outright (R-012); architecturally absent |
| 7 | Grounding gate (G3) | Joint | 1ms | 0.014ms | 0.034ms | |
| 8a | Track A extractive span select | P | 10ms | 0.005ms | 0.008ms | |
| 8b | Track B generation (full completion, not TTFT) | P | 110ms (TTFT) | 1976.5ms | 6429.8ms | See "Track B, standalone" below — this wrapper (`generate()`) doesn't expose first-token timing, only full completion; reported honestly as completion latency, not mislabeled as TTFT |
| 9 | Output guardrail (G5) | Joint | 5ms | 0.055ms | 0.430ms | |
| | **Total to Track A answer (true stage cost)** | | ~30ms | **5.2ms** | 8.9ms | Comfortably under 200ms — the exit criterion IS met, at the stage level |
| | **Total to Track A answer (end-to-end, as currently wired)** | | — | **~213-246ms** | 236-383ms | Exceeds 200ms *only* because of the doomed Track B attempt described above |

## Track B, standalone (bypasses the harness's budget gate entirely)

Measured by calling `generation.sarvam_llm.generate()` directly with real retrieved chunks and a
generous 15s timeout — the number Track B would actually deliver if given a real chance, not what
the 200ms-budget harness measures (which is always "timed out, shed").

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
  this session can fix.

## Response-status mix (500 samples, real query distribution: 60% in-domain / 20% off-topic /
10% unsafe / 10% degenerate, `eval/test_queries.json`)

| Status | Count | % |
|---|---|---|
| answered | 275 | 55% |
| abstained (G3 confidence gate) | 165 | 33% |
| refused (G1/G2) | 60 | 12% |

## Hot-path rules that keep this budget honest

- No network calls on the hot path except the LLM (embeddings, BM25, reranker, guardrails all
  in-process — this is why a hosted vector DB was rejected outright, see `TECH_MENU.md` S6).
- No cold starts in steady state — models load lazily on first real use (`_get_real_retriever()`,
  `_get_real_retriever` singleton pattern), not per-request; the very first request after boot pays
  a real ~5s load cost (`data/index/metadata_aware` + the ONNX embedder session), which is exactly
  why this benchmark's methodology discards a full warm-up pass before timing anything.
- No per-request disk I/O on the hot path — the index is memory-resident (`SQLiteChunkLookup` reads
  are lazy per-chunk, not per-request-bulk; the FAISS/BM25 indexes themselves are fully in memory).

## What's still open

- **`t_e2e_voice` (STT round trip) is not yet measured by this script** — `eval/audio/` has 95 real
  Sarvam-TTS-synthesized 16kHz WAV files ready (`scripts/synthesize_test_audio.py`), matching
  `WS /voice`'s expected input format, but the WebSocket benchmark client itself isn't built yet.
- **`tests/test_latency_regression.py`** (CI gate: Track A stage-cost P50 < 200ms on a 20-query
  smoke subset) — not yet written.
- These numbers are from a local run, not the deployed environment (deployment is deliberately
  parked per `docs/RISKS.md` R4/R-R21) — re-measure once/if a deploy target is chosen, since P-020
  already found local-vs-deployed can differ meaningfully (peak-memory-during-load, not applicable
  to latency directly, but a reminder that "local" and "deployed" are genuinely different
  environments worth re-verifying, not assuming identical).

## Provider RTT probe

Not run as a separate probe — Track B's standalone measurement above (real Sarvam calls, N=30)
supersedes the originally-planned `scripts/probe_latency.py` TCP/TLS/TTFT breakdown for this
project's purposes. ADR-003 (ADR log) was never separately recorded; see `docs/DECISIONS_P.md`
P-012 for the earlier probe data gathered during Track B development.
