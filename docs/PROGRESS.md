# PROGRESS — SHARED

> Edited only at integration syncs (`docs/TEAM_SPLIT.md` §5), by whoever is merging — a hand-written
> combined summary of `PROGRESS_R.md` + `PROGRESS_P.md`. Not a running log; a snapshot.

## ⚠️ Operational change, 2026-08-19: single-operator mode

Workstream P's collaborator (Arunish) is out of weekly Claude Code credits — this session (Shiv,
previously Workstream R only) is now covering **both** workstreams solo for the rest of the build.
`.workstream` still reads `R` (unchanged, historical) but ownership boundaries in `docs/TEAM_SPLIT.md`
§2 no longer restrict what this session touches — read `docs/PROGRESS_P.md`/`docs/DECISIONS_P.md`
for what P already completed (a lot — harness, guardrails, generation, telemetry, API, deployment
prep) before assuming something isn't done. Deployment (R4/R-R21, the free-tier RAM problem) is
**deliberately parked** — attempted a Google Cloud Run migration, hit real setup friction (gcloud
auth issues under Git Bash, GCP billing account setup), and the user redirected to **run everything
locally first, decide where to deploy for free later**. Verified 2026-08-19: the full local stack
(real Sarvam STT/LLM key now available, real retrieval, real guardrails) works end-to-end against
`localhost` — confirmed with a real held-out query returning a real cited answer.

**P6 (latency campaign) done, 2026-08-19** — went from 0% to real numbers in one session:
`eval/test_queries.json` (100 real queries), `eval/audio/` (95 real TTS WAVs),
`scripts/bench_latency.py`, `scripts/make_latency_charts.py`, `tests/test_latency_regression.py`,
`docs/LATENCY_BUDGET.md` filled in with real measurements, ADR-003 and ADR-005 recorded in
`docs/DECISIONS.md`. Headline finding: Track A's true stage cost is P50=5.2ms (exit criterion met)
but the client-observed latency for an answered query is P50=213-246ms because `GenerateStage`
always attempts a doomed Track B call under the 200ms budget before shedding it — reported
honestly, no harness change (user's call). Also found and fixed a real bug in Track B's streaming
handler along the way (`docs/DECISIONS_P.md` P-021) that was silently capping its real success rate
at 0% — now 63.3% (19/30 real calls).

**Memory work, 2026-08-19 (R-023 through R-030) — the RAM problem may now be solved.** Deep
component-by-component audits (ADR-006, R-028) found two real, fixable costs: BM25 loading
unconditionally in dense-only mode (~105MB wasted, fixed in ADR-007) and — the big one — the
tokenizer costing more RAM than the ONNX model itself (262MB vs. 137MB, R-028). Verified a
`sentencepiece`-based replacement reproduces 100.0000% exact token IDs on 1,020 real strings
(R-029), implemented it behind `LiteE5Embedder`'s unchanged interface (R-030), and **production
RSS is now 493.8MB steady-state / 492.9MB peak — under Render's 512MB free-tier budget for the
first time**, down from 1,860MB at the start of this work (-73.4%), with retrieval quality
confirmed unchanged (Recall@1/5/10, MRR@10 match the established baseline to full float
precision). **Verified in a real Docker `-m 512m` environment 2026-08-19 (R-031) — and it FAILS.** Reproduced
2/2: container starts clean (~277MiB, real retriever confirmed loaded, not the stub), but the
first real `/ask` query OOM-kills it (exit 137) every time — the same failure P-020 found live on
Render on 2026-08-18, now confirmed locally. The 493.8MB steady-state number was real but was
never the binding constraint; the binding constraint is the memory spike during first real
embedding-inference + FAISS search, which no fix so far has targeted. R4 is **not** resolved — see
`docs/RISKS.md` R4 and `docs/DECISIONS_R.md` R-031 for the full finding. No code changed in
response per explicit instruction (validation-only, stop-and-report on failure).

**Root cause pinned down 2026-08-19 (R-032), isolated diagnostic probe, directly confirmed against
the real 512MB limit.** Not ONNX inference, not FAISS search, not a leak — all under 2MB each,
every query. It's FAISS+SQLite load (~298MB) plus `ort.InferenceSession`+`SentencePieceProcessor`
construction (~205-218MB, confirmed independent of FAISS) simply stacking to ~503-543MB, never
before measured resident together in one process. A `-m 512m` re-run of the probe dies exactly
between those two steps, nothing else. See `docs/DECISIONS_R.md` R-032 for the full ranked-causes
breakdown. No fix applied — diagnostic only, per instruction.

**Real lever found 2026-08-19 (R-033), offline FAISS index-variant ablation.** `IndexHNSWSQ` +
`ScalarQuantizer.QT_fp16` saves 77.0MB off FAISS's own footprint at zero measured quality change
(Recall@1/5/10, MRR@10 all identical to the rebuilt baseline) — clears the ~60MB target R-032's
gap analysis implied, at no quality cost. An int8 alternative saves more (115MB) but at a real
quality cost, not recommended since fp16 already clears the bar for free. Offline finding only —
not wired in, no new release asset, not deployed. See `docs/DECISIONS_R.md` R-033 for the full
tradeoff table; a real Docker re-verification is the honest next step, not yet done.

**R4 RESOLVED 2026-08-19 (R-034).** Implemented `quantization="sqfp16"` in
`src/vrag/index/dense.py` (opt-in, default unchanged), rebuilt as `index-metadata_aware-v3`
(byte-identical to v2 except `dense/faiss.index`, -76.6MB), updated `Dockerfile`'s one download
line. Real `docker build` + `docker run -m 512m --memory-swap 512m`: startup 204.4MiB, first real
query **survived** at peak 394.2MiB (previously OOM-killed 2/2 at this exact stage, R-031/R-032),
10 real queries survived at peak 397.8MiB, `OOMKilled: false`, real citations confirmed, ~114MB
headroom. Recall@10=0.748/MRR@10=0.45550 vs. the 0.750/0.45627 baseline — within the established
HNSW-rebuild noise floor, not a regression. Query latency 12-42ms per real query, well under the
200ms budget. One open, non-blocking observation flagged for G3's owner: 3/10 real queries
abstained on the confidence gate, possibly fp16 score-precision sensitivity near the calibrated
threshold — not confirmed as a regression. Full record: `docs/DECISIONS_R.md` R-034,
`docs/RISKS.md` R4.

**Both R-034 open observations closed 2026-08-19 (R-035).** Cold start: `is_retrieval_real()`
now runs a real warmup embedding at startup before the app accepts any request — first-query
`retrieve` dropped from 1125.8ms to 42.2ms in a real Docker re-verification (indistinguishable
from steady-state), and `/healthz` is provably unreachable until warmup completes. Grounding
threshold: real 500-query FP32-vs-FP16 comparison via the real unmodified G3 gate found zero
decision changes (371 answered/129 abstained, identical both ways) — the earlier 3/10 abstention
observation was sampling noise, not a real fp16 effect. TAU/MARGIN left untouched, no retuning
needed. Full record: `docs/DECISIONS_R.md` R-035.

Current priority: deployment remains parked by the user's direction, but now has real, favorable
numbers to work with whenever it's picked back up (`docs/RISKS.md` R4). Next up otherwise:
`t_e2e_voice` (the WebSocket voice benchmark — audio assets are ready, the client isn't built yet)
or picking up whatever's next in `docs/BUILD_PLAN.md` P7 (README, frontend polish, manual QA).

**Last updated:** 2026-08-17 (Day 1 sync — merging `workstream-p` into `main`, then `main` into `workstream-r`)
**Current phase:** P0 wrapping up / P1 underway (P is ahead on the walking skeleton; R is ahead on chunking/retrieval code)
**Days remaining:** 5 (deadline 2026-08-22 23:59 IST)
**Build status:** 🟢 both tracks green independently; first cross-branch merge just happened

## Where we are, in one paragraph

Day 1. Workstream P has a real, **live, publicly deployed** walking skeleton:
`https://vrag-voice.onrender.com` verified end to end with genuine Sarvam STT (via Sarvam's own TTS
to generate real Hindi audio, since no physical mic was available), a FastAPI app serving `/ask` and
`WS /voice`, and a frontend wired to real mic capture — all running against the Day-1 `retrieve()`
stub (P's tests: 7/7 passing). Workstream R has all P0 exit criteria met (10,000-query working
subset + 500 frozen held-out pairs built from the real corpus, real dataset stats and a translation
quality spot-check recorded) plus P2/P3 code written and unit-tested ahead of schedule: all 6
chunking strategies, dense/sparse/RRF-fusion index primitives, the E5 embedder, a real concurrent
`HybridRetriever`, a reranker scaffold, and retrieval metrics (89/89 tests passing on R's side before
this merge). Neither track had run the chunking ablation against real data yet as of this sync.
`retrieve()` is still P's Day-1 stub on both branches — R's `HybridRetriever` exists but isn't wired
in, pending the A1 ablation results.

## Phase exit criteria — P0

- [ ] Probe results committed; provider chosen with evidence, recorded as ADR-003 — **blocked**, no API keys on either machine yet
- [x] `t_pipeline` definition agreed and recorded as ADR-004 — confirmed at this sync
- [x] Dataset subset on disk; passage-length distribution known; chunk-count estimate written down (R)
- [x] 500 held-out pairs frozen and committed — `eval/heldout_queries.json` (R)
- [x] `pytest` green, `ruff` clean on both tracks independently

## What works right now (verified, not assumed)

- Live public HTTPS deploy on Render, verified with real Sarvam STT + real Hindi audio round trip (P)
- Real FastAPI app + WebSocket voice endpoint + frontend mic capture (P)
- `retrieve()` stub — both tracks built against the identical joint contract, confirmed compatible
  at merge time (only cosmetic differences: `Field(ge=0,le=1)` validation, empty-query guard)
- All 6 chunking strategies + dense/sparse/fusion + embedder + `HybridRetriever`, unit-tested (R)
- Real 10,000-query working corpus + 500-pair held-out eval set, built from the actual downloaded
  MSMARCO-XI Hindi file (R)

## What is stubbed / faked / TODO

- `retrieve()` is still the Day-1 stub in the merged code — R's real `HybridRetriever` swap is next
- No harness orchestration (deadline propagation, retries, guardrails) wired into the live request
  path yet — `Stage`/`PipelineContext`/`Budget` exist as a shape but `/ask` and `/voice` call
  `retrieve()` directly (P, explicitly Day 2 scope)
- No chunking strategy has real Recall@k/MRR/nDCG numbers yet — `docs/EVAL_RESULTS.md` §1 is empty
- G1–G5 guardrails not started

## Live numbers

| Metric | Value | Measured on |
|--------|-------|-------------|
| p50 t_pipeline | — | — |
| Recall@5 (prod strategy) | — | — |
| Live golden-path round trip (STT→answer) | ~2.4s | 2026-08-17, Render deploy (P-007) |

## Blockers

- No Sarvam/Groq API keys on either dev machine — blocks `scripts/probe_latency.py` and ADR-003. Owner: user.
- Non-blocking: Render `/voice` WebSocket lingers ~20-25s before a clean close (doesn't affect answer
  delivery) — tracked as P-R12 in `docs/RISKS.md`.

## Next session should start by

1. R: run `scripts/eval_chunking.py` across all 6 strategies against the real held-out set, write
   `docs/EVAL_RESULTS.md` §1, promote a winner, wire `HybridRetriever` into `retrieve()`.
2. P: open the live URL on an actual phone over mobile data and click through the real mic UI (the
   one verification gap left from Day 1); begin Day 2 harness hardening.
3. Whoever gets Sarvam/Groq API keys first: run `scripts/probe_latency.py`, record ADR-003.
