# PROGRESS — SHARED

> Edited only at integration syncs (`docs/TEAM_SPLIT.md` §5), by whoever is merging — a hand-written
> combined summary of `PROGRESS_R.md` + `PROGRESS_P.md`. Not a running log; a snapshot.

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
