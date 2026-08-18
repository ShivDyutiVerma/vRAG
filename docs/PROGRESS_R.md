# PROGRESS_R — Workstream R's running status

> Mine, edited freely, any time. Never edited by Workstream P.

**Last updated:** 2026-08-18, Session 02
**Current phase:** P3 exit criteria fully met — full staged ablation (A1-A4) + efSearch curve complete, production wiring applied
**Build status:** 🟢 green — 150/150 tests pass, ruff/mypy clean

## Where I am, in one paragraph

All four planned ablation stages plus the efSearch sweep are done and documented with real measured
numbers, not assumptions: A1 chunking (`metadata_aware` wins, tied with `passage_native`/
`fixed_overlap`, noise-floor validated), A2 embedder (`multilingual-e5-small` wins decisively), A3
retrieval mode (**dense-only wins — a genuine, verified surprise**: hybrid+RRF actually regresses
quality on this corpus because BM25 is comparatively weak on machine-translated Hindi text and naive
RRF has no way to discount it), A4 reranking (**`none` wins outright** — both FlashRank and a
cross-encoder were measured to actively destroy quality on Hindi text, verified via query-level
diagnostics as genuine model/language limitations, not bugs), and the efSearch sweep (**64 confirmed
as the knee of the curve** — same value already hardcoded pre-sweep, now backed by data instead of a
placeholder comment). Full tables, chart, and analysis: `docs/EVAL_RESULTS.md` §1-3. Full ADR trail:
`docs/DECISIONS_R.md` R-001 through R-014.

Production wiring matches the measured winners, not the originally-assumed architecture: after
confirming the deviation with the user (it contradicts `AGENT_BUILD_SPEC.md`'s assumed Phase-3 exit
criterion and CLAUDE.md's original hybrid-retrieval hot-path invariant, both now updated to reflect
this), `HybridRetriever` defaults to `retrieval_mode="dense"` and skips BM25 on the hot path
entirely; reranking stays off by default (already was). A real, verified side effect worth watching:
this also fixed `RetrievedChunk.score`'s scale to match what G3's `TAU` placeholder was actually
calibrated for (flagged for Workstream P in `docs/RISKS.md` P-R18, not touched since G3 is their
module). Also fixed one flaky test in P's file (`tests/test_api.py`) with explicit user
authorization to cross the ownership boundary — documented as R-013.

## Phase exit criteria I'm targeting (P3, my slice) — all met

- [x] Dataset subset + 500 held-out pairs frozen and committed
- [x] `retrieve()` wired to a real, persisted index (`data/index/metadata_aware/`)
- [x] A1 chunking ablation — winner picked, noise-floor validated
- [x] A2 embedder ablation — winner picked, decisively
- [x] A3 retrieval-mode ablation — winner picked (dense-only, a real surprise), wired into production
- [x] A4 reranker ablation — winner picked (none), already the default
- [x] efSearch recall-vs-latency curve (`docs/assets/efsearch_curve.png`) — 64 confirmed as the knee
- [x] `pytest` green (150/150)

## What works right now (verified, not assumed)

- Full ablation trail A1→A4 + efSearch, every run backed by a `eval/ablation_ledger.csv` row, every
  winner backed by query-level diagnostics where the result was surprising enough to warrant one
  (A3, A4) ✅
- `retrieve()` loads the real persisted index and returns real, measured-good results (dense-only,
  Recall@5=0.652 on the frozen 500-query held-out set, efSearch=64) ✅
- `HybridRetriever` supports dense/sparse/hybrid modes, all three unit-tested including the
  "unused modes never call the index they don't need" guarantee ✅
- Shared `score_hits`/`dedupe_doc_ids` in `src/vrag/retrieval/metrics.py` — the R-006 dedup fix is
  now a single tested fix point every eval script (A1-A4, efSearch) routes through ✅
- `CrossEncoderReranker`/`FlashRankReranker` both implemented, tested for wiring correctness (not
  quality — their quality verdict is "actively harmful on Hindi," documented in R-012) ✅
- `DenseIndex.set_ef_search()` — mutates HNSW search-time behavior without a rebuild, unit-tested ✅

## What is stubbed / faked / TODO

- `fixed_overlap`'s hyperparameter sweep (overlap ∈ {0, 0.1, 0.2}) — low priority, tied with the
  winner already
- A candidate RRF mitigation (larger per-lane candidate pool before fusion) — logged as an idea in
  `docs/RISKS.md` R-R14, not tested; wouldn't change the shipped default without a fresh ablation run
- A genuinely Hindi-capable reranker (e.g. `bge-reranker-v2-m3`) was never tried — A4 only tested the
  three TECH_MENU-named candidates, all of which failed for either English-only training or model
  saturation on Hindi

## Blockers

- None currently blocking R's work. `docs/RISKS.md` P-R18 is a flag *for* Workstream P (G3 score
  scale), not a blocker on my side.

## Next session should start by

1. Read `docs/EVAL_RESULTS.md` §1-3 and `docs/DECISIONS_R.md` R-010/R-012/R-014 for the full
   A3/A4/efSearch story before touching retrieval code again — the production default is now
   dense-only, no reranker, efSearch=64, and all three are deliberate, data-driven, documented
   choices, not oversights
2. P3's exit criteria (my slice) are all met — next is likely whatever `docs/BUILD_PLAN.md` names for
   Phase 6+ (ONNX quantisation, real latency benchmarking) or coordinating with Workstream P on the
   G3 calibration flagged in `docs/RISKS.md` P-R18
3. Check `docs/BUILD_PLAN.md`'s later phases for R's remaining ownership before assuming there's
   nothing left — this session focused entirely on closing out P3, not on scoping what's next
