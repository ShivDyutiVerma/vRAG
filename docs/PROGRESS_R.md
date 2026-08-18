# PROGRESS_R — Workstream R's running status

> Mine, edited freely, any time. Never edited by Workstream P.

**Last updated:** 2026-08-18, Session 02
**Current phase:** P3 exit criteria fully met (ahead of schedule); G3 calibration complete and applied jointly with P (TAU+MARGIN both live and verified)
**Build status:** 🟢 green — 151/151 tests pass, ruff/mypy clean

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
calibrated for. Also fixed one flaky test in P's file (`tests/test_api.py`) with explicit user
authorization to cross the ownership boundary — documented as R-013.

**Ran ahead on the schedule** (`docs/TEAM_SPLIT.md` §5 spreads A1-A4 across Days 1-3; all done here
on Day 1) into the next R-scoped item — supporting G3 calibration (`docs/TEAM_SPLIT.md` §5 names
this R's Day-3 task, joint with P). Built `eval/calibration.json` (150 in-domain + 150 genuinely
out-of-index queries, no LLM call needed — reused the already-cached corpus parquet) and ran the
full `TAU` sweep against the real production index. **Finding: `docs/EVAL_PROTOCOL.md`'s calibration
targets (false-refusal<10% AND correct-refusal>80%) are not simultaneously reachable** on this
corpus via pure cosine-similarity gating — verified root cause: MSMARCO-XI passages recur across
different query_ids, so genuinely out-of-index queries often still retrieve a topically-close or
even coincidentally-correct match. Full curve and reference-point table: `docs/DECISIONS_R.md`
R-015, chart at `docs/assets/g3_calibration.png`. Deliberately did **not** apply a chosen TAU/MARGIN
myself — the pick is a real product tradeoff, and `docs/TEAM_SPLIT.md` §5 reserves it as a joint
Day-3/4 decision.

**Partner applied `TAU=0.8835` shortly after** (`docs/DECISIONS_P.md` P-015) — the balanced point
from my curve. Their own commit flagged one remaining gap: `MARGIN=0.05` (the pre-calibration
placeholder) hadn't been re-swept at the new `TAU`. Completed that flagged step same day: verified
live that `MARGIN=0.05` at this `TAU` actually caused **88.0%** false-refusal, not the 19.3% the
commit's design target stated — in-domain top1-vs-top5 gaps are naturally tiny at this operating
point, so even `MARGIN=0.01` alone degrades to 28.7%. Set `MARGIN=0.0` (empirically correct at this
`TAU`, not a shortcut — confirmed live, 2/3 previously-blocked test queries now answer correctly).
Full writeup: `docs/DECISIONS_R.md` R-017. This wasn't a new judgment call on my part — it's the same
calibration data and analysis that already produced `TAU=0.8835`, applied to keep the file
internally consistent with its own stated target.

## Phase exit criteria I'm targeting (P3, my slice) — all met

- [x] Dataset subset + 500 held-out pairs frozen and committed
- [x] `retrieve()` wired to a real, persisted index (`data/index/metadata_aware/`)
- [x] A1 chunking ablation — winner picked, noise-floor validated
- [x] A2 embedder ablation — winner picked, decisively
- [x] A3 retrieval-mode ablation — winner picked (dense-only, a real surprise), wired into production
- [x] A4 reranker ablation — winner picked (none), already the default
- [x] efSearch recall-vs-latency curve (`docs/assets/efsearch_curve.png`) — 64 confirmed as the knee
- [x] `pytest` green (151/151)
- [x] G3 calibration — data gathered (R-015), TAU applied by P, MARGIN corrected same day (R-017),
  both verified live

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
- `fixed_overlap`'s overlap ∈ {0, 0.1, 0.2} swept — confirmed no measurable effect on this corpus
  (R-016), closing the last open item from A1's methodology ✅

## What is stubbed / faked / TODO

- A candidate RRF mitigation (larger per-lane candidate pool before fusion) — logged as an idea in
  `docs/RISKS.md` R-R14, not tested; wouldn't change the shipped default without a fresh ablation run
- A genuinely Hindi-capable reranker (e.g. `bge-reranker-v2-m3`) was never tried — A4 only tested the
  three TECH_MENU-named candidates, all of which failed for either English-only training or model
  saturation on Hindi
- `docs/BUILD_PLAN.md` P3 task 10 ("optional but high-value": Sarvam `transcribe` vs `translit`
  output modes for retrieval quality) — genuinely blocked, not skipped by choice: no `.env`/Sarvam
  key on this machine (`docs/RISKS.md`'s Day 0-1 blockers note). Considered a transliteration-library
  proxy instead of real Sarvam output; rejected — a generic library's romanization scheme has no
  guarantee of matching Sarvam's actual `translit` convention, and reporting a finding based on a
  possibly-wrong proxy would risk a misleading result, not a real one. Needs either R's own Sarvam
  key or running this test from Workstream P's machine.
- A1×A2 confirmation pass — reasoned skip, not run; see `docs/EVAL_RESULTS.md` §2 for why (no genuine
  embedder-side question exists once A2's 38-point gap is accounted for)

## Blockers

- None. G3's calibration (`docs/RISKS.md` R-R19) is fully resolved — TAU and MARGIN both applied
  and verified live.

## Next session should start by

1. Read `docs/EVAL_RESULTS.md` §1-3 and `docs/DECISIONS_R.md` R-010/R-012/R-014 for the full
   A3/A4/efSearch story before touching retrieval code again — the production default is now
   dense-only, no reranker, efSearch=64, and all three are deliberate, data-driven, documented
   choices, not oversights
2. Read `docs/DECISIONS_R.md` R-015/R-017 for the G3 calibration story before touching TAU/MARGIN
   again — both are now live-verified, not placeholders
3. R is running ~2 days ahead of `docs/TEAM_SPLIT.md` §5's schedule (all of A1-A4 + efSearch + G3
   calibration, originally spread across Days 1-3, done on Day 1). Check with the user or
   `docs/BUILD_PLAN.md`'s later phases (P6 ONNX quantisation is R's file, `src/vrag/index/
   embedder.py`) before assuming there's nothing left, but don't pull Day-4-scoped latency work
   forward on its own — `docs/TEAM_SPLIT.md` explicitly warns against that
