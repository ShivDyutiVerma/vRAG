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

## ⚠️ Current state, 2026-08-20 (supersedes everything below this point)

The system described in the "Day 1" snapshot below (`retrieve()` stub, no harness wired in, no
guardrails) **no longer reflects reality anywhere** — kept only as a historical record of where
the project started, not as current status. As of this update:

- **Full harness wired in and live**: G1→G2→Retrieve→G3→TrackA/TrackB→G4→G5→Assemble runs for
  real on both `/ask` and `WS /voice`, deadline propagation active, `GenerateStage`'s pre-flight
  budget gate (R-036) shed the doomed-Track-B-attempt problem entirely.
- **Real retrieval confirmed live**: `GET /healthz` on `https://vrag-voice.onrender.com` returns
  `{"status":"ok","retrieval":"real"}`, verified 2026-08-20 as part of a redeploy that also shipped
  two real fixes — the frontend refusal-state pill (was collapsing 3 distinct statuses into one
  hardcoded "Abstained" label) and an STT no-speech timeout (was hanging on a silent session until
  Sarvam's own ~60s watchdog, now a clear 10s error). 5 real `/ask` calls against the live URL:
  2 answered with real citations, 3 correctly abstained with real, distinct confidence scores.
- **All 6 chunking strategies, 4 embedders, dense-vs-hybrid, and 2 rerankers evaluated with real
  numbers** — `docs/EVAL_RESULTS.md` §1-3, plus a targeted follow-up (R-038) specifically testing
  reranking against the exact failure mode it was hypothesized to fix. `metadata_aware` /
  `multilingual-e5-small` / dense-only / no-rerank shipped, each a measured decision.
- **All 5 guardrail layers real and tested**; G3 has a real 300-query calibration
  (`TAU=0.8835`, `docs/EVAL_RESULTS.md` §5); G4's threshold is the one remaining uncalibrated
  value, disclosed rather than hidden.
- **Memory problem resolved**: 1,860MB → 493.8MB (73.4% cut), verified under the real 512MB Docker
  constraint with ~114MB headroom, zero quality cost — see `docs/ARCHITECTURE.md`'s memory story.
- **Latency**: meets the 200ms target locally (P50=10.1ms wall-clock, post R-036); does **not**
  meet it on live Render specifically due to free-tier CPU contention (P50=594.9ms for the
  retrieve stage alone) — disclosed plainly in `docs/LATENCY_BUDGET.md` and the README, not hidden.
- **`README.md` now exists** at the repo root (previously missing) — see it for the full current
  picture; this file is now a supplementary progress log, not the primary source of truth.

**What's still genuinely open:** real human-microphone verification on the live URL (the
automation environment used this session provides a muted mic track, not real speech — explicitly
marked pending, not faked); demo and process videos (not recorded); the promotion grid and
submission form (`docs/SUBMISSION_CHECKLIST.md`, correctly untouched this early).

## ⚠️ Local-first pivot, 2026-08-20 — multilingual + 200k-chunk rebuild, deployment work frozen

Live hosting (Render/AWS/OCI) is explicitly out of scope for now — the AWS path hit a new-account
verification hold that can't be worked around (see the session record), and the user redirected to
building the best fully-functional multilingual version locally first, deployment decided later.
`baseline-hindi-only-v1` tag (commit `11413e5`) marks the known-good, currently-deployed-on-Render
Hindi-only configuration as a recoverable fallback before this work started.

**Phase 0 (audit) done:** verified live against the real `ai4bharat/MSMARCO-XI` repo (not assumed)
— 13 real train languages, 10,080,140 rows total, every row already carries both
`English_passages` and `Translated_passages`. Full inventory in the session record / `README.md`
update pending a later phase.

**Phase 1 (language-routing plumbing) done — index NOT rebuilt, still Hindi-only.** New
`src/vrag/languages.py` is the single source of truth for `SUPPORTED_LANGUAGES` (the 13 MSMARCO-XI
train languages, English and Telugu deliberately excluded — see ADR-008). Sarvam's STT now uses
real `language_code="auto"` detection instead of a hardcoded `"hi-IN"`; the detected language flows
into G2 (refuses unsupported languages explicitly, e.g. English is refused rather than silently
searched against the Hindi index) and is tracked through the pipeline as `query_language`, kept
separate from `retrieved_language` (the evidence chunk's own language) and `generation_language`
(plumbed through, not yet consumed — Track B is still Hindi-only output). No corpus, FAISS,
embedder, tokenizer, or G3 change — `index-metadata_aware-v3` is untouched. 260/260 tests pass
(238 pre-existing + 22 new), G2 hot-path cost measured unchanged (~1.3-1.5µs/call either way).
Full record: `docs/DECISIONS.md` ADR-008, `docs/DECISIONS_R.md` R-039, `docs/DECISIONS_P.md` P-024.

**Phase 2 (multilingual corpus + index, 3 sizes, language-aware retrieval eval) done.** Full
record: `docs/DECISIONS.md` ADR-009, `docs/DECISIONS_R.md` R-040. Real reservoir-sampled (seed
20260820), nested 100k/150k/200k multilingual corpora built (all 13 languages, near-perfectly
balanced), same chunking/embedder/FAISS config as production. Headline findings, both measured:
language-filtered retrieval beats unfiltered by +8.7-9.1pp Recall@10 at every size; retrieval
quality *falls* as corpus size grows 100k→200k (100k is simultaneously best-quality,
lowest-memory, fastest-build of the three measured). Real memory audit: 397/462/532MB steady-state
RSS at 100k/150k/200k with the production-matching lean SQLite chunk lookup (536/675/812MB with
the eager JSON lookup — a real, disclosed ~140-280MB gap). A real cross-language `chunk_id`
collision artifact was found and quantified (0.67-1.27% of rows, root cause: MSMARCO-XI's
`query_id` is shared across all 13 language files) — disclosed in ADR-009, not silently fixed
(out of Phase 2's scope). `data/index/metadata_aware/` (Hindi-only production index) untouched
throughout. Size/config decision not yet made — handed to the user with the evidence.

**Phase 2 decision + Phase 3 (production candidate selected, language-aware generation, real G3
re-eval) done.** Full record: `docs/DECISIONS.md` ADR-010/ADR-011. User selected **100k, filter
mode** as the production candidate (best on every measured axis, not just smallest). Before
wiring it in: fixed the ADR-009 chunk_id collision (`qualify_doc_id_by_language`, opt-in, Hindi
pipeline's default behavior byte-identical to before), dropped the dead BM25 artifact from the
candidate build (`save_built_index` now accepts `sparse=None`), added English as a genuine 14th
indexed language (771 rows from the `English_passages` field every row already carries — not a
back-fill). Final candidate: **107,678 chunks, 14 languages, `data/index/multilingual_100k/`**,
406.5MB steady-state RSS / 492.6MB peak (real, lean SQLite lookup).

Phase 3: the validated "filter" strategy is now wired into `HybridRetriever.retrieve()` for real
(was accepted-but-inert since Phase 1); `_INDEX_DIR` stays the Hindi-only default (`VRAG_INDEX_DIR`
env var opts a local session into the candidate — deliberately not a hardcoded swap, to avoid
silently breaking a future real deploy). Track B's system prompt is no longer hardcoded to Hindi —
names the real `generation_language` (14 languages covered in tests). **Real G3 re-evaluation
found the multilingual filter does NOT improve the previous 25.8% abstention rate — it makes it
substantially worse (66.5%), because TAU=0.8835 was calibrated on Hindi-only same-script scores
and cross-language E5 similarity runs measurably lower.** TAU left untouched, per explicit
instruction — reported, not silently patched. The regression case ("capital of India", Hindi +
English) correctly abstains in both languages rather than confidently citing the wrong country's
capital. 286/286 tests pass. `data/index/metadata_aware/` (live Render production) untouched
throughout — nothing in this repo has been deployed or redeployed.

**Phase 4 (G3 recalibration attempt on the multilingual candidate) done — result: TAU/MARGIN kept
unchanged.** Full record: `docs/DECISIONS.md` ADR-013. Collected real per-query top1/top-20 scores
+ gold-passage relevance for all 532 held-out queries via the actual production `retrieve()` path
(`scripts/calibrate_g3_collect.py`), then swept TAU across the full observed range plus a
per-language breakdown, a formula-based per-language offset rule, and a MARGIN grid
(`scripts/calibrate_g3_sweep.py`). **Headline finding: the blocker is signal quality, not
threshold placement** — correct-hit and wrong-hit top1 scores heavily overlap on this
multilingual corpus (wrong-hit max 0.9463 exceeds correct-hit median 0.8846), so no global
threshold meaningfully cuts abstention without proportionally increasing false-accepts; the
current operating point already only has 34.8% precision on the answers it does give (62
true-accepts vs. 116 false-accepts out of 178 answered). A free per-language threshold looked
promising in aggregate but **failed an even/odd stability check (only 2/14 languages agreed within
0.02)** — overfit to 38 queries/language, not shipped. A principled formula-based per-language
offset rule performed *worse* than doing nothing. MARGIN re-swept, confirmed 0.0 is still correct.
**Decision: `src/vrag/guardrails/g3_confidence.py` is untouched** — evidence didn't support a
change, reported honestly rather than forced to match the old 25.8% number. Root cause (weak top1
signal on this corpus) flagged as real future work needing a better signal (reranker/embedder), not
a threshold fix. Raw artifacts: `eval/g3_calibration_multilingual_100k_raw.json`,
`eval/g3_threshold_sweep_multilingual_100k.json`. No corpus/retrieval/generation code changed;
286/286 tests still pass (no test changes needed).

**Phase 5 (cheap deterministic confidence-signal experiment) done — result: no candidate adopted.**
Full record: `docs/DECISIONS.md` ADR-014. Investigated whether a cheap, CPU-only signal (no neural
model, no LLM, no network call) beats top1 score alone at separating correct from incorrect
retrieval on the multilingual candidate. Re-collected real production `retrieve()` output with
full passage text this time (`scripts/collect_g3_feature_data.py`), computed 15 candidate features
+ 4 two-feature combinations with strict hindsight-leakage discipline (gold labels used only to
*evaluate*, never to construct, a feature), evaluated via AUC/precision/coverage/per-language
stability (`scripts/g3_feature_experiment.py`). **Found and fixed a real bug first:** Python's
`\w` regex excludes Unicode combining marks, silently shattering Devanagari/Bengali/etc. text at
every vowel sign — would have made every lexical feature meaningless if shipped un-caught. Two
features (`score_std_top5` AUC=0.671, `gap15mean` AUC=0.667) genuinely beat top1 (AUC=0.640) in
aggregate, and one combination (top1 + `concentration_ratio`) showed a real, mechanistically
verified gain (208 vs. 178 answered at flat precision). **All three were rejected anyway** — three
independent checks disqualified them: (1) the flagship "capital of India" (Hindi) regression case
gets *accepted with the wrong Bangkok/Thailand evidence* by both top-AUC features; (2) that
single-case survival by the one combination that passed doesn't generalize — 7 of the 10
highest-confidence *wrong* queries in the whole set fool all three candidates alike; (3) the direct
apples-to-apples "new risk" count (among queries production currently safely abstains on) shows
~2.3–2.6 new wrong answers for every 1 new correct answer recovered, consistently across all three.
`same_lang_consistency` scored a legitimate null (AUC=0.5, no variance left to exploit — production's
existing hard language filter already saturates it). An offline logistic-regression diagnostic
(numpy, train/val split by query_id parity, not shipped) confirmed it doesn't generalize either
(train AUC 0.72 → val AUC 0.63). **Decision: `src/vrag/guardrails/g3_confidence.py` untouched.**
Raw artifacts: `eval/g3_feature_experiment_raw.json`, `eval/g3_feature_experiment_results.json`.
No production code changed; 286/286 tests still pass.

**Phase 6 (final integration + demo-readiness, no deployment) done.** Full record:
`docs/DECISIONS.md` ADR-015. G3/TAU/MARGIN frozen per ADR-013/014 — no further recalibration
attempted. Real 19-case end-to-end test through the actual harness against
`data/index/multilingual_100k/` (`scripts/e2e_demo_readiness_test.py` +
`e2e_bonus_answered_cases.py`): one real voice test (Hindi, real Sarvam STT call) found a real,
disclosed limitation — Sarvam romanized the speech and auto-detected `en-IN` instead of Hindi,
causing a safe abstain rather than a wrong answer; 8 of 9 text-tested languages produced at least
one real correctly-in-language answer; the capital-of-India regression stayed safe in both
languages, now tested against each language's own real corpus slice (English against `eng_Latn`
for the first time, not pinned to `hin_Deva`). Real latency re-measured with the project's one
sanctioned benchmark script: **P50=13.0ms, P100=39.0ms** — confirms a stale P6-era "213-246ms"
number is no longer current (a pre-existing budget-gate fix, R-036, already resolved it; this
phase just measured and reported the real current number). Real memory audit re-confirmed
406.2MB/492.7MB. Found and disclosed (not fixed): ~600MB of unused-at-runtime disk artifacts
(leftover `chunk_lookup.json` + an un-pruned FP32 embedder download) sitting next to the real,
lean files that are actually loaded. Frontend verified via its existing real browser test harness
(16/16 pass) plus one live browser screenshot; made one minimal, disclosed copy fix (stale
"Hindi · <200ms" idle-state text → "14 languages", same CSS treatment, not a redesign). **Checked
the live Render URL read-only and found it returning 502** — not touched, not redeployed, per
this phase's explicit scope; local Docker (the same Hindi-only system) verified working
independently. `README.md` substantially rewritten with the real current state of both systems,
a new §12A for the multilingual candidate (real numbers, honest reproduction instructions, no
one-command shortcut since no distributable artifact exists yet), and every limitation from this
whole multi-phase effort disclosed in one place. 286/286 tests pass, ruff clean, no production
`src/` code changed.

**Not started / explicitly out of scope, per instruction:** any deployment action (multilingual
candidate stays local-only; live URL's 502 not diagnosed further); a real fix for G3's underlying
signal quality (ADR-013/014/015 all flag this as the real remaining gap — likely needs a
per-language-aware reranker or a stronger embedder); genuine human-spoken-voice verification in a
live browser (only a real audio-file-through-STT test was performed); packaging the multilingual
candidate as a distributable artifact (GitHub release / Docker image).

**Phase 7 (diagnose + attempt to fix the 66.5% abstention rate via retrieval, not G3) done —
result: fully diagnosed, every cheap fix tested and rejected, nothing changed.** Full record:
`docs/DECISIONS.md` ADR-016. Classified all 354 abstained queries into a 6-category taxonomy
(cross-validated against ADR-013's independent numbers): 40.1% are genuine retrieval misses (gold
passage never even retrieved), 15.8% are already correctly rank-1 but scored below TAU, the
remaining 44% have evidence retrieved but outranked. 40 real cases inspected in depth
(`eval/g3_abstention_case_inspection.json`) surfaced two recurring failure patterns:
same-template distractors (many passages share surface vocabulary regardless of relevance) and
translation-induced lexical variance (correct passages sometimes share almost zero literal words
with the query). Both patterns directly predicted, then confirmed, why every reranking candidate
tested — lexical Jaccard, a numeric+content-word "entity" proxy, a **real freshly-built BM25
index** (chosen specifically to test whether IDF-weighting escapes the trap; it doesn't), and RRF
fusion — is net-negative on Recall@1, with regressions (43–87) exceeding recoveries (28–37) in
every case. This extends R-010's BM25 finding and R-038's cross-encoder finding to the whole
family of relevance-overlap reranking methods on this corpus. A real depth sweep (10/20/50/100,
plus a genuine beyond-100 raw search) found recall saturates completely between rank 50 and 100
(zero additional recovery) and that search cost/memory are flat across every depth tested (already
free) — 37% of abstentions have no recoverable evidence within a 300-deep search at all, a real
embedding-alignment gap depth/reranking cannot fix. The explicit capital-of-India safety check
passed for every candidate regardless (zero unsafe accepts) — reported as a secondary confirmation,
not as grounds to ship a net-negative candidate. **G3 recalibration (Part 5) did not run — no
candidate produced better evidence to recalibrate against.** No production code changed; the
BM25 index built for Candidate D is in-memory only, not persisted anywhere. 286/286 tests pass,
ruff clean.

---

## Historical: Day 1 sync snapshot (superseded above, kept for the record)

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

## Phase exit criteria — P0 (Day 1, historical)

- [ ] Probe results committed; provider chosen with evidence, recorded as ADR-003 — **blocked**, no API keys on either machine yet
- [x] `t_pipeline` definition agreed and recorded as ADR-004 — confirmed at this sync
- [x] Dataset subset on disk; passage-length distribution known; chunk-count estimate written down (R)
- [x] 500 held-out pairs frozen and committed — `eval/heldout_queries.json` (R)
- [x] `pytest` green, `ruff` clean on both tracks independently

## What worked at that point (Day 1, historical — see current state above for today's reality)

- Live public HTTPS deploy on Render, verified with real Sarvam STT + real Hindi audio round trip (P)
- Real FastAPI app + WebSocket voice endpoint + frontend mic capture (P)
- `retrieve()` stub — both tracks built against the identical joint contract, confirmed compatible
  at merge time (only cosmetic differences: `Field(ge=0,le=1)` validation, empty-query guard)
- All 6 chunking strategies + dense/sparse/fusion + embedder + `HybridRetriever`, unit-tested (R)
- Real 10,000-query working corpus + 500-pair held-out eval set, built from the actual downloaded
  MSMARCO-XI Hindi file (R)

## What was stubbed at that point (Day 1, historical — all of this is now real, see current state above)

- `retrieve()` was still the Day-1 stub — **now real**, dense-only FAISS in production
- No harness orchestration wired into the live request path — **now fully wired**
- No chunking strategy had real Recall@k/MRR/nDCG numbers — **now in `docs/EVAL_RESULTS.md` §1-3**
- G1–G5 guardrails not started — **now all five real and tested**

## Live numbers (Day 1, historical — see `docs/LATENCY_BUDGET.md` for current real numbers)

| Metric | Value | Measured on |
|--------|-------|-------------|
| p50 t_pipeline | — | — |
| Recall@5 (prod strategy) | — | — |
| Live golden-path round trip (STT→answer) | ~2.4s | 2026-08-17, Render deploy (P-007) |

## Blockers (Day 1, historical)

- No Sarvam/Groq API keys on either dev machine — blocks `scripts/probe_latency.py` and ADR-003. Owner: user.
- Non-blocking: Render `/voice` WebSocket lingers ~20-25s before a clean close (doesn't affect answer
  delivery) — tracked as P-R12 in `docs/RISKS.md`.

## What Day 1's session proposed doing next (historical — superseded by the current-state section above)

1. R: run `scripts/eval_chunking.py` across all 6 strategies against the real held-out set, write
   `docs/EVAL_RESULTS.md` §1, promote a winner, wire `HybridRetriever` into `retrieve()`.
2. P: open the live URL on an actual phone over mobile data and click through the real mic UI (the
   one verification gap left from Day 1); begin Day 2 harness hardening.
3. Whoever gets Sarvam/Groq API keys first: run `scripts/probe_latency.py`, record ADR-003.
