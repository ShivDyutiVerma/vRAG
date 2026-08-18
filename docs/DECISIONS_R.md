# Architecture Decision Record — Workstream R

> Mine only. Never edited by Workstream P. Numbered `R-001`, `R-002`, ... Append-only.

## R-001 — Dev environment: system Python 3.13.7, no uv/poetry installed

**Date:** 2026-08-17
**Status:** Accepted
**Context:** `AGENT_BUILD_SPEC.md` §5.2 specifies "Python 3.11+ (3.11 for perf)" but doesn't mandate an
exact minor version. This machine has Python 3.13.7 as the only interpreter, no `uv` or `poetry`
installed.
**Decision:** Build R's `.venv` against system Python 3.13.7 using stdlib `venv` + `pip`, rather than
installing `uv`/`poetry` or downgrading to 3.11.
**Rationale:** 3.13 satisfies "3.11+." Adding a new tool (`uv`/`poetry`) is itself a dependency change
that would need its own ADR and setup time neither of us has spare today; `pip`+`venv` is zero-install
and sufficient for a 5-day project. Verified before committing: `pip install --dry-run` resolved real
cp313-win_amd64 wheels for every planned dependency (faiss-cpu, torch, onnxruntime, transformers,
sentence-transformers, bm25s, datasets) — not assumed.
**Consequences:** If any dependency later turns out to lack a 3.13 wheel and requires a source build
that's too slow/flaky, this ADR gets superseded by one pinning a 3.11/3.12 venv instead — cheap to
reverse, so not over-thought now.

## R-002 — Held-out eval set: heldout queries drawn FROM the indexed pool, not excluded from it

**Date:** 2026-08-17
**Status:** Accepted
**Context:** `docs/BUILD_PLAN.md` P0 task 4 says to freeze `eval/heldout_queries.json` as "500
query→passage pairs, excluded from indexing." Read literally, that would make Recall@k undefined —
if a query's gold passage is never in the search index, it can never be retrieved, so Recall@1/@5/@10
would be zero by construction, not a real quality signal. This contradicts `AGENT_BUILD_SPEC.md`
§7.1's eval protocol, which reports exactly those metrics on the held-out set.
**Decision:** Interpret "excluded" as *excluded from informing chunking-strategy design decisions*
(no eyeballing heldout queries while picking hyperparameters), not *excluded from the physical
index*. Mechanism: draw a working pool of Hindi rows from the corpus (size tuned in R-003); randomly
select 500 (fixed seed, sampled only from rows with a real ground-truth passage — see R-003) as the
frozen heldout eval set; index passages from the **full** pool, heldout included, so every heldout
query's gold passage is a valid retrieval candidate.
**Rationale:** This is standard IR practice (a held-out *query* split, not a held-out *document*
split) and is the only reading under which the required metrics are computable at all. Flagged here
rather than silently assumed, since it's a real ambiguity in the spec — not one of the C1-C7 graded
constraints, so proceeding without asking rather than blocking, but documenting the call so it's
easy to revisit if the interpretation turns out wrong.
**Consequences:** `scripts/build_dataset_subset.py` builds both files from one pass over the same
pool. If this interpretation is wrong, both files need regenerating — cheap, since nothing downstream
has been built against the wrong shape yet.

## R-003 — Dataset spot-check (20 passages) + working-pool size calibrated from real stats

**Date:** 2026-08-17
**Status:** Accepted
**Context:** `docs/BUILD_PLAN.md` P0 task 4 requires reading 20 translated passages and recording
quality impressions before committing to a language/subset size.

**Real stats from `scripts/inspect_dataset.py` (500-row sample of the Hindi train file):**
- Schema confirmed: `query, Answer, query_id, query_type, passages{is_selected,
  English_passages, Translated_passages}, source_lang, target_lang, meta, Eng_Query, Eng_Answer`
- Passage length (whitespace tokens): n=4996, min=5, p50=57, p95=115, max=3711
- Mean *relevant* (`is_selected`) passages per query: **0.67** — most queries have 0 or 1 marked
  relevant passage, not several. Only 12,661/20,500 rows (~62%) in the working pool have at least
  one relevant passage at all — `build_dataset_subset.py`'s heldout sampling must draw only from
  eligible rows, or the requested 500 silently comes up short (it did, on the first run: 311/500).
- Mean *total* translated passages per query: **9.99** — matches MSMARCO's standard top-10
  passage-per-query convention.

**Translation quality impressions (20 passages read by eye):** Generally fluent, grammatically
coherent Hindi — readable as natural sentences, not word-salad. Two recurring, citable artifacts,
useful justification for the G4 groundedness guardrail later:
- **Inconsistent technical-term translation within the same query/passage pair.** query_id=620830
  translates "phloem" as "फ्लूम" (reads like "flume") in the *query*, but the *passage* correctly
  uses "फ्लोएम" (the real Hindi term) — the query and its own gold passage disagree on the
  translation of the one term the question hinges on.
- **Acronyms transliterated letter-by-letter with periods** ("SYSDATE" -> "एस.वाई.एस.डी.ए.टी.ई.",
  "CPA" -> "सी.पी.ए.", "DVR" -> "डी.वी.आर.") rather than left in Latin script, which is unusual
  next to how a native Hindi tech document would typically write them. Not wrong, just a consistent
  MT tell.
- One query (query_id=205237, "hoover al height above sea level") lost "hoover al" (Hoover, Alabama)
  entirely in translation, becoming just "समुद्र तल से ऊंचाई पर" ("at height above sea level") — an
  information-dropping translation, not just a stylistic one.

None of these are severe enough to abandon Hindi (ADR-002 stands), but they're real, and the second
one especially motivates keeping G4's groundedness check strict rather than trusting fluency as a
proxy for correctness.

**Decision — recalibrate working-pool size from 20,500 to 10,000 queries.** At 9.99 passages/query,
20,500 queries yields ~205k passage-native chunks — already past the *top* of
`AGENT_BUILD_SPEC.md` §6.1's 50k-200k target, before accounting for strategies that split passages
further (fixed+overlap, sentence-window) producing even more chunks than passage-native. 10,000
queries yields ~99,900 passage-native chunks — mid-range, leaves headroom for the
chunk-count-multiplying strategies to still land under 200k, and keeps embedding time for the A1
ablation (6 strategies x full-pool embedding, CPU-only) in the "minutes, not an hour per strategy"
range the spec asks for.
**Consequences:** `scripts/build_dataset_subset.py --pool-size 10000` (default) regenerates
`data/working_subset.jsonl` and `eval/heldout_queries.json`. Superseded, not silently changed, if a
later phase needs the full 20,500-row pool (e.g. if 100k chunks proves too small once hierarchical
chunking's child+parent expansion is measured for real).

**Update, 2026-08-18 — the "minutes, not an hour" estimate above was wrong in practice.** Running
the actual A1 ablation, index build (dominated by embedding ~100k passages via plain PyTorch
`sentence-transformers`, no ONNX yet — per `docs/BUILD_PLAN.md` P1's own "PyTorch first, ONNX in P6"
ordering) took 39-46 minutes per strategy for the five strategies near passage-native's chunk count,
96 minutes for `semantic` (extra embedding calls during chunking itself, for boundary detection), and
141 minutes for `sentence_window` (390k chunks, 3.9x passage-native's count). Total wall-clock for
all 6: ~6.5 hours. Recorded here rather than silently revising the earlier estimate, since ONNX int8
quantisation (Phase 6) should cut this dramatically and it's worth remembering *why* the P2 chunking
lab took hours instead of "minutes" as originally hoped.

## R-004 — Chunking strategy: provisionally shipping `metadata_aware`

**Date:** 2026-08-18
**Status:** Provisional — see caveats below, not yet run 3x for a noise floor
**Context:** A1 ablation (all 6 strategies, dense-only e5-small, no rerank) run against the real
10,000-query working pool + 500-pair held-out set. Full table and analysis: `docs/EVAL_RESULTS.md`
§1. Headline: `sentence_window` is a clear loser (Recall@5 0.478 vs 0.640-0.654 for the other five);
the other five are within a 1.4-point Recall@5 band — inside likely noise-floor territory.
**Decision:** Ship `metadata_aware` as the production chunking strategy. It produces the *identical*
chunk boundaries as `passage_native` (same 99,767 chunks, same text per chunk, same embedding/build
cost — fastest of the six at 39min) but tags every chunk with `language`/`source_lang`/`query_type`,
which is free now and may be useful later for G2 language filtering or A3 boosting experiments.
**Rationale:** Per `docs/BUILD_PLAN.md` P2's own guard ("if strategies are statistically tied, ship
the cheapest one and say they were tied") — `metadata_aware` costs nothing over the next-cheapest
option and adds optionality other tied strategies don't.
**Consequences — explicitly provisional, not final:** (1) no noise-floor run yet (winner run 3x,
report spread, per `docs/EVAL_PROTOCOL.md`) — the "marginally highest Recall@5" observation is one
run each, and a noise-floor check could show all five are genuinely indistinguishable, which would
still leave `metadata_aware` as the right pick (cheapest-among-tied) but for a cleaner stated reason;
(2) no hyperparameter sweep for `fixed_overlap` (overlap ∈ {0, 0.1, 0.2}) has run. Neither blocks
wiring `HybridRetriever` into `retrieve()` now — both are cheap to revisit before Phase 7 lock-in if
time allows, and reversing this ADR later costs one re-index, not a redesign.
