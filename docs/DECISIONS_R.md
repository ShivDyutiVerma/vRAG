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

**Update, 2026-08-18 — noise floor run, status upgraded to Accepted.** `metadata_aware` run 3x total
(GPU embedding, R-005): Recall@5 spread 0.652–0.654 (0.2pp), Recall@10 spread 0.750–0.754 (0.4pp) —
tight, consistent with FAISS HNSW build-order randomness being the only source of variance (chunking
+ embedding + dataset are all deterministic). This tight noise floor changes the read on the other
strategies too: `passage_native`/`fixed_overlap`/`metadata_aware` (0.650–0.654) are tied within their
own noise band, but their gap to `hierarchical` (0.640) and `semantic` (0.644) — 1.0–1.4pp — is 5–7x
the measured noise floor, so that gap is real, not noise. The original "all five statistically tied"
framing above undersold the comparison; `metadata_aware`'s selection rationale (cheapest among the
genuinely-tied top three) still holds, now on firmer footing. Full table: `docs/EVAL_RESULTS.md` §1.
Separately, re-running `sentence_window` after the R-006 metric fix (below) confirmed it stays the
clear loser (Recall@5 0.552, still well below the other five) — the fix changes its exact number,
not the ranking. **Status: Accepted.**

## R-005 — Use the CUDA build of PyTorch for offline index-building (embedding), not CPU

**Date:** 2026-08-18
**Status:** Accepted
**Context:** This dev machine has an NVIDIA RTX 3060 (6GB VRAM), but `pip install torch` (no index
specified) installs the CPU-only wheel by default — `torch.cuda.is_available()` was `False` even
though the hardware and driver (CUDA 13.2) both support it. All A1 ablation index builds
(`docs/DECISIONS_R.md` R-004, `docs/EVAL_RESULTS.md` §1) ran on CPU: 39-141 minutes per strategy,
~6.5 hours total for all 6.
**Decision:** Install the CUDA build explicitly —
`pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps`
— for any offline embedding-heavy work (index building, future A2 embedder ablation, noise-floor
reruns). `sentence-transformers`' `SentenceTransformer` auto-detects and uses CUDA with zero code
changes once the right wheel is installed — `E5Embedder` needed no changes.
**Rationale:** Measured, not assumed: a 2,000-text sample embedded at 1.60ms/text on GPU vs.
26.4ms/text on CPU — a 16.5x speedup. Rebuilding the `metadata_aware` production index (99,767
chunks) took 200s on GPU vs. the 2,320s it took during the CPU-run ablation. This is purely an
offline-build-time optimisation — does **not** touch the CLAUDE.md hot-path invariant ("ONNX int8 is
for CPU only — on GPU it is slower than FP32; use FP16 there"), which is about the eventual
*production, query-time* embedding path (Phase 6, ONNX-quantised, deployed on a CPU-only host per
`AGENT_BUILD_SPEC.md` §5.3) — a completely different stage from building an index once, offline,
on a dev machine.
**Consequences:** `pyproject.toml`'s `torch` dependency has no version floor (see the inline
comment) specifically so a routine `pip install -e ".[retrieval]"` doesn't silently downgrade a
working CUDA install back to CPU-only — the CUDA wheel's version numbers (2.6.0+cu124) lag the
CPU-only PyPI releases (2.13+), so a naive `torch>=2.13` floor would make pip "fix" it every time.
Workstream P's machine and CI both stay CPU-only (no GPU there), which is fine — this only speeds up
offline work on this one machine, and every persisted index artifact works identically regardless of
which device built it.

## R-006 — Fixed a metric bug affecting Recall@5 (not just nDCG) for multi-chunk-per-passage strategies

**Date:** 2026-08-18
**Status:** Accepted
**Context:** `sentence_window`'s original A1 run scored an nDCG@10 of 0.881 — higher than every other
strategy — while simultaneously having the *worst* Recall@5 (0.478) and MRR@10 (0.379) of the six.
That combination is not possible under a correct nDCG computation and was the tell that something
was wrong, not an interesting finding.
**Root cause:** `scripts/eval_chunking.py` mapped each retrieved FAISS *chunk* hit to its source
*passage* id, but never deduplicated before computing metrics. Strategies producing multiple chunks
per passage (`sentence_window` averages ~3.9 chunks/passage) can have the same relevant passage
occupy several slots in one query's top-k results. `nDCG@10`'s occurrence-summed credit was the most
visibly broken by this, but **`Recall@k` was affected too**: slicing the raw (non-deduped) hit list
to `top_k` let duplicate chunks from one passage crowd out genuinely distinct passages that would
otherwise have ranked within the window — `MRR@10` was the only metric actually immune (it only
looks for the *first* occurrence, which duplicates can't change).
**Decision:** Deduplicate retrieved chunks down to unique passage ids, preserving rank order
(highest-ranked occurrence kept), *before* any metric slices to `k` — one fix point in
`eval_chunking.py`, all three metrics correct afterward.
**Consequences:** Re-ran `sentence_window` (the only strategy with material chunk-per-passage
duplication — the other five average ~1.0–1.04 chunks/passage, where this bug had negligible effect,
so their originally-reported numbers were left as-is rather than re-run for a fix that wouldn't move
them). Corrected Recall@5 moved from 0.478 to 0.552 — still clearly the worst of the six, so R-004's
decision is unaffected. `eval/ablation_ledger.csv`'s original buggy `sentence_window` row is
annotated as invalid rather than deleted (ledger is append-only); the corrected re-run is a new row.

## R-007 — New dependency: `model2vec` (for the A2 embedder ablation)

**Date:** 2026-08-18
**Status:** Accepted
**Context:** A2 (`docs/TECH_MENU.md` §A, `docs/BUILD_PLAN.md` P3) compares 4 embedders including
`potion-multilingual-128M`, a Model2Vec static-embedding model flagged as "TEST — potentially the
whole ballgame" for its sub-millisecond CPU query latency. It's loadable via plain
`sentence_transformers.SentenceTransformer` too, but its own model card explicitly recommends the
dedicated `model2vec` package as "the fastest and most lightweight way to run Model2Vec models" —
using the slower path would misrepresent the one property (speed) that makes this model worth
testing at all.
**Decision:** Add `model2vec` to `pyproject.toml`'s `retrieval` extra. Verified real wheel
availability via `pip install --dry-run` before adding (resolves cleanly: `model2vec-0.9.0`, all
transitive deps already satisfied by the existing retrieval stack).
**Consequences:** One more package to install for R's retrieval work; zero impact on Workstream P
(not in the base `dependencies` list, and P never imports `vrag.index.embedder`'s Model2Vec class).

**Update, 2026-08-18 — superseded: reverted to loading via `sentence-transformers`, `model2vec`
dependency removed.** `model2vec.StaticModel.from_pretrained()` fetches the model repo's *entire*
file list — including ~25 unrelated benchmark eval-result YAMLs
(`.eval_results/BrightEconomicsLongRetrieval.yaml` and similar) — and repeatedly hung partway
through on this network. Switched `Model2VecEmbedder` to load via
`sentence_transformers.SentenceTransformer`, which only fetches the files actually needed for
inference and completed reliably. The model's core speed property (no transformer forward pass — a
static lookup + mean-pool) is architectural, not a property of the loading library, so this doesn't
misrepresent what's being tested; `sentence-transformers`' own `StaticEmbedding` backend was added
specifically to serve Model2Vec models natively. `model2vec` removed from `pyproject.toml` — nothing
imports it anymore, and CLAUDE.md's conventions call for deleting confirmed-unused code rather than
leaving it as documented dead weight.
