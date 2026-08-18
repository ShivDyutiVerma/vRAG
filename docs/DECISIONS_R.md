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

## R-008 — Reduced batch size for BGE-M3 to fit this machine's 6GB GPU

**Date:** 2026-08-18
**Status:** Accepted
**Context:** During the A2 embedder ablation, `BGEM3Embedder` hit `torch.OutOfMemoryError: CUDA out
of memory` twice in a row (once mid-run in a sequential batch of 3 embedder evaluations, once again
on a clean-GPU-state retry) while embedding the 99,767-chunk working set at
`sentence-transformers`' default `batch_size=32`. BGE-M3 is the largest of the 4 A2 candidates
(568M params, 1024-dim hidden states) and this machine's RTX 3060 Laptop GPU has only 6GB VRAM
(`docs/DECISIONS_R.md` R-005) — the other three candidates (E5-small, potion-multilingual-128M,
Vyakyarth) are all meaningfully smaller and never hit this.
**Decision:** `BGEM3Embedder` passes `batch_size=8` to `.encode()` explicitly, rather than the
library default of 32.
**Rationale:** Smaller batches reduce peak VRAM usage at some throughput cost — an acceptable
trade for a one-time offline index build (AGENT_BUILD_SPEC.md §3.2), not a hot-path constraint.
**Consequences:** If BGE-M3 wins A2 and needs a production embedding pass at a larger scale later,
revisit whether 8 is still necessary or whether it was specific to this ablation's GPU/driver state.
Doesn't affect the other three embedders' batch sizes (left at the library default, no OOM observed).

## R-009 — Embedder: keeping `multilingual-e5-small`

**Date:** 2026-08-18
**Status:** Accepted
**Context:** A2 ablation (`docs/TECH_MENU.md` §S4) — 3 alternative embedders run against the A1
winner's chunking (`metadata_aware`), same held-out set. Full table: `docs/EVAL_RESULTS.md` §2.
Headline: `potion-multilingual-128M` (Recall@5=0.266) and `vyakyarth` (Recall@5=0.274) both lost to
`multilingual-e5-small` (Recall@5=0.653) by 38+ points — not a close call. `bge-m3` was excluded
after ~1 hour of runtime on this machine's 6GB GPU (vs. 1-13 min for the others), confirmed not
hung, cause not fully diagnosed but not worth pursuing given the other three already settle the
question decisively.
**Decision:** Keep `multilingual-e5-small` as the production embedder — no change from A1's
default.
**Rationale:** The gap between e5-small and the two completed alternatives is an order of magnitude
larger than any noise floor could explain (contrast with A1's chunking strategies, several of which
needed an actual noise-floor run to distinguish). `potion`'s speed advantage doesn't offset a
39-point quality loss when the eventual hot-path embed cost is expected to be single-digit
milliseconds after ONNX quantisation anyway (Phase 6) — there's no latency crisis this trade would
be solving. `vyakyarth` underperforming despite being Indic-specialised is a genuine, verified
(not assumed) finding: correct dims, correct normalisation, no missing prefix — being trained for
Indic language tasks generally doesn't imply being trained for retrieval specifically, which is
what E5's large-scale contrastive training targets directly.
**Consequences:** No index rebuild needed — `data/index/metadata_aware/` (already built with
e5-small) remains the live index `retrieve()` loads. A3 (retrieval mode: dense vs. sparse vs.
hybrid) and A4 (rerank) proceed with e5-small + metadata_aware held fixed, per the staged-ablation
design.

## R-010 — Retrieval mode: dense-only, not hybrid (RRF regresses quality on this corpus)

**Date:** 2026-08-18
**Status:** Accepted
**Context:** A3 ablation (`docs/TECH_MENU.md` §A row A3) — chunking=`metadata_aware`,
embedder=`multilingual-e5-small` (A1/A2 winners) held fixed, retrieval mode varied across
dense/sparse/hybrid+RRF against the same 500-query held-out set, reusing the persisted
`data/index/metadata_aware/` index. Full table and analysis: `docs/EVAL_RESULTS.md` §3. Headline,
counter to `docs/TECH_MENU.md` §S8's "BM25 + dense + RRF + rerank is the 2026 production default":
dense-only (Recall@5=0.652) beat hybrid (Recall@5=0.604), which beat sparse-only (Recall@5=0.428).
**Root cause (verified, not assumed):** BM25 sparse search is markedly weaker than dense on this
corpus (Recall@1 0.162 vs. 0.322). RRF fuses purely by rank position within each lane
(`1/(k+rank)`), with no mechanism to discount a lane whose top ranks are less trustworthy — so
sparse's weaker rank-1/2/3 guesses receive the same fusion weight as dense's stronger ones and
displace genuine dense hits from the fused top-`k` (top_k=10 pulled per lane, matching
`HybridRetriever`'s actual production shape, not a larger candidate pool). Confirmed the result
wasn't a wiring bug before accepting it: both `DenseIndex.search`/`SparseIndex.search` return
best-first as `reciprocal_rank_fusion` (`src/vrag/index/fusion.py`) requires, and all three modes
score through the same `score_hits`/R-006-dedup path.
**Decision:** Ship dense-only as the production retrieval mode. A4 (rerank) and beyond proceed
against the dense-only lane, per the staged-ablation rule that each stage carries forward the prior
stage's actual winner, not an assumed one.
**Rationale:** The gap (4.8pp Recall@5 vs. hybrid, 22.4pp vs. sparse-only) is large relative to A1's
noise floor (~0.2-0.4pp) and deterministic (no randomness in query embedding, FAISS HNSW search, or
BM25 search once the index is already built) — not a candidate for a 3x noise-floor rerun.
Dataset-specific explanation, not a general anti-hybrid claim: BM25 depends on literal token overlap
between query and passage, which is weakened by this corpus's machine-translated Hindi text
(inconsistent term translation, transliterated acronyms — `docs/DECISIONS_R.md` R-003) in exactly the
way dense embeddings' semantic matching is not.
**Consequences:** `HybridRetriever`'s RRF fusion path stays implemented and tested (`retrieval_mode=
"hybrid"`), unused in the default config. A candidate mitigation (fuse over a larger per-lane pool,
e.g. top-50, before truncating to top-10) was identified but deliberately not tested in this run —
it changes candidate-pool size, a different ablation axis than retrieval mode, and would void this
run under `CLAUDE.md`'s "never change two variables in one experiment" rule. Logged as a follow-up
idea in `docs/RISKS.md`.

**Update, 2026-08-18 — production wiring applied, after explicit user confirmation.** This ADR
initially left the code change as a follow-up, flagging that it deviates from
`AGENT_BUILD_SPEC.md` line 625's assumed Phase-3 exit criterion ("hybrid beats dense-only on
Recall@5") and from CLAUDE.md's original "dense and sparse run concurrently" hot-path invariant.
Given the size and reliability of the measured gap, asked the user directly rather than resolving
either way unilaterally; confirmed: ship dense-only as the default, keep hybrid mode implemented and
tested but off by default. Applied: `HybridRetriever` (`src/vrag/retrieval/hybrid.py`) now takes a
`retrieval_mode` constructor arg (`"dense"` default, `"sparse"`/`"hybrid"` also supported, each
skipping the search calls the others don't need — dense mode no longer calls BM25 at all);
`src/vrag/retrieval/interface.py` passes `retrieval_mode="dense"` explicitly; CLAUDE.md's hot-path
invariant line updated to state the default and point to this ADR.

**Side effect worth flagging for Workstream P:** switching from RRF-fused scores (bounded to roughly
`0.008-0.033` for a two-list fusion at `k=60`) to raw dense cosine-similarity scores (typically
`~0.3-0.95` for E5 embeddings) changes `RetrievedChunk.score`'s numeric scale. This matters because
`src/vrag/guardrails/g3_confidence.py`'s `TAU = 0.35` placeholder is explicitly calibrated against
"query-document cosine similarity typically runs ~0.30-0.55" (its own docstring) — i.e. G3 was
already written assuming a cosine-similarity-scale score, which the old hybrid-mode RRF scores could
structurally never reach (max ~0.033 < 0.35), meaning `top1 < TAU` was true for every real query
regardless of actual relevance. **Confirmed, not just inferred:** re-ran the local-only `test_api.py`
failure noted earlier this session ("G3 correctly abstains on real low scores") against the live
`/ask` endpoint post-switch — it still abstains, but `refusal_reason` moved from what would have been
a structurally-guaranteed `top1 < TAU` failure to `"Ambiguous match: top result doesn't clearly stand
out"` (the *margin* check, `top1 - top5 < MARGIN=0.05`) — `top1` now clears `TAU` as the docstring
intends. Not fixed here (G3 is Workstream P's module, not touched) — flagged in `docs/RISKS.md`
(P-R18) for P to verify against real G3 calibration; `MARGIN`, not `TAU`, looks like the placeholder
most worth scrutinizing first now.

## R-011 — New dependency: `flashrank` (rerankers' FlashRank backend, for the A4 rerank ablation)

**Date:** 2026-08-18
**Status:** Accepted
**Context:** A4 (`docs/TECH_MENU.md` §S9) tests `FlashRankReranker` (`src/vrag/retrieval/rerank.py`),
which loads via the already-installed `rerankers` library's `model_type="flashrank"` path. Running
it failed immediately: `rerankers` doesn't vendor FlashRank's own inference code, it's a separate
optional backend (`pip install "rerankers[flashrank]"`) that pulls in the `flashrank` PyPI package.
**Decision:** Add `flashrank>=0.2` to `pyproject.toml`'s `retrieval` extra, alongside the already
existing `rerankers>=0.10`.
**Rationale:** `flashrank` is exactly what `docs/TECH_MENU.md` §S9 already named as the one viable
hot-path reranker on CPU-only infra ("sub-20ms for 50 candidates") — this isn't a new capability
being introduced, just the concrete package `rerankers`' own docs require for the backend already
planned. Verified via `pip install "rerankers[flashrank]"`: resolves cleanly, all transitive deps
(`onnxruntime`, `tokenizers`, already-present in the retrieval stack) already satisfied.
**Consequences:** One more package for R's retrieval extra; zero impact on Workstream P (not in base
`dependencies`, P never imports `vrag.retrieval.rerank`).

## R-012 — Reranking: shipping `none` — both tested rerankers actively destroy quality on Hindi text

**Date:** 2026-08-18
**Status:** Accepted
**Context:** A4 ablation (`docs/TECH_MENU.md` §A row A4 / §S9) — chunking/embedder/retrieval mode
held at the A1-A3 winners, reranker varied across none/FlashRank/cross-encoder. Full table and
query-level diagnostics: `docs/EVAL_RESULTS.md` §3. Headline, far larger than any prior ablation
gap: `none` scores Recall@5=0.652; `flashrank` collapses to 0.100 (n=30 sample); `cross-encoder`
collapses to 0.228 (n=500). Verified hard before writing this up, given the size of the effect:
- **FlashRank (`ms-marco-MultiBERT-L-12`) outputs saturate at ~1.000 for every candidate on Hindi
  text, regardless of relevance** — confirmed on 3 real queries where dense's #1-of-50 hit was
  correct: all 10 post-rerank scores read `1.0` and the correct passage was pushed out of the
  top-10 in every case. Isolated to a Hindi-specific model failure, not a bug: the same model
  cleanly separates relevant/irrelevant **English** text (0.999 vs. 0.002-0.006) but produces the
  same near-1.0-for-everything pattern on short, unambiguous **Hindi** text (a directly-relevant
  candidate scored *below* an unrelated "banana" candidate). Also independently disqualifying on
  latency: 12.7s/query (measured on real corpus-length text) — traced to `rerankers`' FlashRank
  backend running CPU-only `onnxruntime` (no `onnxruntime-gpu` installed) on a 12-layer multilingual
  model; ~500x `docs/TECH_MENU.md` §S9's own "sub-20ms" estimate. Not fixable by adding a GPU either
  — `AGENT_BUILD_SPEC.md` §5.3's deploy target is CPU-only.
- **Cross-encoder (`ms-marco-MiniLM-L6-v2`) is fast (150ms/50 candidates, GPU) but English-only
  and equally uninformative on Hindi**, for a different, also-verified reason: same short-Hindi
  isolation test, given one obviously-correct answer and two clearly-unrelated candidates, it ranked
  the correct answer **last**, below both irrelevant ones, with all scores clustered tightly
  (8.32-8.75) — genuine out-of-distribution behaviour for a model never trained on Hindi, not noise
  in the harness.
- **Ruled out as a wiring/candidate-pool problem:** both rerankers received the same
  `(chunk_id, text)` pairs from the identical dense candidate pool that independently scores 0.652
  Recall@5; `score_hits`'s R-006 dedup path is shared and unchanged across all three A4 rows.
**Decision:** Ship `none` (`NoOpReranker`) as the production reranker — no change from
`HybridRetriever`'s existing default. `FlashRankReranker`/`CrossEncoderReranker` stay implemented in
`src/vrag/retrieval/rerank.py` (useful if a genuinely Hindi-capable reranker is swapped in later) but
neither is wired into the request path.
**Rationale:** `docs/TECH_MENU.md` §S9 frames `none` as "SHIP as default — prove rerank earns its
ms," explicitly treating "no measurable difference" as one valid, expected A4 outcome. What was
actually found is stronger and cleaner than that baseline case: both tested candidates were measured
to actively destroy quality, each for an independently verified, model-specific reason (score
saturation vs. English-only training) rather than a marginal or ambiguous result.
**Consequences:** No production code changes needed (the default was already `none`). A genuinely
multilingual/Hindi-capable reranker (e.g. `BAAI/bge-reranker-v2-m3`, TECH_MENU's own BENCH-ONLY tier
for latency reasons, not language fit) was not tested — out of scope for A4's three named
candidates, worth a footnote for anyone revisiting rerankers later. `flashrank`'s dependency
(R-011) stays in `pyproject.toml` since `FlashRankReranker` remains valid, tested code, just unused
by default.
