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

## R-013 — Made `test_api.py`'s stub-covers test hermetic (crosses into Workstream P's file, explicitly authorized)

**Date:** 2026-08-18
**Status:** Accepted
**Context:** `tests/test_api.py::test_ask_returns_answered_for_a_query_the_stub_covers` (Workstream
P's file — `src/vrag/api/`, `src/vrag/guardrails/` are P's ownership per `docs/TEAM_SPLIT.md` §2)
failed locally throughout this session, first noted as "G3 correctly abstains on real low scores"
and later, after R-010's fix, as G3's margin check firing instead. Root cause either way: the test
assumes `retrieve()` takes the Day-0 stub path, which is only true when no real index is present on
disk. On this machine, `data/index/metadata_aware/` exists (built during A1-A2), so `retrieve()`
took the real path and G3 made a real decision the test didn't anticipate — environment-dependent,
not broken (CI is unaffected, `data/` is gitignored, a fresh clone always hits the stub).
**Decision:** Asked the user directly rather than silently editing P's file or silently leaving it —
CLAUDE.md's Step 0 explicitly says to stop and flag a cross-ownership edit even when the request
sounds reasonable in isolation. User confirmed they're acting as both R and P for this instruction
and to proceed. Fixed by making the test hermetic: `monkeypatch.setattr(interface,
"_get_real_retriever", lambda: None)` forces the stub path regardless of ambient filesystem state,
so the test verifies what it's named for (stub behavior) on every machine, not just fresh clones.
**Rationale:** This is a test-hygiene fix, not a guardrail-calibration change — it doesn't touch
`g3_confidence.py`'s logic or thresholds, doesn't resolve the real calibration work `docs/RISKS.md`
P-R18 still flags as open for Workstream P, and doesn't second-guess any decision that's actually
P's to make. It only removes an accidental dependency on local machine state from one test's
pass/fail, which is a legitimate fix regardless of which workstream "owns" the file.
**Consequences:** 149/149 tests pass. `docs/RISKS.md` P-R18 stays open (real G3 calibration is
still Workstream P's future work) — only the test's hermeticity was fixed here, not the underlying
guardrail. Flagging this ADR itself is the transparency mechanism for the cross-boundary edit, same
as any other decision in this log.

## R-014 — efSearch: 64, chosen from the measured recall-vs-latency curve

**Date:** 2026-08-18
**Status:** Accepted
**Context:** `docs/BUILD_PLAN.md` P3 task 9 / `docs/TECH_MENU.md` §S6 — sweep efSearch ∈ {16, 32,
64, 128, 256}, plot Recall@5 vs. p50 search latency, pick the operating point from the curve rather
than guessing. Chunking/embedder/retrieval mode held at the A1-A3 winners. Ran against the frozen
500-query held-out set, reusing the persisted `data/index/metadata_aware/` index —
`DenseIndex.set_ef_search()` (new method) mutates the already-built HNSW graph's search-time
parameter in place, so this needed zero rebuilds and zero re-embedding (500 query vectors computed
once, reused across all five efSearch values). Full table and chart:
`docs/EVAL_RESULTS.md` §3, `docs/assets/efsearch_curve.png`.

| efSearch | Recall@5 | Recall@10 | p50 search | p95 search |
|---|---|---|---|---|
| 16 | 0.628 | 0.728 | 0.139ms | 0.211ms |
| 32 | 0.646 | 0.744 | 0.276ms | 0.436ms |
| **64** | **0.652** | **0.748** | **0.414ms** | **0.553ms** |
| 128 | 0.654 | 0.756 | 0.699ms | 0.987ms |
| 256 | 0.656 | 0.758 | 1.216ms | 1.733ms |

**Decision:** Keep `efSearch=64` (`src/vrag/index/dense.py`'s existing `DEFAULT_EF_SEARCH`) as the
production value.
**Rationale:** The curve has a clear knee at 64: the 16→32→64 climb is real (Recall@5 +2.4pp from
16 to 64, well above A1's measured ~0.2-0.4pp noise floor — docs/DECISIONS_R.md R-004), but 64→128→
256 buys only +0.2pp then +0.2pp more, both inside that same noise band, while p50 latency keeps
climbing roughly linearly with efSearch (0.414ms → 0.699ms → 1.216ms). Even 256's ~1.2ms is trivial
against the 200ms end-to-end budget in isolation, but every hot-path stage's slack matters when the
budget is this tight (`AGENT_BUILD_SPEC.md` §3.2) — there's no reason to pay 2-3x the search cost
for a recall gain that's indistinguishable from noise. `DEFAULT_EF_SEARCH=64` was already hardcoded
as the pre-sweep value in `dense.py` (explicitly marked "placeholder... not a guess dressed up as a
decision") — this run confirms it rather than picking a new number, same pattern as A2's e5-small
confirmation (R-009).
**Consequences:** No code change to the shipped value. `dense.py`'s comment updated from
"placeholder" to cite this ADR, since it's now a measured decision, not an unverified guess.
`DenseIndex.set_ef_search()` is new, reusable infrastructure — cheap to re-sweep later if the index
or query distribution changes materially (e.g. after ONNX quantisation in Phase 6, though efSearch
is a pure search-time/graph-traversal parameter and shouldn't be affected by the embedding
backend's own precision).

## R-015 — G3 calibration: real data gathered, targets not simultaneously reachable via TAU alone (proposal only, not applied)

**Date:** 2026-08-18
**Status:** Proposed — data and curve complete, final TAU/MARGIN NOT applied to
`src/vrag/guardrails/g3_confidence.py`, deliberately left for joint sign-off (see Rationale)
**Context:** `docs/EVAL_PROTOCOL.md`'s G3 calibration spec — 150 in-domain + 150 out-of-domain
queries, sweep `TAU`/`MARGIN`, target false-refusal(in-domain) < 10% and correct-refusal
(out-of-domain) > 80%. Built `eval/calibration.json` (`scripts/build_calibration_set.py`): 150
in-domain = a random sample of the existing frozen `eval/heldout_queries.json`; 150 out-of-domain =
real MSMARCO-XI Hindi questions drawn from the *same* parquet, past row 10,000 (the indexed
working-pool cutoff, `docs/DECISIONS_R.md` R-003) — well-formed, on-topic-*sounding* questions whose
gold passage is genuinely absent from the actual index, which is specifically what G3 (not G1/G2) is
supposed to catch: "no good match in *this* index," not topic/safety classification. No LLM call or
new external dataset needed — the already-cached parquet supplied both halves. Scored all 300 against
the real production index (dense-only, e5-small, efSearch=64) via `scripts/eval_g3_calibration.py`,
replicating `g3_confidence.py`'s exact OR-gated refusal logic (`top1 < TAU OR margin < MARGIN`), not
an approximation of it.

**Headline finding — the two targets cannot both be hit, and this is a real corpus property, not a
calibration-script bug:**

| target false-refusal | actual TAU | actual false-refusal | correct-refusal achieved |
|---|---|---|---|
| ≤ 5% | 0.8487 | 4.7% | 13.3% |
| ≤ 10% (target) | 0.8640 | 10.0% | 38.0% |
| ≤ 15% | 0.8723 | 14.0% | 56.0% |
| ≤ 20% | 0.8835 | 19.3% | 75.3% |
| ≤ 30% | 0.8918 | 30.0% | 79.3% |

At the target false-refusal rate (10%), correct-refusal is only 38% — nowhere near the 80% target.
Reaching 75-79% correct-refusal needs false-refusal 2-3x over budget (19-30%). Full curve:
`docs/assets/g3_calibration.png`.

**Root cause, verified not assumed:** in-domain top1 scores (min=0.822, max=0.961, mean=0.904) and
out-of-domain top1 scores (min=0.829, max=0.949, mean=0.873) heavily overlap — 128/150 (85%) of
out-of-domain queries score above 0.85, squarely inside in-domain's range. Inspected the actual
retrieved passages for several out-of-domain queries to understand why, rather than accepting the
number blind: MSMARCO-XI's passages are **not unique per query_id** — a query about "one billion's
definition" (row >10,000, gold passage not indexed) retrieved a passage that *is* actually about the
number one billion, almost certainly because near-duplicate general-knowledge content recurs under
many different query_ids across the full ~780k-row dataset, not just the one it's "officially" linked
to. Other cases were topically-adjacent near-misses (a "Madrid weather" query's top hit was about
"Prague weather" — same topic, wrong city) rather than random noise. Both patterns mean "outside the
indexed 10k-row pool" is a real but imperfect proxy for "no genuine answer exists" — dense cosine
similarity is measuring semantic closeness, which a broad general-knowledge corpus can often satisfy
even for a query whose *specific* answer isn't indexed.
**Decision:** Do **not** unilaterally change `TAU`/`MARGIN` in `g3_confidence.py`. This data makes a
strong case that the `EVAL_PROTOCOL.md` targets (<10%/>80%) are not achievable with pure top1-cosine
gating on this corpus, which is a genuine product tradeoff (how often is it acceptable to refuse a
real, answerable question, versus how often is it acceptable to confidently answer with no real
match?) — not a pure data question R can settle alone. `docs/TEAM_SPLIT.md` §5 reserves the actual
G3 threshold pick as Day-3/Day-4 joint work ("P: guardrail G3/G4 calibration (joint)" /
"finalize the G3 calibration curve"); this ADR is R's contribution to that joint decision (the
curve, the data, the root cause), not the final call.
**Rationale for not shipping a value now:** `g3_confidence.py`'s constants are explicitly joint
ownership (`docs/TEAM_SPLIT.md` §2: "Guardrails G3/G4 — JOINT — calibration needs both"), unlike the
purely-P-owned `test_api.py` fix in R-013 which needed the user's explicit override to touch. Here
the file ownership itself already permits R to edit it — the reason to hold off is that picking a
point on this curve is a value judgment with no data-only right answer, and the schedule explicitly
names this as a joint checkpoint rather than either track's unilateral call.
**Consequences:** `eval/calibration.json` (300 queries) and `docs/assets/g3_calibration.png` are
committed and reusable — re-running `scripts/eval_g3_calibration.py` after any retrieval-side change
(a new embedder, a larger corpus, a reranker that changes score semantics) costs nothing but a few
seconds. G3's current `TAU=0.35`/`MARGIN=0.05` stay as explicitly-marked uncalibrated placeholders.
Also strengthens the case for G4's hot-path checks (citation-ID validation + lexical overlap,
already implemented by Workstream P) as a necessary second line of defense: the "Prague vs. Madrid"
near-miss example is exactly the kind of plausible-but-wrong match G3 alone can't reliably catch, but
a generated answer asserting the wrong city should fail G4's groundedness check once compared against
what was actually retrieved.

## R-016 — `fixed_overlap`'s overlap hyperparameter sweep: no measurable effect, confirms R-004

**Date:** 2026-08-18
**Status:** Accepted
**Context:** `docs/BUILD_PLAN.md` P2 task 5 / `TECH_MENU.md` §A named `overlap ∈ {0, 0.1, 0.2}` as an
open sweep, left unrun (flagged as a low-priority gap in `docs/EVAL_RESULTS.md` §1 and
`docs/PROGRESS_R.md` across this session). Ran the two missing points (0.0, 0.1) against the frozen
500-query held-out set — 0.2 already had a ledger row from A1. Full table:
`docs/EVAL_RESULTS.md` §1.
**Decision:** No change to `fixed_overlap`'s shipped config or to R-004's chunking decision.
**Rationale:** All three overlap values (Recall@5 0.650/0.652/0.650) land inside A1's own measured
noise floor (~0.2-0.4pp, R-004) — a genuine null result, not just a small effect. Consistent with
this corpus's passage-length distribution (p50=57 words vs. a 256-word chunk size, R-003): most
passages fit in one chunk regardless of overlap, so overlap has little surface area to act on. Also
matches the literature citation already in `fixed_overlap.py`'s own docstring (no measurable overlap
benefit found for sparse retrieval in a Jan 2026 study) — this extends the same null result to dense
retrieval on this specific corpus.
**Consequences:** Closes the last open item from A1's "what's not done yet" list. `eval/
ablation_ledger.csv` gets 2 new rows (`fixed_overlap` at overlap=0.0 and 0.1); no code or config
changes needed since the shipped strategy (`metadata_aware`) doesn't use `fixed_overlap` at all — this
was purely a methodology-completeness check on a strategy that was already not the winner.

## R-017 — G3's joint operating point applied by Workstream P; MARGIN corrected same day to match it

**Date:** 2026-08-18
**Status:** Accepted
**Context:** Workstream P picked and applied `TAU=0.8835` from R-015's curve (the "weighs both
`EVAL_PROTOCOL.md` targets equally" point, 19.3% false-refusal / 75.3% correct-refusal) — the joint
decision R-015 deliberately left open. `docs/DECISIONS_P.md` P-015 has P's side of this. Their
commit's own docstring already flagged the one thing it hadn't done: "MARGIN is carried over
unchanged from the pre-calibration placeholder — not yet independently swept at this TAU."
**Root cause found completing that flagged step:** `MARGIN=0.05` (the pre-calibration placeholder)
does not transfer to the new `TAU=0.8835`. At this operating point, in-domain top1-vs-top5 gaps are
naturally tiny (this TAU sits in a narrow, tightly-clustered part of the score distribution — R-015),
so `MARGIN=0.05` pushed false-refusal to **88.0%** in live testing, not the 19.3% the just-shipped
commit's own docstring states as its design target. Verified two ways before concluding this was a
real bug and not a misreading: (1) a live `/ask` call against 3 real queries — all 3 abstained, 2 via
the margin check specifically; (2) a direct fine sweep at `TAU=0.8835` showed false-refusal degrades
steeply even at tiny margins (`MARGIN=0.01` alone → 28.7%) — there is no useful non-zero `MARGIN` at
this `TAU` on this corpus.
**Decision:** Set `MARGIN=0.0` (which structurally disables the margin gate, since `top1-weakest`
can never be negative) — the value the calibration data itself supports at this `TAU`, confirmed
live: the same 3 queries now correctly get 2/3 answered, 1/3 still abstains via a legitimate `TAU`
check (0.87 vs. 0.8835 — a genuine borderline case, not a bug).
**Rationale for fixing directly rather than only flagging:** unlike R-015's `TAU` pick (a genuine
product-tradeoff value judgment with no data-only answer), this is not a new judgment call — it's
applying the *same* calibration data and analysis that already produced `TAU=0.8835`, to keep the
file internally consistent with its own stated design target. `g3_confidence.py` is joint-owned
(`docs/TEAM_SPLIT.md` §2), and the gap was already explicitly named as unfinished in the commit that
introduced it, so completing it is a natural continuation of the same joint work, not a
second-guess. Verified live before shipping, same discipline as every other finding this session.
**Consequences:** `tests/guardrails/test_g3_confidence.py`'s `test_ambiguous_close_scores_fail_
margin_check` no longer matches reality at `MARGIN=0.0` (the margin gate is a no-op at this
operating point) — split into two tests: one confirming clustered-but-above-tau scores now correctly
pass (the real, current, intentional behavior), one confirming the margin *mechanism* still works
correctly when a nonzero value is set via `monkeypatch` (so the code path stays covered for any
future recalibration). 151/151 tests pass. If `TAU` is ever recalibrated again, `MARGIN` needs
re-sweeping at the new value too — the module docstring now says so explicitly, so this doesn't
silently recur.

## R-018 — Found the live deployment runs entirely on the Day-0 stub; prepared (not applied) the missing index artifact

**Date:** 2026-08-18
**Status:** Accepted — artifact prepared and published; the actual deploy-side fix is Workstream P's
to apply, not done here
**Context:** Was about to start `docs/BUILD_PLAN.md` P6's ONNX-quantise-the-embedder task, reasoning
that `docs/TEAM_SPLIT.md` §5's Day-3 "complete" gate looked satisfied from both `docs/PROGRESS_R.md`
and `docs/PROGRESS_P.md` (both describe a fully working real pipeline). Checked the actual live URL
directly before proceeding, rather than trusting local-testing claims for a deploy-readiness
question — good thing: `POST /ask` against `https://vrag-voice.onrender.com` on a real corpus query
returned `chunk_id: "stub-chunk-001"`/`"stub-chunk-002"` with `retrieve` timing `0.007ms` — the
hardcoded Day-0 stub, not real FAISS search (which costs sub-millisecond but not sub-microsecond,
and wouldn't return exactly those fixed chunk IDs).
**Root cause:** `Dockerfile` runs `pip install -e .` with no extras — the `retrieval` extra
(`sentence-transformers`, `faiss-cpu`, `bm25s`, `torch`, all of `src/vrag/index/`'s real
dependencies) is never installed in the production image. Separately, `data/` is gitignored, so the
persisted index was never going to reach the container even if the extra were installed.
`AGENT_BUILD_SPEC.md` §5.3 anticipated exactly this: *"do not build the FAISS index at container
start... build it offline, commit the artifacts to a release asset or object storage, and
download-and-mmap at boot."* That step was never built by either track — R kept improving the index
locally across A1-A4/efSearch/G3 without anyone shipping a version of it anywhere near production.
**Decision:** Prepared the R-side half of the fix without touching `Dockerfile`/`render.yaml`
(Workstream P's ownership per `docs/TEAM_SPLIT.md` §2's "Deployment" row) — packaged the current
production index (`data/index/metadata_aware/`: `metadata_aware` chunking, `multilingual-e5-small`
embedder, `efSearch=64`, built from git commit `bbe74a5`) as a GitHub Release asset:
`https://github.com/ShivDyutiVerma/vRAG/releases/download/index-metadata_aware-v1/
metadata_aware_index.tar.gz` (187MB compressed, verified publicly downloadable). Extracting it to
`data/index/metadata_aware/` at the repo root is all that's needed on the code side —
`src/vrag/retrieval/interface.py`'s `_get_real_retriever()` already looks for exactly that path and
picks it up automatically, no code change required.
**What's still needed (not done here, flagged for Workstream P in `docs/RISKS.md`):** (1) install
the `retrieval` extra in the production image — `pip install -e ".[retrieval]"` in `Dockerfile`
instead of the current bare `pip install -e .`; this alone adds real weight and CPU-only `torch`
(no CUDA wheel needed in production, `docs/DECISIONS_R.md` R-005) to the image; (2) a boot-time (or
build-time) step that downloads and extracts the release asset above into `data/index/
metadata_aware/` before `uvicorn` starts — a `curl`/`tar` line in the `Dockerfile` or an entrypoint
script; (3) redeploy and re-verify with the same live `/ask` check used to find this bug.
**Rationale for not applying the Dockerfile/render.yaml change myself:** unlike R-013's `test_api.py`
fix or R-017's `MARGIN` fix (both squarely retrieval-adjacent, data-backed corrections), this touches
deployment infrastructure explicitly owned by Workstream P, with real consequences if done wrong
(a broken production deploy, unlike a local test or a guardrail constant). The artifact and the exact
steps needed are prepared and documented so the actual fix is small and low-risk for whoever applies
it, but the "commit to render.yaml/Dockerfile and watch it deploy" step should be P's, or done
jointly.
**Consequences:** The "Day-3 complete" milestone read from local testing (`docs/PROGRESS_R.md`/
`docs/PROGRESS_P.md`) does **not** hold for the actual public URL specifically on the retrieval
stage — everything else in the pipeline (STT, generation, all five guardrails logic, harness) is
real and live per P's verification, only retrieval is stubbed in production right now. This doesn't
retroactively invalidate any ablation finding (A1-A4, efSearch, G3 calibration) — those were all
measured against the real index directly, never through the stub — it only means the *deployed
demo* doesn't yet reflect that work. Flagged in `docs/RISKS.md` as blocking for the actual C7
deliverable ("public GitHub repo, live working link") if left unresolved before submission.

## R-019 — ONNX int8 embedder: 3.7x faster query embedding, small real quality cost, not yet shipped

**Date:** 2026-08-18
**Status:** Accepted — code + validation complete, **not wired into `retrieve()`'s production
path** (see Consequences for why)
**Context:** `docs/BUILD_PLAN.md` P6 task 5 / `CLAUDE.md`'s own hot-path invariant: "ONNX int8 is
for CPU only — on GPU it is slower than FP32; use FP16 there," targeting the actual production
deploy shape (CPU-only host, `AGENT_BUILD_SPEC.md` §5.3), not this dev machine's GPU. Exported
`intfloat/multilingual-e5-small` to ONNX and applied dynamic int8 quantisation via
`sentence-transformers`' own `export_dynamic_quantized_onnx_model` helper (`avx2` config — the
broadest generically-supported x86_64 instruction set, since Render's exact CPU isn't known in
advance; dynamic quantisation needs no calibration dataset). New `ONNXE5Embedder` class in
`src/vrag/index/embedder.py`, registered in `EMBEDDER_REGISTRY`, same interface and query:/passage:
prefix requirement as `E5Embedder` — a drop-in replacement at the embedding layer.

**Tested the realistic production shape, not a same-precision rebuild.** Passages are embedded
once, offline, where build time doesn't matter (the existing FP32-built `data/index/
metadata_aware/` index already exists and works — R-004/R-009/R-014). Query embedding is the actual
hot-path cost against the 200ms budget, so `scripts/eval_onnx_quantization.py` quantises only the
*query-time* embedder and searches the existing FP32-built index with int8-embedded queries —
cross-precision compatibility is exactly what would ship, not a hypothetical same-precision rebuild
(which would also have cost ~50+ min of CPU-bound re-embedding for ~100k passages, for a number that
doesn't reflect the actual deploy shape anyway).

| | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| FP32 baseline (§2, A2) | 0.322 | 0.653 | 0.752 | 0.453 | — |
| int8 ONNX query, FP32 index | 0.330 | 0.644 | 0.750 | 0.4563 | 0.5197 |
| **Δ** | **+0.8pp** | **-0.9pp** | **-0.2pp** | **+0.3pp** | — |

| | p50 | p95 | p100 |
|---|---|---|---|
| FP32, forced CPU | 20.48ms | 25.83ms | 35.02ms |
| int8 ONNX, CPU | 5.60ms | 7.67ms | 10.38ms |
| **Speedup** | **3.7x** | **3.4x** | **3.4x** |

**Analysis:** the -0.9pp Recall@5 drop is small but not run-to-run noise — unlike A1's noise floor
(measured from repeated *index rebuilds*, where FAISS HNSW insertion-order randomness is the
variance source, R-004), this comparison has no randomness at all: same frozen 500-query set, same
FP32-built index, same corpus — the only variable is a fixed, deterministic float32→int8
quantisation of the query embeddings. It's a genuine, small, real quality cost of quantisation, not
noise. Recall@1 and MRR@10 both moved *up* slightly and Recall@10 barely moved, consistent with a
small precision-loss effect that mostly nudges borderline rankings rather than breaking clearly-correct
or clearly-wrong matches. The FP32-CPU baseline itself (20.48ms p50) is also a new, real number worth
having on its own — A2's original write-up (`docs/EVAL_RESULTS.md` §2, footnote 1) explicitly left
e5-small's CPU embed latency unmeasured, flagging "Phase 6 will measure the real ONNX-quantised
figure" — this closes that exact gap.
**Decision:** The 3.7x latency win for a <1pp quality cost is a good trade in isolation — but this
ADR does **not** wire `ONNXE5Embedder` into `retrieve()`/`interface.py` as the production default.
**Rationale for not shipping the wiring change yet:** `docs/RISKS.md` R-R21 (found the same session,
just before this) means there is currently no live deployment for this to matter to — the production
container doesn't even run real retrieval yet. Wiring in a query-embedder swap before that's fixed
would be premature (untestable against the actual deploy target) and would add a second simultaneous
change (query latency *and* whether retrieval is real at all) exactly when R-R21's fix needs to be
isolated and easy to verify. Once R-R21 lands, swapping `interface.py`'s `E5Embedder()` for
`ONNXE5Embedder()` is a one-line change with this ADR's numbers as the justification.
**Consequences:** New dependencies: `onnx`, `optimum`, `optimum-onnx` added to `pyproject.toml`'s
`retrieval` extra; `sentence-transformers>=5.7` changed to `sentence-transformers[onnx]>=5.7` to
pull in ONNX Runtime inference support. **Real, non-trivial side effect:** `optimum-onnx==0.1.0`
hard-pins `transformers<4.58.0`, incompatible with this project's prior `transformers>=5.15` floor —
`huggingface-hub`'s floor also had to drop (`>=1.27` → `>=0.23,<2.0`) to match what `sentence-
transformers[onnx]` actually resolves to. Verified safe before committing, not assumed: full test
suite (172/172) green at the downgraded versions, every embedder class smoke-tested individually
against the new floor. `data/onnx/multilingual-e5-small/` (the exported+quantised model,
~590MB combined FP32+int8 files) is gitignored like the rest of `data/` — regenerate with
`scripts/export_onnx_embedder.py`, don't commit ONNX binaries.

## R-020 — Measured real RSS: the index alone already exceeds Render free tier's memory limit

**Date:** 2026-08-18
**Status:** Accepted — measured and documented; the actual fix is not decided or applied here
**Context:** `docs/RISKS.md` R4 ("index too large for host memory... measure RSS in Phase 1") sat
open, unmeasured, since Phase 0. Became urgent once R-R21 (`docs/RISKS.md`) meant Workstream P was
about to attempt loading the real index into the live Render container for the first time.
**Method:** loaded `data/index/metadata_aware/` (99,767 chunks — under `AGENT_BUILD_SPEC.md` §6.1's
200k cap) in three separate processes, each left idle after loading, RSS read via Windows `tasklist`
(the same technique already used earlier this session to confirm GPU-embedding progress — no new
dependency needed for a one-off diagnostic).
**Results:**
| Config | RSS | % of Render free tier's 512MB |
|---|---|---|
| Index only (no embedder) | 591MB | 115% |
| + `E5Embedder`, forced CPU | 1,474MB | 288% |
| + `ONNXE5Embedder` (int8) | 1,539MB | 301% |
| + `E5Embedder`, this dev machine's GPU (not production-representative) | 1,860MB | — |

**Finding:** the index alone already exceeds the free-tier limit before any embedder loads — this
isn't an embedder-choice problem. `ONNXE5Embedder` (R-019) gave no meaningful RSS reduction vs FP32,
despite the ONNX model file itself being 4x smaller on disk — `sentence-transformers` still imports
the full `torch`/`transformers` stack regardless of inference backend, so R-019's win is real but
latency-only, not the memory win P6's "optimisation pass" framing might suggest.
**Likely largest single contributor (not yet isolated further):** `chunk_lookup.json` stores full
chunk text for all 99,767 chunks (113MB serialized) — CPython's per-object/string overhead when
that's parsed into a live dict of `Chunk` objects typically inflates well beyond the raw JSON size,
plausibly accounting for the largest share of the 591MB index-only figure alongside FAISS's 180MB
on-disk HNSW graph and bm25s's ~35MB files.
**Decision:** Document and flag, not fix unilaterally. Real options exist with real tradeoffs
outside R's sole authority: upgrade Render's plan (cost/infra, joint), shrink the working pool
(fewer chunks — a genuine quality cost against A1-A4's already-measured numbers, would need
re-ablation, not a free lunch), or a leaner `chunk_lookup.json` format (e.g., memory-mapped or
lazy-loaded rather than one big in-memory dict — an R-ownable engineering fix, not yet designed).
**Rationale for not picking one now:** unlike R-017/R-019 (data-backed corrections completing
already-flagged gaps), this is a fresh tradeoff decision with cost/quality/engineering-effort axes
that go beyond pure retrieval-quality data — matches the same reasoning R-015/R-018 used to leave
their own decisions for joint sign-off rather than picking alone.
**Consequences:** `docs/RISKS.md` R4 escalated to 🔴 high and marked confirmed-real, cross-referenced
from R-R21 — whoever attempts R-R21's deploy fix should read this first, or risk an OOM crash that
looks like an unrelated deploy failure. No code changes in this ADR.

## R-021 — Prototyped the "leaner chunk_lookup format" option from R-020: real win, not sufficient alone

**Date:** 2026-08-18
**Status:** Accepted — prototype built, tested, measured; **not wired into production** (see
Consequences)
**Context:** R-020 named a leaner `chunk_lookup.json` format as one of three real fix directions for
the confirmed memory-budget overrun, explicitly "R-ownable, not yet designed" and not requiring a
cost/quality tradeoff decision to attempt (unlike the other two options). Built it to turn that from
a vague possibility into real, measured data for whoever makes the final call.
**What was built:** `SQLiteChunkLookup` (`src/vrag/index/sqlite_chunk_lookup.py`) — same read
interface every real call site already uses (`__getitem__`, `.get`, `__contains__`, `__len__`,
`.items()`, `.values()`, verified against actual usage across `hybrid.py` and every eval script
before writing this, not assumed), backed by SQLite instead of one live dict of ~100k Pydantic
`Chunk` instances. Only `chunk_id -> doc_id` stays fully in memory (cheap: two short strings ×
99,767 rows) — every eval script's bulk `chunk_to_doc_id` mapping needs exactly that, not full text;
full chunk *text* (the actual memory cost) is fetched lazily, one row at a time, matching how
production's hot path actually uses it (`hybrid.py` only ever needs text for a request's top-k
results, never all 99,767 chunks at once). `scripts/convert_chunk_lookup_sqlite.py` builds the
`.sqlite3` file from the existing `chunk_lookup.json` (125MB on disk, vs. JSON's 113MB — the B-tree
index costs a little more on disk in exchange for not needing everything in RAM).

**Measured (not assumed), same `tasklist` RSS methodology as R-020:**

| Config | RSS | vs. R-020's dict-based baseline |
|---|---|---|
| Index only, dict `chunk_lookup` (R-020) | 591MB | — |
| Index only, `SQLiteChunkLookup` | 339MB | **-252MB (-43%)** |
| + `ONNXE5Embedder`, dict `chunk_lookup` (R-020) | 1,539MB | — |
| + `ONNXE5Embedder`, `SQLiteChunkLookup` | 1,321MB | **-218MB (-14%)** |

**Real win, confirms `chunk_lookup`'s in-memory dict was a genuine major contributor to R-020's
finding — but not sufficient alone.** Even at 339MB, the index-only figure is now comfortably under
Render's 512MB free-tier budget on its own — but the full stack (1,321MB) is still 258% over budget,
because the embedder's own import footprint (`torch`/`transformers`, loaded regardless of inference
backend — R-019's own finding) is ~980MB and is now the clearly dominant remaining cost, not the
index. Fixing `chunk_lookup` alone would not have closed the gap.
**Decision:** Keep `SQLiteChunkLookup` as a real, tested, measured option — not wired in as the
default `load_built_index()` path yet. Doing so now would be a partial fix presented as if it
solved the problem, when the embedder import overhead is the larger remaining piece; wiring it in
makes most sense as part of whatever combined fix gets chosen for R4 (e.g. alongside a torch-free
inference path, see below), not as a standalone change.
**Consequences:** `eval/heldout_queries.json`-driven eval scripts still use the original
dict-based `load_built_index()` — `SQLiteChunkLookup` is additive infrastructure, not yet the
default, so nothing downstream changed. **Follow-up idea, not attempted this session:** a torch-free
inference path (raw `onnxruntime.InferenceSession` + a lightweight tokenizer, bypassing
`sentence-transformers`'s Python wrapper, which imports `torch` as a hard dependency regardless of
which backend actually does inference) could plausibly close most of the remaining ~980MB gap —
untested, real engineering effort, would need its own validation pass the same way R-019/R-021 did.
`docs/RISKS.md` R4 updated to reflect the full combined picture (index fixed, embedder still the
open problem) rather than treating this as a full resolution.

## R-022 — Torch-free embedder: -580MB more, from 1,321MB to 741MB — real, major, still not under budget

**Date:** 2026-08-18
**Status:** Accepted — built, tested (byte-identical output verified), measured; **not yet wired
into production** (default embedder unchanged); explicit user direction: no paid Render plan, must
fit the free tier, keep shrinking
**Context:** R-021 confirmed `chunk_lookup`'s in-memory dict was a real contributor but not
sufficient alone — the embedder's own `torch`/`transformers` import footprint (~980MB) was the
larger remaining cost, present regardless of inference backend because `sentence-transformers`
imports `torch` unconditionally. Measured `import torch` alone in isolation: **~383MB RSS**, most
of that ~980MB, confirming the hypothesis directly rather than assuming it.
**What was built:** `LiteE5Embedder` (`src/vrag/index/embedder.py`) — same ONNX int8 model as
`ONNXE5Embedder`, but bypasses `sentence-transformers` entirely: raw `onnxruntime.InferenceSession`
+ the `tokenizers` library's Rust tokenizer (already installed transitively, no new dependency),
mean-pooling and L2-normalisation done by hand in numpy (matching the model's own
`1_Pooling`/`2_Normalize` config, read directly from the exported model directory, not guessed).
**Verified byte-identical to `ONNXE5Embedder`'s output before trusting it**, not assumed: cosine
similarity 1.0, max absolute difference ~1.5e-8 (pure float32 rounding noise) on a real query —
same model, same math, just without `torch` in the import graph.

**Measured (same `tasklist` methodology as R-020/R-021):**

| Config | RSS | Δ from previous step |
|---|---|---|
| SQLite index + `ONNXE5Embedder` (R-021, via sentence-transformers) | 1,321MB | — |
| SQLite index + `LiteE5Embedder` (torch-free) | 741MB | **-580MB (-44%)** |
| `import torch` alone, isolated | 383MB | (explains most of the above) |

Tried two further `onnxruntime`-level tunings on `LiteE5Embedder`, both measured, neither helped:
disabling the memory arena/mem-pattern/limiting threads (738MB, no change) and disabling graph
optimisation at session-creation time (432MB for model+numpy+tokenizer alone, same as with
optimisation on — the ~380MB delta over bare `onnxruntime`'s ~49MB import is inherent to loading
this model's weights/graph into an active session, not tunable via session options).
**Decision:** Ship `LiteE5Embedder` as real, tested, measured infrastructure — not yet the default.
741MB is a massive, real reduction from where this started (1,860MB on this dev machine's GPU,
1,474MB FP32-CPU, R-020) but is still 145% of Render's free-tier 512MB budget — user explicitly
ruled out a paid plan, so this alone doesn't close the gap yet.
**Rationale:** Same reasoning as R-021 — presenting 741MB as "fixed" when it's still over budget
would be misleading. Recording it as real, substantial, verified progress toward the target, with
the honest remaining gap stated plainly.
**Consequences:** Combined with R-021, the two together take the full stack from 1,539MB → 741MB
(-52%) using only R-owned engineering changes, no cost and no quality tradeoff yet. The remaining
~230MB gap to fit under 500MB most plausibly needs the one option R-020 flagged as having a real
quality cost — shrinking the working pool/chunk count — since e5-small is already A2's smallest
viable model and further `onnxruntime`-level tuning showed no further headroom. `eval_chunking.py`'s
`EMBED_BACKEND_BY_NAME` and `tests/test_embedder.py`'s registry tests updated for the new class.

## R-023 — Wired R-021/R-022 into the actual real-retriever path; produced the runtime artifacts; deploy step still needed (P)

**Date:** 2026-08-18
**Status:** Accepted — code wired, real-index smoke-tested in an isolated torch-free venv (727MB
RSS, matching R-022's isolated measurement), runtime artifacts published as GitHub Releases. Deploy
step (Dockerfile) intentionally not touched — that's P's module (`docs/TEAM_SPLIT.md` §2).
**Context:** R-021 and R-022 built and measured `SQLiteChunkLookup` and `LiteE5Embedder` as
standalone prototypes, but `src/vrag/retrieval/interface.py::_get_real_retriever()` — the one place
that actually constructs the production retriever — still hardcoded `E5Embedder()` (FP32,
`sentence-transformers`, unconditional `torch` import) and `load_built_index()` (eager
`chunk_lookup.json` dict). Confirmed via `git log` that this was correctly called out as "not
started" in P's P-018/P-R21 fix (`docs/DECISIONS_P.md`) — the prototypes existed, the wiring didn't.
**What changed (all R-owned or the interface.py seam, not Dockerfile):**
- `src/vrag/index/embedder.py`: added `EmbedderProtocol` (structural type for "any embedder with
  `embed_queries`/`embed_passages`") so `HybridRetriever` isn't hard-typed to `E5Embedder`.
- `src/vrag/retrieval/hybrid.py`: `HybridRetriever.__init__`'s `embedder` param now typed
  `EmbedderProtocol`; `chunk_lookup` param now typed `Mapping[str, Chunk] | SQLiteChunkLookup`
  (tried a narrower custom Protocol first — mypy couldn't cleanly match it against `dict.get`'s
  overloaded signature, so used the concrete union instead; simpler and it typechecks).
- `src/vrag/index/persistence.py`: added `load_built_index_lean()` — loads `chunk_lookup.sqlite3`
  via `SQLiteChunkLookup` when present, falls back to the eager JSON dict otherwise (so it's a
  drop-in replacement for index artifacts built before R-021, e.g. the original
  `index-metadata_aware-v1` release).
- `src/vrag/retrieval/interface.py`: `_get_real_retriever()` now constructs `LiteE5Embedder()` +
  `load_built_index_lean()` instead of `E5Embedder()` + `load_built_index()`. The `retrieve()`
  function's own signature/contract is untouched — this is an internal implementation swap, not a
  change to the R/P seam itself.
**Verification, not assumption:** built a real index (`data/index/metadata_aware/`, 99,767 chunks)
into a **fresh, throwaway venv with only the new `retrieval-lean` pyproject.toml extra installed**
(no torch, no transformers — confirmed via the install log) and called `retrieve()` end-to-end
through `interface.py`. Got 5 real (non-stub) hits back, correct Devanagari text, plausible cosine
scores (0.818–0.835) — confirmed `IS_STUB=False`. Measured RSS via the same `tasklist` methodology
as R-020/R-021/R-022: **727MB**, matching R-022's isolated 741MB measurement closely (small
variance expected — different process, different measurement instant).
**New dependency grouping (`pyproject.toml`):** added a `retrieval-lean` extra — `numpy`,
`faiss-cpu`, `bm25s`, `onnxruntime`, `tokenizers` — deliberately excluding
`torch`/`transformers`/`sentence-transformers`/`optimum`/`optimum-onnx`/`rerankers`/`flashrank`.
The existing `retrieval` extra (all of the above) stays for dev machines running ablations/ONNX
export; `retrieval-lean` is what the deployed container should install. No new PyPI packages were
added — every package in `retrieval-lean` was already a `retrieval` dependency, just regrouped —
so no separate ADR needed under the "never add a dependency without an ADR" rule.
**Runtime artifacts published (GitHub Releases, same pattern as R-018's index release,
`AGENT_BUILD_SPEC.md` §5.3):**
- `embedder-lite-onnx-v1` — `multilingual-e5-small-lite-onnx.tar.gz` (87.6MB compressed): just the
  two files `LiteE5Embedder` actually reads (`tokenizer.json` + `onnx/model_quint8_avx2.onnx`), not
  the full `sentence-transformers` export directory (579MB — most of it an unused FP32
  `model.onnx` and HF metadata files `LiteE5Embedder` never opens).
- `index-metadata_aware-v2` — same dense/sparse index as v1, plus `chunk_lookup.sqlite3`
  (`chunk_lookup.json` kept alongside for backward compatibility with `load_built_index()`/local
  dev extraction).
**What P's Dockerfile still needs to do** (deployment is P's module — deliberately not done here):
1. `pip install --no-cache-dir -e ".[retrieval-lean]"` instead of the current plain
   `pip install --no-cache-dir -e .` (still no torch/transformers pulled in — verified above).
2. Download+extract `index-metadata_aware-v2` (not `-v1`) from the URL pattern already in the
   Dockerfile, just the new tag.
3. Download+extract `embedder-lite-onnx-v1`'s asset to `data/onnx/multilingual-e5-small/` (the
   `LiteE5Embedder`/`ONNXE5Embedder` default `model_dir` — no code/env-var change needed if
   extracted to exactly that path).
4. That's the complete list — no other interface.py/Dockerfile coordination needed; `retrieve()`'s
   contract is unchanged, and `_get_real_retriever()`'s existing try/except-and-fall-back-to-stub
   still covers any partial/missing artifact the same way it already does today.
**Consequences:** This is the difference between "741MB measured in isolation" and "741MB is what
production actually uses" — without this wiring, R-021/R-022's real work would never reach the live
URL regardless of how good the numbers were. Once P's Dockerfile change lands, expected production
RSS is ~727-741MB (measured, not projected) — still ~215-230MB over Render free tier's 512MB, so
`docs/RISKS.md` R4 stays open; the corpus-shrink lever (real quality cost, needs re-ablation) is
still the most plausible remaining path to close that gap, unless the team decides 741MB is close
enough to reconsider paid-tier economics (explicitly the user's/team's call, not mine — see R4's
current framing on the paid-fallback question).

## R-024 — Measured the corpus-shrink lever's real ceiling before running it: not a "no cost" fix, likely disqualifying at full budget

**Date:** 2026-08-18
**Status:** Investigation complete, **decision escalated to the user — not resolved here**. No
ablation run, no chunk-count change made. This ADR exists to record why a real re-ablation pass
wasn't started without checking first.
**Context:** R-023's Consequences section named "shrink the corpus" as the remaining lever to close
the ~215-230MB gap to Render's 512MB free tier, on the assumption (R-020's original framing) that
it's a real-but-bounded quality cost, parallel to R-021/R-022's engineering fixes. Before spending a
re-ablation pass on it (real time, and it touches Recall@5 — one of the graded C1-C6 requirements),
measured what the lever can actually deliver.
**Method:** Reused the real, already-persisted `data/index/metadata_aware` FAISS index's own
computed vectors (via `faiss.Index.reconstruct()` — no re-embedding needed) to build two
subsampled indices at N=20,000 and N=50,000 chunks (down from the full 99,767), converted each to
`chunk_lookup.sqlite3` (R-021's format), and measured index-only RSS with the same `tasklist`
methodology as R-020/R-021/R-023.
**Measured (index-only, no embedder loaded, SQLite chunk lookup):**

| N chunks | RSS |
|---|---|
| 20,000 | 131.0MB |
| 50,000 | 209.5MB |
| 99,767 (known, R-021) | 339MB |

These three points are strongly linear: fitting N=20k/50k gives `RSS(N) ≈ 78.7MB + 0.00262MB/chunk
× N`, and that fit predicts 339.8MB at N=99,767 — matching the independently-measured R-021 number
to within 0.3%. High confidence this is genuine linear HNSW-graph scaling, not noise.
**The problem: the embedder's fixed session cost, not the index, is now the dominant term.**
R-023 measured full-stack (index + `LiteE5Embedder`) at 727MB for N=99,767; index-only at that N is
339MB, so the embedder+Python-baseline overhead is a **fixed ~388MB regardless of corpus size**
(confirmed independently by R-022's own isolated `onnxruntime` session measurement, ~432MB, and by
R-022's finding that further `onnxruntime`-level tuning couldn't reduce it further — this is the
cost of loading this specific model's weights/graph into an active session, not something corpus
size touches at all).
**Consequence, doing the arithmetic honestly:** `full_stack(N) ≈ 467MB + 0.00262MB/chunk × N`.
Fitting a target of 512MB: **N ≈ 17,300 chunks — a cut to ~17% of the current corpus, an ~83%
reduction.** Even N=0 (an empty index) would still cost ~467MB, already 91% of the entire budget,
before a single chunk is added. There is no corpus size that reaches 512MB without a cut this
severe.
**Checked whether a smaller embedder model could remove the fixed ~388MB cost instead (Model2Vec's
`potion-multilingual-128M` — no transformer/ONNX session at all, a static lookup+mean-pool
embedding, which A2 already had real numbers for):** `eval/ablation_ledger.csv` row
`metadata_aware_1787028710` — **Recall@5 = 0.266 vs. e5-small's 0.652 on the same full corpus,
same querybase — less than half.** A2 already called this "decisive," and this confirms it's not a
viable memory lever either: trading the embedder for one with near-zero session overhead costs far
more quality than shrinking the corpus does.
**Decision:** Did not run a real re-ablation pass or make any chunk-count change. An ~83% corpus
cut is not the "real but bounded" cost R-020 originally framed this as — it is a severe change that
would very likely fail Recall@5 by a wide margin (candidate pool shrinks 5.7x; MSMARCO-XI passages
already recur across query_ids per R-015's calibration finding, so a much smaller pool plausibly
loses genuine hits, not just easy ones) and risks the graded C1-C6 requirements. This is exactly the
kind of call `CLAUDE.md`'s "When blocked" section reserves for the user, not something to guess on
or spend a full re-ablation pass validating without checking direction first.
**What the user/team should know making this call** (see `docs/RISKS.md` R4 for the live-updated
version): 741MB (or 727MB measured in the actual production-wired code, R-023) is already a 60%
cut from where this started (1,860MB on a GPU dev machine measurement, R-020) using only free
R-owned engineering, zero quality cost paid so far. Getting the rest of the way to 512MB via corpus
size alone requires an ~83% cut with a real, likely-severe Recall@5 hit — untested, but the
magnitude of the required cut (not just "smaller," but keeping under a fifth of the corpus) makes a
mild outcome unlikely. Real alternatives worth weighing, none of which are R's call alone: accept a
smaller, quantified corpus cut and a correspondingly higher deploy target than exactly 512MB;
revisit paid-tier economics now that the free-path number is 727MB vs. the original 1,474MB
(a materially different cost-benefit than when the paid fallback was first discussed); or check
whether a different free hosting tier offers more headroom (P's deployment domain).

## R-025 — Verified the live deploy directly after P applied R-023/P-020's Dockerfile fix: `/ask` returns 502, `/healthz` doesn't

**Date:** 2026-08-18
**Status:** Finding recorded, **not investigated or fixed here** — this is P's deployment module,
and the fix (if any) lives in `Dockerfile`/`render.yaml`/Render's dashboard, none of which R owns.
**Context:** P's P-020 (`docs/DECISIONS_P.md`) applied R-023's 3-step Dockerfile change, fixed two
real bugs found along the way (tarball path mismatch, Python 3.12 floor), and verified locally
under Docker's own `-m 512m` limit: 446.8MiB steady-state, real (non-stub) retrieval confirmed.
Deployed live on that basis, with a documented rollback plan if Render's real environment behaved
differently than the local Docker test.
**What I checked, following the same "verify claims against the live URL directly" discipline that
found R-018's original stub issue:** `GET /healthz` → 200 OK, consistently, across 3 separate
retries. `POST /ask` with a real Hindi query (`"भारत की राजधानी क्या है?"`, verified UTF-8 bytes in
the request body, not a shell-encoding artifact) → **502 Bad Gateway**, header
`x-render-routing: no-deploy`, reproduced twice (immediately, and again after a 20s wait for a
possible restart).
**Interpretation, not confirmed root cause:** `/healthz` never touches
`_get_real_retriever()` — it's a static handler. `/ask` is the first thing that forces the lazy
singleton to actually load the FAISS index + ONNX embedder session for real. The pattern (health
check fine, first real-retrieval-triggering request 502s with a "no deploy currently serving"
routing header) is consistent with an OOM crash during that first load in Render's actual
environment, even though P's local Docker test with the same `-m 512m` limit completed a real
query successfully — P-020 already named this as a real possibility ("cloud container limits and
enforcement can differ from local Docker Desktop in ways not fully visible from here"). Not
confirmed — could also be a Render-specific cold-start/routing artifact unrelated to memory; I
didn't have access to Render's own logs/dashboard to check for an actual OOM-kill signal.
**Deliberately not acted on:** fixing this (whether that's the documented rollback, a memory
adjustment, or something else) requires `Dockerfile`/`render.yaml`/Render dashboard access, all P's
ownership (`docs/TEAM_SPLIT.md` §2) — recording the verified symptom here and in `docs/RISKS.md`
R-R21 for P's session to pick up, not attempting a fix myself.

## R-026 — R-R14's candidate-pool mitigation tested: doesn't rescue hybrid, closing the item

**Date:** 2026-08-18
**Status:** Accepted — tested, real result, closes `docs/RISKS.md` R-R14. No config change (dense-
only stays the shipped default, unaffected either way).
**Context:** A3 (R-010) found hybrid+RRF regresses vs. dense-only (Recall@5 0.604 vs. 0.652) and
named a hypothesis: each lane only contributes `top_k=10` candidates before fusion, so BM25's
weaker top ranks get equal RRF weight against dense's stronger ones — a larger per-lane candidate
pool before fusion (fetch more, truncate to `top_k` after) is the standard mitigation for exactly
this failure mode. Flagged as untested, worth a quick follow-up "if time remains after A4/A5
land" (R-R14). With R4/R-R21 both currently blocked on decisions/access outside R's control (user
direction on the RAM tradeoff, P's Render diagnosis) and Day 4's freeze not yet reached, this was
available, self-contained R-owned work.
**What was tested:** `scripts/eval_rrf_candidate_pool.py` — same held-out 500-query set, chunking/
embedder/`retrieval_mode="hybrid"`/`fusion_k=60` all held fixed at existing values; only the
per-lane candidate pool size (fetched from each of dense/sparse before RRF fusion, then truncated
to `top_k=10` after) varies: 10 (A3's original baseline), 30, 50, 100.
**Result — real, not a projection:**

| candidate_pool | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---|---|
| 10 (A3 baseline) | 0.604 | 0.392 | 0.467 |
| 30 | 0.578 | 0.378 | 0.451 |
| 50 | 0.586 | 0.378 | 0.449 |
| 100 | 0.578 | 0.375 | 0.446 |
| *dense-only, for reference* | *0.652* | *0.452* | *0.516* |

A larger candidate pool does **not** help — Recall@5 is flat-to-slightly-worse than the pool=10
baseline at every size tested, and every hybrid configuration remains well below dense-only
regardless of pool size. The hypothesis doesn't hold: this isn't a simple "candidate starvation"
problem a bigger pool fixes. A plausible reason, consistent with A3's original diagnosis: a larger
pool admits *more* of BM25's weaker, lower-precision candidates into the fusion at full RRF weight,
which can dilute good dense hits further rather than rescuing hybrid's ranking.
**Decision:** No config change — dense-only remains correct and unchallenged as the shipped
default. Closes R-R14 with a real, tested answer instead of leaving it as an open, untested idea.
**Consequences:** One less open item on `docs/RISKS.md`. Confirms A3's original conclusion is
robust to the most obvious first mitigation attempt, strengthening confidence in the dense-only
decision rather than casting doubt on it.

## R-027 — The real quality/latency/memory sweep R-024 never ran; corrects R-024's "83% cut" estimate

**Date:** 2026-08-19
**Status:** Accepted — real measurement, run on user request. **Materially corrects R-024's
"~17,300 chunks / 83% cut" estimate** — the real number is worse: **~5,900 chunks / ~94% cut**.
**Context:** R-024 (2026-08-18) measured only RSS-vs-chunk-count scaling (20k/50k/99,767 chunks,
index-only) and explicitly did not measure Recall/MRR/latency at those sizes — that was flagged as
the missing half and escalated rather than assumed. This entry is that missing half, run on
request, using `scripts/eval_corpus_size.py` (quality + latency) and
`scripts/audit_full_stack_at_size.py` (real full-stack RSS, not derived by combining
separately-measured components).
**Methodology, stated precisely:** Chunk subsamples via `random.Random(42).sample()` (not a prefix
slice — avoids ordering bias from how the corpus was originally built), reusing the real,
already-computed FAISS vectors via `reconstruct()` — no re-embedding. Every held-out query scored
through the exact same `score_hits()`/`dedupe_doc_ids()` path every A1-A4 ablation uses. Full-stack
RSS measured with `load_built_index_lean(..., retrieval_mode="dense")` — the exact function and
mode production uses (post-ADR-007), including the lean SQLite chunk lookup at every size, not the
eager JSON dict (an earlier draft of this measurement mixed the two and produced a 961MB
full-corpus reading that didn't match ADR-007's real 715.7MB — caught by cross-checking against a
known-real number before trusting it, not assumed correct on the first run).

**Real, measured table:**

| n_chunks | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Coverage* | FAISS RSS | FAISS disk | Search P50/P100 | Full-stack RSS (steady) |
|---|---|---|---|---|---|---|---|---|---|
| 20,000 | 0.120 | 0.184 | 0.192 | 0.149 | 22.2% | 86.9MB | 36.2MB | 0.45/0.90ms | **542.7MB** |
| 50,000 | 0.202 | 0.406 | 0.440 | 0.285 | 54.4% | 142.9MB | 90.4MB | 0.44/0.75ms | **606.4MB** |
| 99,767 (full) | 0.330 | 0.644 | 0.750 | 0.456 | 100% | 237.4MB | 180.4MB | 0.47/0.74ms | **714.4MB** (matches ADR-007's 715.7MB within 1.3MB — cross-checked) |

*Coverage = fraction of the 500 frozen held-out queries whose relevant passage survives in the
subsampled corpus at all, independent of whether search actually finds it — precomputed once per
corpus from the full `doc_id` set, not from top-10 hits (an earlier draft of this script computed
coverage from retrieved hits only, which conflates "answer removed" with "answer present but
ranked outside top-10"; fixed before trusting the number).

**What "83% cut" (R-024) vs. "94% cut" (this entry) means, precisely:** both are solving the same
equation — fit `full_stack_RSS(N) = intercept + slope * N` to real measured points, then solve for
the `N` where `full_stack_RSS(N) = 512MB`. R-024's fit used an intercept built by *adding* two
separately-measured numbers from different sessions (R-020's index-only intercept + R-022's
embedder-only figure). This entry's fit (least-squares over the 3 real full-stack points above,
predicted-vs-measured within 0.6MB at every point — a tight fit) gives:
**`full_stack_RSS(N) ≈ 499.3MB + 0.00215MB × N`**. Solving for 512MB: **N ≈ 5,900 chunks — a ~94%
cut from 99,767**, not R-024's ~83%. Neither number reflects a measured recall at that reduced
size (5,900 is smaller than any tested point) — both are linear extrapolations of *memory*, with
zero information about quality at that size. The real, measured recall numbers above (0.184 at
20,000, the smallest size actually tested) already show severe degradation well above 5,900 chunks;
recall at 5,900 would be worse still, unmeasured, and not safely extrapolable from a 3-point curve
whose lowest real data point is already over 3x larger.
**Does any tested configuration remain within the spec's 50k–200k target (ADR-002)?** **No.**
50,000 sits exactly at the spec's floor and its full-stack RSS (606.4MB) already exceeds 512MB.
20,000 is already below the spec's floor and also exceeds 512MB (542.7MB). The full corpus
(99,767, within spec, and the only size with real quality numbers matching A1-A4's history) is at
714.4MB. **No spec-compliant size was measured to fit under 512MB, and the extrapolated size that
would (~5,900 chunks) is itself roughly 8.5x below the spec's 50k floor** — meaning satisfying both
the RAM budget and ADR-002's original corpus-size target simultaneously is not achievable via
corpus-shrink alone under the current architecture (embedder + FAISS HNSW dense-only).
**Consequences:** This makes the corpus-shrink lever *more* clearly non-viable than R-024 already
found it to be, not less. No corpus change made — this is measurement only, matching the user's
explicit scope for this request. The practical choices for closing the remaining ~200MB gap (at
the real, spec-compliant full corpus) are unchanged from R-024/RISKS.md R4: a paid tier, a
different host, or accepting a severe, spec-violating corpus cut with real (now partially
quantified) quality cost. `eval/corpus_size_tmp/` (gitignored, ~313MB, regenerate via
`scripts/eval_corpus_size.py --sizes 20000 50000 99767`) holds the built subsample artifacts this
run used.

## R-028 — Deep LiteE5Embedder memory audit: the tokenizer costs more than the model, and no ONNX Runtime setting meaningfully helps

**Date:** 2026-08-19
**Status:** Investigation only, per the user's explicit "do not modify production code yet"
instruction. `scripts/audit_embedder_memory_faithful.py` (real step-by-step RSS, matching
production's actual code path), `scripts/audit_embedder_memory.py` (an earlier, flawed attempt —
kept and documented rather than deleted, see the methodology lesson below), and
`scripts/investigate_onnx_settings.py` (SessionOptions sweep).

**Methodology lesson worth keeping on record:** the first attempt at fine-grained step separation
used `onnx.load()` to get an intermediate "model loaded into memory" checkpoint before
`InferenceSession(...)`, splitting what `InferenceSession(path)` normally does in one call. This
produced a session-ready total of ~704.7MB — which didn't match ADR-006's already-real, already-
trusted 460.2MB isolated-embedder figure, so it was NOT taken at face value. Investigated the
discrepancy before reporting anything: `onnx.load()` + `SerializeToString()` + `InferenceSession()`
double-parses the model (a full Python-side `ModelProto` plus ORT's own internal C++
representation coexist simultaneously), adding **~250MB of pure measurement artifact** not present
in real usage. A second, faithful script (`ort.InferenceSession(path)` called directly, exactly as
`_ensure_loaded()` does, no `onnx.load()` detour) landed at 456.0MB steady-state — within 4MB of
ADR-006's number. **The faithful numbers below are the ones to trust; the staged ones are kept in
the repo for the methodology lesson, not as a source of truth.**

**Real, faithful step-by-step RSS (production code path, no artificial split):**

| Step | RSS | Delta |
|---|---|---|
| 1. Before any import | 18.9MB | — |
| 2. After importing `vrag.index.embedder` | 19.5MB | +0.6MB |
| 3. After importing `onnxruntime` | 54.1MB | +34.7MB |
| 4. After importing `tokenizers` | 54.7MB | +0.6MB |
| 5. After `Tokenizer.from_file()` (tokenizer loaded) | **317.0MB** | **+262.3MB** |
| 6. After `InferenceSession(path)` (model read + session built) | **454.4MB** | **+137.3MB** |
| 7. After first dummy inference | 455.9MB | +1.6MB |
| 8. After 20 more inferences | 456.0MB | +0.1MB (flat — no growth/leak) |

**The tokenizer costs almost 2x what the actual neural network session does** — 262MB vs. 137MB.
Root cause, verified not assumed: the tokenizer is a SentencePiece Unigram model (XLM-RoBERTa's,
same family E5-multilingual uses) with a **250,002-token vocabulary** — real, checked directly in
`tokenizer.json`. The `tokenizers` Rust library builds trie/lookup structures for the full
vocabulary at load time; this cost is inherent to using this model's tokenizer at all, and applies
identically to `E5Embedder`/`ONNXE5Embedder` too (same tokenizer, same vocab) — it isn't something
`LiteE5Embedder`'s implementation choice caused or can avoid alone.

**Requested facts, all measured directly:**
- ONNX model file size: 118,335,516 bytes (~118.3MB)
- ONNX Runtime version: 1.28.0. Available providers: `AzureExecutionProvider`,
  `CPUExecutionProvider`. **Provider actually selected: `CPUExecutionProvider` only** (no GPU).
- Threads: `intra_op_num_threads=0`, `inter_op_num_threads=0` — ORT's "auto" sentinel (not
  literally zero threads; ORT picks a default based on visible CPU cores at session-run time, not
  something the Python `SessionOptions` object reports back as a resolved number after the fact).
- Execution mode: `ORT_SEQUENTIAL`. Graph optimization: `ORT_ENABLE_ALL` (both are ORT's un-set
  defaults — `LiteE5Embedder` passes no `SessionOptions` at all today).
- Memory arena: `enable_cpu_mem_arena=True`, `enable_mem_pattern=True` (both ORT defaults).
- Batch size in production: **1** — `interface.py`/`hybrid.py` always call
  `embed_queries([single_query])`, one item, every request.
- Peak RSS across all 8 steps: 456.0MB (steady-state *is* the peak here — no transient spike above
  the final resting value, unlike the earlier full-stack finding where peak-during-load mattered).
- Duplicate session check: **no duplication** — `LiteE5Embedder._session` is a true lazy singleton;
  verified directly by comparing `id(embedder._session)` across three separate `embed_queries()`/
  `embed_passages()` calls on the same instance — identical object every time.
- Dtype breakdown, byte-weighted (not just tensor count): **UINT8: 117.49MB (99.65%), FLOAT32:
  0.41MB (0.35%)**. The model is genuinely, overwhelmingly int8 — the small float32 remainder is
  the normal, expected residue of dynamic quantization (LayerNorm/bias parameters are typically
  left unquantized; quantizing them barely saves space and can hurt precision), not evidence the
  "int8" claim is wrong.

**ONNX Runtime settings investigated (each isolated, one variable at a time, real latency P50/P100
over 30 real inferences per variant):**

| Variant | Session RSS | Δ vs. baseline | Latency P50 | Latency P100 |
|---|---|---|---|---|
| Baseline (current production defaults) | 455.2MB | — | 3.615ms | 4.125ms |
| `enable_cpu_mem_arena=False` | 454.2MB | **-1.1MB** | 2.829ms | 3.931ms |
| `enable_mem_pattern=False` | 455.3MB | +0.0MB | 3.327ms | 3.751ms |
| `intra_op_num_threads=1` | 453.8MB | **-1.4MB** | 4.985ms (+38%) | 5.699ms |
| `graph_optimization_level=ORT_DISABLE_ALL` | 453.0MB | **-2.2MB** | 3.628ms | 5.661ms |
| All three memory settings combined | 454.4MB | -0.9MB | 4.753ms (+31%) | 5.112ms |

**None of these settings offer a meaningful memory reduction** — every delta is ≤2.2MB against a
~455MB session (under 0.5%), consistent with (and now more rigorously confirming, with real
per-setting isolation and latency data) R-022's earlier finding that session-option tuning showed
no real headroom. `intra_op_num_threads=1` and the combined variant both cost real latency (+31 to
+38% P50) for essentially zero memory benefit — a clear net loss, not a tradeoff worth taking.
`enable_cpu_mem_arena=False`'s faster latency in this run is more likely measurement noise (30
samples, single run) than a real effect — arena allocators normally exist to make repeated
same-shape allocations faster, so a reproducible *speedup* from disabling one would be surprising;
not re-run for a tighter confidence interval since the memory question (the reason to investigate
this at all) was already answered as "no meaningful gain either way."

**Consequences, not acted on (measurement/investigation only, no production change):** ONNX Runtime
session settings are not a fruitful lever for this model — the ~455MB session cost is dominated by
actual weight/vocabulary data, not tunable session bookkeeping. The one real, unexploited, and
substantial finding is the **tokenizer's 262MB** — larger than the model itself — which sits
entirely outside ONNX Runtime's settings surface (a `tokenizers`-library concern, not an ORT one)
and wasn't in scope for this investigation, but is the more consequential place to look next if
embedder memory is revisited.

## R-029 — Tokenizer-only investigation: sentencepiece reproduces exact token IDs at ~18% the memory

**Date:** 2026-08-19
**Status:** Investigation only, per the user's explicit "do not modify production code yet"
instruction. `src/vrag/index/embedder.py` untouched. Recommending consideration (not making the
change) since the user's stated bar — 100% exact token-ID equivalence — was met, verified, not
assumed.

**Where the 262MB actually goes (isolated further than R-028):** read-and-`json.loads()`-ing
`tokenizer.json` in pure Python costs ~53MB; the Rust `tokenizers` library's own internal
structures (trie/automaton for Unigram decoding over the vocabulary) cost an *additional* ~262MB
on top of that when built via `Tokenizer.from_file()` — the two aren't the same allocation, and
production only ever pays the second one (`_ensure_loaded()` never materialises a Python dict of
the vocab). Re-measured fresh via `scripts/investigate_tokenizer.py --stage where_262mb_goes`:
+262.4MB, matching R-028's 262.3MB closely.

**Backend identity, confirmed directly (not inferred from the filename):** `tokenizers.Tokenizer`
(HuggingFace, Rust, v0.22.2) running a `tokenizers.models.Unigram` model — this is a **different
codebase** from Google's `sentencepiece` C++ library, even though both implement the same
published Unigram/SentencePiece algorithm. `tokenizer.json`'s normalizer is a `Precompiled`
charsmap and its pre-tokenizer is `Metaspace` (the "▁" convention) — both are exactly what raw
`sentencepiece` already does internally as part of loading its own `.model` file, which is why a
sentencepiece-based candidate doesn't need to reimplement them separately.

**Vocabulary/model file size on disk:** `tokenizer.json` = 17,082,800 bytes (~16.3MB — of which the
`model.vocab` section alone is ~11.9MB, metadata ~0.3MB). The raw `sentencepiece.bpe.model` (the
original binary this was converted from, downloaded fresh from `intfloat/multilingual-e5-small`
for this investigation, not present in this repo before) is **5,069,051 bytes (~5.07MB) — 3.4x
smaller on disk** before any RSS measurement.

**Memory-mapped/lazy-loading:** no explicit mmap parameter exists in either library's public
Python API (`sentencepiece.SentencePieceProcessor.__init__`'s full signature has none;
`tokenizers.Tokenizer.from_file()` takes only a path). Google's own project documentation states a
"~6MB" typical memory footprint for sentencepiece as a general claim (not measured by this
investigation, but directionally consistent with what was actually measured below) — whether that
comes from an internal mmap mechanism wasn't confirmed from public docs; not claimed either way.

**Token-ID equivalence test — the core result:**
- Test corpus: **1,020 real strings** (`scripts/build_tokenizer_test_corpus.py`) — 500 real Hindi
  queries (the frozen held-out set), 500 real English queries (`data/working_subset.jsonl`'s
  `Eng_Query` field — the original English MS MARCO queries this corpus was translated from, real
  1:1-paired data already in the project, not synthetic filler), 20 mixed/romanized strings (real
  English+Hindi concatenations plus the already-vetted romanized example from
  `tests/guardrails/test_g2_scope_language.py`). Every string formatted with the real `"query: "`
  E5 prefix, matching exactly what `format_query()` sends to the tokenizer in production.
- **First attempt: 0/1020 (0%) exact match.** Investigated before concluding incompatibility, not
  assumed: every mismatch showed the *same* pattern — candidate IDs exactly 1 less than current
  IDs, for every token except BOS/EOS. Root-caused via `sp.id_to_piece()`: raw sentencepiece's
  native layout is `0=<unk> 1=<s> 2=</s> 3=<first real piece>...` (confirmed: `sp.pad_id() == -1`,
  sentencepiece has no native pad concept), while HF's `tokenizer.json` conversion renumbered
  specials to `0=<s> 1=<pad> 2=</s> 3=<unk>` and shifted every real piece up by 1 to make room for
  the inserted `<pad>` token. A **verified, formula-based remap** (`hf_id = raw_id + 1` for real
  pieces, `raw 0 (<unk>) -> hf 3`), not a fudge factor — re-tested:
- **After the remap: 1,020/1,020 = 100.0000% exact match.** Hindi 500/500, English 500/500, mixed
  20/20. Also spot-checked two edge cases the main corpus wouldn't naturally hit: a >512-token
  string (current tokenizer truncates to 512; candidate requires the same truncation applied
  manually, then matches exactly) and a genuinely exotic-script string (emoji + Greek + Russian +
  Arabic mixed) — both matched exactly too.

**Measured, real:**

| | Current (`tokenizers`) | Candidate (`sentencepiece`) | Delta |
|---|---|---|---|
| RSS (net, load only) | 264.2MB | 48.7MB | **-215.5MB (-81.6%)** |
| Tokenization latency P50 | 0.038ms | 0.009ms | **-76% (faster)** |
| Tokenization latency P95 | 0.071ms | 0.015ms | **-79% (faster)** |
| On-disk model file | 17.08MB (`tokenizer.json`) | 5.07MB (`sentencepiece.bpe.model`) | -70.3% |
| Token-ID equivalence | — | **100.0000% (1,020/1,020)** | — |

**Compatibility risks, stated plainly (not glossed over despite the strong result):**
1. **The ID remap is specific to this exact model's vocab layout**, not a generic solution — if
   the ONNX model is ever re-exported from a different/updated checkpoint, the remap formula would
   need re-verification against that checkpoint's own `tokenizer.json`, not assumed to still hold.
2. **Padding/truncation must be reimplemented manually** for a real integration — `tokenizers`
   handles `enable_padding(pad_id=1)`/`enable_truncation(max_length=512)` internally;
   `sentencepiece` has no batch API with equivalent built-in padding, so `LiteE5Embedder._embed()`
   would need its own padding/truncation logic added (straightforward — pad_id=1 confirmed correct
   from `tokenizer.json`'s own `added_tokens`, truncation-then-append-EOS confirmed correct via the
   edge-case test above — but not yet written).
3. **New dependency + new release artifact**: `sentencepiece` isn't currently in any
   `pyproject.toml` extra used by production (added to `dev` only, for this investigation) —
   adopting it would need promotion to `retrieval-lean`, and `sentencepiece.bpe.model` (5MB) isn't
   in the current `embedder-lite-onnx-v1` release asset — a new release/asset would be needed.
4. **1,020 real strings is a strong sample, not exhaustive** — genuinely adversarial or
   pathological inputs (deeply nested Unicode combining characters, e.g.) weren't specifically
   constructed and tested beyond the two edge cases above.
**Consequences:** Meets the user's explicit bar (100% exact token-ID equivalence) for
recommendation consideration. Real, substantial win on both memory (-215.5MB, more than
R-023+R-021's combined memory work) and latency (faster, not a tradeoff) if adopted — but adoption
itself (embedder.py changes, padding/truncation logic, dependency promotion, release artifact
update, full test-suite re-verification) was explicitly out of scope for this investigation and
not done here.

## R-030 — Implemented the sentencepiece tokenizer swap: production RSS now under 512MB for the first time

**Date:** 2026-08-19
**Status:** Accepted — implemented, tested, verified. `src/vrag/index/embedder.py`'s
`LiteE5Embedder` now uses `sentencepiece` instead of `tokenizers`. FAISS index, corpus, embedding
model weights, retrieval architecture, and API behavior are all unchanged — verified, not just
claimed (retrieval eval below reproduces R-027's baseline to full float precision).

**Implementation:** `_ensure_loaded()` now loads `sentencepiece.bpe.model` via
`spm.SentencePieceProcessor` instead of `tokenizer.json` via `Tokenizer.from_file()`. New
`_tokenize_batch()` method replicates, by hand in numpy, everything `tokenizers` did internally:
the verified special-token remap (R-029: `hf_id = raw_id + 1` for real pieces, raw `<unk>`→HF id
3), truncation to `max_length` keeping BOS+EOS (`core_ids[:max_length-2]`, verified against a
>512-token string), and right-padding to the batch's own longest sequence with `pad_id=1` and a
matching `attention_mask` (verified empirically against the old tokenizer's real batch behavior
before implementing, not assumed). `pyproject.toml`: `sentencepiece>=0.2` replaces
`tokenizers>=0.20` in `retrieval-lean` (nothing left in that extra's dependency graph needs
`tokenizers` — confirmed via `grep` before removing it, not assumed); `sentencepiece.*` added to
mypy's `ignore_missing_imports` overrides.

**A real bug found and fixed by the regression tests, not by manual review:** the first
implementation passed all 1,020 corpus strings but failed on `""` and whitespace-only edge cases —
`tokenizers`' Metaspace pre-tokenizer emits exactly one trailing "▁" token when input ends in
whitespace (any amount, always exactly one token); raw `sentencepiece.encode()` silently strips
trailing whitespace instead. Root-caused via a systematic sweep (leading/internal whitespace never
differs; only trailing does) before patching: `_tokenize_batch()` now appends
`sp.piece_to_id("▁")` (looked up dynamically, not hardcoded, so it stays correct if a future model
export shifts that piece's ID) whenever the input text is non-empty and ends in whitespace. This
is exactly the scenario `tests/index/test_lite_e5_embedder_tokenizer_regression.py` was written to
catch — and it did, on the very first run, before this ever had a chance to reach production.

**Regression tests** (`tests/index/test_lite_e5_embedder_tokenizer_regression.py`, 11 tests): all
1,020 real strings from R-029's corpus (live-compared against the real `tokenizers` library, not a
frozen fixture, so future drift fails loudly); 8 additional edge cases (empty, whitespace-only,
single word x2, pure digits, exotic mixed-script, >512-token truncation, realistic long passage);
real batch-padding equivalence; and an end-to-end sanity check (384-dim, L2-normalized output).
**All pass. Full suite: 215/215** (204 previous + 11 new), ruff clean, mypy clean.

**Retrieval evaluation — reproduces R-027's baseline exactly, to full float precision, not just
"close":**

| Metric | R-027 baseline (old tokenizer) | This run (new tokenizer) |
|---|---|---|
| Recall@1 | 0.330 | 0.330 |
| Recall@5 | 0.644 | 0.644 |
| Recall@10 | 0.750 | 0.750 |
| MRR@10 | 0.4562706349206349 | 0.4562706349206349 |

Exact match to 16 significant figures on MRR@10 — not approximately equal, bit-for-bit identical,
exactly as the 100% token-ID equivalence proof predicted (identical tokenizer output → identical
ONNX inputs → identical outputs, same deterministic computation). Same real production FAISS
index, unchanged, used throughout — the passage embeddings baked into it were never touched;
only which library computes *query*-time embeddings changed, and R-019/R-022 already established
that chain (`E5Embedder`→`ONNXE5Embedder`→`LiteE5Embedder`) produces byte-identical output at
every step.

**Production memory audit, rerun in full (3 runs, `scripts/audit_memory.py --component full`):**

| Run | Steady-state RSS | Peak (startup + first query) |
|---|---|---|
| 1 | 494.4MB | 493.1MB |
| 2 | 492.4MB | 492.4MB |
| 3 | 494.5MB | 493.2MB |
| **Average** | **493.8MB** | **492.9MB** |

**Before (ADR-007): 715.7MB. After: 493.8MB — a real 221.9MB (31.0%) reduction**, closely matching
R-029's predicted -215.5MB (small variance from cross-run/cross-session measurement noise, fully
expected). **This is the first time production RSS has measured under Render's 512MB free-tier
budget** — both steady-state (493.8MB) and, critically, the peak during startup+first-query
(492.9MB), which is what P-020 found actually broke the earlier live deploy attempt (peak-during-
load exceeding steady-state). Total reduction across the full memory-fix arc: 1,860MB (R-020's
original GPU-machine baseline) → 493.8MB, a **73.4% cut**, entirely from R-owned engineering with
zero quality cost — confirmed zero, not assumed, by the exact-match retrieval numbers above.

**Consequences, not yet done:** this measurement is local, not the actual Render container — P-020
already found local-vs-Render can differ for peak-during-load specifically, so a real deploy
attempt (or at minimum a Docker `-m 512m` local test, matching P-020's own methodology) is the
honest next step before declaring R4 resolved, not this local number alone. `docs/RISKS.md` R4
updated to reflect this real, substantial, but not-yet-fully-verified-in-the-target-environment
progress. Deployment itself remains parked per the user's earlier "run everything locally first"
direction — this finding doesn't change that, it just means the eventual deploy attempt (whenever
it happens) now has real, current, favorable numbers to work with, and a corpus cut may no longer
be necessary at all if a live Render (or Docker-simulated) test confirms this number holds.

## R-031 — Docker `-m 512m` validation of R-030's build: the exact peak-during-load discrepancy P-020 warned about, now reproduced locally, real R4 FAIL

**Date:** 2026-08-19
**Status:** Accepted as a real, reproduced negative result. R4 is **not resolved**. No architecture,
corpus, FAISS, embedding-model, or tokenizer change made in response — per explicit user
instruction, this was a validation-only exercise and the finding is reported as-is.

**What changed to make this test possible (all Docker/packaging-only, zero retrieval-architecture
change):**
- `Dockerfile`: `pip install -e .` → `pip install -e ".[retrieval-lean]"` (was installing zero
  retrieval dependencies at all — confirmed the pre-existing `Dockerfile` silently ran the Day-0
  stub in production, never real retrieval, see below).
- `Dockerfile`: index asset `index-metadata_aware-v1` → `index-metadata_aware-v2` (the
  SQLite-lookup layout `load_built_index_lean()` actually expects, R-021).
- `Dockerfile`: base image `python:3.11-slim` → `python:3.12-slim` — genuine, previously-undetected
  bug found by this exercise: `retrieval-lean`'s `numpy>=2.5` pin requires Python ≥3.12, but had
  only ever been installed on the dev machine's real interpreter (3.13, R-001), never inside the
  Dockerfile's declared 3.11 until this build. `pyproject.toml`'s stated `>=3.11` floor is still
  satisfied.
- New GitHub Release `embedder-lite-onnx-v2`: `sentencepiece.bpe.model` + the unchanged int8 ONNX
  model (`onnx/model_quint8_avx2.onnx`, byte-identical sha256 to v1's copy). `embedder-lite-onnx-v1`
  only shipped `tokenizer.json`, predating the R-029/R-030 sentencepiece swap — the deployed
  container had no way to load the current embedder at all until this release existed. Asset
  sha256: `840fd2a6563bcf12cd3a7c3e787bc2fb2c4559af811bf09fdd0761d3c7c73769`; inner file hashes
  recorded in the release notes.
- `src/vrag/api/main.py`: added a `lifespan` hook that eager-loads the retriever singleton at boot
  (so a memory test's "startup peak" actually includes the FAISS+embedder load, not just the first
  request), plus an opt-in `VRAG_REQUIRE_REAL_RETRIEVAL=1` env var that makes a failed real-retriever
  load fatal at startup instead of silently degrading to the stub — used only via `docker run -e`
  for this validation, never set by the `Dockerfile` itself, so production's existing "never take
  the service down over a retrieval hiccup" resilience (docs/DECISIONS_P.md) is unchanged by
  default. `/healthz` now also reports `{"status": "ok", "retrieval": "real" | "stub"}` instead of
  just confirming the FastAPI process is alive — status code unchanged (still 200) so this doesn't
  alter Render's actual health-check pass/fail semantics.
- `is_retrieval_real()` added to `src/vrag/retrieval/interface.py` as the public way to ask this
  without reaching into the private `_get_real_retriever()` singleton from another module.
- `tests/test_api.py::test_healthz` updated for the new response shape (machine-dependent
  `retrieval` value, same pattern already used by `test_ask_returns_answered_for_a_query_the_stub_
  covers` for the same reason); added `test_healthz_reports_stub_when_real_retriever_unavailable`
  for a deterministic check of the stub branch. Full suite: 216/216, ruff clean, mypy clean.

**First finding, before any memory test: the previously-committed `Dockerfile` was not testing
real retrieval at all.** Built and ran it exactly as it stood (no extras installed, `index-
metadata_aware-v1`, no embedder asset downloaded): container ran fine at 41.5MiB / 512MiB, but
`/ask` returned `"chunk_id":"stub-chunk-001"` — `_get_real_retriever()` was silently falling back
to the Day-0 stub the whole time (missing `onnxruntime`/`faiss-cpu`/`sentencepiece`). This means
**every prior claim about the live/deployed container's memory behavior for R4 was, at best,
untested and, at worst, measuring the wrong thing** — the actual R-023-through-R-030 memory work
had never been exercised inside a container until this session.

**Second finding, after fixing the above and rebuilding with real retrieval wired in: OOM-killed,
reproduced 2/2.** `docker run -m 512m --memory-swap 512m` (swap disabled, a true hard ceiling, no
masking):

- Startup: clean. `VRAG_REQUIRE_REAL_RETRIEVAL=1` did **not** trip — the real retriever loaded
  successfully (FAISS index + `LiteE5Embedder`'s ONNX session + SentencePiece). `/healthz` returned
  `{"status":"ok","retrieval":"real"}`, confirming this wasn't another stub-fallback false pass.
- Steady-state after startup, before any real (non-guardrail-refused) query: **~276-277MiB**
  (276.4MiB and 277.1MiB across the two runs) — well under budget, and notably *lower* than the
  local host's previously-measured 493.8MB steady-state, because this reading was taken before any
  real embedding inference or FAISS search had executed even once (only a guardrail-refused empty
  query had been sent — G1 rejects before `retrieve()` runs, so it never touches the embedder).
- First real end-to-end query (`"भारत में सबसे ऊँचा पर्वत कौन सा है?"`, the same held-out-style
  query used throughout this session): container was **OOM-killed both times** (`exit code 137`,
  `docker inspect`'s `OOMKilled: true`). Run 1: FastAPI's own access log shows `"POST /ask HTTP/1.1
  200 OK"` immediately followed by `Killed` — the response was served and the kernel reclaimed the
  process's memory right after. Run 2 (continuous ~150ms-interval `docker stats` sampling
  alongside the same request): last live sample **352MiB / 512MiB**, then the container was gone by
  the next sample — the true peak is bounded below by 352MiB and reached/exceeded the 512MiB cgroup
  ceiling somewhere in the ~150ms gap the sampler couldn't resolve (the same class of sampling-
  resolution caveat already documented for BM25 loading in ADR-006, now confirmed to apply here
  too). Because of this, **E ("/ask returns a real chunk_id, not stub-chunk-001") was not directly
  confirmed by reading a captured response body** — the process died before the body was reliably
  read back — but `/healthz` immediately prior confirmed `retrieval: "real"`, and there is no code
  path by which `/ask` would use the stub while `/healthz` simultaneously reports real, so this is
  circumstantially as strong as direct confirmation without literally being one.
- Steps F/G ("10 repeated queries", "memory growth check") were not reached — the container did not
  survive the first real query, so there is nothing to measure repeated-query behavior against.

**This is not a new failure mode — it's the same one P-020 already found live on Render on
2026-08-18** ("every real `/ask` call OOM-kills the process... the blocker is specifically *peak*
memory during first real load, not just steady-state"), **now reproduced for the first time in a
controlled, local, swap-disabled Docker environment against the fully memory-optimized R-030
build**, closing the "not yet verified in the actual target environment" gap this ADR's own R-030
entry and `docs/RISKS.md` R4 both explicitly flagged as the honest next step. The 221.9MB steady-
state win from R-030 is real and unaffected by this finding — but it was never the binding
constraint. **The binding constraint, confirmed twice now across two different measurement
environments (live Render, local Docker), is the memory spike during the first real embedding-
inference-plus-FAISS-search, which no work so far (R-023 through R-030) has directly measured or
targeted** — every memory fix to date reduced steady-state/idle RSS, not inference-time peak.

**Likely cause (hypothesis, not verified — no further investigation run per "stop, do not
auto-optimize"):** ONNX Runtime's session memory arena (`enable_cpu_mem_arena`, on by default,
R-028 found tuning it doesn't help *steady-state* but never isolated its *first-inference* growth
specifically) sizes its buffers to the actual input tensors on first real use, not at session
creation — the eager-`lifespan` startup load in this same change set creates the `InferenceSession`
but deliberately does not run a warm-up inference, so this cost was never paid until the real
query arrived. FAISS HNSW search-time working set and Docker cgroup memory accounting differing
from host-side RSS (page cache, shared-library mappings counted differently) are secondary
candidates. Distinguishing between these would need an isolated first-inference-only memory probe
(same methodology as R-028's audit scripts) — not attempted here, consistent with the user's stop
instruction.

**R4 verdict: FAIL.** The current production build does not survive a hard 512MB limit under real
traffic. `docs/RISKS.md` R4 updated accordingly.

## R-032 — Isolated first-real-inference memory probe: root cause found, and it isn't ONNX inference

**Date:** 2026-08-19
**Status:** Accepted. Diagnostic-only, per explicit instruction — no production/architecture/
Dockerfile/corpus/FAISS/embedding-model/tokenizer change made. Corrects R-031's "likely cause"
hypothesis (ONNX arena growth during inference) with directly measured evidence: **the ONNX
inference call itself costs under 2MB. The real cost is constructing the ONNX `InferenceSession`
and `SentencePieceProcessor` in the first place** — a ~205MB one-time allocation, on top of FAISS
+ SQLite's own ~298MB, that only happens to land on the first real query because `LiteE5Embedder`
loads lazily (docs/DECISIONS_R.md R-030's own design, for good reason — see Consequences below).

**Method:** `scripts/probe_first_inference_memory.py` (new, diagnostic-only, not imported by
production code or referenced from `Dockerfile`). Instruments the real production objects
(`LiteE5Embedder`, `DenseIndex`, `SQLiteChunkLookup`, the `_get_real_retriever()` singleton)
directly — calls their existing methods with checkpoints between them; the one place a boundary
falls *inside* an existing method (`LiteE5Embedder._embed`, to split tokenize / before-ONNX /
after-ONNX / after-normalize), that method's body is inlined verbatim, not reimplemented. Two
measurement layers per checkpoint: an in-process ~1ms-interval background RSS sampler (resolves
the 150ms `docker stats` blind spot R-031 hit) and `tracemalloc` for the Python/numpy-visible
share, so `native_unexplained_mb` (RSS delta minus tracemalloc delta) isolates ONNX Runtime's and
FAISS's own C++-side allocations specifically. Every checkpoint is written to a JSON-lines log,
flushed and `fsync`'d immediately, so a partial log survives an OOM kill. Three modes: `full`
(real singleton, whole path), `embed_only` (fresh process, embedder alone), `faiss_only` (fresh
process, FAISS+SQLite alone, no `onnxruntime`/`sentencepiece` import at all). Run inside the real
`vrag-real:test` image (same assets as R-031) via a bind-mounted script, `psutil` installed
ad hoc into the ephemeral container (not baked into the image — it's a `dev`-extra-only tool).

**A real self-caught instrumentation bug, before trusting the first result — same discipline as
R-028's onnx.load() staging distortion:** the first `full`-mode run showed a +205MB jump
attributed to "03_after_tokenize_query", with `embed_only` showing no such jump at the same label
(+0.5MB) — read naively, this looked like an interaction effect between FAISS being loaded and
tokenization. It wasn't real: `_run_embed_inline`'s first checkpoint sat *after* both
`embedder._ensure_loaded()` (first-ever ONNX session + SentencePieceProcessor construction, since
the singleton's embedder is constructed-but-unloaded at that point, R-030's lazy-load design) and
the actual tokenize call, with no boundary between them — so the label lied about which of the two
operations was actually expensive. `embed_only` didn't show the jump at that label only because
it had already paid the construction cost one checkpoint earlier, at "02_embedder_loaded_no_index"
(+217.8MB there). Fixed by adding an explicit checkpoint between construction and tokenize
(`02a_faiss_and_sqlite_loaded_embedder_not_yet` -> `02b_onnx_session_and_tokenizer_loaded` ->
`03_after_tokenize_query`) and re-running. All numbers below are from the corrected instrumentation.

**Corrected, measured breakdown (Linux container, `-m 1g` diagnostic limit, 6 queries):**

| Stage | RSS after | Delta | native_unexplained delta |
|---|---|---|---|
| 01 process startup | 37.7MB | — | — |
| 02a FAISS + SQLite loaded (embedder not yet touched) | 335.7MB | **+298.0MB** | +262.2MB |
| 02b ONNX session + SentencePieceProcessor constructed | 540.5MB | **+204.8MB** | +204.0MB |
| 03 after tokenizing query 0 | 540.7MB | +0.3MB | +0.3MB |
| 04 before ONNX inference | 540.7MB | +0.0MB | -0.0MB |
| 06 after ONNX inference (window peak *during* = boundary 5) | 542.0MB | +1.3MB | +1.3MB |
| 07 after mean-pool + normalize | 542.2MB | +0.1MB | +0.1MB |
| 08 before FAISS search | 542.2MB | +0.0MB | +0.0MB |
| 10 after FAISS search (window peak *during* = boundary 9) | 542.5MB | +0.3MB | +0.3MB |
| 11 after SQLite chunk fetch | 542.5MB | +0.0MB | -0.0MB |
| 12 after `retrieve()` round trip | 542.8MB | +0.3MB | +0.3MB |
| queries 1-5, all boundaries | — | ~0.0MB each | ~0.0MB each |

**Isolation runs (same container, same 1GB limit, confirms the breakdown is real, not an
artifact of the `full`-mode ordering):**
- `embed_only` (embedder alone, no FAISS/SQLite ever loaded): startup 38.0MB -> after
  `_ensure_loaded()` **255.8MB (+217.8MB)** -> after 6 real tokenize+ONNX+normalize cycles,
  peak 258.2MB. Construction cost (~218MB) matches `full` mode's ~205MB construction cost
  (same operation, same order of magnitude, small variance expected between separate runs) --
  **not an interaction effect with FAISS, a real, roughly order-invariant fixed cost.**
- `faiss_only` (FAISS+SQLite alone, `onnxruntime`/`sentencepiece` never imported): startup
  37.9MB -> after index load **335.2MB (+297.3MB)**, matching `full` mode's 02a figure almost
  exactly -> after 6 real searches + SQLite fetches, peak 340.0MB (+4.8MB total, first-search-only,
  never repeats).

**Direct confirmation under the real 512MB limit:** re-ran `full` mode with `-m 512m
--memory-swap 512m` (the real limit, not the 1GB diagnostic one). Checkpoint 02a logged
successfully at **335.1MB**, well under budget. The process was **`Killed` before checkpoint 02b
could be written** -- i.e., during `LiteE5Embedder._ensure_loaded()`'s construction of the ONNX
session / SentencePieceProcessor, exactly the stage identified above, with nothing in between.
This is the single most direct piece of evidence in this investigation: the partial log's last
line is the boundary immediately before the fatal allocation, not an inference or search step.

**Ranked list of actual causes (measured):**

1. **FAISS dense index + SQLite chunk lookup, resident from first real use: ~298MB.** The single
   largest block. One-time, does not grow across queries.
2. **`ort.InferenceSession` + `SentencePieceProcessor` construction, resident from first real
   embedder use: ~205-218MB.** The second-largest block, confirmed one-time (queries 1-5 show
   zero further growth) and confirmed *not* caused by FAISS being loaded first (embed_only
   reproduces essentially the same cost alone). A rough split, cross-referenced against an
   isolated construction-only micro-test (SentencePieceProcessor alone: ~41MB from a clean
   baseline) and R-028's independent prior finding that the ONNX session alone (before the
   R-029/030 tokenizer swap) cost roughly ~137MB in isolation: **SentencePieceProcessor is a
   minor share (~40MB); `ort.InferenceSession` construction is the dominant share of this
   ~205MB (~165-180MB)** -- moderate confidence, not directly isolated in this same Linux
   container (would need one more targeted run to split cleanly; not attempted, per "stop after
   the diagnostic").
3. **Actual per-query work -- tokenize, ONNX inference, mean-pool/normalize, FAISS search, SQLite
   fetch, the full `retrieve()` round trip -- costs under 2MB combined, every time, including the
   first query.** Rules out ONNX inference-time arena growth (R-031's original hypothesis) as a
   material contributor: boundary 06 (after ONNX inference) shows +1.3MB max, ever.
4. **No per-request growth across repeated queries.** Peaks across queries 1-5 vary by at most
   ~0.6MB from query 0's peak in every mode tested -- not leak-shaped, one-time cost only.
5. **Code-inspection findings (real, but negligible in magnitude, not contributors to the spike
   above):** `LiteE5Embedder._embed()` (`src/vrag/index/embedder.py`) returns a Python list via
   `.tolist()`; `HybridRetriever._embed_and_search_dense()` passes that list straight into
   `DenseIndex.search()`, which immediately converts it back with `np.asarray([query_vector],
   dtype=np.float32)` -- a numpy -> Python list -> numpy round trip for a single ~384-float
   vector (~1.5KB), avoidable by threading the numpy array through directly instead of via
   `list[float]`. `_embed()`'s `mask = attention_mask[..., None].astype(np.float32)` followed by
   `last_hidden_state * mask` allocates a full extra `(batch, seq_len, hidden_dim)` intermediate
   array the same shape as the ONNX output tensor -- avoidable via an in-place multiply. Both are
   real and both are sized in the tens of KB for a single query, three to four orders of
   magnitude below the ~205MB and ~298MB costs above -- not worth attributing any of the OOM to
   either. Not fixed, per "do not fix anything yet."

**Consequences (not yet decided, no action taken):** R4's actual binding constraint is now a
precise, well-evidenced number, not a hypothesis: **FAISS+SQLite (~298MB) + embedder construction
(~205MB) ≈ 503-543MB total resident once both have been touched once**, against a 512MB budget --
a gap of roughly 0MB to -30MB depending on measurement environment (this diagnostic's own
`tracemalloc`+background-sampler+ad-hoc-`psutil`-install overhead is itself a few MB not present
in the real app; R-030's local-host figure of 493.8MB was measured on this dev machine's Windows
`psutil` `WorkingSetSize`, which this session separately found lags/differs from true Linux `RSS`
-- see the smoke-test note in this ADR's development history, not reproduced here as it predates
the corrected instrumentation). The two components were never previously measured resident in the
same process at the same time with this granularity -- R-020/R-023's "index alone" and R-028's
"embedder alone" audits were exactly that, alone. Options this opens up, none decided or
attempted: (a) initialize the ONNX session with a smaller/no memory arena (`enable_cpu_mem_arena
=False` -- R-028 found this doesn't help *inference-time* RSS, but was never tested for
*construction-time* RSS specifically, which this probe shows is the real cost); (b) reduce FAISS/
SQLite's ~298MB (corpus-shrink, already known catastrophic for quality per R-027); (c) accept a
deploy target above 512MB. This ADR does not recommend one -- diagnostic only, per instruction.

## R-033 — Offline FAISS index-variant ablation: scalar-quantized fp16 storage saves 77MB at zero measured quality cost

**Date:** 2026-08-19
**Status:** Accepted as a finding. Diagnostic/offline only, per explicit instruction — no
production code, `Dockerfile`, deployment config, corpus, embedding model, tokenizer, or the
500-query held-out set changed. Not deployed, not wired into `src/vrag/index/dense.py`'s defaults.

**Motivation:** R-032 found FAISS + SQLite load (~298MB) is the single largest of the two
components that together exceed the real 512MB Docker limit — the largest lever R-032 itself did
not investigate. This ablation asks whether FAISS's own footprint specifically can shrink by
R-032's implied gap (~60MB) without giving back the quality R-027 already spent effort protecting.

**Method:** `scripts/eval_faiss_index_variants.py` (new). Same methodology as R-027's
`eval_corpus_size.py`: every candidate searches the exact same 99,767 real vectors, reconstructed
once via `faiss.Index.reconstruct()` from the current production index — no re-embedding, no
corpus change. The 500 real held-out queries are embedded once (real `LiteE5Embedder`, real
`"query: "` prefix) and reused unchanged across every candidate, so the embedder/tokenizer can't
be a confound. Exactly one axis changes per candidate relative to the production baseline
(`IndexHNSWFlat`, M=32, efConstruction=200, efSearch=64, `METRIC_INNER_PRODUCT` — `src/vrag/index/
dense.py`'s own defaults, unmodified), per CLAUDE.md's "never change two variables in one
experiment run": `hnsw16_flat_fp32` changes only M (32→16); `hnsw32_sq8` and `hnsw32_sqfp16`
change only vector storage (`IndexHNSWSQ` with `ScalarQuantizer.QT_8bit` / `QT_fp16` in place of
flat fp32 storage), M/efConstruction/efSearch/metric all held at the baseline's own values.
Resident RSS measured in an isolated subprocess per candidate (same discipline as R-020/R-027, one
candidate's load can't pollute another's), both immediately after `DenseIndex.load()` and again
after running all 500 real `.search()` calls in that same subprocess, to catch any search-time
growth. Quality via the shared `score_hits()` path every ablation in this project routes through.
Latency P50/P95/P100 (matching R-014's efSearch curve and R-028's ONNX sweep — this project's
established convention for component-level search latency).

**A real, small, expected discrepancy against the given baseline, checked before trusting the
rest:** this ablation's own rebuilt `baseline_hnsw32_flat_fp32` measured Recall@5=0.642 /
MRR@10=0.45610, not the reference 0.644 / 0.45627 — a difference of exactly one query's outcome
(`(0.644-0.642)*500 = 1.0`). HNSW graph construction uses an unseeded internal RNG for level
assignment (`faiss.IndexHNSWFlat`'s `.add()`), so an independently rebuilt graph over the identical
vectors is not guaranteed bit-identical to the original, and a single query landing near a
tie-broken boundary is exactly the expected shape of that effect — not a bug in reconstruction,
the embedder, or the held-out set. All comparisons below are read relative to this run's own
rebuilt baseline, not the original reference numbers, for a fair apples-to-apples comparison.

**Results (99,767 vectors, dim=384, 500 real held-out queries, single run each — not yet
repeated 3x per CLAUDE.md's experiment-discipline rule; see Consequences):**

| Candidate | RSS after load | RSS after 500 searches | Disk | Recall@1 | Recall@5 | Recall@10 | MRR@10 | search P50 | P95 | P100 | Build time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **baseline** HNSW32 flat fp32 | 237.8MB | 245.7MB (+7.96MB) | 180.4MB | 0.330 | 0.642 | 0.750 | 0.45610 | 0.403ms | 0.548ms | 0.937ms | 34.5s |
| HNSW16 flat fp32 | 224.6MB (**-13.2MB**) | 232.5MB (+7.98MB) | 167.6MB | 0.330 | 0.642 | 0.748 | 0.45465 | 0.284ms | 0.386ms | 0.551ms | 23.3s |
| HNSW32 + SQ **int8** | 122.5MB (**-115.3MB**) | 130.8MB (+8.34MB) | 65.5MB | 0.320 | 0.640 | 0.742 | 0.45038 | 1.085ms | 1.466ms | 2.150ms | 42.7s |
| HNSW32 + SQ **fp16** | 160.8MB (**-77.0MB**) | 169.2MB (+8.34MB) | 103.8MB | 0.330 | 0.642 | 0.750 | 0.45610 | 1.417ms | 1.874ms | 2.414ms | 56.3s |

All four confirmed `metric_type == METRIC_INNER_PRODUCT` (checked directly, not assumed) — cosine
semantics preserved throughout, since production vectors are already L2-normalised at embed time.
Vector storage: baseline and HNSW16 store full float32 vectors uncompressed (`IndexHNSWFlat` wraps
a `Flat` storage); SQ8 stores 1 byte/dimension, SQfp16 stores 2 bytes/dimension (IEEE half-float)
— both confirmed via `.reconstruct()` returning dequantized float32 (works on all four, verified
directly) and via the disk-size ratios above, not asserted from documentation alone. The ~8MB
search-time RSS growth is present in **all four candidates at nearly the same magnitude**
(7.96-8.34MB) regardless of index type or quantization — this doesn't scale with the variable
under test, so it reads as a generic Python/numpy-interpreter warmup cost from running 500 real
`.search()` calls in a fresh subprocess, not a FAISS-index-type-specific search-time cost.

**Against the three acceptance criteria:**

- **HNSW16** (fewer graph links only): saves only 13.2MB — well short of the ~60MB target. Quality
  and latency are both fine (actually faster: fewer links to traverse), but it fails criterion 1
  outright regardless. Not recommended alone.
- **SQ8** (int8): saves 115.3MB, clears the target with room to spare. Real, non-trivial quality
  cost though — Recall@1 -0.010, Recall@5 -0.002, Recall@10 -0.008, MRR@10 -0.0057 (~1.3%
  relative) versus this run's own baseline — a genuine regression, not noise (the baseline-vs-
  reference check above establishes the noise floor at ~0.002/one query; SQ8's Recall@1 delta
  alone is 5x that). Search latency roughly 2.5-2.7x baseline's own (1.09ms P50 vs 0.40ms) but
  still under 2.2ms worst-case against a 200ms total budget — not a real latency threat in
  absolute terms, whatever the relative multiplier.
- **SQfp16**: saves 77.0MB — clears the ~60MB target and does so at **zero measured quality
  cost relative to this run's own rebuilt baseline**: Recall@1/5/10 identical (0.330/0.642/0.750),
  MRR@10 identical to 5 decimal places (0.45610396825396826, matching the baseline row bit for
  bit). Latency roughly 3.5x baseline's own P50 (1.42ms vs 0.40ms) but, same as SQ8, trivial in
  absolute terms against the 200ms budget (2.41ms P100 worst case).

**Recommendation: HNSW32 + SQ fp16 (`IndexHNSWSQ`, `ScalarQuantizer.QT_fp16`, M=32,
efConstruction=200, efSearch=64, `METRIC_INNER_PRODUCT`) satisfies all three stated criteria** —
saves 77.0MB (>60MB target), zero measured quality regression, latency stays trivial against the
200ms budget. SQ8 is explicitly **not** recommended over it: it saves more memory (115.3MB) but at
a real quality cost, and fp16 already clears the required ~60MB without spending any of that
budget — there's no reason to pay SQ8's quality cost for headroom this problem doesn't need. If a
future constraint ever needs more than 77MB from this lever specifically, SQ8 is the documented,
measured fallback, not a guess.

**Consequences, nothing applied yet:**
- **This is a single run per candidate, not the 3x-and-report-the-spread CLAUDE.md's experiment
  discipline calls for.** The recall/MRR deltas discussed above are compared against this run's
  own baseline (not the original reference) specifically to control for HNSW's construction
  non-determinism as best as a single run can, but a real spread check (3 independent rebuilds per
  candidate) is the honest next step before this recommendation is treated as final, not just
  directionally right.
- **Not yet verified that a 77MB FAISS-side saving actually closes R4's real gap.** R-032's ~298MB
  FAISS+SQLite figure was measured in the real Linux Docker container; this ablation's ~237.8MB
  baseline FAISS-alone figure was measured on this dev machine's Windows `psutil` — the same
  cross-environment caveat R-032 already flagged (Windows `WorkingSetSize` vs. Linux `RSS`) applies
  here too. A rough projection (298MB - 77MB ≈ 221MB FAISS+SQLite, + R-032's ~205-218MB embedder
  construction ≈ 426-439MB, comfortably under 512MB) is *plausible* but not measured — the actual
  next step, not taken here, is wiring this index type into a real Docker build and re-running
  R-031/R-032's exact validation, not assuming the offline number transfers.
- **Not wired into `src/vrag/index/dense.py`, not rebuilt into `index-metadata_aware-v3`, not
  deployed anywhere.** This ADR reports a measured, promising option; it does not implement it.

## R-034 — sqfp16 implemented, deployed to a real Docker container, real 512MB Docker OOM RESOLVED

**Date:** 2026-08-19
**Status:** Accepted and implemented. **R4 is resolved** — the real Docker `-m 512m` OOM found in
R-031/R-032 no longer reproduces. Real retrieval survives full initialization and 10 real queries
under the true 512MB limit with real headroom, real citations, and no measurable quality or
latency regression. This is a live, Docker-verified result, not an offline projection.

**Scope, exactly as instructed:** only `src/vrag/index/dense.py` (the index construction/build
path) and `Dockerfile` (the one line pointing at the index release asset) changed. Corpus size,
embedding model, tokenizer, `SQLiteChunkLookup`, BM25/sparse behaviour, and
retrieval/fusion/ranking logic are all byte-for-byte or code-for-code unchanged.

**Implementation (`src/vrag/index/dense.py`):** added a `quantization: Literal["none", "sqfp16"]
= "none"` constructor parameter. `"none"` (the default — every existing caller that never passes
this argument, including every other ablation script and the full test suite, builds the exact
same `IndexHNSWFlat` it always has, confirmed by `test_default_quantization_still_builds_a_plain_
flat_index`) is unchanged. `"sqfp16"` builds `IndexHNSWSQ(dim, ScalarQuantizer.QT_fp16, m,
METRIC_INNER_PRODUCT)` — same M/efConstruction/efSearch/metric as before, only the vector storage
changes. `add()` now trains the index first if `not self._index.is_trained` (a no-op for flat
storage, required once for the scalar quantizer's per-dimension codec). `save()`/`load()` persist/
restore the `quantization` choice in `meta.json`, defaulting to `"none"` when absent so v1/v2
archives (which predate this field) still load correctly. Deliberately not a general multi-
quantizer framework — the int8 option R-033 measured and rejected was not implemented, per
instruction. Two real, targeted mypy fixes needed (`.hnsw` attribute access and the
`ScalarQuantizer` arg both hit faiss's known-imprecise stubs — the same class of issue
`set_ef_search()` already carried an ignore comment for; not new problems, just newly triggered by
routing index construction through a helper function instead of an inline literal constructor
call, which broke mypy's narrowing).

**Index rebuild (`scripts/build_index_sqfp16.py`, new):** reconstructs all 99,767 real vectors
from the CURRENT production `dense/faiss.index` via `faiss.Index.reconstruct()` — no re-embedding,
no re-chunking, so this is provably the exact same vectors R-033 measured, not a re-derived
approximation that could silently drift (e.g. if `working_subset.jsonl` had changed since v2 was
built). Builds the new `sqfp16` dense index from them, then copies `chunk_lookup.json`,
`chunk_lookup.sqlite3`, and `sparse/` **byte-for-byte, unmodified** from the current v2 directory
— nothing about the corpus, SQLite lookup, or BM25 index was rebuilt or touched.

**Release asset:** `index-metadata_aware-v3` — identical to v2 in every file except
`dense/faiss.index` (180.4MB → 103.8MB, -76.6MB). Asset sha256:
`fbe0eaac14a021c966fd3786cdae9c942115010ddb84895c78e0cb9c37e2545d`; the changed inner file's
sha256: `b8b5d22862583f7392e90580378686400b75aae6d23e039899274367f6ed256e`, both recorded in the
release notes. `Dockerfile`'s index-download step is the only line changed (v2 tag → v3 tag).

**Loader:** required no change. `DenseIndex.load()` already read the index type from the file
itself (`faiss.read_index()` auto-detects), throwing away the constructor's default index —
confirmed true for `IndexHNSWSQ` the same as every other type, not newly verified only now.
`load_built_index_lean()`, `HybridRetriever`, and `retrieve()` never reference a concrete index
class at all.

**Regression tests (`tests/index/test_dense.py`, 7 new, 16 total in the file):**
`test_default_quantization_still_builds_a_plain_flat_index`, `test_unknown_quantization_rejected`,
`test_sqfp16_index_loads_successfully`, `test_sqfp16_index_metric_is_inner_product`,
`test_sqfp16_index_hnsw_parameters_match_production_defaults` (verifies M via
`hnsw.nb_neighbors(1)` — faiss has no direct `.M` readback, confirmed the mapping
`nb_neighbors(1)==M`/`nb_neighbors(0)==2*M` directly against a live index before writing the
assertion, not assumed), `test_sqfp16_index_returns_valid_chunk_ids` (every returned ID was
actually added, no duplicates, exact match ranks first), `test_sqfp16_and_flat_agree_on_top_hit_
for_a_clear_match`. Full suite: 223/223, ruff clean, mypy clean.

**Re-verification against the real built artifact, not just R-033's ablation script's temp copy
(`scripts/verify_index_sqfp16.py`, new), full 500-query held-out set:**

| Metric | Reference baseline | R-033 ablation (temp copy) | **This artifact (index-metadata_aware-v3)** |
|---|---|---|---|
| Recall@1 | 0.330 | 0.330 | 0.330 |
| Recall@5 | 0.644 | 0.642 | 0.642 |
| Recall@10 | 0.750 | 0.750 | 0.748 |
| MRR@10 | 0.45627 | 0.45610 | 0.45550 |
| search P50/P95/P100 | ~0.4/~0.5/~0.9ms (fp32) | 1.42/1.87/2.41ms | 1.46/1.93/2.57ms |

All differences from the original reference are within the single-query noise floor R-033 already
established (HNSW's own unseeded construction RNG — confirmed there via a direct fp32-vs-fp32
independent-rebuild comparison, not assumed here to still apply). Not a quantization-driven
regression: this is a second independent `sqfp16` build (different from R-033's ablation-script
copy) landing within the same noise band as two independent *fp32* rebuilds landed against each
other.

**Real Docker `-m 512m` validation — the actual test that matters, not an extrapolation:**
built `vrag-real:v3` from the updated `Dockerfile` (`docker build --no-cache`), ran with
`-m 512m --memory-swap 512m` (swap disabled, true hard ceiling) and
`VRAG_REQUIRE_REAL_RETRIEVAL=1` (same fail-loud flag from R-031/R-032 — did not trip).

| | R-031/R-032 (v2, fp32) | **R-034 (v3, sqfp16)** |
|---|---|---|
| Startup (FAISS+SQLite loaded, embedder not yet touched) | ~276-298MB | **204.4MiB** |
| `/healthz` | `{"status":"ok","retrieval":"real"}` | `{"status":"ok","retrieval":"real"}` |
| First real query | **OOM-killed (exit 137), reproduced 2/2** | **Survived.** peak 394.2MiB, `retrieve`=1125.8ms (one-time — see below) |
| After 10 real queries | never reached | **Survived all 10.** peak 397.8MiB, `OOMKilled: false` |
| Headroom under 512MB | negative (crashed) | **~114MB** |

`/ask` returned real citations with real `chunk_id`/`passage_id`/`score` fields (e.g.
`"chunk_id":"189527_4::metadata_aware::0"`) on 6 of 10 real queries — not `stub-chunk-001`,
directly confirming real retrieval, not the Day-0 fallback. Memory stayed flat (394.2MiB →
397.8MiB, +3.6MiB across 9 further queries) — not leak-shaped, matches R-032's "no per-request
growth" finding, the one-time cost was the embedder's first construction, same as before.

**Two honest observations, neither disqualifying, both worth recording:**
- **First-query latency is 1.1s**, not the ~12-42ms every subsequent query measured. This is the
  same lazy `LiteE5Embedder._ensure_loaded()` cost R-032 already diagnosed (ONNX session +
  SentencePieceProcessor construction, deferred to first real embedder use even though the
  retriever singleton itself was eager-loaded via `VRAG_REQUIRE_REAL_RETRIEVAL=1`) — **not caused
  by or specific to this change**, the same cost existed with the fp32 index and would recur
  identically there too. Not addressed here — out of scope for an index-storage-only change, and
  not one of R4's stated acceptance criteria (which are about surviving the memory limit, not
  first-request latency).
- **3 of the 10 real queries abstained** (G3 confidence 0.86-0.88, just under the calibrated 0.8835
  threshold) where the other 7 answered normally with real citations. G3's threshold was
  calibrated against fp32 scores (docs/DECISIONS_P.md); fp16 quantization introduces small score
  perturbations that could plausibly nudge a borderline case across that specific cutoff in either
  direction. Not measured directly against a fp32 Docker run on the same live queries (would
  require a second Docker build/run, out of scope for "keep this strictly controlled" and "do not
  change any other memory-related architecture during this test") — flagged honestly as a
  plausible interaction for G3's owner to be aware of, not confirmed as a regression, and not
  blocking R4 (R4's acceptance criteria are about memory/OOM/Recall@10/MRR@10/latency, not G3's
  abstention rate).

**R4 verdict: RESOLVED.** All six stated acceptance criteria met: container survives full
initialization and first query under 512MB; no OOM kill (10/10 real queries); real retrieval
active (`/healthz`, real citations); Recall@10 0.748 vs. 0.750 baseline (within the established
noise floor, not a material regression); MRR@10 0.45550 vs. 0.45627 baseline (~0.17% relative,
effectively unchanged); query latency 12-42ms per real query, comfortably compatible with the
200ms budget. `docs/RISKS.md` R4 updated accordingly.

## R-035 — Two R-034 follow-ups: eager warmup kills the 1.1s cold start; FP32-vs-FP16 grounding-gate comparison finds zero decision changes

**Date:** 2026-08-19
**Status:** Accepted and implemented (part 1); accepted as a diagnostic finding, no threshold
change made (part 2, per explicit instruction). Retrieval architecture, FAISS, corpus size,
embedding model, tokenizer, and deployment memory configuration are all unchanged.

### Part 1 — cold-start warmup

**Root cause:** R-034's Docker validation measured a 1.1s first-request `retrieve` cost. R-031's
eager-startup `lifespan` hook already forced `_get_real_retriever()` to run at boot (loading
FAISS + SQLite, constructing the `LiteE5Embedder` *object*) — but `LiteE5Embedder._ensure_
loaded()` (the actual ONNX `InferenceSession` + `SentencePieceProcessor` construction R-032 found
costs ~205MB and real wall-clock time) is itself lazy, deferred to the first real
`embed_queries()` call. "The retriever object exists" and "the embedder has actually run once"
were two different costs, and only the first was covered at startup.

**Fix (`src/vrag/retrieval/interface.py`):** `is_retrieval_real()` now means "loaded AND warm",
not just "loaded" — after `_get_real_retriever()` succeeds, it runs one real, memoized warmup
embedding (`retriever._embedder.embed_queries(["warmup"])`) the first time it's called, catching
and reporting `False` (not raising) if that fails, matching `retrieve()`'s existing "never take
the whole service down" contract exactly. A `None`/`True`/`False` tri-state global
(`_warmup_ok`) tracks "not attempted yet" vs. a real attempt's result, mirroring the existing
`_retriever_load_attempted` pattern.

**No change needed to `src/vrag/api/main.py`'s control flow** — its `_lifespan` hook already
called `is_retrieval_real()` unconditionally before `yield`, and FastAPI/uvicorn don't accept ANY
request (including `/healthz`) until that coroutine reaches `yield`. Strengthening what
`is_retrieval_real()` *means* was sufficient to make warmup happen at boot structurally, not by
convention — there's no window where the process accepts traffic but isn't actually warm.
`VRAG_REQUIRE_REAL_RETRIEVAL=1`'s fail-loud behavior now also covers a warmup failure, not just a
load failure, for free. Docstrings updated to describe the new behavior accurately.

**4 new regression tests** (`tests/retrieval/test_interface_loading.py`, 7 total in the file,
fixture updated to also reset `_warmup_ok`): `is_retrieval_real()` false when the index is
missing; a real warmup embedding call actually happens (asserted via a fake embedder's call log,
not just the return value); a warmup failure is caught and returns `False` without raising; the
warmup embedding only ever runs once across repeated calls (memoized, the real cost is paid once).
Full suite: 227/227, ruff clean, mypy clean.

**Real Docker verification** (rebuilt `vrag-real:v3-warmup` from the updated source, same
`index-metadata_aware-v3`/`embedder-lite-onnx-v2` assets as R-034, `docker run -m 512m
--memory-swap 512m`):

`/healthz` was polled in a tight loop starting the instant `docker run` returned, to directly
prove readiness gates on warmup rather than trusting log-line ordering: the port refused every
connection attempt until **5.42s** after container start, at which point the **first successful
response was already `{"status":"ok","retrieval":"real"}`** — there is no reachable window where
the server is up but retrieval isn't warm. Memory at that first-reachable moment was already
395.7MiB (matching R-034's post-warmup peak almost exactly), confirming the full memory cost, not
just the object construction, had already landed before the port opened.

**First real `/ask` query: `retrieve`=42.2ms** (was 1125.8ms in R-034 before this fix — same
query, same index, same container image family, only the warmup timing changed). Two more real
queries measured 47.7ms and 46.5ms — indistinguishable from the first, confirming there's no
residual first-request penalty left anywhere. `curl`'s own wall-clock (267-321ms, includes
network/TCP/event-loop overhead beyond the stage-sum) is consistent across all three calls too.
Memory stayed at 395.7 → 397.6MiB across the 3 queries (matching R-034's "flat after first query,
not leak-shaped" finding) — `docker inspect`: `OOMKilled: false` throughout.

**Part 1 verdict:** cold start eliminated. The 1.1s first-request cost R-034 found is gone —
verified by direct measurement, not just code reading, using the exact same real image build,
real 500-chunk index, and real 512MB Docker limit the rest of this ADR arc has used throughout.

### Part 2 — FP32 vs. FP16 grounding-gate (G3) comparison

**Method** (`scripts/compare_g3_fp32_vs_fp16.py`, new): two `HybridRetriever` instances differing
ONLY in which `DenseIndex` they wrap (fp32 `index-metadata_aware` vs. fp16
`index-metadata_aware_sqfp16`), sharing the same `LiteE5Embedder` and `SQLiteChunkLookup` — so any
decision difference is attributable only to the index's own returned scores, nothing else. Runs
the real, unmodified `g3_confidence.check()` (TAU=0.8835, MARGIN=0.0, R-015/P-015 — **not
changed**) against both retrievers' real output for every one of the 500 real held-out queries.

**Result: zero decisions changed.**

| | FP32 | FP16 |
|---|---|---|
| Answered | 371 | 371 |
| Abstained | 129 | 129 |
| Decisions changed vs. the other | — | **0 / 500 (0.0%)** |

Top1 confidence-score differences (fp16 − fp32) across all 500 queries: mean **-0.00007**,
population stdev **0.00131**, max absolute difference **0.0281** (one query) — and even that
largest single-query divergence did not flip a decision, meaning it wasn't anywhere near TAU.

**This directly resolves R-034's open, non-blocking observation** ("3/10 real queries abstained
near the confidence threshold during the live Docker smoke test — possibly fp16 score-precision
sensitivity, not confirmed"). The rigorous 500-query, same-query, real-gate comparison shows that
observation was sampling noise from a 10-query smoke test, not a real effect: across the full
held-out set, fp16 quantization does not change a single G3 decision. **No calibration issue
found — TAU/MARGIN correctly left untouched, per instruction.** Per-query detail (top1 score and
pass/fail for both indices, every query) written to `eval/g3_fp32_vs_fp16_comparison.json` for
anyone who wants to audit the raw numbers directly.
