# EVAL_RESULTS.md

> Shared file, split by section: §1–3 (chunking / embedding / retrieval) is Workstream R's to write,
> §4–6 (generation / guardrails / latency) is Workstream P's. Different sections of one file rarely
> conflict if everyone stays in their lane (`docs/TEAM_SPLIT.md` §3). **Every number below must trace
> to a row in `eval/ablation_ledger.csv` or a `traces.jsonl` run — no number gets written here first.**

## §1 — Chunking (A1)

**Setup:** embedder = `multilingual-e5-small` (PyTorch, not yet ONNX-quantised), retrieval mode =
dense-only, no rerank — held fixed per the staged-ablation design (`TECH_MENU.md` §A). Working pool:
10,000 Hindi queries / 99,767 translated passages (`data/working_subset.jsonl`,
`docs/DECISIONS_R.md` R-003). Held-out eval set: the frozen 500 query→passage pairs
(`eval/heldout_queries.json`, `docs/DECISIONS_R.md` R-002). One run per strategy, default config
(no per-strategy hyperparameter sweep yet — see "What's not done" below).

| Strategy | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Chunks | Index build |
|---|---|---|---|---|---|---|---|
| passage_native | 0.322 | 0.650 | 0.750 | 0.452 | 0.517 | 99,767 | 44 min (CPU) |
| fixed_overlap (size=256, overlap=0.2) | 0.322 | 0.650 | 0.750 | 0.452 | 0.521 | 101,008 | 46 min (CPU) |
| **metadata_aware** (n=3, see noise floor below) | 0.322 | **0.653 ± 0.001** | **0.752 ± 0.002** | **0.453 ± 0.001** | 0.518 ± 0.001 | 99,767 | 39 min (CPU) / 3.4 min (GPU) |
| hierarchical (child=128, parent=512) | 0.318 | 0.640 | 0.742 | 0.446 | 0.516 | 103,907 | 41 min (CPU) |
| semantic (percentile=90) | 0.318 | 0.644 | 0.748 | 0.448 | 0.514 | 101,308 | 96 min (CPU) |
| sentence_window (window=2) | 0.310 | 0.552 | 0.554 | 0.405 | 0.436 | 390,288 | 14 min (GPU)¹ |

¹ `sentence_window` was re-run after a metric bug fix (below) and after switching to GPU embedding
(`docs/DECISIONS_R.md` R-005) — its original CPU run took 141 min, this corrected run took 14.3 min.

**Metric bug found and fixed (2026-08-18, `scripts/eval_chunking.py`):** the original `nDCG@10`
implementation summed credit per chunk *occurrence* in the top-k instead of per unique passage.
Strategies producing multiple chunks per passage (`sentence_window` averages ~3.9 chunks/passage)
could have the same relevant passage occupy several slots in one query's top-10, inflating its
score — `sentence_window`'s original nDCG@10 read an impossible 0.881 despite the worst Recall@5 of
the six. Turned out this also affected **Recall@5 itself**, not just nDCG: slicing the raw
(non-deduped) hit list to the top-5 let duplicate chunks from one passage crowd out genuinely
distinct passages that would otherwise have appeared within the top-5 window. Fixed by deduplicating
retrieved chunks down to unique passages *before* any metric slices to `k` — re-running
`sentence_window` afterward moved its Recall@5 from 0.478 to the corrected 0.552 (still clearly the
worst of the six). The other five strategies produce close to 1 chunk per passage each, so the bug
barely affected them and their originally-reported numbers stand.

### Noise floor (`docs/BUILD_PLAN.md` P2 task: "run the winner's config 3x")

`metadata_aware` run 3 times (full 99,767-chunk rebuild each time, GPU embedding):

| Run | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Index build |
|---|---|---|---|---|---|
| 1 (original, CPU) | 0.654 | 0.754 | 0.4539 | 0.5190 | 2320.7s |
| 2 (GPU) | 0.652 | 0.750 | 0.4526 | 0.5171 | 203.5s |
| 3 (GPU) | 0.654 | 0.752 | 0.4535 | 0.5182 | 203.5s |
| **Spread** | **0.2pp** | **0.4pp** | **0.0013** | **0.0019** | — |

The noise floor is tight — 0.2 percentage points on Recall@5, run-to-run, almost certainly from
FAISS HNSW's multi-threaded-insertion build randomness (chunking, embedding, and the dataset are all
deterministic; nothing else varies between runs). This matters for the strategy comparison above:
`metadata_aware`/`passage_native`/`fixed_overlap` (0.650–0.654) sit within their own noise band of
each other, but their gap to `hierarchical` (0.640) and `semantic` (0.644) — roughly 1.0–1.4
percentage points — is **5–7x the measured noise floor**. That gap is real, not noise; the earlier
"all five are statistically tied" framing undersold it.

### Analysis

- **`sentence_window` is a clear loser on this corpus**, even after the metric fix: Recall@5 = 0.552
  vs. 0.64–0.654 for everything else. Splitting already-short passages (p50 = 57 words, §R-003) down
  to individual-sentence retrieval units loses too much context for the embedder to match queries
  correctly; the 3.9x chunk-count blowup (390k vs ~100k) also makes it the most expensive to build
  and would slow dense search proportionally at query time.
- **`passage_native`, `fixed_overlap`, and `metadata_aware` are genuinely tied** (within their own
  ~0.2–0.4pp noise floor) — but this trio is measurably, not just marginally, ahead of `hierarchical`
  and `semantic` given the noise-floor comparison above. This matches `TECH_MENU.md`'s prediction
  that passage-native-shaped strategies would likely win on this corpus, and is unsurprising given
  the passage-length stats (p50=57, p95=115 words) — most passages are already short enough that
  fixed-size windowing or metadata tagging barely change what gets embedded, while small-to-big
  splitting (`hierarchical`) and per-topic sentence grouping (`semantic`) both restructure the text
  enough to cost a real, measurable amount of recall.
- **`metadata_aware` produces the exact same chunk boundaries as `passage_native`** (99,767 chunks,
  identical text per chunk) but tags every chunk with `language`/`source_lang`/`query_type` at zero
  extra chunk-count or build-time cost — it strictly dominates `passage_native` by being free extra
  metadata for later retrieval filtering/boosting (`TECH_MENU.md` S5 #5 — flagged as the most
  dataset-specific strategy, worth highlighting).

### Decision — shipping `metadata_aware`

**Status: accepted, no longer provisional** (`docs/DECISIONS_R.md` R-004, updated after the
noise-floor run above). `metadata_aware` is tied with `passage_native`/`fixed_overlap` within noise,
and measurably ahead of `hierarchical`/`semantic` beyond noise — ships as the production strategy
because it costs nothing over the cheapest tied option and adds metadata other stages can use for
free. Already wired into `retrieve()` via `HybridRetriever` (`src/vrag/retrieval/hybrid.py`).

### What's not done yet

- Hyperparameter sweep for `fixed_overlap` (overlap ∈ {0, 0.1, 0.2}) — not run; strategy is tied with
  the winner already, so this is a low-priority nice-to-have, not a blocker for anything
- A2 (embedder comparison), A3 (retrieval mode: dense vs sparse vs hybrid), A4 (rerank) — next
- efSearch recall-vs-latency curve (§3, `docs/assets/efsearch_curve.png`) — Phase 3 task, not started

## §2 — Embedding (A2)

**Setup:** chunking = `metadata_aware` (A1 winner, §1), dense-only retrieval, no rerank — one
variable (embedder) at a time per the staged-ablation design. Same 99,767-chunk working pool,
same 500-query held-out set. 4 candidates per `docs/TECH_MENU.md` §S4 / `docs/BUILD_PLAN.md` P3.

| Embedder | Dim | Recall@1 | Recall@5 | Recall@10 | MRR@10 | p50 query-embed | Index build |
|---|---|---|---|---|---|---|---|
| **multilingual-e5-small** (A1 baseline) | 384 | 0.322 | **0.653 ± 0.001** (n=3) | 0.752 | 0.453 | not measured¹ | 203s (GPU) |
| potion-multilingual-128M | 256 | 0.084 | 0.266 | 0.334 | 0.160 | **0.71ms** | 85s (GPU) |
| vyakyarth | 768 | 0.092 | 0.274 | 0.370 | 0.169 | 13.23ms | 769s (GPU) |
| bge-m3 | 1024 | — | — | — | — | — | excluded² |

¹ A1's runs (§1) predate `p50_embed_ms` being added to `eval_chunking.py` — added specifically for
A2 per `docs/BUILD_PLAN.md` P3's "record both quality AND query-embed latency per model." A
same-config re-run to backfill this one number stalled after ~10 minutes with no forward progress
(GPU state left over from other same-session runs, most likely) and was killed rather than chased
further — low value for the time cost, since e5-small's massive quality lead already settles A2
regardless of its own embed latency (which was ~2.5ms on CPU per published benchmarks, and is not
the hot-path number that matters anyway — Phase 6 will measure the real ONNX-quantised figure).

² **BGE-M3 excluded on practicality, not quality.** Took ~1 hour on this machine's 6GB laptop GPU
(RTX 3060) before being killed — confirmed not hung (100% GPU utilization, checked repeatedly over
the full hour) — vs. 1-13 minutes for the other three. Even after reducing `batch_size` from the
library default of 32 to 8 to avoid a `CUDA out of memory` error that hit twice at the default
(`docs/DECISIONS_R.md` R-008), it remained far slower than expected for its parameter count (568M —
smaller than some other tested models complete faster). Likely cause: `sentence-transformers`
loading BGE-M3 in its default configuration may compute the sparse and multi-vector representations
alongside dense even though only dense output is used here — not confirmed, not worth the
investigation time given the outcome (exclusion) doesn't change either way.

### Analysis

- **`multilingual-e5-small` wins decisively, not marginally.** Recall@5 of 0.653 vs. 0.266 (potion)
  and 0.274 (Vyakyarth) is not a close call requiring a noise-floor check the way A1's top strategies
  needed one — the gap (38+ percentage points) is enormous relative to any plausible run-to-run
  variance.
- **`potion-multilingual-128M`'s quality collapse confirms `docs/TECH_MENU.md` §S4's own caveat**,
  measured rather than assumed: "Static embeddings lack contextualisation — that's the quality
  tradeoff." Its speed is real (0.71ms vs. whatever e5-small's turns out to be) but a 39-point
  Recall@5 loss is not a trade worth making for a system whose hot-path embed cost is already
  expected to be single-digit milliseconds after ONNX quantisation (Phase 6) — the "expensive"
  option isn't actually expensive enough for this trade to make sense.
- **`vyakyarth` — "the Indic-specialist wildcard" — underperforming a general-purpose model is a
  genuine, non-obvious finding, not a wiring bug.** Verified before concluding this: correct output
  dimension (768, matches its published spec), correctly L2-normalised vectors, no missing
  instruction prefix (confirmed against its model card — it needs none). The likely explanation:
  being trained for Indic language understanding/similarity tasks doesn't automatically confer
  strength at open-domain passage *retrieval* specifically, which is a distinct skill E5's
  large-scale contrastive training (on MS MARCO-style retrieval pairs, at a scale Vyakyarth's
  smaller training run likely didn't match) optimises for directly.
- **BGE-M3's exclusion is itself evidence, not just a gap.** A model that requires disproportionate
  compute for an *offline, one-time* index-build cost is a genuine engineering red flag even before
  asking about its quality — `AGENT_BUILD_SPEC.md` §3.2 explicitly scopes index construction as a
  cost that still has to fit within a 5-day hackathon build budget, not an unconstrained one.

### Decision — keeping `multilingual-e5-small`

A1's default is confirmed, not merely left unchallenged: three real alternatives were built,
smoke-tested, and run against the same frozen eval set, and none came close. No ADR needed to
"switch back" to e5-small since it was never actually replaced — this ran as a genuine ablation
stage, and the incumbent won on its merits.

## §3 — Retrieval mode + reranking (A3, A4)

_Not run yet. Will include the efSearch recall-vs-latency curve (`docs/assets/efsearch_curve.png`)._

## §4 — Generation (A5)

_Not run yet — Workstream P._

## §5 — Guardrails

_Not run yet — Workstream P + joint G3/G4 calibration. Will include
`docs/assets/g3_calibration.png`._

## §6 — Latency (P50/P70/P100)

_Not run yet — Workstream P, Phase 6. Will include `docs/assets/latency_breakdown.png` and
`docs/assets/latency_cdf.png`._
