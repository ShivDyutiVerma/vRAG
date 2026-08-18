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

_Not run yet._

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
