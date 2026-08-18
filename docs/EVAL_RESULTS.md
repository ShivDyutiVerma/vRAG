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
| passage_native | 0.322 | 0.650 | 0.750 | 0.452 | 0.517 | 99,767 | 44 min |
| fixed_overlap (size=256, overlap=0.2) | 0.322 | 0.650 | 0.750 | 0.452 | 0.521 | 101,008 | 46 min |
| metadata_aware | 0.322 | **0.654** | **0.754** | **0.454** | 0.519 | 99,767 | 39 min |
| hierarchical (child=128, parent=512) | 0.318 | 0.640 | 0.742 | 0.446 | 0.516 | 103,907 | 41 min |
| semantic (percentile=90) | 0.318 | 0.644 | 0.748 | 0.448 | 0.514 | 101,308 | 96 min |
| sentence_window (window=2) | 0.318 | 0.478 | 0.570 | 0.379 | ~~0.881~~ invalid¹ | 390,288 | 141 min |

¹ `sentence_window`'s nDCG@10 was computed with a bug (fixed 2026-08-18, `scripts/eval_chunking.py`):
because it produces ~3.9 chunks per passage on average, the same relevant passage frequently appears
multiple times in one query's top-10, and the original nDCG implementation summed credit per
*occurrence* instead of per unique passage — Recall@k and MRR@10 were unaffected (both already
dedupe by nature), only nDCG was inflated. Not re-run (141min cost) since it doesn't change the
ranking — `sentence_window` is already clearly the worst performer on Recall@5, the metric that
matters for picking a winner.

### Analysis

- **`sentence_window` is a clear loser on this corpus.** Recall@5 = 0.478 vs. 0.64–0.65 for
  everything else — a real, large gap, not noise. Splitting already-short passages (p50 = 57 words,
  §R-003) down to individual-sentence retrieval units loses too much context for the embedder to
  match queries correctly; the 3.9x chunk-count blowup (390k vs ~100k) also makes it the most
  expensive to build and would slow dense search proportionally at query time.
- **The other five are within ~1.4 percentage points of each other on Recall@5** (0.640–0.654) —
  inside noise-floor territory (`docs/BUILD_PLAN.md` P2 guard: "declaring a winner that's inside the
  noise band" is the trap to avoid). This matches `TECH_MENU.md`'s prediction that passage-native or
  hierarchical would likely win on this corpus, and is unsurprising given the passage-length stats
  (p50=57, p95=115 words) — most passages are already short enough that fixed-size windowing,
  metadata tagging, or small-to-big splitting barely change what gets embedded.
- **`metadata_aware` produces the exact same chunk boundaries as `passage_native`** (99,767 chunks,
  identical text per chunk) but tags every chunk with `language`/`source_lang`/`query_type` at zero
  extra chunk-count or build-time cost — its Recall@5/10 and MRR are marginally the highest of the
  six (within the noise band, so not a claimed win on quality alone), and it strictly dominates
  `passage_native` by being free extra metadata for later retrieval filtering/boosting
  (`TECH_MENU.md` S5 #5 — flagged as the most dataset-specific strategy, worth highlighting).

### Decision — provisionally shipping `metadata_aware`

**Reasoning:** given the top five strategies are statistically tied on Recall@5, `docs/BUILD_PLAN.md`
P2's own guidance is "ship the cheapest one and say they were tied" — `metadata_aware` costs nothing
over `passage_native` (same chunks, same embedding cost, actually the fastest build of the six at 39
min) and adds metadata that's free to use later (language filtering, query-type-aware boosting) if
G3/A3 experiments want it. Marked **provisional**, not final, because two things from
`docs/EVAL_PROTOCOL.md`'s protocol haven't happened yet: (1) the winning config hasn't been run 3x to
establish a real noise floor — the "0.654 is marginally highest" observation is a single run, and
given the ~1.4pt spread across five strategies, a noise-floor check could easily show they're
genuinely indistinguishable; (2) no hyperparameter sweep (e.g. `fixed_overlap`'s overlap ∈
{0, 0.1, 0.2}) has run yet. Both are worth doing before Phase 7 lock-in, not before wiring
`retrieve()` — this decision is good enough to unblock Workstream P now, and can be revisited without
much cost if the noise-floor run says otherwise (see `docs/RISKS.md` if that happens).

### What's not done yet

- Noise-floor validation (run `metadata_aware` 3x, report the spread) — deferred, see above
- A2 (embedder comparison), A3 (retrieval mode: dense vs sparse vs hybrid), A4 (rerank) — next
- Overlap sub-study for `fixed_overlap` (0/10%/20%) — not run, strategy wasn't the leading candidate
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
