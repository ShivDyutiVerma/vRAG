# TESTING_SUMMARY.md

> One place that shows everything tested and validated, and what each result was used to decide.
> This is the showcase/decision-trail doc — `eval/ablation_ledger.csv` is the raw machine-readable
> data every number here traces back to; `docs/EVAL_RESULTS.md` is the full per-stage ablation
> writeup; `docs/DECISIONS_R.md`/`DECISIONS_P.md` are the dated ADR reasoning behind each call. This
> file exists to answer one question at a glance: **"prove you actually tested this, and show what
> you found."**
>
> Updated as work progresses — not a one-time snapshot. Workstream R's testing is below; Workstream
> P appends their own section rather than editing R's (same rule as everywhere else in `docs/`).

---

## 1. Unit test suite (Workstream R)

**102/102 passing, `ruff` clean, `mypy` clean** — verified before every commit, not just once.

| Area | Test file | Tests | What it actually checks |
|---|---|---|---|
| Chunking protocol | `tests/chunking/test_base.py` | 3 | `Document`/`Chunk` model defaults, no shared-mutable-default bug |
| Strategy registry | `tests/chunking/test_registry.py` | 4 | Registration, duplicate-name rejection, unknown-strategy lookup |
| Fixed+overlap chunker | `tests/chunking/test_fixed_overlap.py` | 9 | Window/stride math, overlap boundary correctness, invalid-param rejection (parametrized over 3 bad values) |
| Passage-native chunker | `tests/chunking/test_passage_native.py` | 3 | 1:1 passage→chunk mapping, empty-doc handling |
| Sentence-window chunker | `tests/chunking/test_sentence_window.py` | 8 | Devanagari danda (`।`) sentence splitting, window bounds don't overrun the doc |
| Semantic chunker | `tests/chunking/test_semantic.py` | 5 | Splits at the actual similarity trough (fake embedder, hand-picked vectors), fails loudly without an embedder |
| Metadata-aware chunker | `tests/chunking/test_metadata_aware.py` | 4 | Dataset tags (`language`/`source_lang`/`query_type`) actually reach the chunk |
| Hierarchical chunker | `tests/chunking/test_hierarchical.py` | 6 | Every child's `parent_chunk_id` resolves to a real parent chunk |
| Dense index (FAISS) | `tests/index/test_dense.py` | 8 | Nearest-neighbour correctness, dimension validation, **save/load round-trip** |
| Sparse index (BM25) | `tests/index/test_sparse.py` | 10 | **Devanagari tokenisation is correct** (see bug #1 below), lexical ranking, save/load round-trip |
| RRF fusion | `tests/index/test_fusion.py` | 5 | Fusion formula matches the literature's definition exactly, rank (not raw score) is what's fused |
| Retrieval metrics | `tests/test_metrics.py` | 11 | Recall@k / MRR@10 / nDCG@10 against hand-computed expected values, not just "does it run" |
| E5 embedder prefixes | `tests/test_embedder.py` | 4 | `"query: "`/`"passage: "` prefix logic (the model call itself needs the real model, tested separately — §3) |
| Hybrid retriever | `tests/retrieval/test_hybrid.py` | 4 | **Dense+sparse genuinely run concurrently** (timing-based proof, not just code inspection), never raises |
| Reranker scaffold | `tests/retrieval/test_rerank.py` | 4 | `NoOpReranker` (the shipped default) preserves order and respects `k` |
| `retrieve()` contract | `tests/test_retrieval_interface.py` | 3 | Shape guarantee Workstream P builds against: count, fields, empty-query behaviour |
| Real/stub fallback | `tests/retrieval/test_interface_loading.py` | 3 | Falls back to stub when no index is built; uses the real `HybridRetriever` when it is; lazy-loads only once |
| Package smoke test | `tests/test_smoke.py` | 1 | Import doesn't explode |

(Workstream P's `tests/test_api.py`, `tests/test_retrieval_stub.py` — 3 + 4 tests — verify the FastAPI
app and their own stub usage; not duplicated here, see `docs/PROGRESS_P.md`.)

### Bugs caught by these tests, not by luck

1. **Devanagari tokenisation bug** (`src/vrag/index/sparse.py`) — first implementation used Python's
   `\w+` regex for "word characters." `tests/index/test_sparse.py::test_hindi_sentence_tokenises_into_expected_word_count`
   failed immediately: `\w+` doesn't include Devanagari's combining vowel signs (matras), so it
   silently fragmented single Hindi words apart at every matra (`दिल्ली` → `['द', 'ल', '्ल']`, not one
   token) — a naive tokenizer would have shipped and quietly halved sparse recall on Hindi text with
   no error raised. Fixed by tokenising on separators (whitespace + punctuation) instead of
   positively matching "word" characters. Full writeup: `src/vrag/index/sparse.py` module docstring.
2. **Recall@5/nDCG@10 metric bug** (`scripts/eval_chunking.py`) — surfaced by an impossible result
   during the A1 ablation, not a unit test: `sentence_window` scored nDCG@10=0.881 (highest of six)
   while having the *worst* Recall@5 (0.478) and MRR@10 (0.379). Root cause: retrieved chunks weren't
   deduplicated to unique passages before metrics sliced to top-k, so strategies producing multiple
   chunks per passage could have one relevant passage occupy several slots in a result list — this
   inflated nDCG (which sums credit per occurrence) and also depressed Recall@5 (duplicate chunks
   crowded out genuinely distinct passages from the top-5 window). Fixed, re-ran the one affected
   strategy, confirmed the ranking didn't change. Full writeup: `docs/DECISIONS_R.md` R-006.

---

## 2. Chunking ablation (A1) — the graded C2 requirement

Full protocol, table, and analysis: **`docs/EVAL_RESULTS.md` §1**. Every number below is one row in
`eval/ablation_ledger.csv` (9 rows total: 6 strategies + 1 corrected re-run + 2 noise-floor repeats).

**Setup:** 10,000-query working pool (99,767 Hindi passages), 500 frozen held-out query→passage
pairs, `multilingual-e5-small` embedder, dense-only retrieval, no rerank (staged-ablation design —
one variable at a time).

| Strategy | Recall@5 | Recall@10 | Chunks | Verdict |
|---|---|---|---|---|
| passage_native | 0.650 | 0.750 | 99,767 | Tied for best |
| fixed_overlap | 0.650 | 0.750 | 101,008 | Tied for best |
| **metadata_aware** | **0.653 ± 0.001 (n=3)** | **0.752 ± 0.002 (n=3)** | 99,767 | **Shipped** — tied for best, free metadata |
| hierarchical | 0.640 | 0.742 | 103,907 | Measurably behind (beyond noise floor) |
| semantic | 0.644 | 0.748 | 101,308 | Measurably behind (beyond noise floor) |
| sentence_window | 0.552 (corrected) | 0.554 (corrected) | 390,288 | Clear loser |

**Noise floor established:** `metadata_aware` run 3x (full index rebuild each time) — Recall@5 spread
0.652–0.654 (0.2 percentage points). This is what makes "measurably behind" in the table above a real
claim rather than a guess: the top-3/bottom-2 gap (1.0–1.4pp) is 5–7x the measured noise floor.

**Decision:** ship `metadata_aware` (`docs/DECISIONS_R.md` R-004, status: Accepted). Already wired
into `retrieve()` via `HybridRetriever` — verified end-to-end, not just unit-tested (§3 below).

---

## 3. Infrastructure / environment validation

Things checked by actually running them, not assumed:

| What | How verified | Result |
|---|---|---|
| Wheel availability for the full ML stack on Python 3.13 | `pip install --dry-run` before committing to the venv | Real cp313 wheels exist for faiss-cpu, torch, onnxruntime, transformers, sentence-transformers, bm25s, datasets (`docs/DECISIONS_R.md` R-001) |
| E5 embedder produces real, correctly-shaped vectors | Sanity embed during the CUDA-torch install check | `embed_dim=384`, matches `multilingual-e5-small`'s known output size |
| `HybridRetriever` dense∥sparse concurrency is real, not just async syntax | Timing-based test with artificial per-branch delays (`tests/retrieval/test_hybrid.py`) | Wall-clock < sum of the two branches' delays — genuine parallelism, not cooperative scheduling around nothing |
| Save/load index persistence round-trips correctly | Build a tiny index → save → load in a fresh process → run a real query through `HybridRetriever` | Identical search results before/after persistence (full smoke test run before committing to the ~39min production build) |
| GPU actually gets used for embedding, not just "should" | `nvidia-smi` utilization check mid-embed | 94% GPU utilization observed live during a real embedding pass |
| GPU embedding speedup (`docs/DECISIONS_R.md` R-005) | 2,000-text sample, timed on CPU vs. GPU | **16.5x faster** — 1.60ms/text (GPU) vs. 26.4ms/text (CPU); full production index (99,767 chunks) rebuilt in 200s vs. the original 2,320s |
| `retrieve()`'s real-index-vs-stub fallback actually switches correctly | `tests/retrieval/test_interface_loading.py`, monkeypatched index paths | Falls back to stub when no index on disk; uses real `HybridRetriever` when one exists; loads only once (lazy singleton) |
| The production index is actually live | Direct call to `retrieve()` after building, checked result identity against the stub | Returned real chunk IDs and RRF-fused scores, not `_STUB_CHUNKS` |

---

## 4. What's not tested yet (tracked so it isn't forgotten)

- A2 (embedder comparison — 4 candidates), A3 (retrieval mode: dense vs. sparse vs. hybrid), A4
  (rerank) — next ablation stages, not started
- `fixed_overlap`'s overlap hyperparameter sweep (0/10%/20%) — low priority, already tied with the winner
- efSearch recall-vs-latency curve — Phase 3 task
- Real Sarvam STT / harness / guardrails — Workstream P's domain, see their own testing notes in
  `docs/PROGRESS_P.md` and `docs/DECISIONS_P.md` (P-007's live-deployment STT verification is the
  equivalent document on their side)
- `scripts/probe_latency.py` — written, blocked on API keys, see `docs/RISKS.md`
