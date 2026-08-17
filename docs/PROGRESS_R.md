# PROGRESS_R — Workstream R's running status

> Mine, edited freely, any time. Never edited by Workstream P.

**Last updated:** 2026-08-17, Session 01
**Current phase:** P0 — Foundations & Probes (chunking work from P2 started early, see below)
**Build status:** 🟢 green — tests pass, ruff/mypy clean

## Where I am, in one paragraph

`.venv` set up on Python 3.13 (R-001), full `src/vrag/` skeleton scaffolded, `retrieve()` interface
+ stub written for Workstream P to build against. All 6 chunking strategies from
`AGENT_BUILD_SPEC.md` §7.1 are implemented with boundary tests (fixed+overlap, passage-native,
sentence-window, semantic, metadata-aware, hierarchical) — done ahead of the strict P0→P2 order
because they don't depend on anything not yet built, and having them ready removes P2 as a
bottleneck later. The E5 embedder wrapper is written (prefix logic unit-tested, actual model call
not yet exercised — see blockers). Currently downloading the Hindi MSMARCO-XI train file
(`train/hintrain.parquet`, 3.7GB, one-time cost) to materialize the working subset + held-out eval
set; `scripts/build_dataset_subset.py` is written and ready to run the moment the download finishes.

## Phase exit criteria I'm targeting (P0, my slice)

- [~] Dataset subset on disk — script ready, download in progress (~20% at last check)
- [ ] 500 held-out pairs frozen and committed (`eval/heldout_queries.json`) — blocked on the same download
- [x] `retrieve()` interface + stub written
- [x] `pytest` green, `ruff` clean, `mypy` clean

## What works right now (verified, not assumed)

- Repo pushed to GitHub, `main` matches local ✅
- `docs/` fully scaffolded ✅
- 50/50 tests passing, ruff clean, mypy clean (`pytest -q`, `ruff check .`, `mypy src`) ✅ 2026-08-17
- All 6 `ChunkingStrategy` implementations registered and unit-tested ✅
- `retrieve()` stub returns well-shaped fake data — contract tests pass ✅
- E5 prefix logic (`format_query`/`format_passage`) unit-tested — the actual model-loading path
  (`E5Embedder.embed_*`) is untested pending model download, see blockers

## What is stubbed / faked / TODO

- `retrieve()` — still the stub from Day 0, real hybrid dense+sparse implementation not started
- `E5Embedder.embed_queries`/`embed_passages` — code written, never actually run against the real
  model yet (network-bound, see blockers)
- No FAISS/BM25 index build script yet (`scripts/build_index.py` doesn't exist)
- `scripts/eval_chunking.py` doesn't exist yet — needs the embedder + index modules first
- `data/working_subset.jsonl` and `eval/heldout_queries.json` don't exist yet — blocked on dataset download

## Blockers

- Downloading `train/hintrain.parquet` (3.7GB) is slow on this network — ~20% done after ~15 min.
  One-time cost (HF-cached after), but blocks materializing the working subset / heldout set until
  it finishes. Not retrying with a different approach — `hf_hub_download` is the reliable path found
  after `datasets.load_dataset` and `hf://` streaming both failed (see `scripts/inspect_dataset.py`
  docstring for the two dead ends).
- API keys (Sarvam/Groq) still absent — blocks `scripts/probe_latency.py` from producing real numbers.

## Next session should start by

1. Check whether `train/hintrain.parquet` finished downloading
2. Run `scripts/build_dataset_subset.py` to materialize the working subset + `eval/heldout_queries.json`
3. Run `scripts/inspect_dataset.py` for the real passage-length distribution + chunk-count estimate,
   record impressions on translation quality in `docs/DECISIONS_R.md`
4. Start `scripts/build_index.py` (FAISS HNSW + bm25s) so `scripts/eval_chunking.py` has something to
   evaluate against
