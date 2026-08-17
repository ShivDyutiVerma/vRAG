# PROGRESS_R — Workstream R's running status

> Mine, edited freely, any time. Never edited by Workstream P.

**Last updated:** 2026-08-17, Session 01
**Current phase:** P0 — Foundations & Probes (P2/P3 code written ahead of schedule, not yet run against real data)
**Build status:** 🟢 green — 89/89 tests pass, ruff/mypy clean

## Where I am, in one paragraph

`.venv` set up on Python 3.13 (R-001), full `src/vrag/` skeleton scaffolded, `retrieve()` interface
+ stub written for Workstream P to build against. All P2/P3 code is written and unit-tested even
though no real data has flowed through it yet: all 6 chunking strategies, dense (FAISS)/sparse
(BM25)/RRF-fusion index primitives, the E5 embedder wrapper, `HybridRetriever` (the real
dense∥sparse concurrent implementation — concurrency itself is unit-tested, not just assumed),
a reranker protocol + NoOp default + FlashRank candidate, retrieval metrics (Recall@k/MRR/nDCG),
`build_index.py`, and `eval_chunking.py` (the A1 ablation runner). Two real corpus/model downloads
are in progress in the background (detached OS processes, survive independently of any one tool
session) — the Hindi MSMARCO-XI parquet (3.7GB, ~33% done) and the multilingual-e5-small model
(~80% done). Once both land, the plan is: `build_dataset_subset.py` → `inspect_dataset.py` (real
numbers + translation spot-check) → `eval_chunking.py` for all 6 strategies → `docs/EVAL_RESULTS.md`
§1 → promote the winner → wire `HybridRetriever` into `interface.py`'s `retrieve()`.

## Phase exit criteria I'm targeting (P0, my slice)

- [~] Dataset subset on disk — script ready, download ~33% (background, detached process)
- [ ] 500 held-out pairs frozen and committed (`eval/heldout_queries.json`) — same blocker
- [x] `retrieve()` interface + stub written
- [x] `pytest` green, `ruff` clean, `mypy` clean (89/89)

## What works right now (verified, not assumed)

- All 6 `ChunkingStrategy` implementations registered, unit-tested, boundary-tested ✅
- Dense/sparse/fusion index primitives unit-tested with synthetic vectors ✅
- Devanagari tokenizer bug caught and fixed via the required unit test (see `docs/DECISIONS_R.md`) ✅
- `HybridRetriever` concurrency directly proven by a timing-based unit test, not inferred ✅
- `retrieve()` stub returns well-shaped fake data — contract tests pass ✅
- E5 prefix logic unit-tested; actual model call untested pending model download (in progress)

## What is stubbed / faked / TODO

- `retrieve()` (`src/vrag/retrieval/interface.py`) — still the Day-0 stub; `HybridRetriever` exists
  but isn't wired in yet, because there's no built index to wire it to until data lands
- `E5Embedder.embed_queries`/`embed_passages` — code written, never run against the real model
- `FlashRankReranker` — code written, untested (needs the `rerankers` FlashRank model, not downloaded)
- `data/working_subset.jsonl` and `eval/heldout_queries.json` — blocked on the dataset download
- No chunking strategy has been run against real data yet — all 6 are eval-ready the moment the
  data lands, but zero real Recall@k numbers exist yet. `docs/EVAL_RESULTS.md` §1 is still empty.

## Blockers

- Two large downloads in progress (dataset ~33%, E5 model ~80%, both climbing steadily). One-time
  cost, HF-cached after. See `docs/RISKS.md` for the two networking issues diagnosed along the way
  (broken IPv6, HF's Xet transfer layer bypassing the IPv4 workaround).
- API keys (Sarvam/Groq) still absent — blocks `scripts/probe_latency.py` from producing real numbers.

## Next session should start by

1. Check whether both downloads finished (`docs/RISKS.md` has the exact paths to check)
2. Run `scripts/build_dataset_subset.py` to materialize the working subset + `eval/heldout_queries.json`
3. Run `scripts/inspect_dataset.py` for the real passage-length distribution + chunk-count estimate,
   record impressions on translation quality in `docs/DECISIONS_R.md`
4. Run `scripts/eval_chunking.py --strategy <name>` for all 6 strategies, write `docs/EVAL_RESULTS.md` §1
5. Promote the winner; wire `HybridRetriever` into `interface.py`'s `retrieve()`, replacing the stub
