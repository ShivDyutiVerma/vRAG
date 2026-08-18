# PROGRESS_R — Workstream R's running status

> Mine, edited freely, any time. Never edited by Workstream P.

**Last updated:** 2026-08-19, Session 05
**Current phase:** Single-operator mode (P's collaborator out of credits, see `docs/PROGRESS.md`). P6 latency campaign done (real numbers, found+fixed a real Sarvam bug). Deep memory work: ADR-006 audit found BM25 loading unconditionally in dense-only mode → fixed (ADR-007, -63MB). R-027 corrected R-024's corpus-shrink estimate with real quality data (worse than thought, non-viable within spec). **R-028/R-029/R-030: found the tokenizer (not the ONNX model) was the dominant embedder cost, verified a sentencepiece replacement at 100.0000% exact token-ID equivalence on 1,020 real strings, implemented it, and full production RSS is now 493.8MB steady-state — under Render's 512MB free-tier budget for the first time.** Not yet verified in the real target environment (Docker `-m 512m` or live Render) — that's the honest next step. Retrieval quality confirmed unchanged (Recall@1/5/10, MRR@10 match R-027's baseline to full float precision)
**Build status:** 🟢 green — 215/215 tests pass, ruff/mypy clean

## Where I am, in one paragraph

All four planned ablation stages plus the efSearch sweep are done and documented with real measured
numbers, not assumptions: A1 chunking (`metadata_aware` wins, tied with `passage_native`/
`fixed_overlap`, noise-floor validated), A2 embedder (`multilingual-e5-small` wins decisively), A3
retrieval mode (**dense-only wins — a genuine, verified surprise**: hybrid+RRF actually regresses
quality on this corpus because BM25 is comparatively weak on machine-translated Hindi text and naive
RRF has no way to discount it), A4 reranking (**`none` wins outright** — both FlashRank and a
cross-encoder were measured to actively destroy quality on Hindi text, verified via query-level
diagnostics as genuine model/language limitations, not bugs), and the efSearch sweep (**64 confirmed
as the knee of the curve** — same value already hardcoded pre-sweep, now backed by data instead of a
placeholder comment). Full tables, chart, and analysis: `docs/EVAL_RESULTS.md` §1-3. Full ADR trail:
`docs/DECISIONS_R.md` R-001 through R-014.

Production wiring matches the measured winners, not the originally-assumed architecture: after
confirming the deviation with the user (it contradicts `AGENT_BUILD_SPEC.md`'s assumed Phase-3 exit
criterion and CLAUDE.md's original hybrid-retrieval hot-path invariant, both now updated to reflect
this), `HybridRetriever` defaults to `retrieval_mode="dense"` and skips BM25 on the hot path
entirely; reranking stays off by default (already was). A real, verified side effect worth watching:
this also fixed `RetrievedChunk.score`'s scale to match what G3's `TAU` placeholder was actually
calibrated for. Also fixed one flaky test in P's file (`tests/test_api.py`) with explicit user
authorization to cross the ownership boundary — documented as R-013.

**Ran ahead on the schedule** (`docs/TEAM_SPLIT.md` §5 spreads A1-A4 across Days 1-3; all done here
on Day 1) into the next R-scoped item — supporting G3 calibration (`docs/TEAM_SPLIT.md` §5 names
this R's Day-3 task, joint with P). Built `eval/calibration.json` (150 in-domain + 150 genuinely
out-of-index queries, no LLM call needed — reused the already-cached corpus parquet) and ran the
full `TAU` sweep against the real production index. **Finding: `docs/EVAL_PROTOCOL.md`'s calibration
targets (false-refusal<10% AND correct-refusal>80%) are not simultaneously reachable** on this
corpus via pure cosine-similarity gating — verified root cause: MSMARCO-XI passages recur across
different query_ids, so genuinely out-of-index queries often still retrieve a topically-close or
even coincidentally-correct match. Full curve and reference-point table: `docs/DECISIONS_R.md`
R-015, chart at `docs/assets/g3_calibration.png`. Deliberately did **not** apply a chosen TAU/MARGIN
myself — the pick is a real product tradeoff, and `docs/TEAM_SPLIT.md` §5 reserves it as a joint
Day-3/4 decision.

**Partner applied `TAU=0.8835` shortly after** (`docs/DECISIONS_P.md` P-015) — the balanced point
from my curve. Their own commit flagged one remaining gap: `MARGIN=0.05` (the pre-calibration
placeholder) hadn't been re-swept at the new `TAU`. Completed that flagged step same day: verified
live that `MARGIN=0.05` at this `TAU` actually caused **88.0%** false-refusal, not the 19.3% the
commit's design target stated — in-domain top1-vs-top5 gaps are naturally tiny at this operating
point, so even `MARGIN=0.01` alone degrades to 28.7%. Set `MARGIN=0.0` (empirically correct at this
`TAU`, not a shortcut — confirmed live, 2/3 previously-blocked test queries now answer correctly).
Full writeup: `docs/DECISIONS_R.md` R-017. This wasn't a new judgment call on my part — it's the same
calibration data and analysis that already produced `TAU=0.8835`, applied to keep the file
internally consistent with its own stated target.

## Phase exit criteria I'm targeting (P3, my slice) — all met

- [x] Dataset subset + 500 held-out pairs frozen and committed
- [x] `retrieve()` wired to a real, persisted index (`data/index/metadata_aware/`)
- [x] A1 chunking ablation — winner picked, noise-floor validated
- [x] A2 embedder ablation — winner picked, decisively
- [x] A3 retrieval-mode ablation — winner picked (dense-only, a real surprise), wired into production
- [x] A4 reranker ablation — winner picked (none), already the default
- [x] efSearch recall-vs-latency curve (`docs/assets/efsearch_curve.png`) — 64 confirmed as the knee
- [x] `pytest` green (172/172)
- [x] G3 calibration — data gathered (R-015), TAU applied by P, MARGIN corrected same day (R-017),
  both verified live

## What works right now (verified, not assumed)

- Full ablation trail A1→A4 + efSearch, every run backed by a `eval/ablation_ledger.csv` row, every
  winner backed by query-level diagnostics where the result was surprising enough to warrant one
  (A3, A4) ✅
- `retrieve()` loads the real persisted index and returns real, measured-good results (dense-only,
  Recall@5=0.652 on the frozen 500-query held-out set, efSearch=64) ✅
- `HybridRetriever` supports dense/sparse/hybrid modes, all three unit-tested including the
  "unused modes never call the index they don't need" guarantee ✅
- Shared `score_hits`/`dedupe_doc_ids` in `src/vrag/retrieval/metrics.py` — the R-006 dedup fix is
  now a single tested fix point every eval script (A1-A4, efSearch) routes through ✅
- `CrossEncoderReranker`/`FlashRankReranker` both implemented, tested for wiring correctness (not
  quality — their quality verdict is "actively harmful on Hindi," documented in R-012) ✅
- `DenseIndex.set_ef_search()` — mutates HNSW search-time behavior without a rebuild, unit-tested ✅
- `fixed_overlap`'s overlap ∈ {0, 0.1, 0.2} swept — confirmed no measurable effect on this corpus
  (R-016), closing the last open item from A1's methodology ✅

## What is stubbed / faked / TODO

- A candidate RRF mitigation (larger per-lane candidate pool before fusion) — logged as an idea in
  `docs/RISKS.md` R-R14, not tested; wouldn't change the shipped default without a fresh ablation run
- A genuinely Hindi-capable reranker (e.g. `bge-reranker-v2-m3`) was never tried — A4 only tested the
  three TECH_MENU-named candidates, all of which failed for either English-only training or model
  saturation on Hindi
- `docs/BUILD_PLAN.md` P3 task 10 ("optional but high-value": Sarvam `transcribe` vs `translit`
  output modes for retrieval quality) — genuinely blocked, not skipped by choice: no `.env`/Sarvam
  key on this machine (`docs/RISKS.md`'s Day 0-1 blockers note). Considered a transliteration-library
  proxy instead of real Sarvam output; rejected — a generic library's romanization scheme has no
  guarantee of matching Sarvam's actual `translit` convention, and reporting a finding based on a
  possibly-wrong proxy would risk a misleading result, not a real one. Needs either R's own Sarvam
  key or running this test from Workstream P's machine.
- A1×A2 confirmation pass — reasoned skip, not run; see `docs/EVAL_RESULTS.md` §2 for why (no genuine
  embedder-side question exists once A2's 38-point gap is accounted for)

**Was about to start P6's ONNX-quantise-the-embedder task**, reasoning R was clear to move into
Day-4-scoped work since both progress docs described a fully working real pipeline. Checked the
actual live URL directly before committing to that — good thing: **the live deployment
(`https://vrag-voice.onrender.com`) runs entirely on the Day-0 stub, not real retrieval at all.**
`Dockerfile` never installs the `retrieval` extra and `data/` is gitignored, so none of A1-A4's work
has ever reached production, despite local testing on both machines showing it working. Full
root-cause and what's needed: `docs/DECISIONS_R.md` R-018, `docs/RISKS.md` R-R21 (marked high
priority — blocks the actual C7 "live working link" deliverable if unresolved by submission).
Prepared the R-side half of the fix without touching `Dockerfile`/`render.yaml` (Workstream P's
ownership): packaged and published the current production index as a GitHub Release asset
(`index-metadata_aware-v1`, 187MB, verified publicly downloadable) — extracting it to
`data/index/metadata_aware/` is all the code side needs, no code changes required.

**Then did the ONNX embedder work anyway** (`docs/DECISIONS_R.md` R-019) — didn't need R-R21 fixed
first to build and *validate* it, only to *ship* it. Exported `multilingual-e5-small` to ONNX,
dynamically int8-quantised (`ONNXE5Embedder`, new class in `src/vrag/index/embedder.py`, registered
in `EMBEDDER_REGISTRY`). Tested the realistic production shape: quantised only the query-time
embedder (passages stay FP32 in the already-built index — build time doesn't matter there, query
latency does). **Result: 3.7x faster query embedding (20.48ms → 5.60ms p50, CPU) for a real but
small -0.9pp Recall@5 cost.** Closes a gap A2's own write-up flagged as deferred to Phase 6.
Deliberately **not wired into `retrieve()`** — R-019 explains why: there's no live deployment for
this to matter to yet (R-R21), and wiring a query-embedder swap in before that's fixed would be
untestable against the real deploy target and would conflate two changes when R-R21's fix needs to
stay isolated and easy to verify. New dependencies (`onnx`, `optimum`, `optimum-onnx`,
`sentence-transformers[onnx]`) added to `pyproject.toml`'s `retrieval` extra — required lowering
`transformers`'s floor and adding a ceiling (`optimum-onnx` hard-pins `<4.58.0`), verified safe
against the full test suite before committing.

**Measured real RSS before P attempted the R-R21 deploy fix** (`docs/DECISIONS_R.md` R-020,
`docs/RISKS.md` R4, escalated to 🔴 high) — good thing: the persisted index *alone*, no embedder
loaded, already uses 591MB RSS, 115% of Render free tier's 512MB limit. With the FP32 embedder the
total is 1.47GB (288% of budget). `ONNXE5Embedder` (R-019) gave no memory relief, only latency —
`torch`/`transformers` load either way regardless of inference backend. Three real fix directions
identified: upgrade Render's plan, shrink the chunk count (real quality cost against A1-A4's
numbers), or a leaner `chunk_lookup.json` format — the last one explicitly R-ownable and not needing
a cost/quality tradeoff decision to attempt, unlike the other two.

**Built and measured that one** (`docs/DECISIONS_R.md` R-021) — `SQLiteChunkLookup`
(`src/vrag/index/sqlite_chunk_lookup.py`), same read interface every real call site uses, backed by
SQLite instead of one live dict of ~100k Pydantic `Chunk` objects. **Real win: index-only RSS drops
591MB → 339MB (-43%, now under budget on its own); full stack with the ONNX embedder drops
1,539MB → 1,321MB (-14%).** But confirms this isn't sufficient alone — the embedder's own
`torch`/`transformers` import footprint (~980MB) is now the clearly dominant remaining cost, not the
index.

**User gave explicit direction: no paid Render plan, keep shrinking under 500MB.** Isolated
`import torch` alone: ~383MB, most of that ~980MB. Built `LiteE5Embedder`
(`docs/DECISIONS_R.md` R-022) — same ONNX int8 model, but bypasses `sentence-transformers` entirely:
raw `onnxruntime.InferenceSession` + `tokenizers`' Rust tokenizer, mean-pooling/normalisation done by
hand in numpy. Verified byte-identical to `ONNXE5Embedder`'s output before trusting it (cosine sim
1.0, diff ~1.5e-8, float32 noise) — same model, same math, no `torch` in the import graph.
**Result: full stack drops 1,321MB → 741MB (-44% more).** Combined with R-021: **1,539MB → 741MB,
-52% total, zero quality cost.** Tried further `onnxruntime` tuning (memory arena, graph
optimisation) — measured, no further headroom. Still 145% of the 512MB budget; the remaining
~230MB gap most plausibly needs the one option with a real quality cost (shrink the chunk count) —
e5-small is already A2's smallest viable model.

**Wired R-021/R-022 into production and shipped the runtime artifacts** (`docs/DECISIONS_R.md`
R-023) — until now the standalone prototypes existed but `interface.py::_get_real_retriever()`
still hardcoded the old `E5Embedder()`+JSON-dict path, so none of the 52% RSS cut would have
actually reached production even once P's Dockerfile fix landed. Added `EmbedderProtocol`
(`embedder.py`) and widened `HybridRetriever`'s type hints (`hybrid.py`) so it accepts
`LiteE5Embedder`/`SQLiteChunkLookup`, not just the original `E5Embedder`/`dict`. Added
`load_built_index_lean()` (`persistence.py`) and a new `retrieval-lean` pyproject.toml extra
(numpy/faiss-cpu/bm25s/onnxruntime/tokenizers — no torch/transformers). **Verified in a fresh
throwaway venv with only `retrieval-lean` installed** (confirmed torch-free via the pip install
log): built the real index, ran `retrieve()` end-to-end through `interface.py`, got 5 real
(non-stub) hits with correct Devanagari text and plausible scores — **727MB RSS measured**,
matching R-022's isolated 741MB number. Published two new GitHub Release artifacts so P's
Dockerfile has something to point at: `embedder-lite-onnx-v1` (87.6MB — just the two files
`LiteE5Embedder` reads, not the full 579MB export dir) and `index-metadata_aware-v2` (adds
`chunk_lookup.sqlite3` to the existing index archive). Full 3-step Dockerfile change P needs is
spelled out in R-023 — did not touch `Dockerfile` itself (P's module).

**P tried R-023's fix live, found a real OOM, rolled back** (`docs/DECISIONS_P.md` P-020) — verified
myself right after via a direct `/ask` call against the live URL (same discipline as R-018's
original discovery): `/healthz` 200 OK, `/ask` 502 (`x-render-routing: no-deploy`), reproduced
twice (R-025, `docs/DECISIONS_R.md`). P's session independently caught the same thing via Render's
own runtime logs, confirmed it's a real OOM (not a routing artifact), rolled back within ~15
minutes, and sharpened the diagnosis: the blocker is **peak memory during `_get_real_retriever()`'s
first load**, not the 727MB steady-state figure — local Docker testing under the same 512MB limit
had shown comfortable steady-state headroom (446.8MB), so the transient spike while the index/ONNX
session initialize is the newly-identified real constraint. Live URL re-verified stable afterward
(serving the stub again, no more 502s).

**Measured the corpus-shrink lever's real ceiling before running it** (R-024) — the embedder's
fixed ~388MB session cost dominates the remaining gap: an *empty* index would still cost ~467MB
(91% of the 512MB budget), so hitting 512MB via corpus-shrink alone needs ~17,300 chunks, an ~83%
cut from 99,767. Checked whether a smaller embedder (Model2Vec) could remove the fixed cost
instead: Recall@5 0.266 vs. e5-small's 0.652 — not viable either. **Escalated to the user rather
than running an ablation on a lever that's very likely disqualifying** — this now touches graded
quality requirements (C1-C6), not a call to make unilaterally. User's direction: check whether a
different free host offers more RAM before accepting a quality cut or revisiting the paid tier.
That's P's deployment domain — flagged in `docs/RISKS.md` R4 for their session, not investigated
here. (P's session has already started on this per their P-020 write-up.)

**Closed R-R14 with a real, tested answer** (R-026) — the standing "larger candidate pool before
RRF fusion" mitigation for hybrid's A3 underperformance was untested; tested it properly
(`scripts/eval_rrf_candidate_pool.py`, pools of 10/30/50/100). Doesn't help — Recall@5 stays
flat-to-worse (0.578-0.604) at every size, all below dense-only's 0.652. No config change; confirms
A3's dense-only decision is robust to the most obvious first mitigation attempt.

## Blockers

- **R-R21 (live deployment stub)** — R's side is fully done (code wired, tested, artifacts
  published, R-023). Blocked on P diagnosing the Render-specific peak-memory-during-load issue
  (P-020) — needs Render's own logs/dashboard, which R has no access to.
- **R4 (RAM gap)** — R's engineering side is done (727MB steady-state, zero quality cost). The
  corpus-shrink lever's real cost is now known and it's severe (R-024, ~83% cut needed). Decision
  escalated to the user, who directed a hosting-alternatives check first — that's P's domain,
  already in motion on their side per P-020.
- Nothing currently blocked on R alone. Both open R4/R-R21 items need P's investigation or the
  user's/team's decision, not more R-side engineering.

## Next session should start by

1. **Check `docs/RISKS.md` R4/R-R21 for what P's session found** on the hosting-alternatives check
   and the peak-memory-during-load diagnosis — both were in progress on P's side as of this
   session's end, not yet resolved.
2. **Re-verify the live URL directly** before trusting any "fixed" claim (same discipline every
   time) — `GET /healthz` then a real `POST /ask` with a genuine query, not just a health check.
3. If real retrieval reaches production, measure actual live RSS/behavior rather than assuming the
   727MB isolated-venv or 446.8MB local-Docker numbers transfer exactly — P-020's OOM already
   showed local ≠ Render for peak memory specifically.
4. If the team ultimately decides on a corpus-shrink cut (R4), that needs a real re-ablation pass
   quantifying the actual Recall@5 cost at whatever size is chosen — not a guess, and not started.
5. Read `docs/EVAL_RESULTS.md` §1-3 and `docs/DECISIONS_R.md` R-010/R-012/R-014/R-015/R-017/
   R-023/R-024/R-025/R-026 for the full A3/A4/efSearch/G3-calibration/memory-fix story before
   touching retrieval code again — all of it is deliberate, data-driven, and documented, not
   oversights.
