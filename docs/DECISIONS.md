# Architecture Decision Record — SHARED

> Append-only. Edited only at integration syncs (`docs/TEAM_SPLIT.md` §5) by whoever is merging —
> concatenate `DECISIONS_R.md` + `DECISIONS_P.md`, sort by date. To reverse a decision, add a new ADR
> that supersedes it; never edit history.

## ADR-001 — STT provider: Sarvam

**Date:** 2026-08-17 (pre-decided in `AGENT_BUILD_SPEC.md` §2, transcribed here as the formal record)
**Status:** Accepted
**Context:** Brief requires Sarvam or ElevenLabs (constraint C1). Corpus (`ai4bharat/MSMARCO-XI`) is Indic-language.
**Decision:** Sarvam.
**Rationale:** Indic-native training, code-mixing support, India-hosted → lower RTT from an Indian
deployment, which matters directly against the 200ms constraint. ElevenLabs rejected — strongest in
English/major European/East Asian languages, not the corpus's language family (`TECH_MENU.md` S2).
**Consequences:** STT stack is Indic-coherent end to end. Locked unless a probe disproves the RTT
assumption (see ADR-003, pending). **Confirmed for real** at the Day 1 sync — Workstream P verified
real Sarvam realtime STT end to end against the live deployment (`docs/DECISIONS_P.md` P-007).

## ADR-002 — Corpus scope: Hindi only for v1

**Date:** 2026-08-17 (pre-decided in `AGENT_BUILD_SPEC.md` §6.1)
**Status:** Accepted
**Context:** `ai4bharat/MSMARCO-XI` covers 13 Indian languages. Indexing all of them burns days for no
grading benefit and no team has validated translation quality across all 13.
**Decision:** Index Hindi (`hi`) only, targeting 50k–200k chunks.
**Rationale:** Best downstream tooling support, largest community validation, easiest to spot-check
translation quality by eye. A second language is explicitly gated to Phase 7, only if everything else
is green (`docs/BUILD_PLAN.md` cut list — second language is cut item #1 if behind schedule).
**Consequences:** All chunking/embedding/retrieval ablation work (A1–A4) runs on the Hindi subset only.

## ADR-003 — Provider RTT probe results

**Date:** 2026-08-19
**Status:** Accepted — real Sarvam key obtained (single-operator session, P's collaborator out of
weekly credits, see the operational note in `docs/PROGRESS.md`); Groq key still empty, that leg
correctly shows as skipped rather than guessed. `scripts/probe_latency.py --n 30`, real output:

**1. Raw TCP + TLS connect time (no API key needed):**

| Provider | Host | P50 (ms) | P95 (ms) | P100 (ms) | Failures |
|----------|------|----------|----------|-----------|----------|
| sarvam | api.sarvam.ai | 98.7 | 119.6 | 121.8 | 0/30 |
| groq | api.groq.com | 67.4 | 88.9 | 92.3 | 0/30 |

**2. Chat completion TTFT:**

| Provider | Model | P50 (ms) | P95 (ms) | P100 (ms) | Failures |
|----------|-------|----------|----------|-----------|----------|
| sarvam | sarvam-105b | 271.3 | 1403.3 | 1965.8 | 0/30 |
| groq | — | — | — | — | SKIPPED — no API key in `.env` |

**Decision:** Sarvam confirmed as the only real candidate (ADR-001's choice stands) — no Groq key
means no real comparison was ever possible, so this isn't a competitive result, just confirmation
that Sarvam's connection physics (~99ms TCP+TLS from this machine) and TTFT (P50=271ms) are real
and measured, not assumed. **Consequence for the 200ms budget:** even Sarvam's P50 TTFT alone
(271ms) exceeds the entire pipeline target before a single retrieval/guardrail stage runs — this is
the quantitative confirmation behind the two-track (Track A always-fast, Track B best-effort)
design, not a new finding but the number that was missing to state it with real evidence. See
`docs/LATENCY_BUDGET.md` for the full P6 latency campaign this probe was run alongside, including a
separate, larger (N=30 real generation calls, not just TTFT-to-first-chunk) Track B measurement
showing P50=1976.5ms full completion.

## ADR-004 — `t_pipeline` metric definition

**Date:** 2026-08-17
**Status:** Accepted — confirmed by both workstreams at the Day 1 sync.
**Decision:**
> `t_pipeline` is measured server-side, from the moment the final transcript is available to the
> moment the first grounded answer token is emitted to the client. It excludes: client→server
> network transit, microphone capture, and speech duration. It includes: input guardrails, query
> embedding, hybrid retrieval, fusion, grounding gate, and answer generation up to first token.
> Index construction (chunking + embedding + index build) is a one-time offline cost, reported
> separately and excluded from `t_pipeline`.
**Rationale:** Defensible and standard for voice-RAG systems; separates one-time indexing cost from
per-request latency; `t_e2e_voice` (mic-stop → first audible/visible answer) is reported as a
secondary honest number so STT cost isn't hidden. Settles the metric definition on Day 1 rather than
under deadline pressure later — the exact failure mode risk R6 in `docs/RISKS.md` warns about.
**Consequences:** This exact wording goes in the README verbatim. Full protocol detail in
`docs/EVAL_PROTOCOL.md`.

## ADR-005 — P6 latency campaign: real numbers, and Track A's answer is fast but the wire hides it

**Date:** 2026-08-19
**Status:** Accepted. Joint ownership per `docs/TEAM_SPLIT.md` §2 ("Latency benchmark — JOINT"); run
solo (single-operator session, see `docs/PROGRESS.md`'s operational note) so recorded directly here
rather than merged from two per-track logs.
**Context:** `docs/BUILD_PLAN.md` P6 — "never cut" graded core, 0% built before this session.
Built `scripts/make_test_queries.py` (100 real queries, 60 in-domain/20 off-topic/10 unsafe/10
degenerate, every one sourced from already-real, already-vetted content — the frozen held-out set,
G3's calibration out-of-domain pool, and the existing guardrail test suites — not invented),
`scripts/synthesize_test_audio.py` (95 real Sarvam-TTS 16kHz WAVs), `scripts/bench_latency.py`
(500 text samples + a standalone 30-call Track B measurement), `scripts/make_latency_charts.py`,
and `tests/test_latency_regression.py`.
**The headline finding:** Track A's true stage cost (retrieve + all guardrails + extraction,
summed from the harness's own `timings_ms`) is **P50=5.2ms, P100=8.9ms** — comfortably under the
200ms target, exit criterion genuinely met. But the client-observed end-to-end latency for an
answered query is **P50=213-246ms** (three runs) — because `GenerateStage` attempts a real Track B
call on every answerable query with whatever budget remains (~190ms), Sarvam's real latency is
~2s, so the attempt always times out, and the circuit breaker never trips (it only counts >=2.0s
"fair chance" failures, never reached under a 200ms budget) — so every request pays that doomed
wait before Track A's already-ready answer returns. **Decision (user, 2026-08-19): report both
numbers honestly, no harness behavior change** — the shedding design is deliberate and correct;
Sarvam being ~10x slower than the total budget is what makes each attempt futile in practice, not
a flaw in the design itself. Matches `docs/BUILD_PLAN.md`'s own guidance to report Track B's real
cost plainly rather than hide it. Full breakdown, charts, and the response-status mix:
`docs/LATENCY_BUDGET.md`.
**A real bug was found and fixed along the way**, not a pre-existing known issue — see
`docs/DECISIONS_P.md` P-021: Track B's streaming handler crashed on an explicit `null` content
delta from Sarvam's SSE stream, silently masking most of Track B's real success rate (0/30 before
the fix, 19/30 = 63.3% after, in this session's own measurement). Recorded in P's log since the
fix lives in P's module; referenced here since it materially changed this ADR's own Track B numbers
mid-investigation.
**Consequences:** `eval/ablation_ledger.csv` has a new `p6_latency_bench_*` row. `eval/test_queries.json`
and `eval/audio/` are now real, committed, reusable assets for any future latency/voice work — no
need to re-synthesize or re-sample. `tests/test_latency_regression.py` guards Track A's true stage
cost (not the wall-clock-with-Track-B-wait number) against future regression, skipping cleanly on a
fresh checkout without the local index artifact rather than failing or faking a pass.

## ADR-006 — Component-by-component memory audit, before any architectural change

**Date:** 2026-08-19
**Status:** Accepted — measurement only, per the user's explicit "do not modify the architecture
yet, do not upgrade Render yet" instruction. `scripts/audit_memory.py`, one isolated subprocess per
component (so one measurement's already-resident memory never pollutes another's), `psutil`-based
in-process RSS sampling (added to the `dev` extra — diagnostic-only, never imported by application
code). Real numbers, not projections; full detail in the chat response to the user, summarized here
for the permanent record.

**Measured (bytes rounded to MB):**

| Component | RSS after load | Notes |
|---|---|---|
| Bare Python + FastAPI | 43MB | |
| Embedder only (`LiteE5Embedder`, production choice) | 460MB | torch/transformers confirmed absent from `sys.modules` |
| FAISS index only | 239MB | 99,767 vectors, dim 384, `IndexHNSWFlat`, float32, 180MB on disk |
| BM25 index only | 125MB | Loaded with `load_corpus=False` — never holds raw chunk text |
| Metadata/corpus only (`SQLiteChunkLookup`) | 58MB | Only `chunk_id->doc_id` resident; text fetched lazily per-row |
| Reranker only (`FlashRankReranker`) | 871MB | **Not in production** (A4 chose `none`, R-012) |
| **Full application** (production wiring, dense-only) | **778MB steady-state**, peak ~776MB during startup+first query | Stable across repeated queries (778MB after 1 query, 778MB after 4) |

**Top 3 real sources of RAM in the production configuration, ranked:**
1. **The embedder (~440MB net over baseline)** — the single largest cost, already minimized per
   R-022 (torch-free, ONNX int8) with no further headroom found there.
2. **The FAISS dense index (~219MB net)** — scales with chunk count (linear fit from R-024:
   ~78.7MB + 0.00262MB/chunk).
3. **The BM25/sparse index (~105MB net) — loaded but architecturally unused.** `persistence.py`'s
   `load_built_index_lean()` unconditionally loads the sparse index regardless of
   `retrieval_mode`, but production runs `retrieval_mode="dense"` (A3 winner, R-010) and never
   calls `sparse.search()`. **This is pure waste under the current architecture — real, immediately
   actionable, and does not require redesigning anything, just skipping unnecessary work** — but
   per the user's explicit instruction, not changed in this session; flagged for a future,
   deliberate decision.

**Answers to the specific questions asked:**
- Chunk count: 99,767. Embedding dim: 384. Vector dtype: float32.
- FAISS index type: `IndexHNSWFlat`, efSearch=64 (M=32/efConstruction=200 at build time, not
  re-queryable from a loaded index object but known from `docs/DECISIONS_R.md` R-001–R014).
- FAISS index file size on disk: 180,399,222 bytes (~172MB).
- BM25 memory: 125MB loaded (never holds raw text).
- Torch/Transformers when ONNX embedder is used: confirmed absent (`"torch" in sys.modules` and
  `"transformers" in sys.modules` both `False`).
- Corpus/chunk text duplication: **no duplication found** — FAISS stores only vectors,
  BM25 loads with `load_corpus=False`, only `SQLiteChunkLookup` holds text, and it's lazy per-row,
  not resident in bulk.
- Peak RSS during startup: ~776MB (first real query included, since that's when the embedder
  session actually initializes under the current lazy-load design).
- Steady-state RSS after warmup: 778MB, stable across repeated queries (no growth/leak observed
  over 4 real calls).
**Methodology caveat, stated honestly:** thread-based sampling under CPython's GIL can miss a true
peak during a CPU-bound, GIL-holding deserialization step — observed once, for BM25's numpy/JSON
load (`peak_rss_during_load` under-reported vs. the `after_load` reading by ~42MB). Where that gap
appeared, the `after_load` reading is the more reliable number; noted per-component, not silently
smoothed over.
**Consequences:** No code changed as a result of this audit (per instruction). The three concrete,
already-identified levers for anyone deciding what to do next: (a) stop loading the unused BM25
index in dense-only mode (~105MB, no architecture change, no quality cost), (b) the embedder is
already at its measured floor (R-022), (c) corpus/chunk-count reduction remains the only lever with
real headroom left, and R-024 already measured that its cost is severe (~83% cut needed to reach
512MB). None of this was acted on in this session.

## ADR-007 — Fixed the unconditional BM25 loading ADR-006 found: ~63MB real reduction, nothing else touched

**Date:** 2026-08-19
**Status:** Accepted. Scoped exactly to the user's instruction: fix only the unconditional BM25
loading; no change to retrieval architecture, corpus size, embedding model, FAISS configuration,
or deployment plan.
**What changed, three files, minimally:**
- `src/vrag/index/persistence.py`: `load_built_index_lean()` gained a `retrieval_mode` parameter
  (default `"dense"`, matching production) — only calls `SparseIndex.load()` when
  `retrieval_mode in ("sparse", "hybrid")`, returns `None` otherwise. `load_built_index()` (the
  non-lean version used by offline ablation scripts, which always need both indexes) is
  untouched, out of scope.
- `src/vrag/retrieval/hybrid.py`: `HybridRetriever.__init__`'s `sparse` param is now
  `SparseIndex | None`; added a fail-fast guard — constructing with `retrieval_mode in ("sparse",
  "hybrid")` and `sparse=None` raises `ValueError` immediately, rather than silently returning `[]`
  the first time a real query hits `_search_sparse()` and crashes on a `None` attribute access.
- `src/vrag/retrieval/interface.py`: introduced a single `_RETRIEVAL_MODE = "dense"` module
  constant so the `load_built_index_lean()` call and the `HybridRetriever(...)` construction can't
  drift out of sync (previously the mode was a literal repeated at two call sites).
**Test coverage added:** `tests/index/test_persistence.py` (dense mode returns `sparse=None`;
sparse/hybrid modes still load a real `SparseIndex`), `tests/retrieval/test_hybrid.py` (dense mode
accepts `sparse=None`; sparse/hybrid modes reject it with a clear error). 204/204 tests green,
ruff/mypy clean.
**Measured, not assumed — 3 runs, real spread:**

| Run | Steady-state RSS (4 queries) |
|---|---|
| 1 | 715.42MB |
| 2 | 715.60MB |
| 3 | 716.23MB |

**Before this fix: 778MB (ADR-006). After: ~715.7MB average — a real ~63MB (8.1%) reduction**,
somewhat less than the ~105MB isolated BM25-only measurement suggested, because isolated
single-component measurements each pay their own ~20-27MB Python interpreter baseline separately;
the marginal cost of one component inside an already-loaded stack is smaller than its isolated
number implies. Still a real, verified, zero-quality-cost win — dense-only retrieval quality is
completely unaffected (A3's Recall@5=0.652 result used real dense search all along; nothing about
*how retrieval scores* changed, only what's resident in memory).
**Consequences:** Gap to Render's 512MB free tier narrows from ~266MB to ~204MB (778MB→512MB was
the old gap; 715.7MB→512MB is the new one) — still open, still requires either the corpus-shrink
lever (R-024: severe, ~83% cut, not attempted) or a paid tier, per the user's still-pending
decision on `docs/RISKS.md` R4. No architecture, corpus, embedding, FAISS, or deployment change was
made here — exactly as instructed.

---

> Further shared ADRs (`retrieve()` contract changes if any, joint G3/G4 calibration decisions) land
> here at future integration syncs. Day-to-day, per-track ADRs live in `docs/DECISIONS_R.md` and
> `docs/DECISIONS_P.md` — read both before assuming this file is current, since it's only touched at
> sync points by design (or, from 2026-08-19, directly by the single-operator session for genuinely
> joint-ownership work — see `docs/PROGRESS.md`).

## ADR-008 — Phase 1: language-aware routing (query_language / retrieved_language split), `retrieve()` gains an inert `language` param

**Date:** 2026-08-20
**Status:** Accepted.
**Context:** This session's language-routing diagnostic (docs/PROGRESS.md) found the STT call
hardcoded to `language_code="hi-IN"`, Sarvam's real `event.language` signal captured then
discarded, and `ctx.data["language"]` conflating two different meanings (the query's language vs.
the retrieved chunk's language) into one key. ADR-002 (Hindi-only corpus scope) stands unchanged —
this ADR is about routing plumbing, not corpus content. Full language inventory (13 real MSMARCO-XI
train languages, verified against live parquet metadata) recorded as this session's Phase 0 audit.
**Decision:**
- New module `src/vrag/languages.py`: `SUPPORTED_LANGUAGES` = the 13 MSMARCO-XI train languages,
  expressed as Sarvam's real BCP-47 codes (verified live against docs.sarvam.ai, not assumed).
  English (`en-IN`, not a MSMARCO-XI *target* language) and Telugu (`te-IN`, MSMARCO-XI has only a
  validation file, no train data) are Sarvam-recognisable but deliberately excluded.
- `src/vrag/stt/sarvam.py`: `language_code` default changed from `"hi-IN"` to `"auto"` — Sarvam's
  real adaptive-detection mode, and the *only* mode in which it populates a transcript event's
  `language` field at all (verified live). `WS /voice` now uses it unconditionally.
- Three distinct concepts in `PipelineContext.data`, never aliased onto one key:
  `query_language` (Sarvam's real detected code, set once at pipeline entry), `retrieved_language`
  (the top retrieved chunk's own `language` metadata — a *different* code space, e.g. `"hin_Deva"`
  vs. `"hi-IN"`), `generation_language` (defaults to `query_language`, plumbed through but not yet
  consumed by Track B — Phase 1 explicitly does not enable multilingual generation).
- `g2_scope_language.check()` gains an optional `language` param: when a real code is given, G2
  decides purely from `SUPPORTED_LANGUAGES` membership (unsupported → refused, with the language
  named in the reason). When `language=None` (the `/ask` text endpoint has no STT signal), the old
  script-presence heuristic runs unchanged — zero behavior change for existing callers.
- `AnswerResponse` gains `query_language: str | None = None` (additive — old clients/tests
  unaffected). The existing `language` field's meaning is unchanged (the answer/evidence's own
  language); its refused/abstained fallback changed from a bare hardcoded `"hi"` to the real
  `query_language` when known, `"hi"` only as the last-resort default for signal-less callers.
- `retrieve()` (the R/P seam, `src/vrag/retrieval/interface.py`) gains an optional `language`
  param, per that file's own docstring requiring an ADR for any signature change. Threaded through
  `HybridRetriever.retrieve()` too. **Deliberately inert in Phase 1** — no filter/boost logic reads
  it yet; Phase 2 is what wires it into the `metadata_aware` chunk tag that already exists for
  exactly this purpose (chunking strategy's own docstring, written before Phase 1, already
  anticipated it).
**Consequences:** Every one of the 13 supported languages now reaches retrieval — but only Hindi
is actually indexed until Phase 2, so G3's existing confidence threshold is the honest backstop for
the other 12 (and for English's exclusion, structurally never reaches retrieval at all). No corpus,
FAISS, embedder, tokenizer, or G3 change. 260/260 tests pass (238 pre-existing + 22 new). Hot-path
cost measured, not assumed: G2's language-aware path is 1.35µs/call vs. 1.48µs/call for the
unchanged script-heuristic path — a real dict lookup replacing two regex searches, marginally
*faster*, not slower.

## ADR-009 — Phase 2: multilingual corpus built and measured at 100k/150k/200k; language-filtering wins; production untouched

**Date:** 2026-08-20.
**Status:** Accepted (measurement + artifacts). **Not yet decided:** which configuration (if any)
replaces the Hindi-only production index — that's explicitly the next decision, not made here.
**Context:** Phase 1 (ADR-008) made the pipeline language-aware but left the index Hindi-only.
Phase 2's job: build a real multilingual corpus/index at three candidate sizes and measure —
not assume — retrieval quality, memory, and whether language-aware retrieval actually helps.

**Sampling.** `scripts/build_multilingual_dataset_subset.py`, seed `20260820`. Real reservoir
sampling (Algorithm R, one sequential pass per language) over all 13 MSMARCO-XI train languages
— not the old "first N rows" bias `build_dataset_subset.py` has. Nested pools by construction
(100k ⊂ 150k ⊂ 200k row samples, not three independent draws) specifically so one held-out set
(494 queries, drawn only from the 100k pool, 38/language) has its gold passages present in all
three corpora — isolates the corpus-size effect from held-out-coverage drift. 771/1157/1542 rows
per language respectively (13 × equal allocation, matching the ~9.9767 chunks/row ratio measured
on the original Hindi build).

**A real data-integrity finding, disclosed rather than hidden:** MSMARCO-XI's `query_id` is a
*global* identifier shared across all 13 language files (the same underlying English query,
translated 13 ways) — not per-language-unique the way the Hindi-only pipeline implicitly assumed.
Independently reservoir-sampling from 13 files can therefore draw the *same* `query_id` from two
different languages, and since `doc_id`/`chunk_id` are derived purely from `query_id` (not
language-qualified), this collides two genuinely different chunks onto one chunk_id string.
Measured directly: 0.668% of rows at 100k, 0.977% at 150k, 1.267% at 200k (confirmed via a live
row-id audit, not estimated — e.g. `query_id=655605` is real Assamese content in one row and real
Malayalam content in another, in the same 100k sample). Effect: the FAISS index still holds both
real vectors (nothing is lost at the vector-storage level), but `chunk_lookup` retains only the
last-written chunk per colliding ID, so a hit on that specific chunk_id occasionally returns the
wrong language's text/doc_id at lookup time — a small, real noise source in the reported Recall/MRR
numbers below, not zero, not large enough to change any conclusion at this sample size. **Fix for
any future rebuild:** qualify `doc_id`/`chunk_id` with the language code
(e.g. `f"{lang}_{query_id}_{i}"`) — not applied here, since Phase 2's own scope explicitly
excludes revisiting the chunking/ID scheme; flagged for whoever builds the next real index.

**Real builds, three sizes** (`scripts/build_multilingual_index.py`, same `metadata_aware`
chunking, same `E5Embedder`, same FAISS config as production — HNSW32/efConstruction=200/
efSearch=64/SQ_fp16 — nothing about embedding model, tokenizer, or FAISS variant changed):

| Size | Rows | Chunks (real) | Build time | Per-language range |
|---|---|---|---|---|
| 100k | 10,023 | 99,981 | 328.9s | 7673–7712 (13 langs) |
| 150k | 15,041 | 150,050 | 825.5s | 11,524–11,565 |
| 200k | 20,046 | 199,982 | 686.8s | 15,365–15,411 |

Language balance within ~0.5% of perfectly even at every size — the equal-allocation sampling
strategy worked as designed.

**Language-aware retrieval, A/B/C, measured against the real 494-query multilingual held-out set**
(`scripts/eval_multilingual_retrieval.py`; k=100 candidate pool narrowed to top-10; "filter" =
hard same-language restriction with a documented empty-result fallback to unfiltered top-k;
"boost" = 1.10x score multiplier on same-language candidates, same k=100→10 window):

| Size | Mode | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| 100k | no_filter | 0.1883 | 0.4757 | 0.5648 | 0.3008 | 0.3605 |
| 100k | filter | 0.2146 | 0.5364 | 0.6518 | 0.3445 | 0.4140 |
| 100k | boost | 0.2146 | 0.5364 | 0.6518 | 0.3445 | 0.4140 |
| 150k | no_filter | 0.1903 | 0.4696 | 0.5385 | 0.2965 | 0.3514 |
| 150k | filter | 0.2126 | 0.5324 | 0.6296 | 0.3387 | 0.4053 |
| 150k | boost | 0.2126 | 0.5324 | 0.6296 | 0.3387 | 0.4053 |
| 200k | no_filter | 0.1781 | 0.4312 | 0.5040 | 0.2758 | 0.3269 |
| 200k | filter | 0.1984 | 0.4980 | 0.5911 | 0.3170 | 0.3797 |
| 200k | boost | 0.1984 | 0.4980 | 0.5911 | 0.3170 | 0.3797 |

**Two real findings, both measured, neither assumed:**
1. **Language filtering wins, consistently, at every size** — +8.7 to +9.1pp Recall@10, +4.1 to
   +4.4pp MRR@10 over no-filter, at 100k/150k/200k alike. Filter's empty-result fallback triggered
   only 0.2–0.4% of the time (the wide k=100 window almost always contains a same-language
   candidate) — a language filter here is safe, not merely helpful.
2. **"filter" and "boost" produced numerically identical results at every size.** Not a bug —
   verified by construction: a 1.10x multiplier on cosine-similarity-range scores was large enough
   to always push every same-language candidate above every cross-language one in the k=100
   window on this eval set, so the soft-boost degenerated into hard-filter behaviour. A smaller
   boost factor would plausibly differentiate them; not swept here (single representative value
   was the scope, per instruction) — worth a follow-up if a *softer* language preference (rather
   than a hard filter) is ever wanted.
3. **Quality DECLINES as corpus size grows, at every mode** — Recall@10 (filter): 0.6518 → 0.6296
   → 0.5911 as the corpus goes 100k → 150k → 200k, despite each larger corpus being a strict
   superset (same 494 queries, same gold passages present throughout, by the nested-pool design).
   Not a coverage artifact — more corpus means more same-language distractor chunks competing for
   the same ranks, and on this corpus that hurts more than the (structurally guaranteed, since
   pools are nested) larger candidate pool helps. **This is the single most consequential finding
   for the eventual size decision: bigger is not better here, on either resource or quality
   grounds.**

**Latency:** trivial at every size/mode — p50 1.6–1.9ms, p100 2.6–4.7ms, filter/boost included
(list-filtering k=100 candidates costs nothing next to the FAISS search itself). Not a
differentiator between configurations.

**Memory, real staged measurement** (`scripts/audit_multilingual_memory.py`, isolated subprocess
per size, matching R-032's methodology). Measured **twice** — once with the eager JSON
chunk_lookup (what `build_multilingual_index.py` produces by default), once after converting to
the lean SQLite-backed lookup (`scripts/convert_chunk_lookup_sqlite.py`, the same one production
actually uses, R-021) — the gap between them is itself a real, reportable number:

| Size | Steady-state RSS (JSON lookup) | Steady-state RSS (SQLite lookup, production-matching) | Peak WSet (SQLite) |
|---|---|---|---|
| 100k | 536.5MB | **396.9MB** | 482.1MB |
| 150k | 674.7MB | **462.2MB** | 549.0MB |
| 200k | 812.5MB | **532.1MB** | 618.9MB |

The eager JSON lookup costs 140–280MB more resident memory than the lean SQLite one, growing with
corpus size, exactly the R-021 finding restated at multilingual scale, now with three real data
points instead of one.

**Disk, real measured sizes** (dense FAISS + sparse BM25 + SQLite chunk_lookup; the redundant JSON
copy dropped from a deployment count since only one lookup format ships):

| Size | Dense | Sparse | SQLite lookup | Total (deployable) | Embedder (fixed, unchanged) |
|---|---|---|---|---|---|
| 100k | 103MB | 55MB | 129.8MB | ~288MB | 583MB |
| 150k | 154MB | 78MB | 194.1MB | ~426MB | 583MB |
| 200k | 205MB | 100MB | 257.8MB | ~563MB | 583MB |

Sparse/BM25 is never queried in production's dense-only mode (ADR-007) — shipping it is a real,
easy ~55-100MB cut available at every size if dense-only stays the shipped mode; not applied here
(out of Phase 2's explicit scope), flagged as a concrete, low-risk lever for whoever builds the
real deployment artifact.

**Production untouched:** `data/index/metadata_aware/` (Hindi-only, `baseline-hindi-only-v1`) was
never read or written by any Phase 2 script. All three candidate builds live in
`data/index/multilingual_{100k,150k,200k}/`, entirely separate paths.

**Artifacts:** `data/multilingual_dataset_manifest.json`, `data/multilingual_index_build_results.json`,
`eval/heldout_queries_multilingual.json` (494 queries, 38/language, held separate from the
production `eval/heldout_queries.json`, which is untouched), `eval/multilingual_retrieval_eval_results.json`,
`eval/multilingual_memory_audit.json` (SQLite-backend numbers; JSON-backend numbers recorded above,
not separately persisted), 9 new rows in `eval/ablation_ledger.csv` (3 sizes × 3 modes, real
measured Recall/MRR/nDCG/latency/RSS/build-time, `git_sha=7258789`).

**Consequences / open decision:** the smallest corpus (100k) is simultaneously the best-quality,
lowest-memory, and fastest-to-build configuration measured — an unusual, genuinely non-obvious
result worth double-checking before committing to it as final (e.g. a repeat run or a larger
held-out set would strengthen confidence that this isn't a one-sample artifact). Language
filtering should be treated as a strong, low-risk candidate for production wiring regardless of
which size is chosen — it improved every metric at every size with near-zero latency cost and a
near-zero fallback rate. Neither decision is made in this ADR; both are handed to the user with
the evidence above.

## ADR-010 — 100k selected as the production candidate; chunk_id collision fixed; BM25 dropped from the candidate artifact

**Date:** 2026-08-20.
**Status:** Accepted. User decision, evidence-driven: 100k over 150k/200k (Recall@10 0.652 vs.
0.630 vs. 0.591, filter mode; 100k also lowest RSS and fastest build — see ADR-009). "Do not
assume bigger is better" — confirmed correct by the data, not just accepted as a principle.

**Chunk-ID collision fixed (docs/DECISIONS_R.md context: `_rows_to_documents` in
`scripts/build_index.py`).** ADR-009 found and quantified the bug (MSMARCO-XI's `query_id` is
global, not per-language — 0.67-1.27% of rows collided across languages). Fix: an opt-in
`qualify_doc_id_by_language: bool = False` parameter — default unchanged (the single-language
Hindi pipeline's doc_id format, and `eval/heldout_queries.json`'s `passage_id` compatibility, are
byte-identical to before), `True` only for the multilingual build path, producing
`f"{target_lang}::{query_id}_{i}"` (e.g. `"asm_Beng::655605_0"`). 6 new regression tests
(`tests/test_build_index_multilingual_ids.py`) pin both the fixed collision behavior and the
unchanged default. `eval/heldout_queries_multilingual.json`'s `passage_id`s were regenerated to
match (same 494 queries, same text, verified by direct text-equality check during regeneration —
not resampled). Rebuilt 100k index: 99,981 chunks, exactly matching `chunk_lookup`'s own entry
count (zero collisions, confirmed directly) — real Recall@10 improved marginally to 0.6559 (was
0.6518, the ~0.4pp gap is exactly the previously-silently-dropped chunks), fallback_rate now
exactly 0.000 (was 0.002).

**BM25/sparse dropped from the candidate build.** `save_built_index()` (`src/vrag/index/
persistence.py`) now accepts `sparse: SparseIndex | None` — `None` skips writing a sparse artifact
at all, additive, every existing caller unaffected. `scripts/build_multilingual_index.py` defaults
to `build_sparse=False` for the production candidate (dense-only is the intentional retrieval
mode, ADR-007 already never loads it into memory regardless — this drops the dead ~55MB disk
weight too). `SparseIndex`/`HybridRetriever`'s sparse/hybrid modes are untouched; `--build-sparse`
flag still produces a real one if a future experiment needs it.

**English added as a 14th genuinely-indexed language.** Using the `English_passages` field every
MSMARCO-XI row already carries (Phase 0 finding) — 771 rows (the same per-language budget every
other language got at the 100k tier), deduped by query_id, drawn from the already-sampled 100k
pool (no new download). 7,697 English chunks appended to the existing dense index via incremental
`DenseIndex.add()` (HNSW supports this natively; the sqfp16 ScalarQuantizer's calibration doesn't
need retraining — E5 embeddings are L2-normalised regardless of input language). Final candidate:
**107,678 chunks, 14 languages**, balanced within ~0.5% of each other. `src/vrag/languages.py`
updated: `en-IN` moves from excluded to `SUPPORTED_LANGUAGES`, and `CURRENTLY_INDEXED_LANGUAGES`
now equals `SUPPORTED_LANGUAGES` for the first time (was a strict subset through Phase 1/2).

**Final candidate memory (real, staged, isolated subprocess, lean SQLite lookup):**
steady-state RSS **406.5MB**, peak WSet **492.6MB** — for comparison, the eager JSON lookup would
cost 536-812MB at these corpus sizes (ADR-009's measured gap). Disk: dense 103MB → grows with the
English append (not re-measured standalone), SQLite lookup 140.2MB, no sparse file.

**Real end-to-end smoke test** (`scripts/smoke_test_multilingual_candidate.py`, via
`VRAG_INDEX_DIR` env-var override — see `src/vrag/retrieval/interface.py`, never a hardcoded
default swap, see that file's docstring for why): `/healthz` reports `retrieval:"real"`, no stub
fallback, 5 languages (Hindi/English/Bengali/Tamil/Marathi) all reach retrieval with correct
`query_language` propagation; separately verified real *answered* responses (Assamese examples)
carry unique, language-qualified, valid `chunk_id`/`passage_id` citations.

**`data/index/metadata_aware/` (Hindi-only, `baseline-hindi-only-v1`, live on Render) untouched
throughout** — confirmed via `git diff`/`ls`, no script in this ADR's scope reads or writes that
path.

## ADR-011 — Phase 3: language filter wired into production retrieval; Track B generation is language-aware; real G3 re-evaluation

**Date:** 2026-08-20. **Status:** Accepted (code + measurement). Deployment still explicitly out
of scope — nothing in this ADR touches Render or the live URL.

**Retrieval: the "filter" strategy ADR-009/ADR-010 measured as the winner is now wired into
`HybridRetriever.retrieve()` for real** (`src/vrag/retrieval/hybrid.py`), not just accepted-but-
inert (ADR-008's original `language` param). A real, mapped `language` searches a wide pool
(`_LANGUAGE_FILTER_WIDE_K=100`, same value Phase 2's measurement used) and restricts to
same-language chunks before truncating to the caller's `k`; falls back to the unfiltered ranking
if no same-language candidate exists in the window (never manufactures a zero-result failure).
`language=None` or an unmapped code (e.g. `te-IN`) is untouched — searches exactly as before. 5
new tests (`tests/retrieval/test_hybrid.py`) cover filtering, fallback, the unfiltered path, and
that the search width actually widens only when filtering is active.

**`_INDEX_DIR` stays the Hindi-only production path by default — deliberately not hardcoded to
the new candidate.** A new `VRAG_INDEX_DIR` env var (unset by default) lets a local session opt
into `data/index/multilingual_100k/`. This was a real correction made mid-session: hardcoding the
new path would mean the *next* `src/`-touching commit that reaches a real deploy silently falls
back to the stub in production, since the multilingual candidate is gitignored with no release
asset. The default swap is deferred to whenever real deployment is actually decided, not bundled
into this ADR.

**Generation: Track B's system prompt is no longer hardcoded to Hindi**
(`src/vrag/generation/sarvam_llm.py`). `_build_system_prompt(generation_language)` names the real
target language + script (e.g. "Hindi (Devanagari script)", "Tamil (Tamil script)") via a new
`LANGUAGE_DISPLAY_NAMES` table in `src/vrag/languages.py`; falls back to Hindi only when no real
signal exists (`None` or an unmapped code) — a documented default, not an assumption baked into
the prompt text itself. `generate()` gains a `generation_language` parameter, threaded from
`ctx.data["generation_language"]` (set once in `ExtractAnswerStage`, ADR-008: defaults to
`query_language`) through `GenerateStage`. 14 new tests
(`tests/generation/test_language_aware_generation.py`) cover the prompt-building function
directly and, via `httpx.MockTransport`, a real `generate()` call's actual wire payload for
Hindi/English/Bengali/Marathi/Kannada/Tamil/Urdu.

**G3 re-evaluated on the real 100k/14-language candidate — two evaluations, kept deliberately
separate** (`scripts/reeval_g3_on_multilingual_100k.py`), because they answer different
questions:

1. **The literal original 500-query Hindi held-out set, rerun as requested, against the new
   index.** Checked directly *before* interpreting results (not assumed): the new index's Hindi
   slice (771 independently reservoir-sampled rows) has **zero** passage_id overlap with the
   original 500 queries' gold passages (drawn from the old pipeline's first-10,000-rows working
   pool, 13x larger, differently sampled). Result: Recall@1/5/10 = 0.0, 456/500 (91.2%) abstained,
   **100% of those abstentions are `not_in_corpus_at_all`** — confirming this is a corpus-coverage
   mismatch from independent resampling, not a retrieval-quality regression. Reported plainly, not
   hidden, matching this project's R-037/R-038 forensic discipline.
2. **The new 532-query multilingual held-out set** (494 + 38 English, one 38-query slice per
   language) — the fair, apples-to-apples measurement, since its gold passages exist in the new
   index by construction (0 `not_in_corpus_at_all`). Real numbers: Recall@10=0.679, MRR@10=0.362 —
   reasonable retrieval quality. **But abstain rate is 66.5% (354/532)** — dramatically higher than
   the Hindi-only baseline's 25.8% (R-034/R-035). Evidence-location breakdown: 199 abstentions have
   the right answer in the top-10 already (evidence found, scored under TAU anyway), 142 more are
   in-corpus but ranked outside top-20. **TAU=0.8835 was calibrated on Hindi-only, same-script
   score distributions (R-015) — cross-language/cross-script E5 similarity runs measurably lower**
   (already observed in this session's earlier language-routing diagnostic: an English query
   scored ~0.024 lower than the equivalent Hindi query for the same intent), and now 12 of 14
   indexed languages contribute non-Hindi-same-script queries to the aggregate.

**Direct answer to "does the multilingual filter materially improve the previous 25.8%
abstention behavior": No — it makes it substantially worse (66.5% vs. 25.8%), despite decent
underlying Recall@10.** TAU is **not** changed in response — per explicit instruction, this
finding is reported, not silently patched. Recalibrating G3 for the new multilingual score
distribution is flagged as necessary future work before this candidate could be considered
demo-ready, not attempted here.

**Regression case, not special-cased:** "भारत की राजधानी क्या है?" (Hindi) and "What is the
capital of India?" (English) both abstain (top1 0.821 and 0.786, both under TAU) — **neither
confidently cites the wrong answer** (the top-ranked-but-incorrect chunk is a Bangkok/Thailand
passage; G3 correctly refuses rather than asserting it). The quality bar requested is met: no
confidently-wrong capital-of-India answer, in either language, on the new candidate.

**No corpus, FAISS, embedder, tokenizer, or G3 change beyond what's described above** — TAU/MARGIN
values themselves are byte-identical to `g3_confidence.py` before this ADR. 286/286 tests pass.

## ADR-013 — Phase 4: G3 recalibration attempted on the multilingual candidate; evidence does not support changing TAU/MARGIN

**Date:** 2026-08-20. **Status:** Accepted — as a decision NOT to change production config.
`src/vrag/guardrails/g3_confidence.py` is byte-identical before and after this ADR. Deployment
still explicitly out of scope.

**Task:** ADR-011 found the multilingual/filter candidate abstains on 66.5% of the 532-query
held-out set (vs. 25.8% Hindi-only baseline), TAU=0.8835 unchanged. This ADR is the requested
follow-up: recalibrate G3 for the new score distribution using evidence, not intuition, and
report the recommended rule — without touching production config.

**Method** (`scripts/calibrate_g3_collect.py` + `scripts/calibrate_g3_sweep.py`): collected
per-query top1/weakest-of-5/top-20 scores and gold-passage relevance for all 532 queries via the
REAL production `retrieve()` path (`VRAG_INDEX_DIR` pointed at `data/index/multilingual_100k/`,
same mechanism as ADR-011/ADR-012's smoke test — not a reimplementation of the filter logic), then
swept TAU over the full observed score range (0.8027–0.9527, step 0.0025, 61 points) plus a
per-language breakdown, a formula-based per-language offset rule, and a small MARGIN grid. Raw
artifact: `eval/g3_calibration_multilingual_100k_raw.json`. Full sweep output:
`eval/g3_threshold_sweep_multilingual_100k.json`.

**Headline finding — the real blocker is signal quality, not threshold placement.** Judged against
gold-passage relevance, the top-1 score for genuinely-correct hits (median 0.8846) and
genuinely-wrong hits (median 0.8686) overlap heavily: wrong-hit scores go as high as 0.9463
(above the correct-hit median) and correct-hit scores go as low as 0.8291 (below the wrong-hit
median). Unlike R-015's original Hindi-only in-domain-vs-OOD calibration (clean separation), top-1
score alone is a weak discriminator here — very plausibly the same cross-lingual E5 depression
ADR-011 already identified, now compounded by MSMARCO-XI's passage reuse across query_ids (noted
in `g3_confidence.py`'s own docstring) landing on a multilingual, mixed-script corpus.

**Consequence, confirmed by the actual sweep, not assumed:** at the current operating point,
answered-but-wrong is already a bigger problem than most people would guess —
**precision_of_accepted = 0.348** (of the 178 queries G3 currently lets through, only 62 are
actually grounded in a correct passage; 116 are false-accepts). Sweeping TAU down across the
entire viable range barely moves this: the best global TAU meeting today's own precision floor
(0.8827, vs. 0.8835 today) only gains 2 answered queries (178→180) — true accepts and false
accepts rise together, roughly in lockstep, across the whole range. There is no global threshold
that substantially cuts abstention without proportionally increasing confidently-wrong answers.

**Per-language free optimization (candidate rule B) looked promising in isolation but failed its
own stability check — reported honestly, not shipped.** Grid-searching an independent TAU per
language against this same 532-query set (38/language) produced dramatic-looking per-language
thresholds (e.g. `eng_Latn`/`guj_Gujr` dropping to 0.830, answering 38/38 and 38/38; `hin_Deva`
*rising* to 0.940, answering only 2/38) and a modest aggregate gain (answered 178→204, abstain
rate 66.5%→61.7%). Before trusting this, split each language's 38 queries by query_id parity
(even/odd) and re-optimized independently on each half: **only 2 of 14 languages (`guj_Gujr`,
`mal_Mlym`) picked thresholds that agree within 0.02 between halves.** The other 12 disagree,
often wildly — this is classic overfitting to a 38-query sample, not a real per-language signal.
**Rule B is not recommended.**

**Normalized per-language offset rule (candidate rule C)** — `TAU_lang = 0.8835 − (global_median −
lang_median)`, clipped to ±0.03, a single formula requiring only each language's own median top-1
score (no free grid search) — is more principled than B but empirically **worse than doing
nothing**: aggregate answered drops to 162 (vs. 178 baseline), abstain rate rises to 69.6%, and
precision drops to 0.315. Reason: `hin_Deva` and `eng_Latn` (the two best-performing languages —
baseline answered 29/38 and 32/38 respectively) have the *highest* median top-1 scores, so the
formula makes them *stricter* (TAU→0.9135) to "equalize" cross-language score meaning — exactly
backwards from what reduces abstention where it's currently working best. **Rule C is not
recommended as specified.**

**MARGIN re-swept at this candidate's distribution, per `g3_confidence.py`'s own stated
requirement** ("if TAU is ever recalibrated, MARGIN must be re-swept too"): a small grid
(0.0–0.03) at both the current and best-global TAU shows precision creeping from 0.348 to at most
0.381 while answered collapses from 178 to 97 at MARGIN=0.03 — the same disproportionate cost
R-015/P-015 found on the original Hindi-only corpus. **MARGIN=0.0 remains the correct pairing.**

**Decision: TAU=0.8835, MARGIN=0.0 are unchanged.** Per the ablation-ledger discipline already
established on this project ("a gap smaller than the noise floor is not a result — say the
options were tied and ship the cheaper one," and R-038's net-negative reranking experiment,
reported honestly and not enabled): the evidence does not support a threshold-only recalibration
that substantially improves the 66.5% abstention rate without either (a) an unvalidated,
overfit-prone per-language rule, or (b) accepting materially worse precision than today's already
weak 0.348. This is a legitimate, different-from-hoped-for finding, reported per the explicit
instruction not to force a match to the old 25.8% number.

**Regression cases, not special-cased:** "भारत की राजधानी क्या है?" (Hindi) and "What is the
capital of India?" (English) — both already abstain under the current TAU (top1 0.821/0.786, both
below 0.8835) rather than confidently citing the wrong country (ADR-011). Every candidate rule
explored above keeps TAU at or above roughly this range in the languages that matter for this
case, so this regression stays safe; it was not used to select or tune any threshold.

**Flagged as real future work, not attempted here (out of Phase 4's scope: threshold-only,
deterministic, no ML model):** the root cause is that top-1 cosine similarity alone doesn't carry
enough signal to separate correct from incorrect retrieval on this multilingual, mixed-script,
passage-reused corpus. Fixing the *abstention* number for real likely requires improving the
*signal* (e.g. a reranker trained/evaluated per-language rather than the single cross-encoder
R-038 already found net-negative on the Hindi-only corpus; a larger or differently-tuned
multilingual embedder; or combining top1 with additional cheap features beyond top1-vs-top5
margin, which this ADR's grid already found unhelpful alone) — not a smarter threshold.

**No corpus, FAISS, embedder, retrieval, or generation code changed by this ADR.**
`src/vrag/guardrails/g3_confidence.py` is untouched. 286/286 tests still pass (no test changes were
needed — no production code changed).

## ADR-014 — Phase 5: cheap deterministic confidence-signal experiment; no candidate cleared the safety bar

**Date:** 2026-08-20. **Status:** Accepted — as a decision NOT to add any new signal to G3.
`src/vrag/guardrails/g3_confidence.py` is byte-identical before and after this ADR. TAU=0.8835,
MARGIN=0.0 untouched. No production code changed. Deployment still out of scope.

**Task:** ADR-013 found top1-score alone is a weak discriminator on the multilingual candidate.
This ADR investigates whether a cheap, deterministic, CPU-only additional signal (no neural model,
no LLM, no network call) can do better — either standalone or in a simple two-feature combination —
using the same 532-query held-out set, with strict hindsight-leakage discipline (every feature is
computed from query text + retrieved hits' text/score/language only; gold labels are used solely
to evaluate features, never to construct them).

**Method:** `scripts/collect_g3_feature_data.py` re-collected the real production `retrieve()`
output (top-20, full passage text + language this time, not just scores) for all 532 queries;
`scripts/g3_feature_experiment.py` computed 15 candidate features and evaluated each via
rank-based AUC (correct vs. wrong top-1), a precision-floor-constrained threshold sweep (same
0.3483 floor as ADR-013, for direct comparability), per-language AUC, and a global even/odd
stability check. Four two-feature combinations (top1+gap, top1+concentration, top1+lexical,
top1+same-language) were grid-swept the same way. A logistic-regression combination was fit purely
as an offline diagnostic (numpy gradient descent — no new dependency; strict train/val split by
query_id parity), per the explicit instruction that it must not be treated as production-ready.
Raw artifacts: `eval/g3_feature_experiment_raw.json`, `eval/g3_feature_experiment_results.json`.

**Real bug found and fixed before it could silently corrupt every lexical feature:** Python's `re`
module's `\w` excludes Unicode combining marks (categories Mn/Mc). Naive `\w+` tokenization
shatters Devanagari/Bengali/Gujarati/etc. text at every vowel sign — "भारत" (4 real characters)
split into 4 garbage single-character tokens. Fixed with a manual scan treating L*/N*/Mn/Mc as
word-continuing, verified against real Hindi text before use. Worth flagging for any future NLP
work on this corpus: `\w+`/`\b` on Indic scripts is silently broken in stdlib `re`.

**Individual features — two beat top1 on AUC:**

| feature | AUC | best answered @ precision floor | per-lang AUC mean±std |
|---|---|---|---|
| `score_std_top5` (top-5 score std-dev) | **0.671** | 221 | 0.642 ± 0.112 |
| `gap15mean` (top1 − mean(top5)) | **0.667** | 203 | 0.634 ± 0.121 |
| `content_overlap_top1` (fixed-tokenizer lexical overlap, tokens ≥4 chars) | 0.644 | 148 | 0.632 ± 0.103 |
| `top1` (ADR-013 baseline) | 0.640 | 180 | 0.622 ± 0.130 |
| `concentration_ratio` (top1 / sum(top5)) | 0.635 | 175 | 0.606 ± 0.140 |
| `lexical_overlap_top1` | 0.626 | 25 | — |
| `gap12` (top1 − top2) | 0.611 | 132 | — |
| `mutual_redundancy_top3` (pairwise Jaccard among top-3 passages) | 0.578 | 8 | — |
| `zscore_top1` | 0.552 | 66 | — |
| `same_lang_consistency` | 0.500 | none | structurally constant — see below |
| `entropy_norm`, `n_hits`, `query_len_tokens` | 0.48–0.54 | none | uninformative |

`same_lang_consistency` scored exactly 0.5 (no threshold clears the floor) because production's
existing hard language filter (ADR-011/012) already forces near-total same-language homogeneity in
what `retrieve()` returns — the feature has almost no variance left to be informative on. A
legitimate null result, not a bug.

**Latency (measured, `time.perf_counter`, not estimated):** all 15 features combined cost 535µs/
query, dominated by tokenizing passage text for the lexical/redundancy features. The two
score-arithmetic features (`concentration_ratio` alone: 1.5µs/query; `score_std_top5` +
`gap15mean` together: 19µs/query) need no text processing at all — negligible against the 200ms
pipeline budget either way, HOTPATH-safe by a wide margin.

**Combinations:** A (top1+gap12), C (top1+lexical), D (top1+same-language) each collapsed to
exactly the top1-only solution (178 answered, 0.3483 precision) — the second feature added zero
filtering power at the grid optimum. **B (top1+concentration_ratio)** was the one real exception:
208 answered, precision 0.351, and concentration_ratio was verified to be doing genuine work, not
just riding along — at combo B's own (lowered) top1 threshold alone, precision would have
collapsed to 0.276 (377 answered), but ANDing with concentration_ratio removes 169 of those
candidates at a 4.5:1 bad:good ratio, restoring precision above the floor.

**Why every one of these was rejected despite the attractive aggregate numbers — three
independent findings, each disqualifying on its own:**

1. **The flagship regression case exposes the two best-AUC features directly.** For "भारत की
   राजधानी क्या है?" (top1=0.8208, top-1 passage is the same wrong Bangkok/Thailand passage
   ADR-011 already flagged), `score_std_top5` and `gap15mean` both **cross their thresholds and
   accept it** — precisely the "capital-of-India answered with wrong-country evidence" failure
   mode this whole recalibration effort exists to prevent. `concentration_ratio` alone accepts it
   too, by a razor-thin margin. Only combination B correctly abstains, and only because its top1
   sub-threshold (0.859) happens to still exceed this specific query's 0.8208 — not because
   `concentration_ratio` itself screens out the bad evidence.

2. **That "save" doesn't generalize.** Among the 10 highest-top1 *wrong* queries in the whole
   532-query set (real confidently-wrong cases, same failure family as the regression case — e.g.
   "side effects of malarone tablets" at top1=0.9463, wrong passage), **7 of 10 are accepted by
   `score_std_top5`, `gap15mean`, AND combination B alike** — all three fail together whenever
   top1 itself is already very high (≳0.93), which is exactly the highest-risk region. (These 10
   were already false-accepted by today's production TAU too, so they're not *new* risk — but they
   prove none of the candidates add real protection where it matters most.)

3. **The apples-to-apples "new risk" number is the most direct evidence.** Restricting to the 354
   queries current production already abstains on (top1 < 0.8835) and asking what each candidate
   would newly flip to "accepted": `score_std_top5` → 26 newly-correct vs. **61 newly-wrong**
   (2.35 wrong per correct); `gap15mean` → 23 vs. **59** (2.57 wrong per correct); combination B →
   24 vs. **59** (2.46 wrong per correct). Every candidate trades roughly 2.3–2.6 new wrong answers
   for every 1 new correct answer recovered — an unfavorable, consistent ratio across all three.

**Per-language stability:** every feature's per-language AUC has substantial spread (std
0.10–0.14). `tam_Taml` is the sharpest warning sign: `score_std_top5` (AUC=0.388) and `gap15mean`
(AUC=0.352) are *inverted* for Tamil — actively anti-predictive on exactly the corpus slice where
they look strongest in aggregate. A single global rule built on these features would help some
languages and actively hurt at least one.

**Offline logistic-regression diagnostic does not generalize**, as instructed to check: train
AUC=0.722, val AUC=0.632 (train/val split by query_id parity) — a 0.09 gap, confirming that
squeezing more out of a learned combination overfits fast at n=532 (118 positives). Reinforces
"don't invent complexity."

**Decision: reject every candidate for production adoption.** No individual feature or simple
combination clears the bar of "meaningful, safety-preserving improvement" — the two strongest
(by AUC) fail the flagship regression case outright, the one combination that survives it fails
the broader stress test the same way, and the real newly-introduced-error ratio (≈2.5:1 bad:good)
is unfavorable across the board. This is not "G3 is unfixable" — `concentration_ratio` is a real,
verified, near-free signal that does genuine filtering work in isolation (documented above for
whoever revisits this) — but nothing explored here is safe enough to ship as-is. Consistent with
this project's standing ablation discipline (R-038's honest net-negative report; ADR-013's "the
evidence does not support a change"): reported honestly rather than forced into an adoption the
data doesn't support.

**No corpus, FAISS, embedder, retrieval, generation, or guardrail code changed by this ADR.**
`src/vrag/guardrails/g3_confidence.py` is untouched. 286/286 tests still pass (no test changes
needed — no production code changed).
