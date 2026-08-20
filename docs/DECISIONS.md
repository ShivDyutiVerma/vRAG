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
