# AGENT_BUILD_SPEC.md
## Voice-Enabled RAG — Master Specification for Claude Code

> **Read this first, fully, before writing any code.**
> This is the single source of truth for the project. If anything you are asked to do
> contradicts this document, stop and ask. If this document is wrong, fix *this document*
> in the same commit as the code change.

**Project codename:** `vrag`
**Deadline:** 2026-08-22, 23:59 IST. **No resubmissions.**
**Build window:** 2026-08-17 → 2026-08-22, split across two people working in parallel — see
`docs/TEAM_SPLIT.md` for the authoritative day-by-day schedule and who owns what.

---

# §0. How to use this document

This spec is written for an AI coding agent working across **multiple sessions with no shared memory
between them**. That constraint drives everything below:

- Work is split into **8 phases** (§9), each with **hard exit criteria**. Never start phase N+1 until phase N's exit criteria pass.
- State lives in `docs/PROGRESS.md`, not in your context window. Read it at session start, write it at session end (§8).
- Decisions are immutable and appended to `docs/DECISIONS.md`. Never silently reverse a prior decision.
- Every session follows the ritual in §8. No exceptions.

**Order of authority when sources conflict:**
`docs/DECISIONS.md` (newest wins) → `AGENT_BUILD_SPEC.md` (this file) → `docs/PROGRESS.md` → code comments → your own judgement.

---

# §1. The mission, in one paragraph

Build a system where a user speaks a question into a browser, and receives a **grounded, cited
answer** derived from the `ai4bharat/MSMARCO-XI` corpus — with the retrieval-and-answer path
completing in **under 200ms server-side**, wrapped in a real orchestration harness, protected by
guardrails that know when to refuse, and backed by measured P50/P70/P100 latency evidence across a
statistically meaningful query set.

The deliverable is judged. Working code is necessary but not sufficient — **the evidence artifacts
(eval tables, latency distributions, architecture write-up) carry equal weight.** Build them as you
go, not at the end.

---

# §2. Non-negotiable constraints

These come directly from the task brief. Violating any of these invalidates the submission.

| ID | Constraint | Source |
|----|-----------|--------|
| C1 | STT must be **Sarvam or ElevenLabs**. Pick exactly one. No Whisper, no Deepgram, no browser SpeechRecognition API. | Brief §1 |
| C2 | Chunking must be **plural and deliberate**. A single fixed-size strategy is an explicit fail. | Brief §2 |
| C3 | Chunking + retrieval + everything through final output **< 200ms**. | Brief §3 |
| C4 | Report **P50 / P70 / P100** across a "reasonable number" of test queries. Not one run. | Brief §4 |
| C5 | Must run inside a **harness**: tool calls, retries, structured I/O, error recovery. Not a raw prompt call. | Brief §5 |
| C6 | Must have **guardrails**: off-topic, unsafe input, hallucination/groundedness checks, and a demonstrable refusal path. | Brief §6 |
| C7 | Deliverables: public GitHub repo, **live working link**, 2 videos. | Brief |

**Decision on C1: use Sarvam.** Rationale recorded as ADR-001 (§6). Do not revisit without a new ADR.

---

# §3. The central engineering problem — read this twice

Most teams will fail C3 (200ms), then quietly report a number that isn't what they measured.
The correct approach is to **define the metric precisely, engineer to it honestly, and report the
full distribution including what falls outside the budget.**

## 3.1 Latency is a network problem before it is a code problem

A round trip from an Indian server to a US-hosted inference API can consume the entire 200ms budget
**before a single token is generated**. No amount of FAISS tuning fixes that.

**Therefore Phase 0's first task is `scripts/probe_latency.py`** — measure real RTT and
time-to-first-token from your actual deployment region to every candidate provider, *before*
committing to an architecture. Architecture follows measurement, not the reverse.

Provider options to probe (do not assume any of these numbers — measure them):
- Sarvam STT (streaming WebSocket + REST) — India-hosted, likely lowest RTT from an Indian deployment
- Sarvam LLM/chat endpoints — same regional advantage
- Groq — very fast inference, but check the geographic RTT penalty
- Any local small model (`Qwen2.5-1.5B-Instruct` class) via `llama.cpp` — zero network RTT, but needs CPU/RAM headroom

If the probe shows a hosted LLM cannot clear the budget from your region, the answer is **not** to
fake the number. It's ADR-worthy: co-locate the app server with the provider region, or move
generation local, or adopt the two-track design below.

## 3.2 The metric definition (write this in the README verbatim)

> **`t_pipeline`** is measured server-side, from the moment the final transcript is available
> to the moment the first grounded answer token is emitted to the client.
> It excludes: client→server network transit, microphone capture, and speech duration.
> It includes: input guardrails, query embedding, hybrid retrieval, fusion, grounding gate,
> and answer generation up to first token.
> Index construction (chunking + embedding + index build) is a **one-time offline cost**,
> reported separately in §7.4 and excluded from `t_pipeline`.

This is defensible and standard. State it plainly and judges will trust the rest of your numbers.
Also report `t_e2e_voice` (mic-stop → first audible/visible answer) as a secondary honest number,
so nobody accuses you of hiding STT cost.

## 3.3 The two-track answer design (this is what actually makes 200ms real)

```
                    ┌─→ TRACK A: extractive answer  (~15–30ms)  ──→ emitted immediately
retrieval done ─────┤
                    └─→ TRACK B: generative synthesis (streams over Track A) 
```

- **Track A (always runs, always fast):** select the best-supporting span from the top-ranked
  passage using a cheap scoring pass. This is a *real, useful, grounded answer*. It reliably lands
  well inside 200ms.
- **Track B (streams in behind it):** the LLM rewrites/synthesises a fluent answer with citations.
  Its first token replaces Track A in the UI when it arrives.

This is not a hack — it's how production voice assistants handle exactly this tradeoff. Document it
as a deliberate architectural choice in `docs/ARCHITECTURE.md`. If Track B's TTFT also lands under
200ms after Phase 6 optimisation, report both and celebrate. If it doesn't, you still meet C3 with a
genuine answer and you've shown engineering maturity rather than corner-cutting.

## 3.4 Deadline propagation (the harness's killer feature)

Every request carries a **budget in milliseconds**. Each stage receives `remaining_ms` and must
either complete within its allocation or degrade. Implement this in `harness/budget.py`:

```
Request enters with budget = 200ms
  ├─ each stage checks remaining_ms before starting
  ├─ if remaining_ms < stage.min_viable_ms → skip stage, mark degraded, continue
  ├─ optional stages (rerank, query expansion) are dropped first
  └─ the response records which stages were skipped and why
```

This means the system **cannot** blow the budget — it sheds quality instead of time. That single
design choice is the strongest technical story in this project. Make sure it appears in the demo
video: show a request where rerank gets dropped under load and the response says so.

---

# §4. Target latency budget

Populate the "Measured" column during Phase 6. These targets are the design contract; they are
hypotheses until probed.

| # | Stage | Target p50 | Min viable | Optional? | Measured |
|---|-------|-----------|-----------|-----------|----------|
| 1 | Input guardrail (local) | 2ms | 2ms | no | _TBD_ |
| 2 | Query normalise + embed (ONNX int8, batch=1) | 8ms | 8ms | no | _TBD_ |
| 3 | Dense search (HNSW) | 3ms | 3ms | no | _TBD_ |
| 4 | Sparse search (BM25) | 5ms | — | **yes** | _TBD_ |
| 5 | RRF fusion | <1ms | <1ms | no | _TBD_ |
| 6 | Cross-encoder rerank (top-20) | 25ms | — | **yes** | _TBD_ |
| 7 | Grounding gate (threshold + margin) | 1ms | 1ms | no | _TBD_ |
| 8a | **Track A extractive span select** | 10ms | 10ms | no | _TBD_ |
| 8b | Track B generation TTFT | 110ms | — | **yes** | _TBD_ |
| 9 | Output verify (first sentence) | 5ms | 5ms | no | _TBD_ |
| | **Total to Track A answer** | **~30ms** | | | _TBD_ |
| | **Total to Track B first token** | **~160ms** | | | _TBD_ |

**Hot-path rules (enforce in code review):**
- No network calls on the hot path except the LLM. Embeddings, BM25, reranker, and all guardrails run **in-process**.
- No cold starts. All models load at application boot and are warmed with a dummy inference before the health check returns ready.
- No per-request disk I/O. Index is memory-resident.
- No Python-level JSON parsing of large payloads on the hot path.

---

# §5. System architecture

## 5.1 Component diagram

```
┌────────────────────────────────────────────────────────────────┐
│  BROWSER (HTTPS required for mic)                              │
│  MediaRecorder → PCM/WAV chunks → WebSocket                    │
│  Renders: live transcript, Track A answer, Track B stream,     │
│           citations, refusal states, live latency HUD          │
└───────────────────────────┬────────────────────────────────────┘
                            │ WS: audio in / events out
┌───────────────────────────▼────────────────────────────────────┐
│  FASTAPI ORCHESTRATOR  ("the harness")                         │
│                                                                │
│  Stage 0  AudioIngest      VAD, buffering, format validation   │
│  Stage 1  Transcribe       Sarvam streaming WS → final text    │
│  ─────────── t_pipeline clock starts here ───────────          │
│  Stage 2  InputGuard       safety + scope + language           │
│  Stage 3  QueryPrep        normalise, embed, (opt) expand      │
│  Stage 4  Retrieve         dense ∥ sparse (parallel)           │
│  Stage 5  Fuse             RRF, (opt) cross-encoder rerank     │
│  Stage 6  GroundGate       confidence + margin → maybe ABSTAIN │
│  Stage 7a ExtractAnswer    Track A — span selection            │
│  Stage 7b Generate         Track B — streaming LLM + tools     │
│  Stage 8  VerifyOutput     groundedness, citation validity     │
│  Stage 9  Assemble         structured response + trace emit    │
└───────────────────────────┬────────────────────────────────────┘
                            │
        ┌───────────────────┼──────────────────┐
        ▼                   ▼                  ▼
  In-process assets    Sarvam APIs        Telemetry sink
  • FAISS HNSW         • STT (WS)         • traces.jsonl
  • BM25 index         • LLM (Track B)    • per-stage ns timings
  • embedder (ONNX)
  • reranker (ONNX)
  • guardrail models
```

## 5.2 Technology decisions

| Layer | Choice | Why |
|-------|--------|-----|
| Language / runtime | Python 3.11+ | ecosystem for RAG; 3.11 for perf |
| Web framework | FastAPI + uvicorn | native async, WebSocket support, Pydantic-integrated |
| STT | **Sarvam** (streaming WebSocket) | corpus is Indic; Indian-hosted → low RTT; code-mixing support |
| Embeddings | `multilingual-e5-small` exported to **ONNX int8** | multilingual incl. Indic; small enough for single-digit-ms inference |
| Dense index | **FAISS** `HNSW32`, `efConstruction=200`, `efSearch` tuned in Phase 6 | best latency/recall for <1M vectors; in-process |
| Sparse index | **`bm25s`** (scipy-sparse backed) | pure-Python `rank_bm25` is too slow for the hot path — benchmark and confirm |
| Fusion | Reciprocal Rank Fusion (RRF, k=60) | no score normalisation needed across heterogeneous scorers |
| Reranker | `bge-reranker-v2-m3` or MiniLM cross-encoder, ONNX | optional stage, budget-gated |
| Generation | Sarvam LLM primary; Groq fallback — **decided by Phase 0 probe** | regional RTT dominates |
| Validation / schemas | Pydantic v2 | typed stage I/O = the "structured I/O" requirement |
| Retries | `tenacity` | per-stage policy, exponential backoff, jitter |
| Frontend | Vanilla TS or minimal React + Vite | mic + WebSocket + streaming render; avoid heavy framework risk |
| Hosting | See §5.3 | HTTPS mandatory |
| Telemetry | JSONL trace records + `polars`/`pandas` for analysis | reproducible latency analytics |
| Tests | `pytest` + `pytest-asyncio` | includes a latency regression test |

### Gotchas the agent must not get wrong
- **E5 models require prefixes.** Queries must be embedded as `"query: {text}"`, passages as `"passage: {text}"`. Omitting this silently degrades recall by a large margin. Encode this in `index/embedder.py` and add a unit test asserting the prefix is applied.
- **Normalise embeddings** and use inner-product (`IndexHNSWFlat` with `METRIC_INNER_PRODUCT`) so IP == cosine.
- **BM25 over Indic text** needs Unicode-aware tokenisation. Do not use `.split()` on whitespace and assume it works for Devanagari. Use a Unicode word-boundary regex at minimum; note the limitation in `docs/DECISIONS.md`.
- **Never `await` inside a lock** on the hot path.
- Dense and sparse retrieval must run **concurrently** (`asyncio.gather`), not sequentially.

## 5.3 Deployment

Requirement: HTTPS (browsers refuse `getUserMedia` on plain HTTP over a network origin).

- **Primary recommendation:** a single container on Render / Railway / Fly.io, region chosen to minimise RTT to the LLM provider (per Phase 0 probe). Serves both the API and the built frontend as static files — one origin, no CORS complexity.
- **Fallback (deploy insurance):** Hugging Face Spaces with a Gradio or FastAPI SDK app. Free HTTPS, dead simple, but less control over region and memory.
- **Deploy at the end of Phase 1**, when the app is still ugly. A working ugly deployment on day 2 is worth more than a beautiful local app on day 8. Redeploy at the end of every subsequent phase.

**Index shipping:** do not build the FAISS index at container start (slow, flaky boot). Build it offline, commit the artifacts to a release asset or object storage, and download-and-mmap at boot. Record the index build hash in the trace so you always know which index produced a given number.

---

# §6. Data layer

## 6.1 Corpus

`ai4bharat/MSMARCO-XI` — MS MARCO translated into 13 Indian languages, with `source_lang` /
`target_lang` / `meta` fields alongside `query`, `answers`, and `passages`.

**Scope decision (ADR-002): index ONE primary language first — Hindi (`hi`).**
Reasons: best downstream tooling support, largest community validation, easiest to sanity-check
translation quality by eye. Add a second language only after Phase 5 exit criteria pass.

**Subset sizing:** target **50k–200k chunks** in the index. Large enough to be a real retrieval
problem, small enough to keep HNSW search in low single-digit ms and to rebuild the index in
minutes during the chunking lab. Do not attempt the full multi-GB corpus — it buys you nothing for
grading and costs you days.

**Spot-check translation quality before committing** to a language (Phase 0 task). MSMARCO-XI is
machine-translated; artifacts are real and they are a legitimate, citable justification for why
your groundedness guardrail exists.

## 6.2 Ground truth for evaluation

The dataset gives you `query → relevant passage(s)` pairs. That is your retrieval ground truth —
you do not need to hand-label anything. Hold out **500 query/passage pairs** that are used for
evaluation only.

---

# §7. The four things being graded — detailed specs

## 7.1 Chunking (C2) — the "vast" requirement

Build a **pluggable strategy interface**, implement six strategies, and **prove which is best with
numbers**. The comparison table is the deliverable, not the code.

### Interface (`src/vrag/chunking/base.py`)

```python
class ChunkingStrategy(Protocol):
    name: str
    def chunk(self, doc: Document) -> list[Chunk]: ...
    def config(self) -> dict:  # serialised into eval results for reproducibility
        ...
```

Every strategy registers itself in `chunking/registry.py` so `scripts/eval_chunking.py` can
enumerate them without code changes.

### Strategies to implement

| # | Strategy | Core idea | Key parameters |
|---|----------|-----------|----------------|
| 1 | **Fixed-size + overlap** | Baseline / control. Token windows with stride. | `size=256`, `overlap=0.2` |
| 2 | **Passage-native** | Use the dataset's own passage boundaries — they're already coherent retrieval units. | none |
| 3 | **Sentence-window** | Retrieve on single sentences, but return sentence ± N neighbours as context. Decouples retrieval granularity from generation granularity. | `window=2` |
| 4 | **Semantic** | Split at embedding-similarity troughs between adjacent sentences. Boundaries follow meaning, not character count. | `percentile_threshold=90` |
| 5 | **Metadata-aware** | Tag every chunk with `language`, `source_lang`, `query_type`. Enables filtered/boosted retrieval by the language Sarvam detected. **Most dataset-specific — highlight this one.** | filter mode vs boost mode |
| 6 | **Hierarchical (small-to-big)** | Index small precise chunks; on hit, return the larger parent passage for generation. | `child=128`, `parent=512` |
| 7 | *(stretch)* **Proposition** | LLM decomposes passages into atomic self-contained facts, offline. Zero hot-path cost. | — |

### Evaluation protocol (`scripts/eval_chunking.py`)

For each strategy × each embedding model, on the 500-query held-out set, report:

| Metric | Why it's here |
|--------|--------------|
| Recall@1 / @5 / @10 | did we find the right passage at all |
| MRR@10 | how high did it rank |
| nDCG@10 | graded ranking quality |
| Chunks produced | index size proxy |
| Index build time (s) | the one-time offline cost |
| Mean search latency (ms) | does a better strategy cost hot-path time |
| P95 chunk length (tokens) | context-window pressure on generation |

Output a markdown table into `docs/EVAL_RESULTS.md` **and** `eval/chunking_results.json`.
The README must state which strategy shipped to production **and why**, citing the numbers.

> Expect passage-native or hierarchical to win on this corpus. Do not pre-write that conclusion —
> run it, report what happens. If the naive baseline wins, that is an interesting, honest finding
> and you should say so.

## 7.2 The harness (C5)

"Harness" means: structured orchestration, not `llm(prompt)`. Implement all six of these:

1. **Typed stages.** Every stage is `async def run(ctx: PipelineContext) -> StageResult` with Pydantic input/output models. Stages are pure with respect to `ctx` — they read and append, never mutate history.
2. **Deadline propagation.** Per §3.4. Each stage declares `min_viable_ms` and `optional: bool`.
3. **Retries with policy.** `tenacity`: exponential backoff + jitter, capped attempts, **only on idempotent stages**, and never in a way that can exceed the request deadline.
4. **Circuit breaker.** If the LLM provider fails N times in a rolling window, trip the breaker and serve Track A only until it half-opens. Do not let one bad provider take the demo down.
5. **Tool calling.** Track B's LLM gets a real tool: `search_corpus(query: str, k: int) -> list[Passage]`. This lets the model issue a follow-up retrieval when the first pass is insufficient — and it satisfies the brief's explicit "tool calls" wording. Cap tool-call depth at 1 to protect the budget.
6. **Structured output + error recovery.** LLM must return JSON matching `AnswerSchema`. On parse failure: one repair attempt, then fall back to Track A. Never surface a raw exception or a truncated JSON blob to the user.

### Canonical response schema (`src/vrag/schemas.py`)

```python
class Citation(BaseModel):
    chunk_id: str
    passage_id: str
    score: float
    text_span: str

class AnswerResponse(BaseModel):
    status: Literal["answered", "abstained", "refused", "degraded"]
    answer: str | None
    track: Literal["extractive", "generative"]
    citations: list[Citation]
    confidence: float                # calibrated, 0–1
    refusal_reason: str | None
    language: str
    stages_skipped: list[str]        # deadline-shed stages
    trace_id: str
    timings_ms: dict[str, float]     # per-stage, always populated
```

Returning `timings_ms` on every response makes the latency HUD in the UI trivial and makes the demo
video self-evidencing. Do this.

## 7.3 Guardrails (C6) — five layers

Guardrails on the hot path must be **local and fast**. Any guardrail requiring a network LLM call
belongs in the offline evaluation, not the request path.

| Layer | Runs | Mechanism | Budget | Failure action |
|-------|------|-----------|--------|----------------|
| **G1 Input safety** | pre-retrieval | keyword/regex denylist + small local classifier | ~2ms | `refused`, safe message |
| **G2 Scope & language** | pre-retrieval | detected language ∉ supported set, or query is empty/degenerate | ~1ms | `refused`, ask to rephrase |
| **G3 Retrieval confidence** | post-fusion | `top1_score < τ` **OR** `(top1 − top5) < margin` | <1ms | `abstained` |
| **G4 Groundedness** | post-generation | citation IDs must exist in retrieved set; answer↔context lexical overlap ≥ threshold | ~5ms | drop to Track A, or `abstained` |
| **G5 Output safety / PII** | pre-emit | redaction pass on the final string | ~2ms | redact + flag |

### G3 must be calibrated, not guessed

This is where you separate yourself from teams that hardcode `if score < 0.5`. Build a calibration
set: **150 in-domain queries + 150 deliberately out-of-domain queries**. Sweep τ and the margin,
plot the tradeoff, pick the operating point, and record:

- Abstention rate on in-domain queries (false refusals — want low)
- Abstention rate on out-of-domain queries (correct refusals — want high)
- The chosen τ and margin, with justification

Put the curve in `docs/EVAL_RESULTS.md`. This single artifact will land harder than any amount of
guardrail prose.

### G4 two-tier design
- **Hot path:** cheap lexical — verify every cited `chunk_id` was actually retrieved, and check n-gram overlap between answer and cited spans. Fails fast, costs ~5ms.
- **Offline eval:** run a proper NLI entailment model over a sample of answers to report a real hallucination rate. Shows you know the lexical check is an approximation, and quantifies how good an approximation it is.

### Demo requirement
The demo video **must** show all three refusal modes firing live: an unsafe input (G1), an
out-of-scope question (G3), and a case where generation was ungrounded and the system fell back
(G4). The brief explicitly asks to see the system know when *not* to answer. Script these into the
demo, do not hope they happen.

## 7.4 Latency analytics (C4)

### Test query set (`scripts/make_test_queries.py`)

Build **100 queries**, stratified and version-controlled in `eval/test_queries.json`:

| Bucket | Count | Purpose |
|--------|-------|---------|
| In-domain, held out from index build | 60 | the main latency + accuracy population |
| Out-of-domain / off-topic | 20 | exercises G3 abstention path |
| Unsafe / adversarial | 10 | exercises G1 |
| Ambiguous / degenerate (empty, single word, noise) | 10 | exercises error recovery |

**Generate spoken versions with TTS** so the end-to-end voice benchmark is reproducible and
repeatable rather than depending on someone talking into a laptop 100 times. Sarvam's TTS is right
there and keeps the stack coherent. Store the audio in `eval/audio/`.

### Benchmark runner (`scripts/bench_latency.py`)

- Runs all 100 queries, **N=5 repetitions each** (500 samples) to separate signal from jitter.
- Discards a warm-up pass (JIT, page cache, connection pool establishment) — and says so in the report.
- Reads `traces.jsonl` and emits:
  - **P50 / P70 / P100** for `t_pipeline` — *note: P100 is the maximum observed, i.e. the worst case. It is not a typo for P99, and it will be ugly. Report it honestly; the brief asked for it specifically because it reveals whether you understand tail latency.*
  - The same percentiles **per stage**, so the bottleneck is visible
  - A stacked bar chart of the stage breakdown → `docs/assets/latency_breakdown.png`
  - A histogram / CDF of `t_pipeline` → `docs/assets/latency_cdf.png`
  - Separate tables for Track A and Track B
  - `t_e2e_voice` reported separately with STT cost broken out

### Latency regression test
`tests/test_latency_regression.py` fails the build if `p50(t_pipeline) > 200ms` on a fixed 20-query
smoke subset. Wire it into CI. This prevents a Phase 7 "polish" commit from silently destroying your
headline number two days before the deadline.

---

# §8. The `docs/` folder — session continuity system

This is the machinery that lets a memoryless agent work coherently across a multi-day build with
more than one person touching the code.

## 8.1 Structure

```
docs/
├── PRD.md                     # what & why; requirements → acceptance criteria
├── ARCHITECTURE.md            # system design; kept current with the code
├── BUILD_PLAN.md              # the 8 phases with exit criteria (from §9)
├── PROGRESS.md                # ⭐ LIVING STATE — the most important file
├── DECISIONS.md               # ADR log; append-only, numbered, dated
├── CONVENTIONS.md             # code style, error handling, naming patterns
├── API_CONTRACTS.md           # stage interfaces + HTTP/WS schemas
├── EVAL_PROTOCOL.md           # exactly how chunking + latency are measured
├── EVAL_RESULTS.md            # ⭐ generated tables + charts — a graded artifact
├── LATENCY_BUDGET.md          # §4's table, updated with real measurements
├── RISKS.md                   # live risk register with owners
├── SUBMISSION_CHECKLIST.md    # ⭐ including the promotion requirements
├── SESSION_LOG/
│   └── YYYY-MM-DD-session-NN.md
└── assets/                    # charts, diagrams, screenshots
```

## 8.2 Session ritual — mandatory

**At the START of every session, in this order:**
1. Read `/CLAUDE.md`
2. Read `docs/PROGRESS.md` — this tells you where you actually are
3. Read `docs/DECISIONS.md` — so you don't reverse a settled call
4. Read the current phase section in `docs/BUILD_PLAN.md`
5. Run the test suite. **If it's red, fixing it is the session's first task**, before any new feature.
6. State back to the user: current phase, what you're doing this session, exit criteria you're targeting.

**At the END of every session:**
1. Update `docs/PROGRESS.md` (template below) — do this even if the session was unproductive
2. Append a `docs/SESSION_LOG/YYYY-MM-DD-session-NN.md` entry
3. Append any new ADRs to `docs/DECISIONS.md`
4. Update `docs/LATENCY_BUDGET.md` if any numbers were measured
5. Commit with a message referencing the phase: `[P3] hybrid retrieval + RRF fusion`
6. Push

## 8.3 File templates

### `docs/PROGRESS.md` — the critical one

```markdown
# PROGRESS

**Last updated:** YYYY-MM-DD HH:MM IST by session NN
**Current phase:** P3 — Retrieval Quality
**Days remaining:** N
**Build status:** 🟢 green / 🟡 tests failing / 🔴 blocked

## Where we are, in one paragraph
<Plain prose. Assume the reader knows nothing. What works end to end right now?>

## Phase exit criteria — current phase
- [x] Hybrid retrieval implemented
- [ ] RRF fusion tuned on held-out set
- [ ] Recall@5 ≥ 0.80

## What works right now (verified, not assumed)
- Sarvam STT streaming → transcript ✅ verified <date>
- FAISS HNSW index over 120k Hindi chunks ✅
- ...

## What is stubbed / faked / TODO
- Reranker is a no-op passthrough — Phase 5
- G4 groundedness returns True unconditionally — Phase 5
> ⚠️ Anything stubbed MUST be listed here. An undocumented stub is how a demo dies.

## Live numbers
| Metric | Value | Measured on |
|--------|-------|-------------|
| p50 t_pipeline | — | — |
| Recall@5 (prod strategy) | — | — |

## Blockers
- <blocker> — owner, what's needed to unblock

## Next session should start by
1. <specific first action>
2. <second>
```

### `docs/DECISIONS.md` — ADR log

```markdown
# Architecture Decision Record

## ADR-001 — STT provider: Sarvam
**Date:** 2026-08-14 · **Status:** Accepted
**Context:** Brief requires Sarvam or ElevenLabs. Corpus is Indic-language.
**Decision:** Sarvam.
**Rationale:** Indic-native training; code-mixing support; Indian hosting → lower RTT
from an Indian deployment, which matters given the 200ms constraint.
**Consequences:** Stack is Indic-coherent end to end. Locked unless a probe disproves the
RTT assumption.

## ADR-002 — Corpus scope: Hindi only for v1
...
```

Rules: **append-only.** To reverse ADR-00N, write ADR-00M that supersedes it and mark the old one
`Superseded by ADR-00M`. Never edit history.

### `docs/PRD.md`

Sections: Problem · Users & scenario · Functional requirements (FR-1…N, each mapped to a C-constraint)
· Non-functional requirements (latency, availability, safety) · **Acceptance criteria per requirement**
· Explicit non-goals · Evidence artifacts the submission must contain.

### `docs/CONVENTIONS.md`

Must specify at minimum:
- Type hints mandatory on all public functions; `mypy` clean
- All timings in **nanoseconds internally** (`time.perf_counter_ns()`), converted to ms only at serialisation. Float ms accumulates error across nine stages.
- No bare `except:`; every caught exception is logged with `trace_id`
- No secrets in code — `.env` + `pydantic-settings`, `.env.example` committed
- New dependency ⇒ new ADR
- Every new module ⇒ a test file
- Hot-path functions carry a `# HOTPATH` comment and may not perform network or disk I/O

### `docs/SESSION_LOG/YYYY-MM-DD-session-NN.md`

```markdown
# Session NN — YYYY-MM-DD
**Phase:** P3 · **Duration:** ~Xh
## Goal
## Done
## Not done / deferred (and why)
## Surprises & learnings
## Numbers measured this session
## Follow-ups created
```

---

# §9. Build plan — 8 phases

> **⚠️ Superseded schedule.** This section's dates assumed an Aug-14 solo start. The project is
> now split across two people from Aug 17 — **`docs/TEAM_SPLIT.md` is the authoritative day-by-day
> plan.** The phase *content* below (what P0, P1, etc. actually involve) is still correct and still
> what needs to happen — just mapped onto the compressed two-person schedule in `TEAM_SPLIT.md`
> instead of the single-person dates shown here.

> Each phase has **exit criteria**. They are gates, not suggestions. Do not proceed on vibes.

## Phase 0 — Foundations & Probes
**Goal: know the physics before designing around them.**
1. Init repo, `pyproject.toml`, pre-commit (ruff + mypy), CI skeleton
2. Scaffold the entire `docs/` folder from §8 templates
3. `scripts/probe_latency.py` — measure RTT + TTFT from your deployment region to: Sarvam STT, Sarvam LLM, Groq, and a local model. **Record results in `docs/DECISIONS.md` as ADR-003.**
4. Download a Hindi subset; spot-check 20 translations by eye; record quality impressions
5. Obtain and verify all API keys work

**Exit:** provider decision made and recorded with evidence · `docs/` fully scaffolded · dataset subset on disk · one green trivial test.

## Phase 1 — Walking Skeleton & First Deploy
**Goal: something real, end to end, deployed publicly — however ugly.**
1. Naive fixed-size chunking over ~10k passages → FAISS flat index
2. Embedder wired (with the e5 prefixes — write the test first)
3. Single-stage retrieval → single LLM call → text answer
4. Minimal browser page: mic → WS → transcript → answer
5. **Deploy to production HTTPS. Verify mic works on the live URL from a phone.**

**Exit:** a public URL where a stranger can speak a Hindi question and get a real answer. Nothing stubbed in the STT path. Screenshot in `docs/assets/`.

> This is the highest-risk phase. Deployment problems discovered during code freeze are fatal;
> discovered on the first day of this phase, they're an afternoon.

## Phase 2 — Chunking Lab
1. `ChunkingStrategy` protocol + registry
2. Implement strategies 1–6
3. `scripts/eval_chunking.py` with all metrics from §7.1
4. Run the full matrix; write `docs/EVAL_RESULTS.md`
5. Promote the winner to production config

**Exit:** ≥5 strategies implemented and evaluated · comparison table committed · production strategy chosen with written justification · Recall@5 ≥ 0.75 on the winner.

## Phase 3 — Retrieval Quality
1. BM25 sparse index with Unicode-aware tokenisation
2. Parallel dense ∥ sparse via `asyncio.gather`
3. RRF fusion; tune `k`
4. Optional cross-encoder rerank behind a budget flag
5. HNSW `efSearch` sweep: recall vs latency curve → `docs/assets/`

**Exit:** hybrid beats dense-only on Recall@5 · efSearch operating point chosen from the measured curve, not guessed.

## Phase 4 — Harness Hardening
1. Refactor to typed `Stage` objects with a `PipelineContext`
2. `budget.py` — deadline propagation + graceful degradation
3. `tenacity` retry policies per stage; circuit breaker on the LLM
4. `search_corpus` tool exposed to the generation model
5. Structured JSON output + repair-then-fallback
6. `TraceRecord` emission to `traces.jsonl` with ns-precision per-stage timings
7. **Track A / Track B split implemented**

**Exit:** every stage typed · deadline shedding demonstrably works (test that forces a 50ms budget and asserts optional stages are skipped and the response still returns) · traces written for every request.

## Phase 5 — Guardrails
1. G1–G5 implemented per §7.3
2. Calibration set built (150 in-domain + 150 out-of-domain)
3. τ and margin swept; operating point chosen; curve plotted
4. Offline NLI hallucination-rate evaluation on a sample
5. Adversarial test suite in `tests/test_guardrails.py`

**Exit:** all three refusal modes reproducible on demand · calibration curve committed · false-refusal rate on in-domain queries < 10% · every guardrail on the hot path measured at < 10ms.

## Phase 6 — Latency Campaign
1. Build the 100-query test set + TTS audio
2. `bench_latency.py`, full run, N=5
3. Fill in the Measured column of §4 / `docs/LATENCY_BUDGET.md`
4. Optimise the top bottleneck; re-measure; repeat until the budget holds or you run out of ideas
5. ONNX/int8 the embedder if not already; warm all models at boot; confirm zero cold starts
6. Latency regression test wired into CI
7. Charts generated into `docs/assets/`

**Exit:** P50/P70/P100 reported for total **and** per stage · Track A p50 comfortably < 200ms · CI regression test green · charts committed.

## Phase 7 — Polish, Docs, Evidence
1. Frontend: live transcript, citations, refusal states, **latency HUD**, degradation indicator
2. README: architecture diagram, chunking table, latency table, guardrail calibration, honest limitations section
3. `docs/ARCHITECTURE.md` reconciled with the code as actually built
4. Full end-to-end manual test pass; edge cases; mobile browser check
5. Load a second language if — and only if — everything above is green

**Exit:** README is submission-quality · a stranger could clone and run it from the README alone · no stubs remain in `PROGRESS.md`.

## Phase 8 — Freeze & Submit
**Code freeze ~20:00 IST on the last full build day — see `docs/TEAM_SPLIT.md` §5 for the exact
date and time on the real two-person schedule.** After that: documentation, videos, and submission only.
1. Record demo video (script it — see §10)
2. Cut the process video from footage captured all week
3. Final deploy; verify live link from a device on mobile data, not just your own wifi
4. Promotion posts (§10.3) — track per member per platform
5. Submit the form

**Exit:** everything in `docs/SUBMISSION_CHECKLIST.md` ticked with a name against it.

---

# §10. Submission artifacts

## 10.1 README structure (this is graded — treat it as a deliverable)

```
1. What this is + live link + 60-second quickstart
2. Architecture diagram + request lifecycle walkthrough
3. Chunking: strategies implemented, comparison table, what shipped and why
4. Latency: metric definition (§3.2 verbatim), budget table, P50/P70/P100,
   per-stage breakdown, charts, and what we'd do next with more time
5. Harness: stage model, deadline propagation, retries, circuit breaker, tools
6. Guardrails: five layers, calibration curve, measured refusal rates
7. Limitations & honest notes  ← include this. It signals maturity.
8. Reproduce our numbers: exact commands
```

## 10.2 Video scripts

**Demo video** — must show, in order:
1. Speaking a Hindi question → live transcript appearing
2. Answer with citations + the latency HUD showing the real number
3. An out-of-scope question → visible abstention
4. An unsafe input → visible refusal
5. A forced-low-budget request → optional stages shed, response says which
6. Two seconds on the eval tables

**Process video (90s)** — cut from footage captured *throughout the week*. Screen recordings of
debugging, the chunking comparison table filling in, a whiteboard, a disagreement about the 200ms
interpretation. Start capturing on day 0. Footage staged on Aug 21 looks staged.

## 10.3 Promotion checklist — track it as engineering work

Every team member posts **both** videos to **Instagram, X, and LinkedIn**, each post tagged
`#RAGInGoa`. At least one Instagram account public. A shared team post does **not** satisfy this.

Build the grid in `docs/SUBMISSION_CHECKLIST.md`:

| Member | IG (v1) | IG (v2) | X (v1) | X (v2) | LI (v1) | LI (v2) | IG public? |
|--------|---------|---------|--------|--------|---------|---------|-----------|

Given "no resubmissions," a missed checkbox here costs the same as a broken build.

---

# §11. Rules for the coding agent

**Always**
- Read `docs/PROGRESS.md` before doing anything
- Write the test before the hot-path optimisation, so you can prove the optimisation worked
- Measure before optimising; commit the measurement
- Update `docs/PROGRESS.md` at session end, even if the session went badly
- Prefer boring, working technology over clever technology, every single time

**Never**
- Never mock, stub, or hardcode the STT path in committed code. The one thing that must be undeniably real is that a human voice produces a real transcript.
- Never report a latency number you did not measure with `bench_latency.py`.
- Never silently reverse a decision in `DECISIONS.md`.
- Never add a dependency without an ADR.
- Never leave a stub undocumented in `PROGRESS.md`.
- Never put a network call on the hot path other than the LLM.
- Never commit secrets. `.env` is gitignored from commit #1.
- Never start a new phase with a red test suite.

**When blocked or uncertain:** stop, write the ambiguity into `docs/RISKS.md`, and ask the user.
Do not guess on anything that touches C1–C7.

---

# §12. Session kickoff prompts

Paste these into Claude Code at the start of each session. They're written to trigger the ritual in §8.2.

**P0** — `Read AGENT_BUILD_SPEC.md fully, then CLAUDE.md. Execute Phase 0. Priority order: scaffold docs/ from the §8 templates, then write and run scripts/probe_latency.py. Report probe results as ADR-003 before proposing any architecture changes.`

**P1** — `Session start ritual per §8.2. Execute Phase 1 — walking skeleton. Non-negotiable: the STT path must be real Sarvam, not mocked. Ship a public HTTPS deployment before this session ends, however ugly.`

**P2** — `Session start ritual. Execute Phase 2 — chunking lab. Implement the ChunkingStrategy protocol and strategies 1–6 from §7.1, then run the full eval matrix and write docs/EVAL_RESULTS.md. Report actual numbers; do not assume which strategy wins.`

**P3** — `Session start ritual. Execute Phase 3 — hybrid retrieval. Dense and sparse must run concurrently. Produce the efSearch recall-vs-latency curve and pick the operating point from the data.`

**P4** — `Session start ritual. Execute Phase 4 — harness hardening. Focus on budget.py deadline propagation and the Track A / Track B split. Include the forced-50ms-budget degradation test.`

**P5** — `Session start ritual. Execute Phase 5 — guardrails G1–G5 per §7.3. Calibrate G3 empirically on the 150+150 set and commit the curve. All three refusal modes must be reproducible on demand for the demo.`

**P6** — `Session start ritual. Execute Phase 6 — latency campaign. Build the 100-query set with TTS audio, run bench_latency.py at N=5, fill in the measured column of docs/LATENCY_BUDGET.md, then optimise the single largest bottleneck and re-measure.`

**P7** — `Session start ritual. Execute Phase 7 — polish and evidence. README to submission quality per §10.1. Reconcile docs/ARCHITECTURE.md with what was actually built. Clear every stub from PROGRESS.md.`

**P8** — `Session start ritual. Code freeze is in effect — documentation, verification, and submission only. Walk docs/SUBMISSION_CHECKLIST.md item by item and verify each against reality, not against what we believe.`

---

# §13. Risk register (seed `docs/RISKS.md` with these)

| ID | Risk | Impact | Mitigation |
|----|------|--------|-----------|
| R1 | Provider RTT makes 200ms unreachable | 🔴 high | Phase 0 probe; two-track design; consider local generation |
| R2 | Deployment/mic fails late | 🔴 high | Deploy in Phase 1, redeploy every phase |
| R3 | Rate limits break the benchmark run | 🟡 med | Retries in the bench script; run overnight if needed |
| R4 | Index too large for host memory | 🟡 med | Cap at 200k chunks; measure RSS in Phase 1 |
| R5 | MT artifacts in corpus hurt answer quality | 🟡 med | Spot-check in Phase 0; it's also the justification for G4 |
| R6 | Team disagreement on the 200ms interpretation resurfaces late | 🟡 med | Settle as ADR on day 0; §3.2 is the wording |
| R7 | A member misses a promotion post | 🔴 high | Named grid in SUBMISSION_CHECKLIST.md, checked Aug 21 |
| R8 | Polish commits break the latency number | 🟡 med | CI latency regression test from Phase 6 |
| R9 | Scope creep into a second language before core is done | 🟡 med | Hard gate: not before Phase 7 exit |
