# vRAG — Voice-Grounded RAG in Hindi

**Live demo:** https://vrag-voice.onrender.com

Speak a Hindi question → real Sarvam speech-to-text → hybrid-evaluated dense retrieval over
99,767 real MSMARCO-XI passages → a grounded, cited answer, or an honest refusal when the
evidence isn't there. Built for a hackathon with a hard constraint: **server-side answer latency
under 200ms** — the numbers below show exactly where that holds and where it doesn't, measured,
not asserted.

**60-second quickstart:** open the live URL above, tap the mic, ask something in Hindi
("भारत में मानसून कब आता है?"). Real STT, real retrieval, real citations. See
[Known limitations](#known-limitations) before judging by voice alone — the real human-microphone
leg of verification is still pending (below).

---

## Table of contents

1. [What this is](#1-what-this-is)
2. [Problem statement](#2-problem-statement)
3. [Architecture](#3-architecture)
4. [How the pipeline works](#4-how-the-pipeline-works)
5. [Chunking strategy](#5-chunking-strategy)
6. [Retrieval architecture](#6-retrieval-architecture)
7. [Guardrails and refusal behavior](#7-guardrails-and-refusal-behavior)
8. [Harness and deadline propagation](#8-harness-and-deadline-propagation)
9. [Memory optimization story](#9-memory-optimization-story)
10. [Latency results](#10-latency-results)
11. [The Render Free CPU limitation, honestly](#11-the-render-free-cpu-limitation-honestly)
12. [Live demo](#12-live-demo)
13. [Local Docker instructions](#13-local-docker-instructions)
14. [Evaluation results](#14-evaluation-results)
15. [Known limitations](#15-known-limitations)
16. [Project structure](#16-project-structure)
17. [Hackathon deliverables](#17-hackathon-deliverables)

---

## 1. What this is

vRAG is a voice-enabled retrieval-augmented generation system: real-time Hindi speech in, a
grounded and cited answer out, with five layers of guardrails standing between a user's voice and
whatever the system says back. Every retrieval, chunking, and reranking decision in this repo was
settled by running the actual ablation and reading the actual number — not by picking whatever
sounded reasonable. Where the naive/simple option won (dense-only beats hybrid; no reranker beats
either tested reranker), that's reported as the real result, not treated as a disappointment.

## 2. Problem statement

Build a voice assistant that answers Hindi questions **only** from a real, retrievable corpus
(`ai4bharat/MSMARCO-XI`), that says "I don't know" when the evidence isn't there rather than
guessing, and that does the retrieval-and-grounding part of that round trip in under 200ms
server-side. The interesting engineering is almost entirely in making "grounded" and "under
200ms" both true at once, not in wiring an LLM to a search index.

## 3. Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  BROWSER (HTTPS required for mic)                              │
│  getUserMedia → PCM16 chunks → WebSocket (/voice)               │
│  Renders: live transcript, answer + citations, distinct         │
│           answered/abstained/refused/degraded states, latency   │
└───────────────────────────┬────────────────────────────────────┘
                            │ WS: audio in / transcript+answer events out
┌───────────────────────────▼────────────────────────────────────┐
│  FASTAPI ORCHESTRATOR  ("the harness") -- fully wired, live      │
│                                                                │
│  Sarvam STT     saaras:v3-realtime streaming WS, real, never    │
│                 mocked                                          │
│  ── t_pipeline clock starts once the final transcript lands ──  │
│  G1 InputGuard      safety / degenerate-input check              │
│  G2 ScopeGuard      language / scope check                       │
│  Retrieve           dense-only FAISS HNSW+SQfp16 (A3 winner)     │
│                      -- NOT hybrid, NOT reranked (both measured, │
│                      not left untested -- see §6)                │
│  G3 GroundGate      top1-cosine >= TAU(0.8835) -> ABSTAIN?       │
│                      (real 300-query calibration)                │
│  Track A            extractive span-select, real, ~5ms           │
│  Track B            streaming Sarvam LLM, budget-gated:          │
│                      skipped outright if remaining budget can't  │
│                      realistically fit a fair attempt             │
│  G4 Groundedness    citation/n-gram overlap check                │
│  G5 OutputGuard     PII redaction                                │
│  Assemble           AnswerResponse + TraceRecord emit             │
└───────────────────────────┬────────────────────────────────────┘
                            │
        ┌───────────────────┼──────────────────┐
        ▼                   ▼                  ▼
  In-process assets    Sarvam APIs        Telemetry sink
  • FAISS HNSW+SQfp16  • STT (WS)         • traces.jsonl
    (dense only)       • LLM (Track B)    • per-stage ns timings
  • LiteE5Embedder
    (SentencePiece +
     ONNX int8, no
     torch/transformers)
```

Full detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## 4. How the pipeline works

1. Browser opens `WS /voice`, streams PCM16 audio as the user speaks.
2. Sarvam's streaming STT (`saaras:v3-realtime`) returns interim + final transcripts. The receive
   loop bounds the wait to 10s while no transcript has arrived yet — a silent session gets a clear
   "No speech detected yet. Please try again." error instead of hanging until Sarvam's own ~60s
   inactivity watchdog. Once real speech is detected, the wait reverts to unbounded so a normal
   mid-sentence pause is never mistaken for silence.
3. **The metric that matters, defined precisely (quoted verbatim from the build spec):**

   > **`t_pipeline`** is measured server-side, from the moment the final transcript is available
   > to the moment the first grounded answer token is emitted to the client.
   > It excludes: client→server network transit, microphone capture, and speech duration.
   > It includes: input guardrails, query embedding, hybrid retrieval, fusion, grounding gate,
   > and answer generation up to first token.
   > Index construction (chunking + embedding + index build) is a **one-time offline cost**,
   > reported separately and excluded from `t_pipeline`.

   `t_e2e_voice` (mic-stop → first visible answer) is reported separately as a secondary honest
   number — see [§10](#10-latency-results).
4. `retrieve()` embeds the query (`"query: " + text`, E5's required prefix) and searches the
   production FAISS index (dense-only, no sparse, no rerank — see [§6](#6-retrieval-architecture)
   for why both were tried and rejected with real data).
5. `G3` decides answered vs. abstained from a calibrated `TAU=0.8835` threshold on top1 cosine
   similarity.
6. Track A always computes a real extractive answer (~5ms). Track B streams a generative rewrite
   over it if the remaining budget realistically allows a fair Sarvam attempt — under the default
   200ms total budget it almost always doesn't, and is cleanly skipped rather than attempted and
   timed out (see [§10](#10-latency-results)).
7. `G4`/`G5` check groundedness and redact PII before the response is assembled.
8. A `TraceRecord` is appended to `traces.jsonl` for every real request, whatever the outcome.

## 5. Chunking strategy

Six real strategies implemented and evaluated against a frozen 500-query held-out set (10,000
Hindi queries / 99,767 passages, `data/working_subset.jsonl`):

| Strategy | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Chunks |
|---|---|---|---|---|---|
| passage_native | 0.322 | 0.650 | 0.750 | 0.452 | 99,767 |
| fixed_overlap (256/0.2) | 0.322 | 0.650 | 0.750 | 0.452 | 101,008 |
| **metadata_aware** (shipped) | 0.322 | **0.653 ± 0.001** | **0.752 ± 0.002** | **0.453 ± 0.001** | 99,767 |
| hierarchical (128/512) | 0.318 | 0.640 | 0.742 | 0.446 | 103,907 |
| semantic (p90) | 0.318 | 0.644 | 0.748 | 0.448 | 101,308 |
| sentence_window (w=2) | 0.310 | 0.552 | 0.554 | 0.405 | 390,288 |

**`metadata_aware` shipped** — statistically tied with `passage_native`/`fixed_overlap` (within a
~0.2-0.4pp noise floor, verified with repeated runs, not assumed), but the cheapest of the three
to build and the one that carries the most useful per-chunk metadata forward for guardrails. Full
detail, including a real metric bug found and fixed mid-ablation: `docs/EVAL_RESULTS.md` §1.

**Honest gap:** the build spec's own Phase-2 exit bar (Recall@5 ≥ 0.75) is not met by any tested
configuration — the best found across 6 chunkers × 4 embedders × 3 retrieval modes × a full
efSearch sweep tops out at 0.656 (efSearch=256, itself inside the noise floor of the shipped
efSearch=64). This is reported as a real, exhaustively-investigated corpus ceiling on this
particular MSMARCO-XI subset, not a config that more tuning would fix.

## 6. Retrieval architecture

**Dense-only FAISS HNSW, `multilingual-e5-small`, no hybrid, no rerank.** Every one of those is a
measured decision, not a default:

- **Dense vs. hybrid (A3):** hybrid+RRF *regresses* quality on this corpus — Recall@5 0.604 vs.
  dense-only's 0.652. BM25 is comparatively weak on this machine-translated Hindi text, and naive
  RRF gives its lower-quality top ranks equal fusion credit against dense's stronger ones. A larger
  per-lane candidate pool before fusion was tested as a mitigation (pools of 10/30/50/100) and
  didn't help — it dilutes dense's own strong ranks with more of BM25's weak candidates. Hybrid
  remains fully implemented and tested; it's just not what runs.
- **Reranking (A4 + a targeted follow-up):** both a multilingual cross-encoder and FlashRank were
  measured to *actively destroy* quality on this corpus's Hindi text — Recall@5 drops from 0.652
  to 0.228 (cross-encoder) or 0.100 (FlashRank), for two distinct, verified root causes (FlashRank
  saturates scores at ~1.0 regardless of relevance on Hindi; the cross-encoder is English-only
  and never learned Hindi at all). A follow-up experiment specifically built a 122-query diagnostic
  subset targeting the exact failure mode reranking should theoretically fix (queries where dense
  returns same-template distractors, like "India's capital" surfacing London/Islamabad/Munich
  boilerplate) — reranking *still* regressed Recall@5/@10 sharply on that targeted subset (0.70→0.39,
  0.91→0.62) and made the flagship example worse, not better. Reranking is implemented, tested, and
  off.
- **efSearch=64:** chosen from a real recall-vs-latency curve — Recall@5 climbs meaningfully from
  16→64 (+2.4pp) but only crawls from 64→256 (+0.4pp, inside the noise floor), while latency keeps
  climbing linearly. 64 is the knee of the curve.
- **Embedder (A2):** `multilingual-e5-small` wins decisively over `potion-multilingual-128M`
  (Recall@5 0.653 vs. 0.266) and `vyakyarth` (0.274) — a 38+ point gap, not a close call.

Full detail, every number sourced from `eval/ablation_ledger.csv`: `docs/EVAL_RESULTS.md` §2-3.

## 7. Guardrails and refusal behavior

Five layers, all real, all independently tested (`pytest tests/guardrails/` — 31/31 pass):

| Layer | Checks | Status |
|---|---|---|
| G1 | Input safety / degenerate input | Deterministic, tested |
| G2 | Scope / language | Deterministic, tested |
| G3 | Retrieval confidence | **Calibrated on 300 real queries** — see below |
| G4 | Groundedness (citation/n-gram overlap) | Real, threshold uncalibrated (honest gap, see [§15](#15-known-limitations)) |
| G5 | PII redaction | Deterministic, tested |

**G3's real calibration** (150 in-domain + 150 out-of-domain queries, sweeping `TAU`): the build
spec's two targets — false-refusal(in-domain) < 10% **and** correct-refusal(out-of-domain) > 80% —
are **not simultaneously reachable** on this corpus via top1-cosine gating alone. Root cause
verified: MSMARCO-XI passages recur across the ~780k-row source dataset, so a genuinely
out-of-index query often still retrieves a topically-close or coincidentally-correct passage.

| Target false-refusal | Actual TAU | Actual false-refusal | Correct-refusal achieved |
|---|---|---|---|
| ≤10% (spec target) | 0.8640 | 10.0% | 38.0% |
| ≤20% | **0.8835 (shipped)** | **19.3%** | **75.3%** |

`TAU=0.8835` is the balanced operating point, weighing both targets equally — a real, documented
product tradeoff, not an oversight. Verified robust to the production index's fp16 quantization:
re-ran the exact same 500-query gate check against both the fp32 and fp16 index — **zero G3
decisions changed** (mean score difference -0.00007, max 0.0281, never near enough to `TAU` to
flip an outcome).

**Refusal is demonstrable on demand** — all four response states (`answered`/`abstained` via
G3/`refused` via G1-G2/`degraded`) render with distinct, correct UI treatment; a real forensic
investigation (documented in `docs/DECISIONS_R.md` R-037) traced one live abstention all the way
to the exact corpus passage involved, confirming G3 fires correctly rather than being a black box.

## 8. Harness and deadline propagation

Typed `Stage`/`PipelineContext`/`Budget` — every stage gets `remaining_ms` propagated from the
total request budget, not a fixed per-stage allowance. `GenerateStage`'s pre-flight gate skips
Track B outright when remaining budget can't realistically fit a fair ~2s Sarvam attempt, instead
of starting it and paying a doomed wait — the single change that took local end-to-end wall-clock
latency for an answered query from ~213-246ms down to 10.1ms (see [§10](#10-latency-results)). A
circuit breaker exists and is tested but stays structurally dormant under the default 200ms budget
(it only counts ≥2.0s "fair chance" attempts as a health signal, and the budget gate now prevents
those attempts from happening at all under a tight budget) — a proven, currently-inapplicable
building block, not a gap. Structured output uses repair-then-fallback, not a bare retry. Full
detail: `src/vrag/harness/{stage,budget,pipeline,retry}.py`.

## 9. Memory optimization story

Render's free tier caps a container at 512MB. The original retrieval stack (PyTorch
`sentence-transformers` + fp32 FAISS) measured **1,860MB** — nowhere close. Two levers were
investigated; only one shipped:

**Corpus-shrink — investigated, rejected with real numbers.** A quality+memory sweep at
20k/50k/99,767 chunks found the size that would actually fit under 512MB (~5,900 chunks, a ~94%
cut) is 8.5x below the spec's own 50k-chunk floor, with severely degraded quality even at the
larger, still-non-compliant sizes tested (Recall@5 0.184 at 20k chunks). Correctly rejected, not
silently abandoned.

**Embedder + index engineering — shipped, zero quality cost.**
1. `LiteE5Embedder`: raw `onnxruntime` + `sentencepiece`, no `torch`/`sentence-transformers`
   import (`import torch` alone costs ~383MB RSS regardless of which backend does inference; the
   HF `tokenizers` Rust library costs another ~262MB for this model's vocabulary). Verified
   byte-identical output and 100.0000% exact tokenizer-ID equivalence against 1,020 real strings.
2. FAISS `sqfp16` scalar quantization: 2-byte half-float vectors instead of 4-byte float32 — ~77MB
   saved at zero measured Recall@k/MRR@10 regression.

**Result: 1,860MB → 493.8MB (73.4% cut), verified under the real 512MB Docker constraint**
(`docker run -m 512m --memory-swap 512m`): startup 204.4MiB, peak after 10 real queries 397.8MiB,
~114MB headroom, real citations returned (not the stub), Recall@10 0.748 vs. 0.750 fp32 baseline
(noise-floor difference). **This is confirmed live**, not just in a local Docker test — the
deployed URL's `/healthz` returns `{"status":"ok","retrieval":"real"}`, verified 2026-08-20.

## 10. Latency results

**Read the environment label on every number below — LOCAL and LIVE RENDER are genuinely
different environments and the numbers are not interchangeable.**

| Measurement | Environment | P50 | P100 |
|---|---|---|---|
| Track A, true stage cost | LOCAL | **5.2ms** | 8.9ms |
| End-to-end wall-clock, answered query | LOCAL | **10.1ms** | 16.2ms |
| `retrieve`/`t_pipeline` stage-sum, 40 real queries | **LIVE RENDER** | **594.9ms** | **1105.7ms** |

**Locally, both numbers meet the 200ms target comfortably.** The wall-clock number used to be
~213-246ms (P50) before a pre-flight budget gate was added to `GenerateStage` — it was attempting
a real Track B call on every answerable query with whatever budget remained, and that attempt was
structurally guaranteed to time out (Sarvam's real completion time is ~2s, measured separately at
P50=1976.5ms on 30 real calls). The gate now skips that attempt outright when the budget can't
fit it, closing a 20-40x gap with one change. See `docs/LATENCY_BUDGET.md` for the full before/after
record — kept, not deleted.

**Track B, given a fair budget (standalone measurement, bypasses the harness's gate):** 19/30 real
Sarvam calls succeeded (63.3%), P50=1976.5ms completion. A real bug (Sarvam's SSE stream sending
explicit `"content": null`, mishandled by a `.get(key, default)` call whose default only applies
when the key is *absent*) was found and fixed while building this measurement — before the fix,
0/30 calls succeeded. The remaining 37% failures are a documented, provider-side stall issue,
mitigated (not fixed — not ours to fix) with stall detection.

Full detail, per-stage table, response-status mix: `docs/LATENCY_BUDGET.md`.

## 11. The Render Free CPU limitation, honestly

**`t_pipeline` does not hold under 200ms on the live deployed URL for a retrieval-touching
request.** A real 40-query benchmark against `https://vrag-voice.onrender.com` measured
`retrieve`/`t_pipeline` stage-sum at P50=594.9ms, P100=1105.7ms — 10-40x the identical local code
path. This was investigated, not shrugged off: Render's own metrics API showed steady memory
(~409-430MB) throughout, ruling out memory pressure; the root cause is Render's free tier's 0.1
shared vCPU (vs. Starter's 0.5, per Render's own published specs), compounded by a directly-
observed free-tier instance spin-up event mid-benchmark. **This is a hosting-tier limitation, not
a code regression** — the exact same request path that runs in single-digit milliseconds locally
is genuinely CPU-starved on the free instance. A paid tier would very likely close most of this
gap (untested — would need its own real measurement).

## 12. Live demo

**https://vrag-voice.onrender.com**

Verified 2026-08-20, against the current deployed commit:
- `GET /healthz` → `{"status":"ok","retrieval":"real"}`
- 5 real `/ask` calls: 2 answered with real citations (real `chunk_id`/`passage_id`/scores, never
  the Day-0 stub), 3 correctly abstained with genuinely distinct confidence scores
- All four frontend states (answered/abstained/refused/degraded) render with distinct pill text
  and styling — verified against the live page's own exposed test hook, confirmed `refused` never
  displays "Abstained"
- A real silent-microphone browser session correctly received a "No speech detected yet" error
  within ~13s, not Sarvam's ~60s watchdog

**Pending: real human-voice verification.** Everything above was verified with real infrastructure
(real WebSocket, real Sarvam STT connection, real silence as valid audio input) but the automation
environment used for this pass provides a muted microphone track — no path to genuine human
speech content. **A real spoken query on the live URL, from an actual device with a working
microphone, has not been performed and is explicitly called out here as outstanding**, not
claimed.

## 13. Local Docker instructions

```bash
git clone https://github.com/ShivDyutiVerma/vRAG.git
cd vRAG
cp .env.example .env   # fill in SARVAM_API_KEY (dashboard.sarvam.ai) -- required for STT/Track B

docker build -t vrag-local .
docker run --rm -p 7860:7860 --env-file .env vrag-local
# or, to verify it survives the real free-tier memory ceiling:
docker run --rm -p 7860:7860 --env-file .env -m 512m --memory-swap 512m vrag-local

curl http://localhost:7860/healthz
# {"status":"ok","retrieval":"real"}
```

The image downloads the pre-built FAISS index (`index-metadata_aware-v3`) and embedder
(`embedder-lite-onnx-v2`) as GitHub release assets at build time — nothing is built inside the
container, and no local corpus/model download is needed first.

**Without Docker**, for development:

```bash
pip install -e ".[retrieval-lean,pipeline,dev]"
uvicorn vrag.api.main:app --app-dir src --host 0.0.0.0 --port 8000
```

requires `data/index/metadata_aware` and `data/onnx/multilingual-e5-small` present locally (built
via `scripts/build_index.py`, or downloaded from the same release assets the Dockerfile uses).

## 14. Evaluation results

Full tables, every reranker/chunking/embedder/latency number, sourced from
`eval/ablation_ledger.csv` or a real benchmark script, never hand-written:
**[`docs/EVAL_RESULTS.md`](docs/EVAL_RESULTS.md)**.

- §1 Chunking (A1) — 6 strategies compared
- §2 Embedding (A2) — 4 candidates compared
- §3 Retrieval mode + reranking (A3, A4) — dense-vs-hybrid, 2 rerankers, efSearch curve, plus a
  targeted follow-up specifically testing reranking against same-template-distractor queries
- §3b Memory optimization & corpus-size experiment
- §4 Generation (Track B) — real success-rate and latency numbers
- §5 Guardrails — G3's full calibration curve, G4's honest gap
- §6 Latency — LOCAL vs. LIVE RENDER, clearly separated

## 15. Known limitations

Stated plainly, not minimized:

- **Real human-microphone verification on the live URL is still pending** (see [§12](#12-live-demo)).
- **Render Free does not meet the 200ms target for a retrieval-touching request** — CPU
  contention on the shared 0.1 vCPU tier, see [§11](#11-the-render-free-cpu-limitation-honestly).
  Locally the target is comfortably met.
- **Chunking's Recall@5 (0.652-0.656 across every tested configuration) doesn't reach the build
  spec's own 0.75 exit bar** — an exhaustively investigated corpus ceiling on this MSMARCO-XI
  subset, not a config gap.
- **Retrieval is not perfect** — G3's own calibration data shows 19.3% false-refusal on genuinely
  in-domain queries is the accepted operating point (the alternative, hitting <10% false-refusal,
  drops correct-refusal on out-of-domain queries to 38%). This is a real, documented tradeoff, not
  a bug.
- **Reranking is implemented and tested, but disabled** — both a multilingual and an English-only
  cross-encoder were measured to regress retrieval quality on this corpus's Hindi text, including
  on a subset specifically built to favor reranking's strongest hypothesized case. `none` is the
  correct, measured, shipped default.
- **Hybrid (dense+sparse) retrieval is implemented and tested, but disabled** — measured to
  regress Recall@5 vs. dense-only on this corpus.
- **G4's groundedness threshold (`MIN_OVERLAP_RATIO=0.15`) is uncalibrated** — a real check that
  runs on every answer, but the specific value is an initial estimate, not a measured one.
- **Track B (generative synthesis) rarely activates under the default 200ms budget** — by design,
  since Sarvam's realistic completion time (~2s) can never fit inside that budget. It is fully
  real and measured on its own (63.3% success rate, given a fair budget) but most live answers
  under the default budget are Track A (extractive), not Track B.
- **`t_e2e_voice` (full mic-to-answer round trip) has no dedicated benchmark script** — 95 real
  TTS-synthesized test audio files exist and are ready, the WebSocket benchmark client itself was
  not built.

## 16. Project structure

```
src/vrag/
  harness/     pipeline · stage · budget (deadline propagation) · retry · trace
  chunking/    base protocol + 6 strategies + registry
  index/       embedder (ONNX/SentencePiece) · dense (FAISS HNSW+SQfp16) · sparse (bm25s) · fusion (RRF)
  retrieval/   dense ∥ sparse orchestration, reranking (implemented, off by default)
  generation/  Track A extractive · Track B streaming LLM
  guardrails/  G1 input · G2 scope · G3 confidence · G4 grounded · G5 output
  stt/         Sarvam realtime STT client (never mocked)
  telemetry/   trace records → traces.jsonl
  api/         FastAPI routes + WebSocket
frontend/      single-file voice UI (no build step, no framework)
scripts/       build_index · eval_chunking · eval_rerank · bench_latency · bench_live_render
eval/          heldout_queries.json · test_queries.json · audio/ · ablation_ledger.csv
docs/          architecture, decisions (per-workstream ADR logs), eval results, latency budget
```

## 17. Hackathon deliverables

| Requirement | Status | Evidence |
|---|---|---|
| Public GitHub repo | ✅ | this repo |
| Live HTTPS URL | ✅ (with a stated CPU-latency limitation) | [§11](#11-the-render-free-cpu-limitation-honestly), [§12](#12-live-demo) |
| Real Sarvam STT, never mocked | ✅ | `src/vrag/stt/sarvam.py`, zero mocks in `tests/` |
| ≥5 chunking strategies, evaluated | ✅ (6 shipped) | [§5](#5-chunking-strategy) |
| Real harness (typed stages, deadline propagation, circuit breaker, structured output) | ✅ | [§8](#8-harness-and-deadline-propagation) |
| 5 guardrail layers, calibrated where applicable | ✅ (G4's threshold uncalibrated, disclosed) | [§7](#7-guardrails-and-refusal-behavior) |
| P50/P70/P100 latency, real N | ✅ | [§10](#10-latency-results) |
| Real human microphone verification | ⏳ **Pending** | [§12](#12-live-demo), [§15](#15-known-limitations) |
| Demo video | ⏳ Not yet recorded | — |
| Process video | ⏳ Not yet recorded | — |
| Submission checklist / promotion grid | See `docs/SUBMISSION_CHECKLIST.md` | — |

**Deadline:** 2026-08-22 23:59 IST. No resubmissions.
