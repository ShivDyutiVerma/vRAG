# vRAG — Voice-Grounded RAG, 14 Languages (Local) / Hindi (Deployed)

**Two real, distinct systems exist in this repo right now — read this before judging either one:**

| | **Deployed (live)** | **Local candidate (newer, not deployed)** |
|---|---|---|
| Languages | Hindi only | 14 (13 MSMARCO-XI languages + English) |
| Corpus | 99,767 chunks | 107,678 chunks |
| Where it runs | Render free tier — **currently returning 502**, checked 2026-08-20, not touched (see below) | Your machine only — `data/index/multilingual_100k/`, gitignored, no GitHub release asset |
| How to run it | `docker run` — verified working locally today, see [§13](#13-local-docker-instructions) | `VRAG_INDEX_DIR=...` + local rebuild, see [§12A](#12a-the-multilingual-candidate--local-only-not-deployed) |

Speak a question → real Sarvam speech-to-text → dense retrieval over real MSMARCO-XI passages →
a grounded, cited answer, or an honest refusal when the evidence isn't there. Built for a
hackathon with a stated latency target: **server-side answer latency under 200ms** — the numbers
throughout this document show exactly where that holds and where it doesn't, measured, not
asserted, and every claim below is dated to when it was actually checked.

**Quickstart, honestly:** the live URL above is currently down (502, see [§12](#12-live-demo)).
The fastest way to see the real system working today is local Docker (§13, Hindi-only, verified
minutes before this was written) or the local multilingual candidate (§12A, 14 languages, more
setup). See [Known limitations](#15-known-limitations) before judging by voice alone — real
human-microphone verification is still pending for both systems (text/audio-file paths are real
and verified; a live mic in an actual browser has not been tested end-to-end by a human).

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
12A. [The multilingual candidate — local only, not deployed](#12a-the-multilingual-candidate--local-only-not-deployed)
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

**⚠️ Re-checked 2026-08-20 (Phase 6 of this session, after the multilingual work below): the live
URL currently returns `502 Bad Gateway` on `GET /healthz`, from Render's own edge, on two attempts
15 seconds apart (to rule out a simple cold-start hiccup — a cold start times out slowly, it
doesn't fail fast with a clean 502).** Cause not diagnosed — could be the free-tier instance being
fully decommissioned after extended inactivity, a crash, or something else. **Not touched, not
redeployed, per this phase's explicit instructions** (deployment work is out of scope). Reported
here rather than left as a silent contradiction with the "✅ Live HTTPS URL" verification below,
which was real and true when it was performed, earlier the same day:

Verified 2026-08-20 (earlier the same day, before the deployment went unreachable), against the
current deployed commit:
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

## 12A. The multilingual candidate — local only, not deployed

Built across a separate work session (six phases, `docs/DECISIONS.md` ADR-008 through ADR-015).
**Never deployed. Lives only at `data/index/multilingual_100k/` on the machine that built it —
gitignored, no GitHub release asset, no Docker image built for it.** Everything below is real and
measured; none of it changes what's live at the URL above (still Hindi-only, currently down, see
§12).

**What it is:** the same harness, guardrails, and API — pointed at a bigger, real multilingual
corpus instead of the Hindi-only one, via an opt-in `VRAG_INDEX_DIR` environment variable (unset
by default, so a fresh clone/CI/the existing Docker image behave exactly as before this work).

| | Value | Source |
|---|---|---|
| Chunks | 107,678 | real build, `data/index/multilingual_100k/` |
| Languages | 14 (as, bn, en, gu, hi, kn, ml, mr, ne, or, pa, sa, ta, ur) | `src/vrag/languages.py` |
| Retrieval | Dense-only FAISS HNSW32+SQfp16, **language-filtered** (+8.7-9.1pp Recall@10 over unfiltered, every size tested) | ADR-009/ADR-011 |
| Steady RSS (warm, lean SQLite lookup) | **406.2MB** | isolated-subprocess audit, re-run 2026-08-20 |
| Peak RSS | **492.7MB** | same audit |
| Startup RAM (index + lookup loaded, embedder not yet warm) | ~208MB | same audit |
| Pipeline latency, P50/P70/P95/P100 (local, 500 real samples) | **13.0 / 13.7 / 15.5 / 39.0ms** | `scripts/bench_latency.py`, re-run 2026-08-20 against this candidate |
| `retrieve` stage, P50/P100 | 11.2 / 36.5ms | same run |
| Recall@1 / @5 / @10, MRR@10 (532-query real held-out set, filtered) | 0.222 / 0.573 / 0.679 / 0.362 | ADR-013 |
| G3 abstention rate (same 532-query set) | **66.5%** — substantially higher than the Hindi-only baseline's 25.8% | ADR-013 |

**Why abstention is high, and why `TAU=0.8835` was deliberately left unchanged:** two dedicated
recalibration phases (ADR-013, ADR-014) investigated this and found the real cause is a **weak
underlying signal**, not a bad threshold. Top-1 cosine score barely separates correct from
incorrect retrieval on this multilingual, mixed-script corpus (correct-hit median 0.885, wrong-hit
median 0.869 — heavy overlap; wrong-hit scores go as high as 0.946). A full TAU sweep (61 points
across the observed range) found no threshold that cuts abstention without proportionally
increasing wrong answers. A follow-up experiment tried 15 cheap deterministic signals (score
dispersion, lexical/entity overlap, language consistency, score concentration — no neural model,
no LLM) plus 4 combinations: two signals beat top1 on aggregate AUC, but every candidate — checked
against the exact "capital of India" failure case this whole effort exists to prevent — either
answered it with the wrong evidence directly, or only avoided that by chance in a way that failed
7 of the 10 highest-confidence *wrong* queries in the set. The real, apples-to-apples number:
every candidate signal traded **~2.3-2.6 new wrong answers for every 1 new correct answer
recovered** on queries the current system safely abstains on today. All rejected. **TAU=0.8835,
MARGIN=0.0 remain exactly as calibrated for the Hindi-only corpus** — this is reported as a real,
unresolved limitation (fixing it needs a better *signal*, e.g. a per-language reranker or a
stronger embedder — both out of scope here), not silently patched to look better.

**Real end-to-end test, 2026-08-20 (`scripts/e2e_demo_readiness_test.py` +
`scripts/e2e_bonus_answered_cases.py`, 19 real cases total, results in
`eval/e2e_demo_readiness_results.json` / `eval/e2e_bonus_answered_results.json`):**

- **Voice tested for four languages (Hindi, English, Bengali, Tamil), and a critical, previously-
  undiscovered finding came out of it (Phase 9, ADR-018) — read this before relying on voice for
  anything but English.** All four, via Sarvam's real STT with `language_code="auto"` (the
  production default), were **auto-detected as English** — Bengali and Tamil audio got
  transliterated into English-script approximations instead of transcribed in their real script.
  Isolated the cause directly: re-running the same audio with an **explicit** (non-auto) language
  code transcribes both correctly, in the real script, near-perfectly. **The STT model can
  transcribe every language tested — only auto-detection fails.** This matters because the entire
  language-routing architecture (`query_language` → G2 → language-filtered retrieval →
  `generation_language`) depends on auto-detection for real voice input; if it defaults to
  English, every downstream stage — independently verified correct via text, in every phase —
  never receives the right input for any language but English. Not fixed here (no STT
  configuration change was authorized); flagged as the single most urgent open item from this
  whole local build. **Also disclosed here plainly: no voice test in this project, this one
  included, has used genuine human-spoken microphone input** — every audio file (this test's and
  Phase 6's) is Sarvam-TTS-synthesized speech through real STT, not a human recording. That
  distinction matters more than usual given what this test found. Every other language was tested
  via text (permitted — voice requires physically speaking many languages, which wasn't done;
  **no voice result is fabricated**).
- **Answered, with correct in-language generation** (real, not translated to Hindi): English,
  Bengali, Marathi, Tamil, Kannada, Urdu, Gujarati, Assamese — 8 of the 9 text-tested languages
  produced at least one real grounded answer in its own script during this test pass (Hindi
  answered too, via case 13 below). Every `generation_language` matched the query's language
  exactly.
- **The flagship regression case, both languages, both correctly safe:** "भारत की राजधानी क्या
  है?" (Hindi, top1=0.821) and "What is the capital of India?" (English, tested against the real
  English corpus slice this time, top1=0.817) — both abstain. Neither confidently cites the wrong
  country's capital (the Hindi query's top-1 hit is still the same Bangkok/Thailand passage found
  in earlier phases; the English query's top-1 hit is an unrelated "world's most expensive cities"
  passage — a different wrong passage, same correct outcome).
- **A real successful grounded answer** (the "this is what working looks like" case): "क्या मोनो
  खाँसी का कारण बनता है?" (Hindi) → answered correctly in Hindi, confidence 0.885, citing the
  actually-relevant passage — not cherry-picked far above threshold, a realistic pass.
- **Unsupported language correctly refused before any retrieval call**: a Telugu-script test
  query (`te-IN` — Sarvam can transcribe Telugu, but MSMARCO-XI has no Telugu training data to
  index it against) was refused by G2 in 0.08ms, `retrieve()` never called.
- **A genuinely unanswerable question correctly abstained**, not hallucinated: a constructed,
  out-of-domain English question found no confident evidence and abstained (top1=0.808).

**A real disk-footprint finding, cleaned up in Phase 9 (with one real mistake along the way) —
not a RAM finding** (the RAM numbers above are the true runtime footprint either way): the local
`data/index/multilingual_100k/` directory is 365MB on disk; only ~246MB of it (`dense/` +
`chunk_lookup.sqlite3`) is ever loaded at runtime. The remaining 120MB, `chunk_lookup.json`, was
initially flagged as dead weight — it isn't: it's a real, active dependency of several legitimate
offline tools (`add_english_to_multilingual_index.py`, `eval_multilingual_retrieval.py`, and
others), just not something production loads at request time, so it's **kept**. The embedder
directory (`data/onnx/multilingual-e5-small/`) genuinely was bloated — 583MB, of which only the
118MB int8 `model_quint8_avx2.onnx` + 5MB `sentencepiece.bpe.model` are ever read at runtime
(`LiteE5Embedder`) — the 470MB FP32 `model.onnx` was confirmed unreferenced anywhere and deleted
(regenerable via the existing `scripts/export_onnx_embedder.py` if ever needed again). Its
neighbor `tokenizer.json` (17MB) was deleted in the same pass on the same "unreferenced" check —
**that check was wrong**: an `src/`-only grep missed that a real regression test
(`tests/index/test_lite_e5_embedder_tokenizer_regression.py`, 11 tests) depends on it as a
byte-identical comparison baseline. Caught immediately by running the full suite, fixed by
regenerating the file directly from the HF tokenizer. **Real net saving: 448MB** (embedder
directory 583MB → 135MB), with the mistake-and-fix disclosed rather than only the clean outcome.
Neither directory is committed to git or shipped in the Docker image, so none of this affects the
deployed system.

**How to run this candidate locally** (no Docker image exists for it yet — this is the real,
current, multi-step process, not a one-command shortcut):

```bash
git clone https://github.com/ShivDyutiVerma/vRAG.git
cd vRAG
cp .env.example .env   # fill in SARVAM_API_KEY
pip install -e ".[retrieval-lean,pipeline,dev]"

# Build the multilingual corpus + index yourself -- there is no pre-built artifact to download.
# This downloads real MSMARCO-XI data (~10M rows across 13 languages) and takes real wall-clock
# time and bandwidth; it is not a quick judge-friendly path today.
python scripts/build_multilingual_dataset_subset.py
python scripts/build_multilingual_index.py --size 100k
python scripts/add_english_to_multilingual_index.py

# Point the app at the candidate you just built and run it
VRAG_INDEX_DIR="$(pwd)/data/index/multilingual_100k" \
  uvicorn vrag.api.main:app --app-dir src --host 127.0.0.1 --port 8000

curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  --data-binary '{"query": "What is the capital of India?", "k": 5, "language": "en-IN"}'
# (use --data-binary with a real UTF-8 JSON body for non-ASCII queries -- inline shell -d args can
# mangle non-Latin scripts on some shells/locales, confirmed while writing this section)
```

## 13. Local Docker instructions

**This section builds the deployed Hindi-only system.** For the newer 14-language local candidate,
see [§12A](#12a-the-multilingual-candidate--local-only-not-deployed) instead — no Docker image
exists for it yet. Re-verified working 2026-08-20, independent of the live URL's current 502
(§12): `docker run` locally against the already-built `vrag-real:v3` image, real `/healthz` →
`{"status":"ok","retrieval":"real"}`, real `/ask` call → real retrieval (cold P50 ~1.4s on the
very first query, warm ~27ms after).

```bash
git clone https://github.com/ShivDyutiVerma/vRAG.git
cd vRAG
cp .env.example .env   # fill in SARVAM_API_KEY (dashboard.sarvam.ai) -- required for STT/Track B
# GROQ_API_KEY may be left empty -- generation uses Sarvam's own LLM endpoint
# (src/vrag/generation/sarvam_llm.py); nothing in src/ reads GROQ_API_KEY.

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
- **The live deployed URL currently returns 502** — checked 2026-08-20, not diagnosed further, not
  touched (deployment work explicitly out of scope this session). See [§12](#12-live-demo). Local
  Docker (Hindi-only) and the local multilingual candidate ([§12A](#12a-the-multilingual-candidate--local-only-not-deployed))
  both verified working independently the same day.
- **The multilingual candidate ([§12A](#12a-the-multilingual-candidate--local-only-not-deployed))
  abstains far more often than the Hindi-only system** — 66.5% vs. 25.8%, on real per-language
  evidence. **Four independent investigations** (cheap retrieval reranking, an embedding-model
  head-to-head against two real alternatives, and now a shrinkage-based per-language threshold)
  all reach the same conclusion: no safe fix exists at the threshold or retrieval-reranking layer.
  Every cheap fix tried was rejected as unsafe or unstable. Genuinely unresolved — flagged, not
  hidden; the real fix needs a better embedding model or a per-language-aware reranker, out of
  scope for every pass attempted so far.
- **The multilingual candidate is not packaged for distribution** — no GitHub release asset, no
  Docker image. Reproducing it requires rerunning the real data-download-and-build pipeline
  locally (real time and bandwidth), documented in [§12A](#12a-the-multilingual-candidate--local-only-not-deployed),
  not a one-command judge path today.
- **Real voice input auto-detects as English for every non-English language tested (Hindi, Bengali,
  Tamil) — a critical, previously-undiscovered finding.** Sarvam's real STT, in its production
  `language_code="auto"` mode, mis-detected all three as English, transliterating Bengali/Tamil
  speech into English-script approximations. Isolated directly: the *same* audio, given an
  *explicit* (non-auto) language code, transcribes correctly in the real script — the recognizer
  itself works, only auto-detection fails. Since the entire language-routing pipeline depends on
  auto-detection for real voice input, this means real spoken queries in any language but English
  risk being silently mis-routed today. Not fixed (no STT config change authorized where this was
  found); the single most urgent open item from this build. **Also disclosed: no voice test in
  this project has used genuine human speech** — every audio file, including this one, is
  Sarvam-TTS-synthesized through real STT, not a human recording. Full detail in
  [§12A](#12a-the-multilingual-candidate--local-only-not-deployed).

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
| Live HTTPS URL | ⚠️ **Currently down (502), checked 2026-08-20, not yet fixed** — was verified working earlier the same day; local Docker (same system) verified working today as a fallback | [§11](#11-the-render-free-cpu-limitation-honestly), [§12](#12-live-demo) |
| Real Sarvam STT, never mocked | ✅ | `src/vrag/stt/sarvam.py`, zero mocks in `tests/` |
| ≥5 chunking strategies, evaluated | ✅ (6 shipped) | [§5](#5-chunking-strategy) |
| Real harness (typed stages, deadline propagation, circuit breaker, structured output) | ✅ | [§8](#8-harness-and-deadline-propagation) |
| 5 guardrail layers, calibrated where applicable | ✅ (G4's threshold uncalibrated, disclosed) | [§7](#7-guardrails-and-refusal-behavior) |
| P50/P70/P100 latency, real N | ✅ | [§10](#10-latency-results) |
| Multilingual support (14 languages) | ✅ **local candidate only, not deployed** — real, measured, higher abstention rate disclosed | [§12A](#12a-the-multilingual-candidate--local-only-not-deployed) |
| Real human microphone verification | ⏳ **Pending** — one real STT-via-audio-file test performed (Hindi, revealed a real language-detection limitation); live browser mic by an actual human still not tested | [§12](#12-live-demo), [§12A](#12a-the-multilingual-candidate--local-only-not-deployed), [§15](#15-known-limitations) |
| Demo video | ⏳ Not yet recorded | — |
| Process video | ⏳ Not yet recorded | — |
| Submission checklist / promotion grid | See `docs/SUBMISSION_CHECKLIST.md` | — |

**Deadline:** 2026-08-22 23:59 IST. No resubmissions.
