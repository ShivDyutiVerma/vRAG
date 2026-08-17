# ARCHITECTURE.md

> Kept current with the code as actually built (reconciled fully in Phase 7 — see
> `AGENT_BUILD_SPEC.md` §9 P7). As of Day 1, the request-lifecycle diagram below is still the
> **target** design; the harness stages exist as shapes (`Stage`/`PipelineContext`/`Budget`) but
> aren't yet wired into the live request path — `/ask` and `/voice` call `retrieve()` directly
> today. The deploy section reflects the real, live deployment.

## System design

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
│  Stage 2  InputGuard       G1 safety + G2 scope/language       │
│  Stage 3  QueryPrep        normalise, embed (E5 "query: " pfx) │
│  Stage 4  Retrieve         dense ∥ sparse (asyncio.gather)     │
│  Stage 5  Fuse             RRF k=60, (opt) cross-encoder rerank│
│  Stage 6  GroundGate       G3 confidence + margin → ABSTAIN?   │
│  Stage 7a ExtractAnswer    Track A — span selection            │
│  Stage 7b Generate         Track B — streaming LLM + tools     │
│  Stage 8  VerifyOutput     G4 groundedness, G5 PII redaction   │
│  Stage 9  Assemble         AnswerResponse + trace emit         │
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

Full rationale for every box above is in `AGENT_BUILD_SPEC.md` §5. This file exists so a reader
doesn't have to open the 780-line spec to see the shape of the system.

## Request lifecycle walkthrough

1. Browser opens a WebSocket, streams PCM audio as the user speaks.
2. Sarvam's streaming STT (`saaras:v3-realtime`) returns interim + final transcripts. The
   `t_pipeline` clock starts the instant the final transcript is available — everything before that
   (mic capture, STT itself) is reported separately as `t_e2e_voice`, never folded into the headline
   number (see `AGENT_BUILD_SPEC.md` §3.2 for why, verbatim wording goes in the README).
3. The harness (`src/vrag/harness/pipeline.py`) runs the transcript through typed stages over a
   `PipelineContext`, each stage getting `remaining_ms` from `budget.py`'s deadline propagation.
   **Not yet wired into the live request path as of Day 1** — see the status note at the top.
4. `retrieve()` (the R/P seam, `src/vrag/retrieval/interface.py`) runs dense + sparse concurrently,
   fuses with RRF, optionally reranks if the budget allows. Real implementation
   (`HybridRetriever`, `src/vrag/retrieval/hybrid.py`) built and unit-tested; not yet swapped in for
   the Day-1 stub — pending the A1 chunking-ablation winner.
5. `GroundGate` (G3) decides answered vs. abstained from calibrated τ/margin thresholds.
6. Track A always computes a fast extractive answer. Track B streams a generative one over it if the
   budget and provider allow; the UI swaps Track A's answer for Track B's first token when it lands.
7. `VerifyOutput` runs G4 (citation-ID + n-gram grounding check) and G5 (PII redaction) before the
   response is assembled and emitted with `timings_ms` populated for every stage that ran.
8. A `TraceRecord` is appended to `traces.jsonl` regardless of outcome (answered/abstained/refused/degraded).

## Two-track answer design

Track A (extractive, always runs, ~15-30ms) is emitted immediately and reliably lands inside the
200ms budget. Track B (generative, streams in behind it) replaces Track A in the UI when its first
token arrives, if it arrives in time. See `AGENT_BUILD_SPEC.md` §3.3 for the full rationale — this
is the design that makes the 200ms constraint honest rather than fudged. **Day 1 status:** `/ask`
and `WS /voice` return a simplified Track A only (the top retrieved chunk's full text, not real
span-selection) — `stages_skipped` on every response honestly lists everything not yet wired in.

## Module boundaries

See `docs/TEAM_SPLIT.md` §2 for the authoritative ownership table. In one line: everything upstream of
`retrieve()` (chunking, embedding, dense/sparse index, hybrid fusion, reranking) is Workstream R;
everything from `retrieve()`'s return value onward (harness, guardrails except G3/G4, generation,
telemetry, API, frontend, deployment) is Workstream P. The only jointly-owned code path is
`src/vrag/retrieval/interface.py` itself and the G3/G4 guardrail calibration (needs retrieval scores
from R and generation output from P).

## Deploy runbook

**Live now:** `https://vrag-voice.onrender.com` — Render (Docker runtime, `render.yaml` Blueprint at
repo root, `Dockerfile` builds `pip install -e .` + serves via `uvicorn vrag.api.main:app`). Verified
end to end Day 1 with real Sarvam STT audio (`docs/DECISIONS_P.md` P-007), health-checked at
`/healthz`.

**Why Render, not Hugging Face Spaces (the spec's original insurance target):** attempted HF Spaces
first per `AGENT_BUILD_SPEC.md` §5.3; hit a real paywall (`402 Payment Required` — HF's free tier no
longer covers Docker/Gradio Spaces, policy changed since the spec was written). Switched to Render —
the spec's own primary recommendation, not just a fallback. Full record: `docs/DECISIONS_P.md` P-002
(superseded) and P-005. The same `Dockerfile` still works unmodified on HF Spaces if that access
changes later.

**Known non-blocking issue:** the `/voice` WebSocket lingers ~20-25s before a clean close on Render
(doesn't affect answer delivery — both `transcript_final` and `answer_final` arrive correctly within
~2.4s). Tracked as P-R12 in `docs/RISKS.md`.

**Index shipping (once Workstream R has one):** do not build FAISS/BM25 at container start — build
offline, commit as a release asset or object storage, download-and-mmap at boot. Record the index
build hash in every trace.
