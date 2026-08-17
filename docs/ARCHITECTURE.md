# ARCHITECTURE.md

> Kept current with the code as actually built (reconciled fully in Phase 7 — see
> `AGENT_BUILD_SPEC.md` §9 P7). Right now (Day 0, Phase 0) this reflects the **design**, not yet a
> built system — nothing under `src/vrag/` exists yet.

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
4. `retrieve()` (the R/P seam, `src/vrag/retrieval/interface.py`) runs dense + sparse concurrently,
   fuses with RRF, optionally reranks if the budget allows.
5. `GroundGate` (G3) decides answered vs. abstained from calibrated τ/margin thresholds.
6. Track A always computes a fast extractive answer. Track B streams a generative one over it if the
   budget and provider allow; the UI swaps Track A's answer for Track B's first token when it lands.
7. `VerifyOutput` runs G4 (citation-ID + n-gram grounding check) and G5 (PII redaction) before the
   response is assembled and emitted with `timings_ms` populated for every stage that ran.
8. A `TraceRecord` is appended to `traces.jsonl` regardless of outcome (answered/abstained/refused/degraded).

## Module boundaries

See `docs/TEAM_SPLIT.md` §2 for the authoritative ownership table. In one line: everything upstream of
`retrieve()` (chunking, embedding, dense/sparse index, hybrid fusion, reranking) is Workstream R;
everything from `retrieve()`'s return value onward (harness, guardrails except G3/G4, generation,
telemetry, API, frontend, deployment) is Workstream P. The only jointly-owned code path is
`src/vrag/retrieval/interface.py` itself and the G3/G4 guardrail calibration (needs retrieval scores
from R and generation output from P).

## Deploy runbook

_To be filled in at the end of Phase 1 once the first deploy actually happens (owned by Workstream P,
`AGENT_BUILD_SPEC.md` §5.3). Placeholder intentionally left empty rather than guessed — an
undocumented runbook is more honest than a wrong one._
