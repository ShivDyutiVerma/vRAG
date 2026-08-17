# ARCHITECTURE.md

> Kept current with the code as actually built. As of this writing (Day 1), this describes the
> *target* architecture from `AGENT_BUILD_SPEC.md` §5 — reconcile this file with reality at the
> end of every phase, per that file's own instruction (§7 P7 task 3).

## Request lifecycle

```
BROWSER (HTTPS required for mic)
  MediaRecorder → PCM/WAV chunks → WebSocket
  Renders: live transcript, Track A answer, Track B stream, citations, refusal states, latency HUD
        │ WS: audio in / events out
        ▼
FASTAPI ORCHESTRATOR ("the harness")
  Stage 0  AudioIngest      VAD, buffering, format validation
  Stage 1  Transcribe       Sarvam streaming WS → final text
  ─────────── t_pipeline clock starts here ───────────
  Stage 2  InputGuard       G1 safety + G2 scope/language
  Stage 3  QueryPrep        normalise, embed, (opt) expand
  Stage 4  Retrieve         dense ∥ sparse (parallel) — calls retrieve() (Workstream R)
  Stage 5  Fuse             RRF, (opt) cross-encoder rerank
  Stage 6  GroundGate       G3 confidence + margin → maybe ABSTAIN
  Stage 7a ExtractAnswer    Track A — span selection
  Stage 7b Generate         Track B — streaming LLM + tools
  Stage 8  VerifyOutput     G4 groundedness, G5 output safety, citation validity
  Stage 9  Assemble         structured AnswerResponse + trace emit
        │
   ┌────┼──────────┐
   ▼    ▼          ▼
In-process assets   Sarvam APIs        Telemetry sink
 FAISS HNSW          STT (WS)           traces.jsonl
 BM25 index          LLM (Track B)      per-stage ns timings
 embedder (ONNX)
 reranker (ONNX)
```

## Module boundaries

See `docs/TEAM_SPLIT.md` §2 for the authoritative ownership table. In short: everything upstream
of `src/vrag/retrieval/interface.py::retrieve()` is Workstream R; everything else — STT, harness,
generation, guardrails (mostly), telemetry, API, frontend, deployment — is Workstream P (this
track).

## The seam: `retrieve()`

```python
# src/vrag/retrieval/interface.py
class RetrievedChunk(BaseModel):
    chunk_id: str
    passage_id: str
    text: str
    score: float
    language: str

async def retrieve(query: str, k: int = 5) -> list[RetrievedChunk]:
    """Never raises. Returns [] on internal failure."""
```

Day 1: stub, returns fake data (see `src/vrag/retrieval/interface.py`). Day 2 sync: swapped for
Workstream R's real implementation — a one-line import change if both sides hold the signature.

## Two-track answer design

Track A (extractive, always runs, ~15-30ms) is emitted immediately and reliably lands inside the
200ms budget. Track B (generative, streams in behind it) replaces Track A in the UI when its first
token arrives, if it arrives in time. See `AGENT_BUILD_SPEC.md` §3.3 for the full rationale — this
is the design that makes the 200ms constraint honest rather than fudged.

## Deploy runbook

Target (Day 1, insurance deploy): Hugging Face Spaces, Docker SDK, single container serving the
FastAPI app + built frontend on one origin (no CORS). HTTPS is automatic on Spaces, which satisfies
the mic-access requirement immediately. Region/latency-optimised hosting (Render/Railway/Fly.io per
the Phase 0 probe) is a later-phase upgrade if the Spaces free tier's region hurts `t_pipeline`.

Index shipping (once Workstream R has one): do not build FAISS at container start. Build offline,
commit as a release asset, download-and-mmap at boot. Record the index build hash in every trace.
