# API_CONTRACTS.md

## The R/P seam — `retrieve()`

The single contract between the two workstreams. Agreed jointly on Day 0 (Aug 17), documented
identically in `docs/TEAM_SPLIT.md` §1 and `docs/FRIEND_BRIEF.md` §3. **Changing this signature after
Day 0 is a joint decision, recorded as a new ADR in the shared `docs/DECISIONS.md` immediately.**

```python
# src/vrag/retrieval/interface.py

class RetrievedChunk(BaseModel):
    chunk_id: str
    passage_id: str
    text: str
    score: float          # fused/reranked relevance score, 0-1
    language: str          # ISO code detected/tagged at index time

async def retrieve(query: str, k: int = 5) -> list[RetrievedChunk]:
    """The one function Workstream P's harness calls. Workstream R owns the implementation.
    Must be safe to call concurrently. Must not raise — on internal failure, return []
    and let the harness's grounding gate handle the empty-result case."""
    ...
```

Workstream P builds against a stub returning 2-3 fake `RetrievedChunk` objects from hour one.
Workstream R builds the real implementation in isolation (done as of Day 1 —
`src/vrag/retrieval/hybrid.py`'s `HybridRetriever`, unit-tested including its dense∥sparse
concurrency, pending the A1 chunking-ablation winner before being wired in). Swap is a one-line
import change, scheduled for the Day 2 (Aug 19) sync per `docs/TEAM_SPLIT.md` §5.

## Canonical response schema — `AnswerResponse`

Owned by Workstream P (`src/vrag/schemas.py`), consumed by the frontend and the eval scripts.

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

Constraint: in the LLM's structured-output schema (Track B), the **reasoning field must precede the
answer field** — the model should think before it commits (`AGENT_BUILD_SPEC.md` §7.2 / `TECH_MENU.md` S11).

## HTTP

### `POST /ask`
Text-only entry point (bypasses STT) — for testing the rest of the pipeline without a microphone.

Request:
```json
{"query": "भारत में सबसे ऊँचा पर्वत कौन सा है?", "k": 5}
```

Response: `AnswerResponse` (see schema above / `AGENT_BUILD_SPEC.md` §7.2):
```json
{
  "status": "answered",
  "answer": "...",
  "track": "extractive",
  "citations": [{"chunk_id": "...", "passage_id": "...", "score": 0.0, "text_span": "..."}],
  "confidence": 0.0,
  "refusal_reason": null,
  "language": "hi",
  "stages_skipped": [],
  "trace_id": "...",
  "timings_ms": {"transcribe": 0.0, "retrieve": 0.0, "...": 0.0}
}
```

### `GET /health` (Workstream P calls it `/healthz` on the live deploy — reconcile the name here
with `src/vrag/api/main.py` once harness wiring lands)
Returns ready only after all models + the JSON schema are warmed at boot.

## WebSocket

### `WS /voice`
Client streams raw PCM (`pcm_s16le`, 16kHz) audio frames. Server streams back JSON events:

```json
{"type": "transcript_partial", "text": "..."}
{"type": "transcript_final", "text": "..."}
{"type": "stage", "name": "retrieve", "status": "started" | "done" | "skipped"}
{"type": "answer_extractive", "answer": "...", "citations": [...], "timings_ms": {...}}
{"type": "answer_generative_delta", "delta": "..."}
{"type": "answer_final", "answer_response": { /* AnswerResponse */ }}
{"type": "refused", "reason": "...", "layer": "G1" | "G2" | "G3" | "G4"}
{"type": "error", "detail": "..."}
```

Full request/response JSON shapes are defined by `AnswerResponse` above; no separate schema drift is
allowed — the WS event stream emits partial/final versions of the same model, not an ad hoc shape.

**Day 1 status (verified live on Render, `docs/DECISIONS_P.md` P-007):** `transcript_partial` /
`transcript_final` from real Sarvam STT work end to end; the answer emitted is currently a
simplified Track A placeholder (top retrieved chunk's full text) built from the stub `retrieve()`
result — no harness orchestration, guardrails, rerank, or grounding gate wired through yet. Every
`AnswerResponse` honestly lists the missing stages in `stages_skipped`.

## `search_corpus` tool (exposed to Track B's LLM)

```python
async def search_corpus(query: str, k: int) -> list[Passage]
```

A thin wrapper around `retrieve()`. Cap tool-call depth at 1 — the model may issue exactly one
follow-up retrieval if the first pass is insufficient, never more (protects the latency budget).
