# API_CONTRACTS.md

## `retrieve()` — the cross-workstream seam

```python
# src/vrag/retrieval/interface.py

class RetrievedChunk(BaseModel):
    chunk_id: str
    passage_id: str
    text: str
    score: float          # 0-1, fused/reranked relevance
    language: str

async def retrieve(query: str, k: int = 5) -> list[RetrievedChunk]:
    """Never raises. Returns [] on internal failure — the guardrail layer treats
    an empty list as 'nothing relevant found' and routes to abstention."""
```

Day 1: stub returning fake data. Day 2 sync: swapped for the real implementation. Changing this
signature after Day 0 requires a joint ADR in `docs/DECISIONS.md` — not a unilateral edit.

## HTTP

### `POST /ask`
Text-only debug entry point (bypasses STT), for testing the rest of the pipeline without a
microphone.

Request:
```json
{"query": "भारत में सबसे ऊँचा पर्वत कौन सा है?", "k": 5}
```

Response: `AnswerResponse` (see `src/vrag/schemas.py` / `AGENT_BUILD_SPEC.md` §7.2):
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

Day 1 scope: `transcript_partial`/`transcript_final` from real Sarvam STT, then a placeholder
answer built from the stub `retrieve()` result — no harness orchestration wired through yet.
