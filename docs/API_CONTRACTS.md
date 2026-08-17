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
Workstream R builds the real implementation in isolation. Swap is a one-line import change,
scheduled for the Day 2 (Aug 19) sync per `docs/TEAM_SPLIT.md` §5.

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

## HTTP / WebSocket surface (Workstream P, `src/vrag/api/`)

| Route | Protocol | Purpose |
|-------|----------|---------|
| `POST /ask` | HTTP | Text-only entry point, for testing without a mic |
| `WS /voice` | WebSocket | Audio in (PCM chunks) → transcript/answer/citation/refusal events out |
| `GET /health` | HTTP | Returns ready only after all models + the JSON schema are warmed at boot |

Full request/response JSON shapes are defined by `AnswerResponse` above; no separate schema drift is
allowed — the WS event stream emits partial/final versions of the same model, not an ad hoc shape.

## `search_corpus` tool (exposed to Track B's LLM)

```python
async def search_corpus(query: str, k: int) -> list[Passage]
```

A thin wrapper around `retrieve()`. Cap tool-call depth at 1 — the model may issue exactly one
follow-up retrieval if the first pass is insufficient, never more (protects the latency budget).
