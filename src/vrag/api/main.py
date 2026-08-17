"""FastAPI app: POST /ask (text debug entry point) + WS /voice (real mic -> Sarvam -> transcript).

Day 1 scope (see docs/PROGRESS_P.md): STT is real, retrieval is the Day-1 stub from
src/vrag/retrieval/interface.py, and the "answer" is Track A only — literally the best-supporting
retrieved passage, no generation/guardrails/harness orchestration wired through yet. That wiring
is explicitly Day 2 work per docs/TEAM_SPLIT.md §5. Nothing here pretends otherwise: every
response's `stages_skipped` lists what isn't implemented yet.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.websockets import WebSocketState

from vrag.retrieval.interface import retrieve
from vrag.schemas import AnswerResponse, Citation
from vrag.stt.sarvam import stream_transcribe

# Uvicorn configures its own loggers but not the root logger, so module-level logger.info() calls
# (e.g. in vrag.stt.sarvam) are silently dropped without this. Needed for visibility on a hosted
# deploy where we can't attach a debugger — see docs/DECISIONS_P.md P-006.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

logger = logging.getLogger(__name__)

app = FastAPI(title="vrag", version="0.1.0")

_FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"

# NOT-YET-IMPLEMENTED stages, honestly declared rather than silently skipped.
_DAY1_STAGES_SKIPPED = ["input_guardrail", "rerank", "grounding_gate", "generate", "output_guardrail"]


class AskRequest(BaseModel):
    query: str
    k: int = 5


async def build_placeholder_answer(query: str, k: int = 5) -> AnswerResponse:
    """Track A only: best-supporting retrieved span, verbatim. See module docstring for scope."""
    trace_id = uuid.uuid4().hex
    start_ns = time.perf_counter_ns()
    chunks = await retrieve(query, k=k)
    retrieve_ms = (time.perf_counter_ns() - start_ns) / 1_000_000

    if not chunks:
        return AnswerResponse(
            status="abstained",
            answer=None,
            track="extractive",
            citations=[],
            confidence=0.0,
            refusal_reason="No relevant passages found for this query.",
            language="hi",
            stages_skipped=_DAY1_STAGES_SKIPPED,
            trace_id=trace_id,
            timings_ms={"retrieve": retrieve_ms},
        )

    top = chunks[0]
    return AnswerResponse(
        status="answered",
        answer=top.text,
        track="extractive",
        citations=[
            Citation(chunk_id=c.chunk_id, passage_id=c.passage_id, score=c.score, text_span=c.text)
            for c in chunks[:2]
        ],
        confidence=top.score,
        refusal_reason=None,
        language=top.language,
        stages_skipped=_DAY1_STAGES_SKIPPED,
        trace_id=trace_id,
        timings_ms={"retrieve": retrieve_ms},
    )


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
async def ask(req: AskRequest) -> AnswerResponse:
    return await build_placeholder_answer(req.query, k=req.k)


@app.websocket("/voice")
async def voice(websocket: WebSocket) -> None:
    """Browser mic -> raw PCM16 frames over this socket -> real Sarvam STT -> transcript events
    relayed back, per docs/API_CONTRACTS.md. On a final transcript, runs retrieval and returns a
    Track A placeholder answer."""
    await websocket.accept()
    logger.info("WS /voice: accepted")

    async def browser_audio_chunks():
        n = 0
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                logger.info("WS /voice: browser disconnected after %d chunks", n)
                return
            if message.get("type") == "websocket.disconnect":
                logger.info("WS /voice: browser disconnect message after %d chunks", n)
                return
            audio_bytes = message.get("bytes")
            if audio_bytes:
                n += 1
                yield audio_bytes
                continue
            text = message.get("text")
            if text:
                try:
                    control = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if control.get("event") == "stop":
                    logger.info("WS /voice: received stop control after %d chunks", n)
                    return

    try:
        async for event in stream_transcribe(browser_audio_chunks(), language_code="hi-IN"):
            logger.info("WS /voice: transcript event: %s", event)
            if event.type == "partial":
                await websocket.send_json({"type": "transcript_partial", "text": event.text})
            elif event.type == "final":
                await websocket.send_json({"type": "transcript_final", "text": event.text})
                if event.text.strip():
                    answer = await build_placeholder_answer(event.text)
                    await websocket.send_json(
                        {"type": "answer_final", "answer_response": answer.model_dump()}
                    )
            elif event.type == "error":
                await websocket.send_json({"type": "error", "detail": event.text})
        logger.info("WS /voice: stream_transcribe generator finished normally")
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 - last resort so nothing hangs/vanishes silently
        logger.exception("WS /voice: unhandled error")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "error", "detail": str(exc)})
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()


if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
