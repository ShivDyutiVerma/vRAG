"""FastAPI app: POST /ask (text debug entry point) + WS /voice (real mic -> Sarvam -> transcript).

Day 2 scope (see docs/PROGRESS_P.md): STT is real, retrieval is the real R/P seam (falls back to
the Day-1 stub shape automatically when no index is present locally, see
src/vrag/retrieval/interface.py), and requests now run through the real harness — G1/G2 input
guardrails, deadline-propagated Budget, Track A answer extraction, G5 output redaction — via
`src/vrag/harness/stages.py::default_stages()`. Still missing: G3 confidence gate and G4
groundedness (joint work with Workstream R, needs real retrieval scores — Day 3 per
docs/TEAM_SPLIT.md §5), Track B generation, retries/circuit-breaker on a real network-calling
stage (retrieve() never raises by contract, so there's nothing to retry yet — see
docs/DECISIONS_P.md).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.websockets import WebSocketState

from vrag.harness.budget import Budget
from vrag.harness.pipeline import PipelineContext, run_pipeline
from vrag.harness.stages import default_stages
from vrag.retrieval.interface import is_retrieval_real
from vrag.schemas import AnswerResponse
from vrag.stt.sarvam import stream_transcribe
from vrag.telemetry.trace import build_trace_record, emit_trace

# Uvicorn configures its own loggers but not the root logger, so module-level logger.info() calls
# (e.g. in vrag.stt.sarvam) are silently dropped without this. Needed for visibility on a hosted
# deploy where we can't attach a debugger — see docs/DECISIONS_P.md P-006.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

logger = logging.getLogger(__name__)

@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Forces `retrieve()`'s lazy retriever singleton (src/vrag/retrieval/interface.py) to load
    at boot instead of on first `/ask` — otherwise a Docker memory test's "startup peak" would
    miss the FAISS+embedder load entirely (it'd happen on the first request instead) and
    /healthz would report ready before retrieval actually is. `retrieve()`'s own "never raises"
    contract is untouched — this only decides *when* the existing lazy load happens, not whether
    a failure degrades to the stub.

    VRAG_REQUIRE_REAL_RETRIEVAL=1 (unset by default, never set in the Dockerfile) makes a failed
    load fatal at startup instead of silently degrading — opt-in, for validation runs where a
    healthy-looking stub-backed container would be a false pass (docs/RISKS.md R4), not a
    production default; the graceful degradation P built for a flaky/partial deploy stays intact
    unless this is explicitly requested."""
    real = is_retrieval_real()
    if not real and os.environ.get("VRAG_REQUIRE_REAL_RETRIEVAL") == "1":
        raise RuntimeError(
            "VRAG_REQUIRE_REAL_RETRIEVAL=1 but the real retriever failed to load — see the "
            "logged exception above for the cause. Refusing to start with stub-only retrieval."
        )
    yield


app = FastAPI(title="vrag", version="0.1.0", lifespan=_lifespan)

_FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"

# Target budget per AGENT_BUILD_SPEC.md §4's "Total to Track A answer" row. Generous relative to
# the ~30ms target until Phase 6 actually measures the real per-stage numbers — today's job is
# proving the mechanism works, not hitting the final number.
_DEFAULT_BUDGET_MS = 200.0


class AskRequest(BaseModel):
    query: str
    k: int = 5


async def build_answer(
    query: str, k: int = 5, budget_ms: float = _DEFAULT_BUDGET_MS
) -> AnswerResponse:
    """Runs the real harness pipeline (G1 -> G2 -> Retrieve -> Track A -> G5 -> Assemble) and
    fires a trace emission in the background — never awaited before the response is built, so
    disk I/O never sits on the hot path (docs/CONVENTIONS.md)."""
    ctx = PipelineContext(query=query, budget=Budget(total_ms=budget_ms))
    ctx.data["k"] = k
    await run_pipeline(ctx, default_stages())
    response: AnswerResponse = ctx.data["answer_response"]

    async def _emit() -> None:
        try:
            emit_trace(build_trace_record(ctx, budget_ms))
        except Exception:
            logger.exception("Failed to emit trace record")

    asyncio.create_task(_emit())
    return response


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "retrieval": "real" if is_retrieval_real() else "stub"}


@app.post("/ask", response_model=AnswerResponse)
async def ask(req: AskRequest) -> AnswerResponse:
    return await build_answer(req.query, k=req.k)


@app.websocket("/voice")
async def voice(websocket: WebSocket) -> None:
    """Browser mic -> raw PCM16 frames over this socket -> real Sarvam STT -> transcript events
    relayed back, per docs/API_CONTRACTS.md. On a final transcript, runs the full harness
    pipeline (build_answer) and returns the resulting AnswerResponse."""
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
                    answer = await build_answer(event.text)
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
