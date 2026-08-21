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
from vrag.languages import LANGUAGE_DISPLAY_NAMES, SUPPORTED_LANGUAGES, is_supported
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
    """Forces `retrieve()`'s lazy retriever singleton — FAISS + SQLite load, ONNX session +
    SentencePieceProcessor construction, and one real warmup embedding
    (`is_retrieval_real()`, src/vrag/retrieval/interface.py, R-035) — to happen at boot, before
    `yield`, instead of on first `/ask`. FastAPI/uvicorn don't accept ANY request (including
    `/healthz`) until this coroutine reaches `yield`, so this is what makes "no cold starts" and
    "`/healthz` only ready after real warmup" true structurally, not just by convention: there is
    no window where the process is accepting traffic but retrieval isn't actually warm yet.
    Before R-035, this only forced the retriever *object* to load — `LiteE5Embedder._ensure_
    loaded()` was still lazy, so the first real query paid a real ~1.1s ONNX/SentencePiece
    construction cost that this hook didn't cover (docs/DECISIONS_R.md R-032/R-035). `retrieve()`'s
    own "never raises" contract is untouched — this only decides *when* the existing lazy load
    (now including warmup) happens, not whether a failure degrades to the stub.

    VRAG_REQUIRE_REAL_RETRIEVAL=1 (unset by default, never set in the Dockerfile) makes a failed
    load-or-warmup fatal at startup instead of silently degrading — opt-in, for validation runs
    where a healthy-looking stub-backed container would be a false pass (docs/RISKS.md R4), not a
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

# Phase 10 (docs/DECISIONS.md ADR-019): Sarvam's `language_code="auto"` was found to default to
# English for every non-English language tested (real Hindi/Bengali/Tamil audio, ADR-018) — the
# underlying STT transcribes correctly once given an explicit code, only auto-detection fails.
# `/voice` now requires the client to select a language up front instead of relying on
# auto-detection to discover it after the fact. Hindi is the default when no selection is sent
# (backward compatibility with a client that predates this change), matching this project's
# original Hindi-only scope.
_DEFAULT_VOICE_LANGUAGE = "hi-IN"


class AskRequest(BaseModel):
    query: str
    k: int = 5
    # Phase 1 (docs/DECISIONS.md ADR-009): optional Sarvam-format language hint (e.g. "hi-IN") —
    # this text endpoint has no real STT behind it, so there's no real signal unless a caller
    # supplies one directly (useful for testing G2's language routing without a live Sarvam call).
    # None (the default) preserves the exact pre-Phase-1 behavior: G2 falls back to its script
    # heuristic, unchanged.
    language: str | None = None


async def build_answer(
    query: str,
    k: int = 5,
    budget_ms: float = _DEFAULT_BUDGET_MS,
    language: str | None = None,
) -> AnswerResponse:
    """Runs the real harness pipeline (G1 -> G2 -> Retrieve -> Track A -> G5 -> Assemble) and
    fires a trace emission in the background — never awaited before the response is built, so
    disk I/O never sits on the hot path (docs/CONVENTIONS.md).

    `language`: the query's language — the caller-supplied hint for `/ask`, or (since Phase 10,
    ADR-019) the user's explicitly *selected* language for `/voice`, no longer Sarvam's own
    auto-detected one. None when no signal exists at all. Stored as `query_language` — never
    overloaded onto the "language" key the rest of the pipeline uses for the answer's own
    language (see src/vrag/languages.py)."""
    ctx = PipelineContext(query=query, budget=Budget(total_ms=budget_ms))
    ctx.data["k"] = k
    ctx.data["query_language"] = language
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


@app.get("/languages")
async def languages() -> list[dict[str, str]]:
    """Phase 10 (docs/DECISIONS.md ADR-019): the frontend's language selector is populated from
    this endpoint rather than a hardcoded list, so it can never drift out of sync with
    `src/vrag/languages.py`'s `SUPPORTED_LANGUAGES` — the same source G2/retrieval already use."""
    out = []
    for code in SUPPORTED_LANGUAGES:
        name, script = LANGUAGE_DISPLAY_NAMES.get(code, (code, ""))
        out.append({"code": code, "name": name, "script": script})
    out.sort(key=lambda row: row["name"])
    return out


@app.post("/ask", response_model=AnswerResponse)
async def ask(req: AskRequest) -> AnswerResponse:
    return await build_answer(req.query, k=req.k, language=req.language)


async def _read_selected_language(websocket: WebSocket) -> tuple[str, bytes | None]:
    """Phase 10 (docs/DECISIONS.md ADR-019): the client's first WS message must be
    `{"event": "start", "language": "<supported code>"}`, sent before any audio. Returns the
    resolved language (falling back to `_DEFAULT_VOICE_LANGUAGE` for a missing/unsupported code —
    requirement 9's "backward compatibility" default) and, for an old client that skips the start
    control and sends audio immediately, that first audio chunk so it isn't silently dropped."""
    try:
        message = await websocket.receive()
    except WebSocketDisconnect:
        return _DEFAULT_VOICE_LANGUAGE, None
    if message.get("type") == "websocket.disconnect":
        return _DEFAULT_VOICE_LANGUAGE, None

    text = message.get("text")
    if text:
        try:
            control = json.loads(text)
        except json.JSONDecodeError:
            control = {}
        if control.get("event") == "start":
            candidate = control.get("language")
            if isinstance(candidate, str) and is_supported(candidate):
                return candidate, None
            logger.info(
                "WS /voice: missing/unsupported language in start control (%r) — defaulting to %s",
                candidate,
                _DEFAULT_VOICE_LANGUAGE,
            )
            return _DEFAULT_VOICE_LANGUAGE, None
        return _DEFAULT_VOICE_LANGUAGE, None

    # No start control at all (client predates Phase 10) — first message is presumably audio;
    # don't drop it, hand it back to be replayed as the stream's first chunk.
    logger.info("WS /voice: no start control received — defaulting to %s", _DEFAULT_VOICE_LANGUAGE)
    return _DEFAULT_VOICE_LANGUAGE, message.get("bytes")


@app.websocket("/voice")
async def voice(websocket: WebSocket) -> None:
    """Browser mic -> raw PCM16 frames over this socket -> real Sarvam STT -> transcript events
    relayed back, per docs/API_CONTRACTS.md. On a final transcript, runs the full harness
    pipeline (build_answer) and returns the resulting AnswerResponse.

    Phase 10 (docs/DECISIONS.md ADR-019): the client selects its language up front (see
    `_read_selected_language`) instead of relying on Sarvam's `language_code="auto"` detection,
    which ADR-018 found defaults to English for every non-English language tested. `query_language`
    is always the user's selection, never Sarvam's own detected language — see the mismatch check
    below."""
    await websocket.accept()
    selected_language, first_audio_chunk = await _read_selected_language(websocket)
    logger.info("WS /voice: accepted, selected_language=%s", selected_language)

    async def browser_audio_chunks():
        n = 0
        if first_audio_chunk:
            n += 1
            yield first_audio_chunk
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
        # Explicit, not "auto" (ADR-019) -- requirement 10: auto-detection is no longer the
        # primary routing mechanism. Sarvam only populates a transcript event's `language` field
        # in "auto" mode (verified, ADR-009) -- in explicit mode `event.language` is structurally
        # None on real Sarvam responses, so the mismatch check below is a defensive/dormant safety
        # net (kept per requirement 12), not an active signal under normal operation.
        audio_stream = browser_audio_chunks()
        async for event in stream_transcribe(audio_stream, language_code=selected_language):
            logger.info("WS /voice: transcript event: %s", event)
            if event.type == "partial":
                await websocket.send_json({"type": "transcript_partial", "text": event.text})
            elif event.type == "final":
                await websocket.send_json({"type": "transcript_final", "text": event.text})
                if event.language and event.language != selected_language:
                    logger.warning(
                        "WS /voice: Sarvam-reported language (%s) differs from user selection "
                        "(%s) — routing is NOT switched, query_language stays %s",
                        event.language,
                        selected_language,
                        selected_language,
                    )
                if event.text.strip():
                    answer = await build_answer(event.text, language=selected_language)
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
