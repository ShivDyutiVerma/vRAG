"""Real Sarvam streaming STT client (saaras:v3-realtime). Never mocked — see CLAUDE.md hard rules.

Server-to-server WebSocket: this backend holds the Sarvam API key and proxies browser audio to
Sarvam, relaying transcript events back over our own /voice WebSocket. The browser never sees the
Sarvam key. Endpoint/schema per docs.sarvam.ai/api-reference/speech-to-text/transcribe/realtime/ws
(fetched 2026-08-17; log any breaking API changes here or in docs/DECISIONS_P.md if they surface).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from typing import Literal
from urllib.parse import urlencode

import websockets
from pydantic import BaseModel
from websockets.asyncio.client import connect as ws_connect

from vrag.config import settings

logger = logging.getLogger(__name__)

_SARVAM_WS_BASE = "wss://api.sarvam.ai/speech-to-text-realtime/ws"


class TranscriptEvent(BaseModel):
    type: Literal["partial", "final", "error"]
    text: str
    language: str | None = None


def _build_url(language_code: str, sample_rate: int) -> str:
    params = {
        "language_code": language_code,
        "model": "saaras:v3-realtime",
        "stream_type": "balanced",
        "mode": "transcribe",
        "endpointing": "vad",
        "encoding": "linear16",
        "sample_rate": sample_rate,
    }
    return f"{_SARVAM_WS_BASE}?{urlencode(params)}"


async def stream_transcribe(
    audio_chunks: AsyncIterator[bytes],
    *,
    language_code: str = "hi-IN",
    sample_rate: int = 16000,
) -> AsyncIterator[TranscriptEvent]:
    """Pipe raw PCM16 audio chunks to Sarvam's realtime STT and yield transcript events as they
    arrive. Real network call — this is intentionally the one thing on the request path that is
    never stubbed, per CLAUDE.md hard rules."""
    if not settings.sarvam_api_key:
        raise RuntimeError(
            "SARVAM_API_KEY is not set. Add it to .env (see .env.example) — the STT path "
            "must always use a real key, never a mock."
        )

    url = _build_url(language_code, sample_rate)
    headers = {"API-SUBSCRIPTION-KEY": settings.sarvam_api_key}

    logger.info("Connecting to Sarvam realtime STT...")
    try:
        sarvam_ws = await asyncio.wait_for(ws_connect(url, additional_headers=headers), timeout=10)
    except TimeoutError as exc:
        logger.error("Timed out connecting to Sarvam STT after 10s")
        raise RuntimeError("Timed out connecting to Sarvam STT (10s)") from exc
    logger.info("Connected to Sarvam realtime STT")

    async with sarvam_ws:

        async def _sender() -> None:
            n = 0
            try:
                async for chunk in audio_chunks:
                    n += 1
                    await sarvam_ws.send(
                        json.dumps(
                            {"event": "audio_input", "audio": base64.b64encode(chunk).decode()}
                        )
                    )
            finally:
                logger.info("Audio sender done (%d chunks), sending 'end' to Sarvam", n)
                await sarvam_ws.send(json.dumps({"event": "end"}))
                logger.info("Sent 'end' to Sarvam")

        sender_task = asyncio.create_task(_sender())
        # Grace period to keep waiting for a trailing message after we've sent "end" and the
        # sender has nothing left to feed Sarvam. Sarvam doesn't always close its side promptly
        # (observed on silence-only audio) — without this bound the request-side generator would
        # hang forever instead of completing, which is unacceptable on a request path.
        end_grace_s = 3.0
        try:
            while True:
                timeout = end_grace_s if sender_task.done() else None
                logger.info("Waiting for Sarvam message (timeout=%s)", timeout)
                try:
                    raw = await asyncio.wait_for(sarvam_ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    logger.info("Grace period elapsed with no further message, ending stream")
                    break  # sender is done and nothing arrived within the grace period
                logger.info("Received from Sarvam: %s", raw)
                msg = json.loads(raw)
                event = msg.get("event")
                if event == "transcript.partial":
                    yield TranscriptEvent(
                        type="partial", text=msg.get("text", ""), language=msg.get("language")
                    )
                elif event == "transcript.final":
                    yield TranscriptEvent(
                        type="final", text=msg.get("text", ""), language=msg.get("language")
                    )
                elif event == "error":
                    logger.error("Sarvam STT error: %s", msg)
                    yield TranscriptEvent(type="error", text=msg.get("message", "unknown error"))
        except websockets.ConnectionClosed:
            pass
        finally:
            if not sender_task.done():
                sender_task.cancel()
