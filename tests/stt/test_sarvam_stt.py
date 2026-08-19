"""Tests for src/vrag/stt/sarvam.py's no-speech timeout and sender shutdown guard (R-038).

The evidence behind these fixes (real live voice-path diagnosis: a silent session left the UI
stuck in LISTENING with no error for the full ~71s until Sarvam's own inactivity_timeout fired at
~60s, plus an unretrieved asyncio Task exception from sending "end" on an already-closed Sarvam
socket) lives in docs/DECISIONS_R.md R-038, not here — these tests exercise the resulting logic in
isolation with a fake Sarvam connection. No live network — the real network call in
stream_transcribe() (ws_connect) is monkeypatched, per this project's established convention for
unit-testing internal control flow around a network-calling function (see
tests/generation/test_sarvam_llm.py); the *production* STT path itself is never mocked.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import pytest
import websockets

from vrag.stt import sarvam
from vrag.stt.sarvam import NO_SPEECH_TIMEOUT_S, TranscriptEvent, stream_transcribe


class _FakeSarvamWS:
    """Minimal fake of the object stream_transcribe() actually uses: async context manager,
    `.send()`, `.recv()`. `recv_schedule` is a list of (delay_s, item) pairs consumed in order by
    `.recv()`, where `item` is either a raw message string or an Exception instance to raise.
    Once the schedule is exhausted, `.recv()` waits forever (mirroring a real open-but-silent
    Sarvam socket) so tests can exercise wait_for()'s own timeout behavior."""

    def __init__(self, recv_schedule: list[tuple[float, str | Exception]]) -> None:
        self._recv_schedule = list(recv_schedule)
        self.sent: list[str] = []

    async def __aenter__(self) -> _FakeSarvamWS:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def recv(self) -> str:
        if not self._recv_schedule:
            await asyncio.Event().wait()  # never resolves; caller bounds this with wait_for
            raise AssertionError("unreachable")
        delay, item = self._recv_schedule.pop(0)
        if delay:
            await asyncio.sleep(delay)
        if isinstance(item, Exception):
            raise item
        return item


def _transcript_msg(kind: str, text: str, language: str = "hi-IN") -> str:
    return json.dumps({"event": f"transcript.{kind}", "text": text, "language": language})


async def _endless_audio() -> AsyncIterator[bytes]:
    while True:
        yield b"\x00\x00" * 100
        await asyncio.sleep(0.01)


def _patch_connect(monkeypatch: pytest.MonkeyPatch, fake_ws: _FakeSarvamWS) -> None:
    async def _fake_connect(url: str, **kwargs: object) -> _FakeSarvamWS:
        return fake_ws

    monkeypatch.setattr(sarvam, "ws_connect", _fake_connect)
    monkeypatch.setattr(sarvam.settings, "sarvam_api_key", "fake-key-for-tests")


@pytest.mark.asyncio
async def test_no_speech_timeout_constant_is_ten_seconds() -> None:
    """The user-facing spec is a 10s bound; guard the real configured value directly."""
    assert NO_SPEECH_TIMEOUT_S == 10.0


@pytest.mark.asyncio
async def test_silent_session_yields_no_speech_error_at_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session where Sarvam never sends a transcript must surface the structured error at the
    (shrunk, for test speed) no-speech timeout — not hang, and not wait for Sarvam's own 60s
    watchdog, which this fake never simulates at all."""
    monkeypatch.setattr(sarvam, "NO_SPEECH_TIMEOUT_S", 0.05)
    fake_ws = _FakeSarvamWS(recv_schedule=[])  # never sends anything back
    _patch_connect(monkeypatch, fake_ws)

    events = []
    start = asyncio.get_event_loop().time()
    async for event in stream_transcribe(_endless_audio(), language_code="hi-IN"):
        events.append(event)
    elapsed = asyncio.get_event_loop().time() - start

    assert len(events) == 1
    assert events[0] == TranscriptEvent(
        type="error", text="No speech detected yet. Please try again."
    )
    # generous upper bound -- proves it resolved at OUR timeout, nowhere near a 60s watchdog
    assert elapsed < 2.0


@pytest.mark.asyncio
async def test_normal_transcript_flow_is_unaffected_by_no_speech_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once a first transcript arrives inside the no-speech window, the wait must revert to
    unbounded -- a real mid-session gap longer than the (shrunk) no-speech timeout must NOT be
    mistaken for silence and must not truncate the session."""
    monkeypatch.setattr(sarvam, "NO_SPEECH_TIMEOUT_S", 0.05)
    fake_ws = _FakeSarvamWS(
        recv_schedule=[
            (0.0, _transcript_msg("partial", "नमस्ते")),
            # deliberately longer than the shrunk no-speech timeout -- must not be cut off
            (0.2, _transcript_msg("final", "नमस्ते दुनिया")),
            # Sarvam closes once it has nothing more to say -- ends the loop deterministically
            # instead of racing an exhausted schedule against an unbounded post-transcript wait.
            (0.0, websockets.ConnectionClosedOK(None, None)),
        ]
    )
    _patch_connect(monkeypatch, fake_ws)

    events = []
    async for event in stream_transcribe(_endless_audio(), language_code="hi-IN"):
        events.append(event)

    assert [e.type for e in events] == ["partial", "final"]
    assert events[0].text == "नमस्ते"
    assert events[1].text == "नमस्ते दुनिया"


@pytest.mark.asyncio
async def test_final_transcript_closes_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A normal end-to-end lifecycle -- audio ends, a final transcript arrives, Sarvam has nothing
    more to say -- must terminate via the existing grace-period path, with no error and no hang."""

    async def _short_audio() -> AsyncIterator[bytes]:
        yield b"\x00\x00" * 100

    fake_ws = _FakeSarvamWS(
        recv_schedule=[
            # small real delay: gives the event loop a tick to actually run the background
            # sender task (a bare `create_task` only schedules it) before recv() resolves
            (0.01, _transcript_msg("final", "पूरा वाक्य")),
            # Sarvam closes its side once done -- deterministic clean end, no race against a
            # real-time grace period.
            (0.0, websockets.ConnectionClosedOK(None, None)),
        ]
    )
    _patch_connect(monkeypatch, fake_ws)

    events = []
    async for event in stream_transcribe(_short_audio(), language_code="hi-IN"):
        events.append(event)

    assert len(events) == 1
    assert events[0].type == "final"
    assert events[0].text == "पूरा वाक्य"
    assert fake_ws.sent  # the sender did send an "end" event on its own clean shutdown
    assert json.loads(fake_ws.sent[-1]) == {"event": "end"}


@pytest.mark.asyncio
async def test_already_closed_sarvam_connection_produces_no_unhandled_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If Sarvam's socket is already closed by the time the sender's cleanup tries to say 'end'
    (e.g. Sarvam's own inactivity watchdog fired first), that must not surface as an unretrieved
    background-task exception. Simulate exactly the scenario Fix 2 targets: normal audio sends
    succeed throughout, but by the time the sender's cleanup tries to send the final "end" event,
    Sarvam has already closed its side -- and separately, the receive loop also discovers the
    connection is closed."""
    unhandled: list[dict[str, object]] = []
    loop = asyncio.get_event_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, context: unhandled.append(context))

    class _CloseOnEndWS(_FakeSarvamWS):
        async def send(self, data: str) -> None:
            if json.loads(data).get("event") == "end":
                raise websockets.ConnectionClosedError(None, None)
            self.sent.append(data)

    async def _two_chunks() -> AsyncIterator[bytes]:
        yield b"\x00\x00" * 100
        yield b"\x00\x00" * 100

    fake_ws = _CloseOnEndWS(
        # small real delay: gives the event loop a tick to actually run the background sender
        # task (a bare `create_task` only schedules it) before recv() resolves
        recv_schedule=[(0.01, websockets.ConnectionClosedError(None, None))]
    )
    _patch_connect(monkeypatch, fake_ws)

    try:
        events = []
        async for event in stream_transcribe(_two_chunks(), language_code="hi-IN"):
            events.append(event)
        # give the background sender task's finally-block a chance to run and, if buggy,
        # report an unretrieved exception via the loop's exception handler
        await asyncio.sleep(0.05)
    finally:
        loop.set_exception_handler(previous_handler)

    assert events == []  # recv() closed immediately -- nothing to yield, no crash either
    assert len(fake_ws.sent) == 2  # both real audio chunks went out fine before the close
    assert unhandled == [], f"unhandled task exception(s) leaked: {unhandled}"


@pytest.mark.asyncio
async def test_sarvam_error_event_still_relayed_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-existing behavior, unaffected by this fix: a real error event FROM Sarvam (not our
    synthesized no-speech error) must still come through as-is."""
    sarvam_error_msg = json.dumps({"event": "error", "message": "some real Sarvam-side error"})
    fake_ws = _FakeSarvamWS(
        recv_schedule=[
            (0.0, sarvam_error_msg),
            # error events don't count as a transcript, so without this the loop would fall
            # through to the (real, unshrunk) no-speech timeout next -- close deterministically
            # instead, matching how Sarvam actually behaves once it has nothing more to send.
            (0.0, websockets.ConnectionClosedOK(None, None)),
        ]
    )
    _patch_connect(monkeypatch, fake_ws)

    events = []
    async for event in stream_transcribe(_endless_audio(), language_code="hi-IN"):
        events.append(event)

    assert len(events) == 1
    assert events[0] == TranscriptEvent(type="error", text="some real Sarvam-side error")
