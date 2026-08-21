"""Phase 10 (docs/DECISIONS.md ADR-019): explicit user-selected language for /voice, replacing
Sarvam's `language_code="auto"` (found in ADR-018 to default to English for every non-English
language tested). Covers the /languages endpoint, the WS start-control protocol, and that
query_language is always the user's selection -- never silently overwritten by whatever Sarvam's
own (structurally absent, in explicit mode) `event.language` says.

`stream_transcribe` is monkeypatched at the `vrag.api.main` import site with a small recording
fake -- the same class of test this project already uses for STT (tests/stt/test_sarvam_stt.py
mocks `ws_connect`, one layer lower; here we mock one layer higher, at the seam `main.py` actually
calls, since what's under test is the WS handler's own language-selection logic, not
stream_transcribe's internal event parsing, which already has its own dedicated tests). The real
STT path is never mocked in production code -- this is test-only, per CLAUDE.md's hard rule.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import vrag.api.main as main
from vrag.harness.budget import Budget
from vrag.harness.pipeline import PipelineContext, run_pipeline
from vrag.harness.stages import default_stages
from vrag.languages import SUPPORTED_LANGUAGES
from vrag.stt.sarvam import TranscriptEvent

client = TestClient(main.app)


class _RecordingFakeSTT:
    """Records every `language_code` it was called with and every audio chunk it actually
    consumed (so "the first chunk isn't silently dropped" is verifiable), then yields one final
    transcript event with a configurable (possibly mismatched) `language`."""

    def __init__(self, final_text: str = "test query", final_language: str | None = None) -> None:
        self.language_codes_seen: list[str] = []
        self.chunks_seen: list[bytes] = []
        self.final_text = final_text
        self.final_language = final_language

    async def __call__(self, audio_chunks, language_code: str = "auto"):
        self.language_codes_seen.append(language_code)
        async for chunk in audio_chunks:
            self.chunks_seen.append(chunk)
        yield TranscriptEvent(type="final", text=self.final_text, language=self.final_language)


def _drain_to_answer(ws) -> dict:
    while True:
        msg = ws.receive_json()
        if msg["type"] == "answer_final":
            return msg["answer_response"]
        if msg["type"] == "error":
            raise AssertionError(f"WS /voice returned an error: {msg['detail']}")


@pytest.mark.parametrize(
    "code",
    ["hi-IN", "en-IN", "bn-IN", "ta-IN", "gu-IN", "mr-IN", "kn-IN"],
)
def test_selected_language_is_passed_explicitly_to_stt(monkeypatch, code):
    """Requirement: Hindi/English/Bengali/Tamil selected -> hi-IN/en-IN/bn-IN/ta-IN reaches the
    STT call, plus 3 more supported languages (Gujarati/Marathi/Kannada) for breadth."""
    fake = _RecordingFakeSTT()
    monkeypatch.setattr(main, "stream_transcribe", fake)
    with client.websocket_connect("/voice") as ws:
        ws.send_json({"event": "start", "language": code})
        ws.send_bytes(b"\x00\x01" * 50)
        ws.send_json({"event": "stop"})
        answer = _drain_to_answer(ws)
    assert fake.language_codes_seen == [code]
    assert answer["query_language"] == code


def test_unsupported_language_cannot_be_selected_falls_back_to_hindi(monkeypatch):
    """The frontend only ever offers SUPPORTED_LANGUAGES, but the backend must not trust that --
    a malformed/tampered start control with an unsupported code falls back to Hindi (requirement
    9's backward-compatibility default), not silently accepted."""
    fake = _RecordingFakeSTT()
    monkeypatch.setattr(main, "stream_transcribe", fake)
    with client.websocket_connect("/voice") as ws:
        ws.send_json({"event": "start", "language": "fr-FR"})
        ws.send_bytes(b"\x00\x01" * 50)
        ws.send_json({"event": "stop"})
        answer = _drain_to_answer(ws)
    assert fake.language_codes_seen == ["hi-IN"]
    assert answer["query_language"] == "hi-IN"


def test_no_start_control_defaults_to_hindi_and_preserves_first_audio_chunk(monkeypatch):
    """A client that predates Phase 10 sends audio immediately, no start control at all --
    defaults to Hindi (backward compatibility) and the first audio chunk must not be dropped."""
    fake = _RecordingFakeSTT()
    monkeypatch.setattr(main, "stream_transcribe", fake)
    first_chunk = b"\x11\x22" * 50
    second_chunk = b"\x33\x44" * 50
    with client.websocket_connect("/voice") as ws:
        ws.send_bytes(first_chunk)
        ws.send_bytes(second_chunk)
        ws.send_json({"event": "stop"})
        _drain_to_answer(ws)
    assert fake.language_codes_seen == ["hi-IN"]
    assert fake.chunks_seen == [first_chunk, second_chunk]


def test_sarvam_reported_language_mismatch_does_not_override_query_language(monkeypatch):
    """Requirement 12: if Sarvam ever DOES return a language different from the user's selection,
    routing must not silently switch -- query_language stays the user's selection."""
    fake = _RecordingFakeSTT(final_language="en-IN")  # Sarvam claims English regardless
    monkeypatch.setattr(main, "stream_transcribe", fake)
    with client.websocket_connect("/voice") as ws:
        ws.send_json({"event": "start", "language": "ta-IN"})  # user selected Tamil
        ws.send_bytes(b"\x00\x01" * 50)
        ws.send_json({"event": "stop"})
        answer = _drain_to_answer(ws)
    assert answer["query_language"] == "ta-IN"
    assert answer["query_language"] != "en-IN"


def test_languages_endpoint_matches_supported_languages_source_of_truth():
    resp = client.get("/languages")
    assert resp.status_code == 200
    body = resp.json()
    codes = {row["code"] for row in body}
    assert codes == SUPPORTED_LANGUAGES
    for row in body:
        assert row["name"], f"{row['code']} has no display name"


def test_voice_stop_control_still_ends_the_stream_correctly(monkeypatch):
    """Regression: the pre-existing stop-control protocol (unrelated to the new start control)
    must keep working -- WS /voice behavior beyond language selection is unaffected."""
    fake = _RecordingFakeSTT()
    monkeypatch.setattr(main, "stream_transcribe", fake)
    with client.websocket_connect("/voice") as ws:
        ws.send_json({"event": "start", "language": "hi-IN"})
        ws.send_bytes(b"\x00\x01" * 20)
        ws.send_json({"event": "stop"})
        msg1 = ws.receive_json()
        assert msg1["type"] == "transcript_final"
        answer = _drain_to_answer(ws)
    assert answer["status"] in ("answered", "abstained", "refused", "degraded")


def test_existing_g2_behavior_unchanged_for_an_explicitly_selected_language(monkeypatch):
    """G2's real language-code validation (src/vrag/guardrails/g2_scope_language.py) still runs
    exactly as before -- a selected, supported language still reaches retrieval, never refused
    for a language reason."""
    fake = _RecordingFakeSTT(final_text="भारत की राजधानी क्या है?")
    monkeypatch.setattr(main, "stream_transcribe", fake)
    with client.websocket_connect("/voice") as ws:
        ws.send_json({"event": "start", "language": "hi-IN"})
        ws.send_bytes(b"\x00\x01" * 20)
        ws.send_json({"event": "stop"})
        answer = _drain_to_answer(ws)
    assert answer["status"] != "refused"
    assert "retrieve" in answer["timings_ms"]


def test_answer_response_schema_unchanged(monkeypatch):
    """No regression to AnswerResponse's shape -- every pre-existing field is still present."""
    fake = _RecordingFakeSTT()
    monkeypatch.setattr(main, "stream_transcribe", fake)
    with client.websocket_connect("/voice") as ws:
        ws.send_json({"event": "start", "language": "hi-IN"})
        ws.send_bytes(b"\x00\x01" * 20)
        ws.send_json({"event": "stop"})
        answer = _drain_to_answer(ws)
    for field in (
        "status",
        "answer",
        "track",
        "citations",
        "confidence",
        "refusal_reason",
        "language",
        "query_language",
        "stages_skipped",
        "trace_id",
        "timings_ms",
    ):
        assert field in answer


@pytest.mark.asyncio
async def test_generation_language_follows_selected_query_language(monkeypatch):
    """Offline/in-process verification (not through the WS layer): with a real selected
    query_language and the deterministic Day-0 stub retriever (forced, matching this project's
    existing test convention), ExtractAnswerStage's generation_language must default to the
    selected query_language -- confirmed directly against ctx.data, since generation_language
    isn't part of the public AnswerResponse schema."""
    import vrag.retrieval.interface as interface

    monkeypatch.setattr(interface, "_get_real_retriever", lambda: None)

    ctx = PipelineContext(query="test query", budget=Budget(total_ms=200.0))
    ctx.data["query_language"] = "ta-IN"
    await run_pipeline(ctx, default_stages())

    assert ctx.data.get("query_language") == "ta-IN"
    assert ctx.data.get("generation_language") == "ta-IN"
