"""Phase 9C (docs/DECISIONS.md ADR-018): local voice-path validation for Hindi, English, Bengali,
Tamil. Full real path: real audio -> real Sarvam realtime STT -> detected language -> query_language
-> real G2 -> real language-filtered retrieval -> real G3 -> Track A (Track B budget-gated, same
as every other phase) -> answer in the detected language -> citations. Uses build_answer() itself
(src/vrag/api/main.py), not a hand-rolled substitute, so this is exactly what /voice's WebSocket
handler does internally minus the WebSocket transport framing (pure I/O plumbing, identical either
way -- established in Phase 6).

**Honesty correction, made explicit here rather than left implicit:** NO voice test in this
project to date -- this one included -- has used genuine human-spoken microphone input. The
existing eval/audio/*.wav files (used for the P6 latency campaign and Phase 6's "real voice" test)
are themselves Sarvam-TTS-synthesized (scripts/synthesize_test_audio.py), not human recordings.
This script is consistent with that: Hindi reuses the existing real (TTS) audio file; English,
Bengali, and Tamil are freshly synthesized via Sarvam's real TTS API (real network call, real
audio bytes) using real MSMARCO-XI queries (not hand-typed). Every one of these is a genuine
audio-file round trip through real STT -- validating the full pipeline integration for real -- but
none of it is a substitute for an actual human microphone test, which remains unperformed for
every language, disclosed plainly.

Usage: python scripts/phase9_voice_validation.py
Output: eval/phase9_voice_validation_results.json
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
import wave
from io import BytesIO
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ["VRAG_INDEX_DIR"] = str(REPO_ROOT / "data" / "index" / "multilingual_100k")
sys.path.insert(0, str(REPO_ROOT / "src"))

from vrag.api.main import build_answer  # noqa: E402
from vrag.config import settings  # noqa: E402
from vrag.stt.sarvam import stream_transcribe  # noqa: E402

_TTS_URL = "https://api.sarvam.ai/text-to-speech"
_TTS_MODEL = "bulbul:v2"
_TTS_SPEAKER = "anushka"

HELD_PATH = REPO_ROOT / "eval" / "heldout_queries_multilingual.json"
HINDI_WAV = REPO_ROOT / "eval" / "audio" / "query_000.wav"

# Real MSMARCO-XI queries (not hand-typed), reused from earlier phases for continuity.
CASES = [
    {"label": "english", "query_id": 115346, "tts_lang": "en-IN"},
    {"label": "bengali", "query_id": 1170381, "tts_lang": "bn-IN"},
    {"label": "tamil", "query_id": 871988, "tts_lang": "ta-IN"},  # a known real true-accept case
]


def synthesize_wav_pcm(text: str, tts_lang: str) -> bytes:
    """Real Sarvam TTS call -> raw PCM16 bytes (WAV container stripped, matching stream_transcribe's
    expected input shape -- same as how the existing real Hindi WAV is consumed)."""
    resp = httpx.post(
        _TTS_URL,
        headers={"api-subscription-key": settings.sarvam_api_key},
        json={
            "text": text,
            "target_language_code": tts_lang,
            "speaker": _TTS_SPEAKER,
            "model": _TTS_MODEL,
            "speech_sample_rate": 16000,
        },
        timeout=30,
    )
    resp.raise_for_status()
    audio_b64 = resp.json()["audios"][0]
    wav_bytes = base64.b64decode(audio_b64)
    with wave.open(BytesIO(wav_bytes), "rb") as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2 and w.getnchannels() == 1
        return w.readframes(w.getnframes())


async def run_voice_case(label: str, pcm: bytes, expected_query: str | None = None) -> dict:
    chunk_size = 3200  # ~100ms at 16kHz/16-bit mono, same chunking as Phase 6's real voice test

    async def audio_chunks():
        for i in range(0, len(pcm), chunk_size):
            yield pcm[i : i + chunk_size]

    t0 = time.perf_counter()
    final_text = None
    detected_language = None
    async for event in stream_transcribe(audio_chunks(), language_code="auto"):
        if event.type == "final":
            final_text = event.text
            detected_language = event.language
            break
        if event.type == "error":
            return {"label": label, "stt_error": event.text, "expected_query": expected_query}
    stt_ms = (time.perf_counter() - t0) * 1000

    if not final_text:
        return {
            "label": label,
            "stt_error": "no final transcript event received",
            "expected_query": expected_query,
        }

    t0 = time.perf_counter()
    response = await build_answer(final_text, language=detected_language)
    pipeline_ms = (time.perf_counter() - t0) * 1000

    return {
        "label": label,
        "expected_query": expected_query,
        "stt_transcript": final_text,
        "stt_detected_language": detected_language,
        "stt_ms": stt_ms,
        "pipeline_ms": pipeline_ms,
        "total_voice_to_answer_ms": stt_ms + pipeline_ms,
        "status": response.status,
        "answer_language": response.language,
        "query_language_recorded": response.query_language,
        "confidence": response.confidence,
        "answer": response.answer,
        "citations": [c.model_dump() for c in response.citations],
        "refusal_reason": response.refusal_reason,
    }


async def main() -> None:
    held = json.loads(HELD_PATH.read_text(encoding="utf-8"))
    held_by_id = {q["query_id"]: q for q in held}

    results = []

    print("[hindi] using existing real (TTS-synthesized) audio file...")
    with wave.open(str(HINDI_WAV), "rb") as w:
        pcm = w.readframes(w.getnframes())
    manifest = json.loads(
        (REPO_ROOT / "eval" / "audio" / "manifest.json").read_text(encoding="utf-8")
    )
    r = await run_voice_case("hindi", pcm, expected_query=manifest[0]["query"])
    results.append(r)
    print(
        f"  transcript={r.get('stt_transcript')!r} "
        f"detected={r.get('stt_detected_language')} status={r.get('status')}"
    )

    for case in CASES:
        q = held_by_id[case["query_id"]]
        print(f"\n[{case['label']}] synthesizing real TTS audio for: {q['query'][:60]!r}...")
        pcm = synthesize_wav_pcm(q["query"], case["tts_lang"])
        r = await run_voice_case(case["label"], pcm, expected_query=q["query"])
        results.append(r)
        print(
            f"  transcript={r.get('stt_transcript')!r} detected={r.get('stt_detected_language')} "
            f"status={r.get('status')} answer_lang={r.get('answer_language')}"
        )

    out_path = REPO_ROOT / "eval" / "phase9_voice_validation_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(results)} cases -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
