"""P6 latency campaign, task 2 (docs/BUILD_PLAN.md): synthesize spoken versions of every query in
eval/test_queries.json via Sarvam's own TTS, so the voice benchmark (t_e2e_voice, STT broken out)
is reproducible from committed audio files instead of depending on someone speaking 100 queries by
hand each time the benchmark is re-run.

Degenerate queries with no real text content (empty string, pure punctuation/digit noise) are
skipped -- there's nothing meaningful to speak, and scripts/bench_latency.py exercises those
directly as text queries instead (matches how they're used in guardrail testing).

Usage: python scripts/synthesize_test_audio.py
Requires SARVAM_API_KEY in .env.
"""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path

import httpx

from vrag.config import settings

REPO_ROOT = Path(__file__).resolve().parent.parent
QUERIES_PATH = REPO_ROOT / "eval" / "test_queries.json"
AUDIO_DIR = REPO_ROOT / "eval" / "audio"

_TTS_URL = "https://api.sarvam.ai/text-to-speech"
_MODEL = "bulbul:v2"
_SPEAKER = "anushka"
_LANGUAGE = "hi-IN"
_SAMPLE_RATE = 16000  # matches src/vrag/stt/sarvam.py's stream_transcribe default -- the bench
# script feeds these files straight into WS /voice, so they must already be at the rate STT expects

# Nothing meaningful to synthesize -- no letters/digits at all (empty, whitespace, pure
# punctuation/symbol noise). Mixed-script and repeated-word degenerate queries DO have real text
# and are synthesized normally.
_NO_SPEAKABLE_CONTENT = re.compile(r"^[\s\W]*$", re.UNICODE)


def _has_speakable_content(text: str) -> bool:
    return not _NO_SPEAKABLE_CONTENT.match(text)


def synthesize(client: httpx.Client, text: str) -> bytes:
    resp = client.post(
        _TTS_URL,
        headers={"api-subscription-key": settings.sarvam_api_key},
        json={
            "text": text,
            "target_language_code": _LANGUAGE,
            "speaker": _SPEAKER,
            "model": _MODEL,
            "speech_sample_rate": _SAMPLE_RATE,
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    audios = resp.json()["audios"]
    return base64.b64decode(audios[0])


if __name__ == "__main__":
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    rows = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))

    manifest = []
    with httpx.Client() as client:
        for i, row in enumerate(rows):
            text = row["query"]
            if not _has_speakable_content(text):
                print(f"[{i:3d}/{len(rows)}] skip (no speakable content): {row['category']}")
                manifest.append({**row, "audio_file": None})
                continue

            out_path = AUDIO_DIR / f"query_{i:03d}.wav"
            try:
                audio_bytes = synthesize(client, text)
                out_path.write_bytes(audio_bytes)
                print(f"[{i:3d}/{len(rows)}] {row['category']:10s} -> {out_path.name} "
                      f"({len(audio_bytes)} bytes)")
                manifest.append({**row, "audio_file": out_path.name})
            except httpx.HTTPStatusError as e:
                print(f"[{i:3d}/{len(rows)}] FAILED: {e.response.status_code} "
                      f"{e.response.text[:200]}")
                manifest.append({**row, "audio_file": None})
            time.sleep(0.1)  # be polite to the API, not a hard rate limit requirement

    (AUDIO_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    n_synthesized = sum(1 for m in manifest if m["audio_file"])
    print(f"\nSynthesized {n_synthesized}/{len(rows)} audio files to {AUDIO_DIR}")
