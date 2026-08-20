"""Phase 6 (docs/DECISIONS.md ADR-015): full end-to-end local test of the multilingual candidate,
run through the REAL harness (PipelineContext + run_pipeline + default_stages() -- the exact
sequence api/main.py's build_answer() calls), pointed at data/index/multilingual_100k/ via
VRAG_INDEX_DIR. No production code is modified; this script only calls existing public functions
and keeps the returned PipelineContext for introspection (retrieved chunks/scores/language even on
an abstained query, which AnswerResponse alone doesn't expose).

Real queries are used throughout, drawn from eval/heldout_queries_multilingual.json (the actual
MSMARCO-XI-derived held-out set already used for every prior calibration phase) -- not hand-typed,
except the single unsupported-language case (Telugu has no MSMARCO-XI train data at all, so no
real Telugu query exists to draw from; used only to exercise G2's language-code refusal, which
does not depend on the text's content -- see src/vrag/guardrails/g2_scope_language.py).

Case 1 (Hindi) is the one REAL VOICE test: reads an existing real 16kHz mono PCM16 WAV
(eval/audio/query_000.wav, from the P6 latency campaign) and pipes it through the real
stream_transcribe() Sarvam realtime STT call (real network call, real API key) -- exactly the
same stream_transcribe() -> build_answer() sequence src/vrag/api/main.py's /voice WebSocket
handler uses internally. The WebSocket transport itself (chunk framing over the socket) is not
exercised -- that is I/O plumbing, not pipeline logic, and is identical either way. All other
cases are text-path with an explicit language hint (permitted per Phase 6 instructions).

Usage: python scripts/e2e_demo_readiness_test.py
Output: eval/e2e_demo_readiness_results.json
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ["VRAG_INDEX_DIR"] = str(REPO_ROOT / "data" / "index" / "multilingual_100k")
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from vrag.harness.budget import Budget  # noqa: E402
from vrag.harness.pipeline import PipelineContext, run_pipeline  # noqa: E402
from vrag.harness.stages import default_stages  # noqa: E402
from vrag.retrieval.interface import is_retrieval_real  # noqa: E402
from vrag.stt.sarvam import stream_transcribe  # noqa: E402

# Real queries are looked up by query_id from eval/heldout_queries_multilingual.json at run time
# (not hand-transcribed into this file) -- every one of these was verified during test-case
# selection to be a clean, natural query (a few of Urdu's raw held-out entries turned out to be
# corrupted MSMARCO-XI data artifacts -- repeated garbage tokens -- and were deliberately skipped
# in favor of a clean one, query_id 977954, at selection time).
_HELDOUT = json.loads(
    (REPO_ROOT / "eval" / "heldout_queries_multilingual.json").read_text(encoding="utf-8")
)
_HELDOUT_BY_ID = {q["query_id"]: q["query"] for q in _HELDOUT}

TEXT_CASES = [
    {"id": 2, "label": "english_text", "query_id": 115346, "language": "en-IN"},
    {"id": 3, "label": "bengali_text", "query_id": 1170381, "language": "bn-IN"},
    {"id": 4, "label": "marathi_text", "query_id": 878826, "language": "mr-IN"},
    {"id": 5, "label": "tamil_text", "query_id": 427054, "language": "ta-IN"},
    {"id": 6, "label": "kannada_text", "query_id": 559678, "language": "kn-IN"},
    {"id": 7, "label": "urdu_text", "query_id": 977954, "language": "ur-IN"},
    {"id": "8a", "label": "gujarati_text", "query_id": 201165, "language": "gu-IN"},
    {"id": "8b", "label": "assamese_text", "query_id": 77149, "language": "as-IN"},
    {
        "id": 9,
        "label": "unsupported_telugu_text",
        "query_id": None,
        "query_literal": "ఇది ఏమిటి?",  # "idi emiti?"
        "language": "te-IN",
        "note": "hand-typed (Telugu has no MSMARCO-XI train data, so no real corpus query "
        "exists) -- tests G2's language-code refusal, which does not depend on text content "
        "(src/vrag/guardrails/g2_scope_language.py checks the language code first).",
    },
    {
        "id": 10,
        "label": "unanswerable_out_of_domain",
        "query_id": None,
        "query_literal": (
            "What did the vRAG project's lead engineer eat for breakfast on August 20th, 2026?"
        ),
        "language": "en-IN",
        "note": "constructed to be genuinely absent from a 2019-era MS-MARCO-derived corpus",
    },
    {
        "id": 11,
        "label": "capital_of_india_hindi",
        "query_id": None,
        "query_literal": "भारत की राजधानी क्या है?",
        "language": "hi-IN",
    },
    {
        "id": 12,
        "label": "capital_of_india_english",
        "query_id": None,
        "query_literal": "What is the capital of India?",
        "language": "en-IN",
        "note": "filtered against the real English (eng_Latn) slice this time, not pinned to "
        "hin_Deva as earlier diagnostic scripts did -- this is what a real English speaker "
        "actually gets",
    },
    {
        "id": 13,
        "label": "strong_evidence_success_demo",
        "query_id": 169261,
        "language": "hi-IN",
        "note": "real MSMARCO-XI query with the gold passage genuinely at rank 1 (top1=0.885, "
        "recall@1=1.0) -- the deliberate 'this is what a real successful answer looks like' case",
    },
]

for _case in TEXT_CASES:
    if _case.get("query_id") is not None:
        _case["query"] = _HELDOUT_BY_ID[_case["query_id"]]
    else:
        _case["query"] = _case.pop("query_literal")


async def run_case(query: str, language: str | None) -> dict:
    ctx = PipelineContext(query=query, budget=Budget(total_ms=200.0))
    ctx.data["k"] = 5
    ctx.data["query_language"] = language
    t0 = time.perf_counter()
    await run_pipeline(ctx, default_stages())
    total_ms = (time.perf_counter() - t0) * 1000

    response = ctx.data.get("answer_response")
    chunks = ctx.data.get("chunks") or []
    return {
        "status": response.status if response else None,
        "answer": response.answer if response else None,
        "answer_language": response.language if response else None,
        "query_language_recorded": response.query_language if response else None,
        "confidence": response.confidence if response else None,
        "refusal_reason": response.refusal_reason if response else None,
        "refusal_layer": ctx.data.get("refusal_layer"),
        "citations": (
            [c.model_dump() for c in response.citations] if response and response.citations else []
        ),
        "retrieved_chunks_top5": [
            {
                "chunk_id": c.chunk_id,
                "passage_id": c.passage_id,
                "score": c.score,
                "language": c.language,
                "text_preview": c.text[:120],
            }
            for c in chunks
        ],
        "top1_score": chunks[0].score if chunks else None,
        "retrieved_language": chunks[0].language if chunks else None,
        "generation_language": ctx.data.get("generation_language"),
        "timings_ms": response.timings_ms if response else None,
        "total_wall_ms": total_ms,
    }


async def run_voice_case() -> dict:
    wav_path = REPO_ROOT / "eval" / "audio" / "query_000.wav"
    manifest = json.loads(
        (REPO_ROOT / "eval" / "audio" / "manifest.json").read_text(encoding="utf-8")
    )
    expected_query = manifest[0]["query"]

    with wave.open(str(wav_path), "rb") as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2 and w.getnchannels() == 1
        pcm = w.readframes(w.getnframes())

    chunk_size = 3200  # ~100ms at 16kHz/16-bit mono

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
            return {
                "stt_error": event.text,
                "expected_query_from_manifest": expected_query,
                "wav_file": str(wav_path.relative_to(REPO_ROOT)),
            }
    stt_ms = (time.perf_counter() - t0) * 1000

    if not final_text:
        return {
            "stt_error": "no final transcript event received",
            "expected_query_from_manifest": expected_query,
            "wav_file": str(wav_path.relative_to(REPO_ROOT)),
        }

    pipeline_result = await run_case(final_text, detected_language)
    return {
        "wav_file": str(wav_path.relative_to(REPO_ROOT)),
        "expected_query_from_manifest": expected_query,
        "stt_transcript": final_text,
        "stt_detected_language": detected_language,
        "stt_ms": stt_ms,
        **pipeline_result,
    }


async def main() -> None:
    print(f"VRAG_INDEX_DIR={os.environ['VRAG_INDEX_DIR']}")
    assert is_retrieval_real(), "STUB FALLBACK -- refusing to run demo-readiness test on stub"

    results = []

    print("\n[1] Hindi VOICE (real Sarvam STT)...")
    try:
        voice_result = await run_voice_case()
        voice_result["id"] = 1
        voice_result["label"] = "hindi_voice"
        results.append(voice_result)
        print(f"    transcript={voice_result.get('stt_transcript')!r}")
        print(f"    detected_language={voice_result.get('stt_detected_language')}")
        print(f"    status={voice_result.get('status')}")
    except Exception as exc:
        print(f"    VOICE TEST FAILED: {exc!r}")
        results.append({"id": 1, "label": "hindi_voice", "error": repr(exc)})

    for case in TEXT_CASES:
        print(f"\n[{case['id']}] {case['label']}...")
        result = await run_case(case["query"], case["language"])
        result["id"] = case["id"]
        result["label"] = case["label"]
        result["query"] = case["query"]
        result["language_supplied"] = case["language"]
        if "note" in case:
            result["note"] = case["note"]
        results.append(result)
        print(
            f"    status={result['status']} answer_language={result['answer_language']} "
            f"top1_score={result['top1_score']}"
        )

    out_path = REPO_ROOT / "eval" / "e2e_demo_readiness_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(results)} cases -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
