"""Phase 3 (docs/DECISIONS.md ADR-012): Track B's system prompt is no longer hardcoded to Hindi
-- it names the real generation_language for each request. Tests both the pure prompt-building
function directly, and (via httpx.MockTransport, same pattern as test_sarvam_llm.py) that a real
generate() call actually sends the right language-specific prompt over the wire for Hindi, English,
and at least 3 additional indexed languages.
"""

from __future__ import annotations

import json

import httpx
import pytest

from vrag.generation.sarvam_llm import _build_system_prompt, _messages_for, generate
from vrag.retrieval.interface import RetrievedChunk


def _chunk(lang: str = "hin_Deva") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1", passage_id="p1", text="some context", score=0.9, language=lang
    )


# --- _build_system_prompt: pure function, no network needed ---


@pytest.mark.parametrize(
    "code,expected_name,expected_script",
    [
        ("hi-IN", "Hindi", "Devanagari"),
        ("en-IN", "English", "Latin"),
        ("bn-IN", "Bengali", "Bengali"),
        ("ta-IN", "Tamil", "Tamil"),
        ("ur-IN", "Urdu", "Perso-Arabic"),
    ],
)
def test_system_prompt_names_the_real_generation_language(code, expected_name, expected_script):
    prompt = _build_system_prompt(code)
    assert expected_name in prompt
    assert expected_script in prompt


def test_system_prompt_no_longer_hardcodes_hindi_as_the_product_language():
    """The old prompt said 'a Hindi voice product' and 'never answer in English' unconditionally
    -- both must be gone now that generation_language decides this per request."""
    hindi_prompt = _build_system_prompt("hi-IN")
    english_prompt = _build_system_prompt("en-IN")
    assert "Hindi voice product" not in hindi_prompt
    assert "Hindi voice product" not in english_prompt
    assert "never answer in English" not in hindi_prompt


def test_none_generation_language_falls_back_to_hindi():
    """No real signal (e.g. a direct /ask call with no language hint) -- Hindi remains the
    fallback, but as a documented default, not a hardcoded assumption."""
    prompt = _build_system_prompt(None)
    assert "Hindi" in prompt
    assert "Devanagari" in prompt


def test_unmapped_generation_language_falls_back_to_hindi():
    prompt = _build_system_prompt("xx-XX")
    assert "Hindi" in prompt


def test_messages_for_embeds_the_language_specific_prompt():
    messages = _messages_for("query text", [_chunk()], "ta-IN")
    assert messages[0]["role"] == "system"
    assert "Tamil" in messages[0]["content"]


# --- generate(): real HTTP payload inspection via MockTransport, matching the existing pattern ---


def _capture_transport(captured: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        payload = {
            "choices": [
                {
                    "delta": {
                        "content": json.dumps(
                            {
                                "reasoning": "ok",
                                "answer": "ans",
                                "cited_chunk_ids_csv": "c1",
                                "needs_more_context": False,
                            }
                        )
                    }
                }
            ]
        }
        body = f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n".encode()
        return httpx.Response(200, content=body)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,expected_name",
    [
        ("hi-IN", "Hindi"),
        ("en-IN", "English"),
        ("bn-IN", "Bengali"),
        ("mr-IN", "Marathi"),
        ("kn-IN", "Kannada"),
    ],
)
async def test_real_generate_call_sends_the_correct_language_in_the_system_prompt(
    monkeypatch, code, expected_name
):
    from vrag.generation import sarvam_llm

    monkeypatch.setattr(sarvam_llm.settings, "sarvam_api_key", "fake-key-for-tests")
    captured: list[dict] = []
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real_async_client(transport=_capture_transport(captured))
    )

    result = await generate("some question", [_chunk()], generation_language=code)

    assert result is not None
    assert len(captured) == 1
    system_content = captured[0]["messages"][0]["content"]
    assert expected_name in system_content
