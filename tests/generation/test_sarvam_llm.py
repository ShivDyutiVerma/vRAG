"""Tests for src/vrag/generation/sarvam_llm.py's streaming call and stall detection (P-017).
Uses httpx.MockTransport to fabricate real SSE responses — no live network, deterministic, fast.

The evidence behind these tests (real streamed runs against the live Sarvam API showing the
whitespace-padding failure pattern, and the threshold chosen from them) lives in
docs/DECISIONS_P.md P-017, not here — these tests exercise the resulting logic in isolation.
"""

from __future__ import annotations

import json

import httpx
import pytest

from vrag.generation import sarvam_llm
from vrag.generation.sarvam_llm import (
    STALL_THRESHOLD_CHUNKS,
    _call_once_streaming,
    _GenerationStalled,
    generate,
)
from vrag.retrieval.interface import RetrievedChunk


def _sse_chunk(content: str) -> str:
    payload = {"choices": [{"delta": {"content": content}}]}
    return f"data: {json.dumps(payload)}"


def _sse_client(lines: list[str]) -> httpx.AsyncClient:
    body = "\n\n".join(lines).encode() + b"\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_reconstructs_full_content_from_chunks():
    lines = [_sse_chunk(part) for part in ['{"a', 'nswer": "hi', 'i"}']] + ["data: [DONE]"]
    async with _sse_client(lines) as client:
        result = await _call_once_streaming(client, messages=[])
    assert result == '{"answer": "hii"}'


@pytest.mark.asyncio
async def test_tolerates_a_few_whitespace_chunks_then_recovers():
    # A couple of whitespace-only deltas between fields is normal JSON formatting (e.g. the
    # newline+indent Sarvam emits between fields) -- well below STALL_THRESHOLD_CHUNKS, must not
    # raise.
    lines = (
        [_sse_chunk('{"a": "b",')]
        + [_sse_chunk("\n"), _sse_chunk("  ")]
        + [_sse_chunk('"c": "d"}')]
        + ["data: [DONE]"]
    )
    async with _sse_client(lines) as client:
        result = await _call_once_streaming(client, messages=[])
    assert result == '{"a": "b",\n  "c": "d"}'


@pytest.mark.asyncio
async def test_raises_stalled_after_threshold_consecutive_whitespace_chunks():
    real_content = '{"reasoning": "x", "answer": "y",'
    lines = [_sse_chunk(real_content)] + [_sse_chunk(" ")] * (STALL_THRESHOLD_CHUNKS + 5)
    async with _sse_client(lines) as client:
        with pytest.raises(_GenerationStalled) as exc_info:
            await _call_once_streaming(client, messages=[])
    # partial_content captures what was accumulated before the stall was detected, useful for a
    # repair message -- should contain the real content, not be empty.
    assert real_content in exc_info.value.partial_content


@pytest.mark.asyncio
async def test_empty_deltas_count_toward_the_stall_threshold_same_as_whitespace():
    real_content = '{"a": "b"'
    lines = [_sse_chunk(real_content)] + [_sse_chunk("")] * (STALL_THRESHOLD_CHUNKS + 1)
    async with _sse_client(lines) as client:
        with pytest.raises(_GenerationStalled):
            await _call_once_streaming(client, messages=[])


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1", passage_id="p1", text="असली संदर्भ पाठ", score=0.9, language="hi"
    )


@pytest.mark.asyncio
async def test_generate_retries_once_after_a_stall_and_succeeds(monkeypatch):
    monkeypatch.setattr(sarvam_llm.settings, "sarvam_api_key", "fake-key-for-test")

    calls = {"n": 0}

    async def _fake_call_once_streaming(client, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _GenerationStalled(partial_content='{"reasoning": "x",')
        return json.dumps(
            {"reasoning": "संक्षिप्त कारण", "answer": "असली उत्तर", "cited_chunk_ids_csv": "c1"}
        )

    monkeypatch.setattr(sarvam_llm, "_call_once_streaming", _fake_call_once_streaming)

    result = await generate("some query", [_chunk()])

    assert calls["n"] == 2
    assert result is not None
    assert result.answer == "असली उत्तर"
    assert result.cited_chunk_ids == ["c1"]


@pytest.mark.asyncio
async def test_generate_returns_none_after_a_stall_on_both_attempts(monkeypatch):
    monkeypatch.setattr(sarvam_llm.settings, "sarvam_api_key", "fake-key-for-test")

    async def _always_stalls(client, messages):
        raise _GenerationStalled(partial_content='{"reasoning": "x",')

    monkeypatch.setattr(sarvam_llm, "_call_once_streaming", _always_stalls)

    result = await generate("some query", [_chunk()])

    assert result is None
