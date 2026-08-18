"""Tests for the search_corpus tool (AGENT_BUILD_SPEC.md §7.2 item 5, docs/DECISIONS_P.md P-019):
the real OpenAI-style tool-calling follow-up generate() escalates to when a structured answer
signals needs_more_context. Real tool-calling behavior (Sarvam correctly emits tool_calls, and
that combining tools with response_format:json_schema breaks) was verified live against the API
before writing this — see P-019; these tests exercise the resulting orchestration logic in
isolation, no live network.
"""

from __future__ import annotations

import json

import httpx
import pytest

from vrag.generation import sarvam_llm
from vrag.generation.sarvam_llm import (
    _call_tool_decision,
    generate,
    search_corpus,
)
from vrag.generation.schemas import GeneratedAnswer
from vrag.retrieval.interface import RetrievedChunk


def _chunk(chunk_id: str = "c1", text: str = "असली संदर्भ पाठ") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, passage_id="p1", text=text, score=0.9, language="hi")


def _answer(needs_more_context: bool, cited_csv: str = "c1") -> GeneratedAnswer:
    return GeneratedAnswer(
        reasoning="संक्षिप्त कारण",
        needs_more_context=needs_more_context,
        answer="असली उत्तर",
        cited_chunk_ids_csv=cited_csv,
    )


# --- search_corpus itself -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_corpus_delegates_to_retrieve_real_stub_path():
    # Real integration, not mocked: this dev environment has no persisted index, so retrieve()
    # takes the Day-0 stub path -- search_corpus should transparently return that.
    results = await search_corpus("कोई भी प्रश्न", k=2)
    assert len(results) <= 2
    for r in results:
        assert isinstance(r, RetrievedChunk)


@pytest.mark.asyncio
async def test_search_corpus_clamps_a_runaway_k():
    results = await search_corpus("कोई भी प्रश्न", k=999)
    assert len(results) <= 10  # clamped, not passed straight through to retrieve()


# --- _call_tool_decision --------------------------------------------------------------------


def _tool_call_response(name: str, arguments: str) -> httpx.AsyncClient:
    function = {"name": name, "arguments": arguments}
    tool_call = {"id": "call_1", "type": "function", "function": function}
    message = {"tool_calls": [tool_call]}
    body = json.dumps({"choices": [{"message": message}]}).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _no_tool_call_response() -> httpx.AsyncClient:
    message = {"content": "just an answer", "tool_calls": None}
    body = json.dumps({"choices": [{"message": message}]}).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_call_tool_decision_parses_a_well_formed_tool_call():
    async with _tool_call_response("search_corpus", '{"query": "हिमालय पर्वत", "k": 3}') as client:
        result = await _call_tool_decision(client, messages=[])
    assert result == {"query": "हिमालय पर्वत", "k": 3}


@pytest.mark.asyncio
async def test_call_tool_decision_defaults_k_when_missing():
    async with _tool_call_response("search_corpus", '{"query": "हिमालय पर्वत"}') as client:
        result = await _call_tool_decision(client, messages=[])
    assert result == {"query": "हिमालय पर्वत", "k": 5}


@pytest.mark.asyncio
async def test_call_tool_decision_returns_none_when_no_tool_call_made():
    async with _no_tool_call_response() as client:
        result = await _call_tool_decision(client, messages=[])
    assert result is None


@pytest.mark.asyncio
async def test_call_tool_decision_returns_none_on_malformed_arguments():
    async with _tool_call_response("search_corpus", "not valid json") as client:
        result = await _call_tool_decision(client, messages=[])
    assert result is None


# --- generate()'s orchestration of the whole follow-up --------------------------------------


@pytest.mark.asyncio
async def test_generate_skips_tool_flow_when_context_already_sufficient(monkeypatch):
    monkeypatch.setattr(sarvam_llm.settings, "sarvam_api_key", "fake-key-for-test")
    monkeypatch.setattr(sarvam_llm, "_generate_structured", _make_structured(_answer(False)))

    tool_decision_called = False

    async def _should_not_be_called(*args, **kwargs):
        nonlocal tool_decision_called
        tool_decision_called = True
        raise AssertionError("tool decision must not be called when needs_more_context is False")

    monkeypatch.setattr(sarvam_llm, "_call_tool_decision", _should_not_be_called)

    result = await generate("query", [_chunk()])

    assert tool_decision_called is False
    assert result is not None
    assert result.answer.needs_more_context is False
    assert result.chunks == [_chunk()]


@pytest.mark.asyncio
async def test_generate_escalates_and_expands_chunks_on_needs_more_context(monkeypatch):
    monkeypatch.setattr(sarvam_llm.settings, "sarvam_api_key", "fake-key-for-test")

    call_count = {"n": 0}

    async def _fake_generate_structured(client, messages):
        call_count["n"] += 1
        # First call signals it needs more context; the second (post-tool) call gives the
        # real final answer, now citing a chunk that only exists in the expanded set.
        if call_count["n"] == 1:
            return _answer(True, cited_csv="c1")
        return _answer(False, cited_csv="c2")

    monkeypatch.setattr(sarvam_llm, "_generate_structured", _fake_generate_structured)

    async def _fake_tool_decision(client, messages):
        return {"query": "हिमालय पर्वत श्रृंखला", "k": 3}

    monkeypatch.setattr(sarvam_llm, "_call_tool_decision", _fake_tool_decision)

    async def _fake_search_corpus(query, k):
        assert query == "हिमालय पर्वत श्रृंखला"
        assert k == 3
        return [_chunk("c2", text="नया प्राप्त संदर्भ")]

    monkeypatch.setattr(sarvam_llm, "search_corpus", _fake_search_corpus)

    result = await generate("query", [_chunk("c1")])

    assert call_count["n"] == 2
    assert result is not None
    assert result.answer.cited_chunk_ids == ["c2"]
    # Expanded set contains both the original and the tool-fetched chunk -- this is what
    # GenerateStage validates G4 against, so a tool-fetched citation isn't wrongly "invented".
    assert {c.chunk_id for c in result.chunks} == {"c1", "c2"}


@pytest.mark.asyncio
async def test_generate_falls_back_to_first_answer_when_tool_decision_yields_nothing(monkeypatch):
    monkeypatch.setattr(sarvam_llm.settings, "sarvam_api_key", "fake-key-for-test")
    monkeypatch.setattr(sarvam_llm, "_generate_structured", _make_structured(_answer(True)))

    async def _no_tool_call(client, messages):
        return None

    monkeypatch.setattr(sarvam_llm, "_call_tool_decision", _no_tool_call)

    result = await generate("query", [_chunk()])

    assert result is not None
    assert result.answer.needs_more_context is True  # the original (unescalated) answer
    assert result.chunks == [_chunk()]


@pytest.mark.asyncio
async def test_generate_falls_back_to_first_answer_when_search_corpus_finds_nothing(monkeypatch):
    monkeypatch.setattr(sarvam_llm.settings, "sarvam_api_key", "fake-key-for-test")
    monkeypatch.setattr(sarvam_llm, "_generate_structured", _make_structured(_answer(True)))
    monkeypatch.setattr(
        sarvam_llm, "_call_tool_decision", _make_tool_decision({"query": "x", "k": 5})
    )

    async def _empty_search(query, k):
        return []

    monkeypatch.setattr(sarvam_llm, "search_corpus", _empty_search)

    result = await generate("query", [_chunk()])

    assert result is not None
    assert result.chunks == [_chunk()]


@pytest.mark.asyncio
async def test_generate_falls_back_to_first_answer_when_followup_structured_call_fails(monkeypatch):
    monkeypatch.setattr(sarvam_llm.settings, "sarvam_api_key", "fake-key-for-test")

    call_count = {"n": 0}

    async def _fake_generate_structured(client, messages):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _answer(True)
        return None  # the follow-up structured call fails outright

    monkeypatch.setattr(sarvam_llm, "_generate_structured", _fake_generate_structured)
    monkeypatch.setattr(
        sarvam_llm, "_call_tool_decision", _make_tool_decision({"query": "x", "k": 5})
    )

    async def _fake_search(query, k):
        return [_chunk("c2")]

    monkeypatch.setattr(sarvam_llm, "search_corpus", _fake_search)

    result = await generate("query", [_chunk("c1")])

    assert result is not None
    assert result.answer.needs_more_context is True  # kept the first answer
    assert result.chunks == [_chunk("c1")]  # not the expanded set -- the follow-up never landed


def _make_structured(answer: GeneratedAnswer):
    async def _fake(client, messages):
        return answer

    return _fake


def _make_tool_decision(decision: dict):
    async def _fake(client, messages):
        return decision

    return _fake
