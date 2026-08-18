"""Proves GenerateStage correctly validates G4 against the *expanded* chunk set (original +
whatever the search_corpus follow-up fetched, docs/DECISIONS_P.md P-019) rather than only
RetrieveStage's original list -- the real bug this guards against: a genuinely tool-fetched
citation getting wrongly flagged as "invented" by G4 because the stage only knew about the
original, smaller chunk list.
"""

from __future__ import annotations

import pytest

from vrag.generation import circuit_breaker as cb
from vrag.generation.sarvam_llm import GenerationResult
from vrag.generation.schemas import GeneratedAnswer
from vrag.harness.budget import Budget
from vrag.harness.pipeline import PipelineContext
from vrag.harness.stages import GenerateStage
from vrag.retrieval.interface import RetrievedChunk


def _chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, passage_id=f"p-{chunk_id}", text=text, score=0.9, language="hi"
    )


@pytest.mark.asyncio
async def test_citation_from_tool_fetched_chunk_is_accepted_not_flagged_invented(monkeypatch):
    monkeypatch.setattr(cb, "TRACK_B_BREAKER", cb.CircuitBreaker())

    original_chunk = _chunk("c1", "मूल संदर्भ जिसमें उत्तर नहीं है")
    tool_fetched_chunk = _chunk("c2", "असली उत्तर यहाँ है")

    async def _fake_generate(*args, **kwargs):
        answer = GeneratedAnswer(
            reasoning="संक्षिप्त कारण",
            needs_more_context=False,  # already resolved by the time GenerateStage sees it
            answer="असली उत्तर",
            cited_chunk_ids_csv="c2",  # cites the tool-fetched chunk, NOT the original
        )
        # The full set generate() actually used -- original + tool-fetched, exactly what
        # sarvam_llm.generate()'s real orchestration returns after a successful follow-up.
        return GenerationResult(answer=answer, chunks=[original_chunk, tool_fetched_chunk])

    monkeypatch.setattr("vrag.harness.stages.generate_track_b", _fake_generate)

    ctx = PipelineContext(query="test query", budget=Budget(total_ms=15_000))
    # RetrieveStage's original, smaller set -- deliberately does NOT include c2, proving G4 uses
    # result.chunks (the expanded set), not ctx.data["chunks"] as it stood before this stage ran.
    ctx.data["chunks"] = [original_chunk]

    result = await GenerateStage().run(ctx)

    assert result.skipped is False
    assert ctx.data["track"] == "generative"
    assert ctx.data["answer_text"] == "असली उत्तर"
    assert [c.chunk_id for c in ctx.data["citations"]] == ["c2"]
    # ctx.data["chunks"] itself is updated to the expanded set too (telemetry/trace visibility).
    assert {c.chunk_id for c in ctx.data["chunks"]} == {"c1", "c2"}
