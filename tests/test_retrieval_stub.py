"""Smoke test for the Day-1 retrieve() stub. Real implementation lands per docs/TEAM_SPLIT.md §5;
this only guards the contract shape so Workstream P's harness doesn't silently break against it."""

import pytest

from vrag.retrieval.interface import RetrievedChunk, retrieve


@pytest.mark.asyncio
async def test_retrieve_returns_requested_count():
    chunks = await retrieve("भारत में सबसे ऊँचा पर्वत कौन सा है?", k=2)
    assert len(chunks) == 2
    assert all(isinstance(c, RetrievedChunk) for c in chunks)


@pytest.mark.asyncio
async def test_retrieve_chunks_have_required_fields():
    chunks = await retrieve("test query", k=5)
    for chunk in chunks:
        assert chunk.chunk_id
        assert chunk.passage_id
        assert chunk.text
        assert 0.0 <= chunk.score <= 1.0
        assert chunk.language


@pytest.mark.asyncio
async def test_retrieve_empty_query_returns_empty_list():
    assert await retrieve("", k=5) == []
    assert await retrieve("   ", k=5) == []


@pytest.mark.asyncio
async def test_retrieve_never_raises_on_odd_input():
    # k=0 and a huge k should degrade gracefully, never throw
    assert await retrieve("query", k=0) == []
    assert len(await retrieve("query", k=1000)) <= 1000
