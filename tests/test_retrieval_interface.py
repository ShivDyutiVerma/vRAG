"""Contract test for the R/P seam — asserts the shape Workstream P builds against never breaks
silently. Not a test of retrieval quality (that's eval_chunking.py's job)."""

import pytest

from vrag.retrieval.interface import RetrievedChunk, retrieve


@pytest.mark.asyncio
async def test_retrieve_returns_requested_shape() -> None:
    results = await retrieve("भारत में सबसे ऊँचा पर्वत कौन सा है?", k=5)
    assert isinstance(results, list)
    for chunk in results:
        assert isinstance(chunk, RetrievedChunk)
        assert 0.0 <= chunk.score <= 1.0
        assert chunk.language


@pytest.mark.asyncio
async def test_retrieve_respects_k() -> None:
    results = await retrieve("test query", k=1)
    assert len(results) <= 1


@pytest.mark.asyncio
async def test_retrieve_never_raises_on_empty_query() -> None:
    results = await retrieve("", k=5)
    assert isinstance(results, list)
