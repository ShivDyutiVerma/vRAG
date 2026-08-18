"""Contract test for the R/P seam — asserts the shape Workstream P builds against never breaks
silently. Not a test of retrieval quality (that's eval_chunking.py's job)."""

import json

import pytest

import vrag.retrieval.interface as interface
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


def test_get_real_retriever_falls_back_to_stub_when_index_files_exist_but_load_fails(
    monkeypatch, tmp_path
) -> None:
    """Real scenario, not simulated: this dev environment has no `retrieval` extras installed
    (no faiss/sentence-transformers), so pointing `_INDEX_DIR` at a directory that merely
    *contains* a `chunk_lookup.json` (the only file the existence check looks for) reproduces the
    exact "index files present, real dependencies missing" case the P-018 defensive fix targets
    -- e.g. a container where the index artifact landed before the leaner embedder deps did."""
    (tmp_path / "chunk_lookup.json").write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setattr(interface, "_INDEX_DIR", tmp_path)
    monkeypatch.setattr(interface, "_retriever", None)
    monkeypatch.setattr(interface, "_retriever_load_attempted", False)

    result = interface._get_real_retriever()

    assert result is None
