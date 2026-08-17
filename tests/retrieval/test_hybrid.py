"""Tests for HybridRetriever, using fake dense/sparse/embedder doubles — no real FAISS/bm25s/model
needed. The concurrency test is the important one: it directly proves dense and sparse run in
parallel, not sequentially, per CLAUDE.md's hot-path invariant and BUILD_PLAN.md P3's exit
criterion ("assert wall-clock < sum of parts").
"""

from __future__ import annotations

import time

import pytest

from vrag.chunking.base import Chunk
from vrag.retrieval.hybrid import HybridRetriever

SLEEP_SECONDS = 0.15


class _SlowDense:
    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        time.sleep(SLEEP_SECONDS)
        return [("a", 0.9)]


class _SlowSparse:
    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        time.sleep(SLEEP_SECONDS)
        return [("a", 5.0)]


class _FakeEmbedder:
    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class _RaisingDense:
    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        raise RuntimeError("simulated FAISS failure")


def _lookup() -> dict[str, Chunk]:
    return {
        "a": Chunk(
            chunk_id="a", doc_id="passage-1", text="कुछ पाठ", metadata={"language": "hi"}
        )
    }


@pytest.mark.asyncio
async def test_dense_and_sparse_run_concurrently_not_sequentially() -> None:
    retriever = HybridRetriever(
        dense=_SlowDense(), sparse=_SlowSparse(), embedder=_FakeEmbedder(), chunk_lookup=_lookup()
    )
    t0 = time.perf_counter()
    await retriever.retrieve("भारत की राजधानी क्या है")
    elapsed = time.perf_counter() - t0
    # sequential would take ~2*SLEEP_SECONDS; concurrent should take ~1*SLEEP_SECONDS
    assert elapsed < SLEEP_SECONDS * 2


@pytest.mark.asyncio
async def test_result_shape_maps_chunk_to_retrieved_chunk() -> None:
    retriever = HybridRetriever(
        dense=_SlowDense(), sparse=_SlowSparse(), embedder=_FakeEmbedder(), chunk_lookup=_lookup()
    )
    results = await retriever.retrieve("query", k=5)
    assert len(results) == 1
    assert results[0].chunk_id == "a"
    assert results[0].passage_id == "passage-1"
    assert results[0].language == "hi"


@pytest.mark.asyncio
async def test_empty_query_returns_empty_list_without_calling_indexes() -> None:
    retriever = HybridRetriever(
        dense=_SlowDense(), sparse=_SlowSparse(), embedder=_FakeEmbedder(), chunk_lookup=_lookup()
    )
    assert await retriever.retrieve("   ", k=5) == []


@pytest.mark.asyncio
async def test_internal_failure_returns_empty_list_never_raises() -> None:
    retriever = HybridRetriever(
        dense=_RaisingDense(),
        sparse=_SlowSparse(),
        embedder=_FakeEmbedder(),
        chunk_lookup=_lookup(),
    )
    assert await retriever.retrieve("query", k=5) == []
