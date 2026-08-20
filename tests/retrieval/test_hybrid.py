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
    # retrieval_mode="hybrid" explicitly -- "dense" (the shipped default, docs/DECISIONS_R.md
    # R-010) never calls sparse at all, so it wouldn't exercise this concurrency guarantee.
    retriever = HybridRetriever(
        dense=_SlowDense(),
        sparse=_SlowSparse(),
        embedder=_FakeEmbedder(),
        chunk_lookup=_lookup(),
        retrieval_mode="hybrid",
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


@pytest.mark.asyncio
async def test_default_mode_is_dense_and_never_calls_sparse() -> None:
    class _FailingSparse:
        def search(self, query: str, k: int) -> list[tuple[str, float]]:
            raise AssertionError("sparse.search should never be called in dense mode")

    retriever = HybridRetriever(
        dense=_SlowDense(),
        sparse=_FailingSparse(),
        embedder=_FakeEmbedder(),
        chunk_lookup=_lookup(),
    )
    results = await retriever.retrieve("query", k=5)
    assert len(results) == 1
    assert results[0].chunk_id == "a"


@pytest.mark.asyncio
async def test_sparse_mode_never_calls_dense() -> None:
    class _FailingDense:
        def search(self, vector: list[float], k: int) -> list[tuple[str, float]]:
            raise AssertionError("dense.search should never be called in sparse mode")

    retriever = HybridRetriever(
        dense=_FailingDense(),
        sparse=_SlowSparse(),
        embedder=_FakeEmbedder(),
        chunk_lookup=_lookup(),
        retrieval_mode="sparse",
    )
    results = await retriever.retrieve("query", k=5)
    assert len(results) == 1
    assert results[0].chunk_id == "a"


def test_unknown_retrieval_mode_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="unknown retrieval_mode"):
        HybridRetriever(
            dense=_SlowDense(),
            sparse=_SlowSparse(),
            embedder=_FakeEmbedder(),
            chunk_lookup=_lookup(),
            retrieval_mode="bogus",
        )


def test_dense_mode_accepts_sparse_none() -> None:
    """ADR-006: dense mode (the shipped default) never touches sparse, so callers can skip
    loading the BM25 index entirely (~105MB saved) and pass None instead of a real SparseIndex."""
    retriever = HybridRetriever(
        dense=_SlowDense(),
        sparse=None,
        embedder=_FakeEmbedder(),
        chunk_lookup=_lookup(),
        retrieval_mode="dense",
    )
    assert retriever is not None  # construction alone must not raise


@pytest.mark.parametrize("mode", ["sparse", "hybrid"])
def test_sparse_or_hybrid_mode_rejects_sparse_none(mode: str) -> None:
    """Fail fast at construction rather than silently returning [] the first time a real query
    hits _search_sparse() and crashes on a None attribute access."""
    with pytest.raises(ValueError, match="requires a real SparseIndex"):
        HybridRetriever(
            dense=_SlowDense(),
            sparse=None,
            embedder=_FakeEmbedder(),
            chunk_lookup=_lookup(),
            retrieval_mode=mode,
        )


@pytest.mark.asyncio
async def test_score_is_clamped_into_zero_one_range() -> None:
    class _OutOfRangeDense:
        def search(self, vector: list[float], k: int) -> list[tuple[str, float]]:
            return [("a", 1.5)]

    retriever = HybridRetriever(
        dense=_OutOfRangeDense(),
        sparse=_SlowSparse(),
        embedder=_FakeEmbedder(),
        chunk_lookup=_lookup(),
    )
    results = await retriever.retrieve("query", k=5)
    assert results[0].score == 1.0


# --- Phase 3 (docs/DECISIONS.md ADR-012): language filtering, wiring what ADR-009's `language`
# param left inert. Phase 2 measured this exact strategy (+8.7-9.1pp Recall@10 over no-filter).


def _multilingual_lookup() -> dict[str, Chunk]:
    return {
        "hin1": Chunk(
            chunk_id="hin1", doc_id="p1", text="hindi text", metadata={"language": "hin_Deva"}
        ),
        "hin2": Chunk(
            chunk_id="hin2", doc_id="p2", text="hindi text 2", metadata={"language": "hin_Deva"}
        ),
        "ben1": Chunk(
            chunk_id="ben1", doc_id="p3", text="bengali text", metadata={"language": "ben_Beng"}
        ),
        "tam1": Chunk(
            chunk_id="tam1", doc_id="p4", text="tamil text", metadata={"language": "tam_Taml"}
        ),
    }


class _MixedLanguageDense:
    """Returns a realistic mixed-language ranking -- highest score first, only one of which is
    Hindi -- so a filter test can prove it's actually re-prioritising, not just passing through."""

    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        return [("ben1", 0.95), ("tam1", 0.90), ("hin1", 0.85), ("hin2", 0.80)][:k]

    last_k: int | None = None


@pytest.mark.asyncio
async def test_language_filter_restricts_to_matching_language_and_reprioritises() -> None:
    retriever = HybridRetriever(
        dense=_MixedLanguageDense(),
        sparse=_SlowSparse(),
        embedder=_FakeEmbedder(),
        chunk_lookup=_multilingual_lookup(),
    )
    results = await retriever.retrieve("query", k=2, language="hi-IN")
    assert [r.chunk_id for r in results] == ["hin1", "hin2"]
    assert all(r.language == "hin_Deva" for r in results)


@pytest.mark.asyncio
async def test_no_language_filters_returns_unfiltered_ranking_unchanged() -> None:
    """language=None must behave exactly as before Phase 1/2/3 -- no filtering at all."""
    retriever = HybridRetriever(
        dense=_MixedLanguageDense(),
        sparse=_SlowSparse(),
        embedder=_FakeEmbedder(),
        chunk_lookup=_multilingual_lookup(),
    )
    results = await retriever.retrieve("query", k=2, language=None)
    assert [r.chunk_id for r in results] == ["ben1", "tam1"]


@pytest.mark.asyncio
async def test_unmapped_language_code_returns_unfiltered_ranking() -> None:
    """A Sarvam code with no SARVAM_TO_TARGET_LANG entry (e.g. an unsupported/unmapped code)
    must not crash or silently filter everything out -- falls back to unfiltered search."""
    retriever = HybridRetriever(
        dense=_MixedLanguageDense(),
        sparse=_SlowSparse(),
        embedder=_FakeEmbedder(),
        chunk_lookup=_multilingual_lookup(),
    )
    results = await retriever.retrieve("query", k=2, language="fr-FR")
    assert [r.chunk_id for r in results] == ["ben1", "tam1"]


@pytest.mark.asyncio
async def test_language_filter_falls_back_when_no_same_language_candidate_in_window() -> None:
    """A genuine 'no same-language candidate found' case must not manufacture a zero-result
    failure -- falls back to the unfiltered ranking (docs/DECISIONS.md ADR-012)."""
    lookup_no_urdu = _multilingual_lookup()  # no urd_Arab entries at all

    class _NoUrduDense:
        def search(self, vector: list[float], k: int) -> list[tuple[str, float]]:
            return [("ben1", 0.95), ("tam1", 0.90)][:k]

    retriever = HybridRetriever(
        dense=_NoUrduDense(),
        sparse=_SlowSparse(),
        embedder=_FakeEmbedder(),
        chunk_lookup=lookup_no_urdu,
    )
    results = await retriever.retrieve("query", k=2, language="ur-IN")
    # fell back to the unfiltered ranking rather than returning []
    assert [r.chunk_id for r in results] == ["ben1", "tam1"]


@pytest.mark.asyncio
async def test_language_filter_widens_search_k_for_the_candidate_pool() -> None:
    """When filtering is active, the underlying search must ask for a wide candidate pool
    (_LANGUAGE_FILTER_WIDE_K), not just the caller's small requested k -- otherwise a
    same-language chunk ranked #6 would never even be seen by the filter."""
    requested_ks: list[int] = []

    class _RecordingDense:
        def search(self, vector: list[float], k: int) -> list[tuple[str, float]]:
            requested_ks.append(k)
            return [("hin1", 0.9)]

    retriever = HybridRetriever(
        dense=_RecordingDense(),
        sparse=_SlowSparse(),
        embedder=_FakeEmbedder(),
        chunk_lookup=_multilingual_lookup(),
    )
    await retriever.retrieve("query", k=5, language="hi-IN")
    assert requested_ks == [100]  # widened, not the requested k=5

    requested_ks.clear()
    await retriever.retrieve("query", k=5, language=None)
    assert requested_ks == [5]  # unfiltered path is untouched
