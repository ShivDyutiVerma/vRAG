"""Tests for the reranker protocol's SHIP default. FlashRankReranker isn't tested here — it needs
a real model download, same reasoning as E5Embedder's embed_* methods being untested at the unit
level (see tests/test_embedder.py)."""

from vrag.retrieval.rerank import NoOpReranker


def test_noop_reranker_preserves_incoming_order() -> None:
    reranker = NoOpReranker()
    candidates = [("a", "text a"), ("b", "text b"), ("c", "text c")]
    result = reranker.rerank("query", candidates, k=3)
    assert [chunk_id for chunk_id, _score in result] == ["a", "b", "c"]


def test_noop_reranker_respects_k() -> None:
    reranker = NoOpReranker()
    candidates = [("a", "x"), ("b", "y"), ("c", "z")]
    result = reranker.rerank("query", candidates, k=2)
    assert len(result) == 2


def test_noop_reranker_empty_candidates() -> None:
    assert NoOpReranker().rerank("query", [], k=5) == []


def test_noop_reranker_name_is_none() -> None:
    assert NoOpReranker().name == "none"
