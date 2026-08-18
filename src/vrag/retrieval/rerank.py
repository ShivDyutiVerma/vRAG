"""Reranking (TECH_MENU.md S9, docs/BUILD_PLAN.md P3 task A4). "None" is the SHIP default — a
reranker only belongs in the pipeline if the A4 ablation proves it earns its added milliseconds,
not because reranking is generally a good idea. FlashRank is the only cross-encoder TECH_MENU rates
as viable on CPU (sub-20ms for 50 candidates); heavier rerankers (bge-reranker-v2-m3 at ~240ms) are
BENCH-ONLY, not hot-path candidates, for this project's budget.

Deliberately NOT wired into HybridRetriever yet — A4 hasn't run, so there's no data-backed reason to
add this stage to the request path. When A4 picks a winner, the winner's `rerank()` gets called
between fusion and the final RetrievedChunk list in hybrid.py; "none" winning is a valid, expected
outcome (TECH_MENU.md S9: "prove rerank earns its ms").
"""

from __future__ import annotations

from typing import Any, Protocol


class Reranker(Protocol):
    name: str

    def rerank(
        self, query: str, candidates: list[tuple[str, str]], k: int
    ) -> list[tuple[str, float]]:
        """candidates: [(chunk_id, text), ...]. Returns [(chunk_id, score), ...], best first,
        length <= k."""
        ...


class NoOpReranker:
    """The SHIP default (TECH_MENU.md S9) — passes the top-k candidates through unchanged, in
    their incoming (fusion) order, with a uniform placeholder score. Exists so the pipeline can
    always call `reranker.rerank(...)` without a None-check at every call site."""

    name = "none"

    def rerank(
        self, query: str, candidates: list[tuple[str, str]], k: int
    ) -> list[tuple[str, float]]:
        return [(chunk_id, 1.0) for chunk_id, _text in candidates[:k]]


class FlashRankReranker:
    """A4 candidate — TECH_MENU.md S9's only viable hot-path cross-encoder on CPU. Lazy-loads
    the `rerankers` library's FlashRank backend on first use, same pattern as E5Embedder, so
    importing this module never triggers a model download by itself."""

    name = "flashrank"

    def __init__(self, model_name: str = "ms-marco-MultiBERT-L-12") -> None:
        self._model_name = model_name
        self._ranker: Any = None

    def _ranker_instance(self) -> Any:
        if self._ranker is None:
            from rerankers import Reranker as RerankersReranker

            self._ranker = RerankersReranker(self._model_name, model_type="flashrank")
        return self._ranker

    def rerank(
        self, query: str, candidates: list[tuple[str, str]], k: int
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []
        chunk_ids = [chunk_id for chunk_id, _text in candidates]
        texts = [text for _chunk_id, text in candidates]
        result = self._ranker_instance().rank(query=query, docs=texts, doc_ids=chunk_ids)
        return [(r.doc_id, float(r.score)) for r in result.top_k(k)]


class CrossEncoderReranker:
    """A4's third candidate — TECH_MENU.md S9's `cross-encoder/ms-marco-MiniLM-L6-v2` (~1,800
    docs/sec, nDCG@10 74.30 on TREC DL19). Loaded via `rerankers`' `model_type="cross-encoder"`
    (routes to its TransformerRanker backend), same lazy-load pattern as FlashRankReranker."""

    name = "cross-encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._ranker: Any = None

    def _ranker_instance(self) -> Any:
        if self._ranker is None:
            from rerankers import Reranker as RerankersReranker

            self._ranker = RerankersReranker(self._model_name, model_type="cross-encoder")
        return self._ranker

    def rerank(
        self, query: str, candidates: list[tuple[str, str]], k: int
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []
        chunk_ids = [chunk_id for chunk_id, _text in candidates]
        texts = [text for _chunk_id, text in candidates]
        result = self._ranker_instance().rank(query=query, docs=texts, doc_ids=chunk_ids)
        return [(r.doc_id, float(r.score)) for r in result.top_k(k)]
