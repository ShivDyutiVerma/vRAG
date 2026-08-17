"""Hybrid dense ∥ sparse retrieval (AGENT_BUILD_SPEC.md §5.2, docs/BUILD_PLAN.md P3). Implements
the real body behind `retrieve()` (src/vrag/retrieval/interface.py) once an index is built — that
file's stub gets swapped for `HybridRetriever.retrieve` at the Day 2 sync per docs/TEAM_SPLIT.md.

Dense and sparse MUST run concurrently, not sequentially (CLAUDE.md hot-path invariant). Both FAISS
and bm25s release the GIL during their C-level search calls, so running each in a thread-pool
executor via `asyncio.gather` gives genuine wall-clock parallelism, not just cooperative scheduling
around I/O. Sparse search doesn't depend on the query embedding at all, so embedding + dense search
run as one sequential unit on one thread while sparse search runs concurrently on another — not
"embed, then run two searches," which would serialize the one step that doesn't need to be.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from vrag.chunking.base import Chunk
from vrag.index.dense import DenseIndex
from vrag.index.embedder import E5Embedder
from vrag.index.fusion import DEFAULT_K, reciprocal_rank_fusion
from vrag.index.sparse import SparseIndex
from vrag.retrieval.interface import RetrievedChunk


class HybridRetriever:
    def __init__(
        self,
        dense: DenseIndex,
        sparse: SparseIndex,
        embedder: E5Embedder,
        chunk_lookup: dict[str, Chunk],
        fusion_k: int = DEFAULT_K,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._embedder = embedder
        self._chunk_lookup = chunk_lookup
        self._fusion_k = fusion_k
        self._executor = executor or ThreadPoolExecutor(max_workers=2)

    def _embed_and_search_dense(self, query: str, k: int) -> list[tuple[str, float]]:
        vector = self._embedder.embed_queries([query])[0]
        return self._dense.search(vector, k)

    def _search_sparse(self, query: str, k: int) -> list[tuple[str, float]]:
        return self._sparse.search(query, k)

    async def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Never raises — matches the retrieve() contract in interface.py. Internal failures
        collapse to an empty result so the harness's grounding gate can abstain."""
        if not query.strip():
            return []

        loop = asyncio.get_event_loop()
        try:
            dense_hits, sparse_hits = await asyncio.gather(
                loop.run_in_executor(self._executor, self._embed_and_search_dense, query, k),
                loop.run_in_executor(self._executor, self._search_sparse, query, k),
            )
        except Exception:  # noqa: BLE001 — contract requires [] on any internal failure
            return []

        fused = reciprocal_rank_fusion([dense_hits, sparse_hits], k=self._fusion_k)

        results = []
        for chunk_id, score in fused[:k]:
            chunk = self._chunk_lookup.get(chunk_id)
            if chunk is None:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    passage_id=chunk.doc_id,
                    text=chunk.text,
                    score=score,
                    language=str(chunk.metadata.get("language", "hi")),
                )
            )
        return results
