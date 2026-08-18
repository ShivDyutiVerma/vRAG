"""Retrieval orchestration for `retrieve()` (src/vrag/retrieval/interface.py). Supports three modes
— "dense" (default), "sparse", "hybrid" (RRF-fused dense+sparse) — selected via `retrieval_mode`.

**Default is "dense", not "hybrid."** The A3 ablation (docs/DECISIONS_R.md R-010,
docs/EVAL_RESULTS.md §3) measured dense-only beating hybrid+RRF on this corpus (Recall@5 0.652 vs.
0.604) — BM25 is comparatively weak on this machine-translated Hindi text, and naive equal-weight
RRF gives its lower-quality top ranks the same fusion credit as dense's, displacing genuine dense
hits. This is a deliberate, data-driven deviation from `AGENT_BUILD_SPEC.md` line 625's assumed
Phase-3 exit criterion ("hybrid beats dense-only on Recall@5") and CLAUDE.md's original hot-path
invariant text — confirmed with the user before shipping (R-010's "Consequences", updated).

`retrieval_mode="hybrid"` remains fully implemented and tested (`tests/retrieval/test_hybrid.py`)
for anyone revisiting this call — in that mode, dense and sparse MUST run concurrently, not
sequentially. Both FAISS and bm25s release the GIL during their C-level search calls, so running
each in a thread-pool executor via `asyncio.gather` gives genuine wall-clock parallelism, not just
cooperative scheduling around I/O. Sparse search doesn't depend on the query embedding at all, so
embedding + dense search run as one sequential unit on one thread while sparse search runs
concurrently on another — not "embed, then run two searches," which would serialize the one step
that doesn't need to be.
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
        retrieval_mode: str = "dense",
    ) -> None:
        if retrieval_mode not in ("dense", "sparse", "hybrid"):
            raise ValueError(f"unknown retrieval_mode: {retrieval_mode!r}")
        self._dense = dense
        self._sparse = sparse
        self._embedder = embedder
        self._chunk_lookup = chunk_lookup
        self._fusion_k = fusion_k
        self._executor = executor or ThreadPoolExecutor(max_workers=2)
        self._retrieval_mode = retrieval_mode

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
            if self._retrieval_mode == "dense":
                hits = await loop.run_in_executor(
                    self._executor, self._embed_and_search_dense, query, k
                )
            elif self._retrieval_mode == "sparse":
                hits = await loop.run_in_executor(self._executor, self._search_sparse, query, k)
            else:  # "hybrid" — dense and sparse MUST run concurrently, see module docstring
                dense_hits, sparse_hits = await asyncio.gather(
                    loop.run_in_executor(self._executor, self._embed_and_search_dense, query, k),
                    loop.run_in_executor(self._executor, self._search_sparse, query, k),
                )
                hits = reciprocal_rank_fusion([dense_hits, sparse_hits], k=self._fusion_k)
        except Exception:  # noqa: BLE001 — contract requires [] on any internal failure
            return []

        results = []
        for chunk_id, score in hits[:k]:
            chunk = self._chunk_lookup.get(chunk_id)
            if chunk is None:
                continue
            # Clamp into RetrievedChunk.score's [0,1] contract. In "dense" mode this is a rare
            # edge case (cosine similarity on E5 vectors is virtually always in ~0.3-0.95 for real
            # text, but isn't mathematically guaranteed non-negative). In "sparse" mode BM25 scores
            # are genuinely unbounded above 1.0 -- clamping there is lossy, but "sparse" isn't the
            # shipped default (see module docstring) and G3 (src/vrag/guardrails/g3_confidence.py)
            # requires a bounded score to mean anything as a confidence signal.
            clamped_score = max(0.0, min(1.0, score))
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    passage_id=chunk.doc_id,
                    text=chunk.text,
                    score=clamped_score,
                    language=str(chunk.metadata.get("language", "hi")),
                )
            )
        return results
