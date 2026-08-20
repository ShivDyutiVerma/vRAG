"""Retrieval orchestration for `retrieve()` (src/vrag/retrieval/interface.py). Supports three modes
— "dense" (default), "sparse", "hybrid" (RRF-fused dense+sparse) — selected via `retrieval_mode`.

Language filtering (docs/DECISIONS.md ADR-012, wiring what ADR-009's `language` param left inert):
when a real `language` (Sarvam BCP-47 code, e.g. `"hi-IN"`) is passed and maps to a known target
language, `retrieve()` searches a WIDE candidate pool (`_LANGUAGE_FILTER_WIDE_K`) and restricts to
chunks whose `language` metadata matches, before truncating to the caller's requested `k` — this is
exactly the "filter" strategy Phase 2 measured as the winner (+8.7-9.1pp Recall@10 over no-filter,
every corpus size, `docs/DECISIONS.md` ADR-009/ADR-012), not a new untested mechanism. Falls back to
the unfiltered wide results if the filter would return nothing (a genuine "no same-language
candidate in the window" case must not manufacture a zero-result failure — G3 downstream is what
decides "not enough evidence", not this filter). `language=None` or an unmapped code searches
exactly as before Phase 1/2/3 — this is fully backward compatible.

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
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

from vrag.chunking.base import Chunk
from vrag.index.dense import DenseIndex
from vrag.index.embedder import EmbedderProtocol
from vrag.index.fusion import DEFAULT_K, reciprocal_rank_fusion
from vrag.index.sparse import SparseIndex
from vrag.index.sqlite_chunk_lookup import SQLiteChunkLookup
from vrag.languages import SARVAM_TO_TARGET_LANG
from vrag.retrieval.interface import RetrievedChunk

# Candidate-pool width before narrowing to the caller's requested k, when language-filtering is
# active (docs/DECISIONS.md ADR-012) -- same value Phase 2's real measurement used
# (scripts/eval_multilingual_retrieval.py's WIDE_K); a narrower window risks an empty same-language
# filter result on a roughly-evenly-split 14-language corpus purely from bad luck, not genuine
# absence.
_LANGUAGE_FILTER_WIDE_K = 100


class HybridRetriever:
    def __init__(
        self,
        dense: DenseIndex,
        # None is only valid when retrieval_mode="dense" -- ADR-006 found the sparse/BM25 index
        # was being loaded unconditionally (~105MB) even though dense-only mode never queries it.
        # Callers that don't need "sparse"/"hybrid" can skip loading it entirely and pass None.
        sparse: SparseIndex | None,
        embedder: EmbedderProtocol,
        # Mapping[str, Chunk] covers plain dicts (offline build/eval); SQLiteChunkLookup (R-021)
        # isn't a Mapping (no __iter__, deliberately — see its docstring) so it's named explicitly.
        chunk_lookup: Mapping[str, Chunk] | SQLiteChunkLookup,
        fusion_k: int = DEFAULT_K,
        executor: ThreadPoolExecutor | None = None,
        retrieval_mode: str = "dense",
    ) -> None:
        if retrieval_mode not in ("dense", "sparse", "hybrid"):
            raise ValueError(f"unknown retrieval_mode: {retrieval_mode!r}")
        if retrieval_mode in ("sparse", "hybrid") and sparse is None:
            raise ValueError(
                f"retrieval_mode={retrieval_mode!r} requires a real SparseIndex, got None -- "
                "fail fast here rather than silently degrading to [] on first real query"
            )
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
        # __init__ already guarantees sparse is not None whenever this can be reached
        # (retrieval_mode in ("sparse", "hybrid")) -- assert narrows the type for mypy.
        assert self._sparse is not None
        return self._sparse.search(query, k)

    async def retrieve(
        self, query: str, k: int = 5, language: str | None = None
    ) -> list[RetrievedChunk]:
        """Never raises — matches the retrieve() contract in interface.py. Internal failures
        collapse to an empty result so the harness's grounding gate can abstain.

        `language` (docs/DECISIONS.md ADR-009/ADR-012): a real, mapped Sarvam code searches a
        wide candidate pool and restricts to same-language chunks before truncating to `k` — see
        module docstring for the measured evidence behind this. `None` or an unmapped code (e.g.
        `te-IN`, which has no indexed corpus) searches exactly as before, unfiltered."""
        if not query.strip():
            return []

        target_lang = SARVAM_TO_TARGET_LANG.get(language) if language else None
        search_k = max(k, _LANGUAGE_FILTER_WIDE_K) if target_lang else k

        loop = asyncio.get_event_loop()
        try:
            if self._retrieval_mode == "dense":
                hits = await loop.run_in_executor(
                    self._executor, self._embed_and_search_dense, query, search_k
                )
            elif self._retrieval_mode == "sparse":
                hits = await loop.run_in_executor(
                    self._executor, self._search_sparse, query, search_k
                )
            else:  # "hybrid" — dense and sparse MUST run concurrently, see module docstring
                dense_hits, sparse_hits = await asyncio.gather(
                    loop.run_in_executor(
                        self._executor, self._embed_and_search_dense, query, search_k
                    ),
                    loop.run_in_executor(self._executor, self._search_sparse, query, search_k),
                )
                hits = reciprocal_rank_fusion([dense_hits, sparse_hits], k=self._fusion_k)
        except Exception:  # noqa: BLE001 — contract requires [] on any internal failure
            return []

        if target_lang:
            same_language_hits = [
                (chunk_id, score)
                for chunk_id, score in hits
                if (chunk := self._chunk_lookup.get(chunk_id)) is not None
                and chunk.metadata.get("language") == target_lang
            ]
            # Documented fallback (module docstring): never manufacture a zero-result failure
            # merely because this particular search window had no same-language candidate.
            hits = same_language_hits if same_language_hits else hits

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
