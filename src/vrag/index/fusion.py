"""Reciprocal Rank Fusion (TECH_MENU.md S8, k=60). Combines a dense-search ranked list and a
sparse-search ranked list into one ranking without normalising two incomparable score scales
(cosine similarity vs. BM25 score) — each result gets `1 / (k + rank)` from every list it appears
in, summed. See docs/GLOSSARY.md for the plain-language explanation.
"""

from __future__ import annotations

DEFAULT_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]], k: int = DEFAULT_K
) -> list[tuple[str, float]]:
    """Each input list is [(chunk_id, score), ...] already sorted best-first; the scores
    themselves are ignored — only rank position matters, which is the whole point of RRF."""
    fused: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, (chunk_id, _score) in enumerate(ranked_list, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
