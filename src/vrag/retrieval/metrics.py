"""Retrieval quality metrics for the chunking/embedding/retrieval-mode ablations
(AGENT_BUILD_SPEC.md §7.1, docs/EVAL_PROTOCOL.md). Relevance is judged at the PASSAGE level (via
`doc_id`, which every ChunkingStrategy sets to the source passage id) — a chunking strategy that
splits one passage into five chunks shouldn't be penalised or rewarded relative to one that keeps
it whole, so "was a relevant passage found" is what's scored, not "was this exact chunk_id found."
"""

from __future__ import annotations

import math


def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    if not relevant_doc_ids:
        return 0.0
    top_k = set(retrieved_doc_ids[:k])
    return 1.0 if top_k & relevant_doc_ids else 0.0


def reciprocal_rank(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int = 10) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int = 10) -> float:
    """Binary relevance nDCG — every relevant passage counts equally (no graded relevance
    signal available in MSMARCO-XI's is_selected flag beyond 0/1)."""
    if not relevant_doc_ids:
        return 0.0

    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in relevant_doc_ids:
            dcg += 1.0 / math.log2(rank + 1)

    n_relevant_in_range = min(len(relevant_doc_ids), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, n_relevant_in_range + 1))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
