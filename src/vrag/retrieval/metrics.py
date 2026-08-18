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


def dedupe_doc_ids(raw_doc_ids: list[str]) -> list[str]:
    """Collapse a ranked list of doc_ids down to first-occurrence-only, preserving order.

    Required before any of the functions above see a doc_id list built from *chunk* hits —
    strategies/indexes that can return multiple chunks from the same passage (e.g.
    sentence_window, or a single passage matched by both dense and sparse in a hybrid fusion)
    would otherwise let one passage occupy several slots in the top-k window, which skews
    Recall@k (crowds out genuinely distinct passages) and badly inflates nDCG@10 (which sums
    credit per occurrence). Root-caused and fixed as docs/DECISIONS_R.md R-006 — this function is
    the one fix point every eval script must route through, so the bug can't reappear per-caller.
    """
    return list(dict.fromkeys(raw_doc_ids))


def score_hits(
    hits: list[tuple[str, float]],
    chunk_to_doc_id: dict[str, str],
    relevant_doc_ids: set[str],
) -> dict[str, float]:
    """One query's full metric set from raw (chunk_id, score) search hits — the shared path every
    ablation eval script (A1-A4) should call, so `dedupe_doc_ids` is never skipped by accident."""
    raw_doc_ids = [
        chunk_to_doc_id[chunk_id] for chunk_id, _score in hits if chunk_id in chunk_to_doc_id
    ]
    doc_ids = dedupe_doc_ids(raw_doc_ids)
    return {
        "recall@1": recall_at_k(doc_ids, relevant_doc_ids, k=1),
        "recall@5": recall_at_k(doc_ids, relevant_doc_ids, k=5),
        "recall@10": recall_at_k(doc_ids, relevant_doc_ids, k=10),
        "mrr@10": reciprocal_rank(doc_ids, relevant_doc_ids, k=10),
        "ndcg@10": ndcg_at_k(doc_ids, relevant_doc_ids, k=10),
    }
