"""Tests for retrieval metrics — hand-computed expected values, not just "does it run"."""

from vrag.retrieval.metrics import (
    dedupe_doc_ids,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    score_hits,
)


def test_recall_at_k_hit_within_range() -> None:
    assert recall_at_k(["a", "b", "c"], {"b"}, k=3) == 1.0


def test_recall_at_k_miss_outside_range() -> None:
    assert recall_at_k(["a", "b", "c"], {"z"}, k=3) == 0.0


def test_recall_at_k_relevant_present_but_beyond_k() -> None:
    assert recall_at_k(["a", "b", "z"], {"z"}, k=2) == 0.0


def test_recall_at_k_no_relevant_docs_is_zero_not_error() -> None:
    assert recall_at_k(["a", "b"], set(), k=5) == 0.0


def test_reciprocal_rank_first_position() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_reciprocal_rank_third_position() -> None:
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == 1 / 3


def test_reciprocal_rank_no_hit_is_zero() -> None:
    assert reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0


def test_ndcg_perfect_ranking_is_one() -> None:
    # single relevant doc, ranked first -> perfect nDCG
    assert ndcg_at_k(["a", "x", "y"], {"a"}, k=3) == 1.0


def test_ndcg_relevant_doc_lower_ranked_scores_less_than_one() -> None:
    score = ndcg_at_k(["x", "y", "a"], {"a"}, k=3)
    assert 0.0 < score < 1.0


def test_ndcg_no_relevant_found_is_zero() -> None:
    assert ndcg_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0


def test_ndcg_multiple_relevant_docs_both_found() -> None:
    # two relevant docs, both in top 2 -> perfect nDCG
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3) == 1.0


def test_dedupe_doc_ids_keeps_first_occurrence_order() -> None:
    assert dedupe_doc_ids(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_dedupe_doc_ids_empty_list() -> None:
    assert dedupe_doc_ids([]) == []


def test_score_hits_maps_chunks_to_docs_and_dedupes_before_scoring() -> None:
    # two chunks from the same passage "p1" occupy ranks 1-2; a distinct relevant passage "p2"
    # sits at raw rank 3. Without dedup, recall@1 would miss p2 entirely even though it's the
    # second *unique* passage in the list — this is exactly the R-006 bug this function fixes.
    hits = [("c1", 0.9), ("c2", 0.8), ("c3", 0.7)]
    chunk_to_doc_id = {"c1": "p1", "c2": "p1", "c3": "p2"}
    result = score_hits(hits, chunk_to_doc_id, relevant_doc_ids={"p2"})
    assert result["recall@1"] == 0.0  # p2 is the 2nd unique doc, not within top-1
    assert result["recall@5"] == 1.0
    assert result["mrr@10"] == 1 / 2  # p2 ranks 2nd among unique docs


def test_score_hits_ignores_chunk_ids_not_in_lookup() -> None:
    hits = [("unknown", 0.9), ("c1", 0.8)]
    chunk_to_doc_id = {"c1": "p1"}
    result = score_hits(hits, chunk_to_doc_id, relevant_doc_ids={"p1"})
    assert result["recall@1"] == 1.0
