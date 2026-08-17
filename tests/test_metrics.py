"""Tests for retrieval metrics — hand-computed expected values, not just "does it run"."""

from vrag.retrieval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


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
