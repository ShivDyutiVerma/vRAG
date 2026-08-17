"""Tests for Reciprocal Rank Fusion — the fusion formula itself, and that it correctly rewards a
chunk_id appearing in both lists over one appearing in only one."""

from vrag.index.fusion import reciprocal_rank_fusion


def test_single_list_preserves_order() -> None:
    result = reciprocal_rank_fusion([[("a", 0.9), ("b", 0.5), ("c", 0.1)]])
    assert [chunk_id for chunk_id, _ in result] == ["a", "b", "c"]


def test_chunk_in_both_lists_outranks_chunk_in_one() -> None:
    dense = [("a", 0.9), ("b", 0.8)]
    sparse = [("a", 5.0), ("c", 4.0)]
    result = reciprocal_rank_fusion([dense, sparse])
    assert result[0][0] == "a"  # appears first in both lists


def test_fusion_score_matches_rrf_formula() -> None:
    # "a" is rank 1 in both lists, k=60 -> score = 1/61 + 1/61
    result = reciprocal_rank_fusion([[("a", 1.0)], [("a", 1.0)]], k=60)
    assert result[0][1] == 2 / 61


def test_empty_lists_yield_empty_result() -> None:
    assert reciprocal_rank_fusion([[], []]) == []


def test_scores_within_input_lists_are_ignored_only_rank_matters() -> None:
    # wildly different raw scores, same rank position -> same RRF contribution
    result_a = reciprocal_rank_fusion([[("x", 1000.0)]])
    result_b = reciprocal_rank_fusion([[("x", 0.001)]])
    assert result_a[0][1] == result_b[0][1]
