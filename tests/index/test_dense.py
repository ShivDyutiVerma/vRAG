"""Tests for DenseIndex using tiny synthetic vectors — no real embedder needed, just checks the
FAISS wrapper's contract (add/search/length, dimension validation, empty-index behaviour)."""

from pathlib import Path

import pytest

from vrag.index.dense import DenseIndex


def test_search_returns_nearest_by_inner_product() -> None:
    index = DenseIndex(dim=2)
    index.add(
        ["a", "b", "c"],
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]],
    )
    results = index.search([1.0, 0.0], k=2)
    result_ids = [chunk_id for chunk_id, _ in results]
    assert result_ids[0] == "a"  # exact match should rank first
    assert "c" in result_ids  # near match should also appear in top 2


def test_len_reflects_added_vectors() -> None:
    index = DenseIndex(dim=2)
    assert len(index) == 0
    index.add(["a", "b"], [[1.0, 0.0], [0.0, 1.0]])
    assert len(index) == 2


def test_search_on_empty_index_returns_empty_list() -> None:
    index = DenseIndex(dim=2)
    assert index.search([1.0, 0.0], k=5) == []


def test_mismatched_chunk_ids_and_vectors_length_rejected() -> None:
    index = DenseIndex(dim=2)
    with pytest.raises(ValueError, match="same length"):
        index.add(["a", "b"], [[1.0, 0.0]])


def test_wrong_dimension_vector_rejected() -> None:
    index = DenseIndex(dim=3)
    with pytest.raises(ValueError, match="dim"):
        index.add(["a"], [[1.0, 0.0]])


def test_adding_empty_batch_is_a_no_op() -> None:
    index = DenseIndex(dim=2)
    index.add([], [])
    assert len(index) == 0


def test_k_larger_than_index_size_does_not_error() -> None:
    index = DenseIndex(dim=2)
    index.add(["a"], [[1.0, 0.0]])
    results = index.search([1.0, 0.0], k=10)
    assert len(results) == 1


def test_set_ef_search_changes_search_time_parameter_without_rebuild() -> None:
    index = DenseIndex(dim=2)
    index.add(["a", "b", "c"], [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    index.set_ef_search(16)
    assert index._index.hnsw.efSearch == 16
    index.set_ef_search(256)
    assert index._index.hnsw.efSearch == 256
    # search still works correctly after mutating efSearch, not just accepted silently
    results = index.search([1.0, 0.0], k=1)
    assert results[0][0] == "a"


def test_save_load_round_trip_preserves_search_behaviour(tmp_path: Path) -> None:
    index = DenseIndex(dim=2)
    index.add(["a", "b", "c"], [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    index.save(tmp_path / "idx")

    loaded = DenseIndex.load(tmp_path / "idx")
    assert len(loaded) == len(index)
    assert loaded.dim == index.dim
    original = index.search([1.0, 0.0], k=2)
    restored = loaded.search([1.0, 0.0], k=2)
    assert [c for c, _ in original] == [c for c, _ in restored]
