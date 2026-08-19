"""Tests for DenseIndex using tiny synthetic vectors — no real embedder needed, just checks the
FAISS wrapper's contract (add/search/length, dimension validation, empty-index behaviour)."""

from pathlib import Path

import faiss
import pytest

from vrag.index.dense import DEFAULT_EF_CONSTRUCTION, DEFAULT_EF_SEARCH, DEFAULT_M, DenseIndex


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


# sqfp16 quantization (docs/DECISIONS_R.md R-033/R-034) -- IndexHNSWSQ + ScalarQuantizer.QT_fp16,
# same M/efConstruction/efSearch/metric as the default "none" (flat fp32) path. A handful of
# normalised synthetic vectors, same style as the tests above -- no real embedder needed, this is
# purely about DenseIndex's own construction/persistence contract for the new option.
_SQFP16_CHUNK_IDS = [f"chunk-{i}" for i in range(10)]
_SQFP16_VECTORS = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.9, 0.1, 0.0, 0.0],
    [0.1, 0.9, 0.0, 0.0],
    [0.0, 0.9, 0.1, 0.0],
    [0.0, 0.0, 0.9, 0.1],
    [0.7, 0.7, 0.0, 0.0],
    [0.0, 0.0, 0.7, 0.7],
]


def _build_sqfp16_index() -> DenseIndex:
    index = DenseIndex(dim=4, quantization="sqfp16")
    index.add(_SQFP16_CHUNK_IDS, _SQFP16_VECTORS)
    return index


def test_default_quantization_still_builds_a_plain_flat_index() -> None:
    """Regression guard: adding the `quantization` param must not silently change what every
    existing caller (that never passes it) builds."""
    index = DenseIndex(dim=4)
    assert isinstance(index._index, faiss.IndexHNSWFlat)
    assert not isinstance(index._index, faiss.IndexHNSWSQ)


def test_unknown_quantization_rejected() -> None:
    with pytest.raises(ValueError, match="quantization"):
        DenseIndex(dim=4, quantization="int8")  # type: ignore[arg-type]


def test_sqfp16_index_loads_successfully(tmp_path: Path) -> None:
    index = _build_sqfp16_index()
    index.save(tmp_path / "idx")

    loaded = DenseIndex.load(tmp_path / "idx")
    assert isinstance(loaded._index, faiss.IndexHNSWSQ)
    assert len(loaded) == len(_SQFP16_CHUNK_IDS)
    assert loaded.dim == 4
    assert loaded._quantization == "sqfp16"


def test_sqfp16_index_metric_is_inner_product(tmp_path: Path) -> None:
    index = _build_sqfp16_index()
    index.save(tmp_path / "idx")
    loaded = DenseIndex.load(tmp_path / "idx")
    assert loaded._index.metric_type == faiss.METRIC_INNER_PRODUCT


def test_sqfp16_index_hnsw_parameters_match_production_defaults(tmp_path: Path) -> None:
    index = _build_sqfp16_index()
    index.save(tmp_path / "idx")
    loaded = DenseIndex.load(tmp_path / "idx")
    # nb_neighbors(1) is the upper-level neighbor count, which faiss sets equal to M directly
    # (level 0 stores 2*M instead, verified directly against a live IndexHNSWFlat before writing
    # this assertion, not assumed from documentation).
    assert loaded._index.hnsw.nb_neighbors(1) == DEFAULT_M
    assert loaded._index.hnsw.efConstruction == DEFAULT_EF_CONSTRUCTION
    assert loaded._index.hnsw.efSearch == DEFAULT_EF_SEARCH


def test_sqfp16_index_returns_valid_chunk_ids(tmp_path: Path) -> None:
    """Every returned chunk_id must be one that was actually added -- guards against an index
    corruption or an off-by-one in the quantized code path silently returning garbage IDs."""
    index = _build_sqfp16_index()
    index.save(tmp_path / "idx")
    loaded = DenseIndex.load(tmp_path / "idx")

    results = loaded.search([1.0, 0.0, 0.0, 0.0], k=5)
    assert results, "expected at least one hit"
    result_ids = [chunk_id for chunk_id, _score in results]
    assert all(cid in _SQFP16_CHUNK_IDS for cid in result_ids)
    assert len(set(result_ids)) == len(result_ids)  # no duplicate hits
    # the exact vector [1,0,0,0] was added as "chunk-0" -- it should rank first even under
    # fp16 quantization noise, since it's an exact match, not a near one.
    assert result_ids[0] == "chunk-0"


def test_sqfp16_and_flat_agree_on_top_hit_for_a_clear_match() -> None:
    """Not bit-identical scores (that's the whole point of quantization), but quantization noise
    shouldn't flip which chunk is nearest for a query with one obviously-correct answer."""
    flat = DenseIndex(dim=4, quantization="none")
    flat.add(_SQFP16_CHUNK_IDS, _SQFP16_VECTORS)
    sqfp16 = _build_sqfp16_index()

    query = [0.0, 0.0, 0.0, 1.0]  # exact match to chunk-3
    assert flat.search(query, k=1)[0][0] == "chunk-3"
    assert sqfp16.search(query, k=1)[0][0] == "chunk-3"
