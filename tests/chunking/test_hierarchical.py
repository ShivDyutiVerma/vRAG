"""Boundary-behaviour tests for HierarchicalChunker: parent/child relationship must be correct,
since retrieval indexes children only and generation depends on the parent_chunk_id link."""

import pytest

from vrag.chunking.base import Document
from vrag.chunking.strategies.hierarchical import HierarchicalChunker


def _doc(n_words: int) -> Document:
    return Document(
        doc_id="d1",
        text=" ".join(f"w{i}" for i in range(n_words)),
        language="hi",
        source_lang="en",
    )


def test_produces_one_parent_and_correct_children_for_single_parent_window() -> None:
    chunker = HierarchicalChunker(child=10, parent=50)
    chunks = chunker.chunk(_doc(50))
    parents = [c for c in chunks if c.metadata["is_parent"]]
    children = [c for c in chunks if not c.metadata["is_parent"]]
    assert len(parents) == 1
    assert len(children) == 5  # 50 words / 10-word children


def test_every_child_parent_chunk_id_points_to_an_actual_parent() -> None:
    chunker = HierarchicalChunker(child=10, parent=50)
    chunks = chunker.chunk(_doc(120))
    parent_ids = {c.chunk_id for c in chunks if c.metadata["is_parent"]}
    child_parent_refs = {c.parent_chunk_id for c in chunks if not c.metadata["is_parent"]}
    assert child_parent_refs.issubset(parent_ids)


def test_parent_has_no_parent_chunk_id() -> None:
    chunker = HierarchicalChunker(child=10, parent=50)
    chunks = chunker.chunk(_doc(50))
    parent = next(c for c in chunks if c.metadata["is_parent"])
    assert parent.parent_chunk_id is None


def test_multiple_parent_windows_for_long_doc() -> None:
    chunker = HierarchicalChunker(child=10, parent=50)
    chunks = chunker.chunk(_doc(120))
    parents = [c for c in chunks if c.metadata["is_parent"]]
    assert len(parents) == 3  # ceil(120/50)


def test_child_must_be_smaller_than_parent() -> None:
    with pytest.raises(ValueError, match="smaller"):
        HierarchicalChunker(child=100, parent=50)


def test_empty_doc_yields_no_chunks() -> None:
    assert HierarchicalChunker().chunk(_doc(0)) == []
