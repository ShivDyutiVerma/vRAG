"""Boundary-behaviour tests for FixedOverlapChunker, per BUILD_PLAN.md P2 exit criteria
("each with a unit test asserting chunk-boundary behaviour")."""

import pytest

from vrag.chunking.base import Document
from vrag.chunking.strategies.fixed_overlap import FixedOverlapChunker


def _doc(n_words: int) -> Document:
    return Document(
        doc_id="d1",
        text=" ".join(f"w{i}" for i in range(n_words)),
        language="hi",
        source_lang="en",
    )


def test_short_doc_produces_one_chunk() -> None:
    chunker = FixedOverlapChunker(size=256, overlap=0.2)
    chunks = chunker.chunk(_doc(50))
    assert len(chunks) == 1
    assert chunks[0].text == _doc(50).text


def test_long_doc_produces_multiple_overlapping_chunks() -> None:
    chunker = FixedOverlapChunker(size=100, overlap=0.2)
    chunks = chunker.chunk(_doc(350))
    assert len(chunks) > 1
    # stride = 100 * 0.8 = 80, so consecutive windows share 20 words
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[80:] == second_words[:20]


def test_zero_overlap_windows_do_not_share_words() -> None:
    chunker = FixedOverlapChunker(size=100, overlap=0.0)
    chunks = chunker.chunk(_doc(250))
    assert len(chunks) == 3
    assert set(chunks[0].text.split()).isdisjoint(chunks[1].text.split())


def test_empty_doc_produces_no_chunks() -> None:
    chunker = FixedOverlapChunker()
    assert chunker.chunk(_doc(0)) == []


def test_config_is_reproducible() -> None:
    chunker = FixedOverlapChunker(size=128, overlap=0.1)
    cfg = chunker.config()
    assert cfg["size"] == 128
    assert cfg["overlap"] == 0.1


@pytest.mark.parametrize("bad_overlap", [-0.1, 1.0, 1.5])
def test_invalid_overlap_rejected(bad_overlap: float) -> None:
    with pytest.raises(ValueError, match="overlap"):
        FixedOverlapChunker(overlap=bad_overlap)


def test_invalid_size_rejected() -> None:
    with pytest.raises(ValueError, match="size"):
        FixedOverlapChunker(size=0)
