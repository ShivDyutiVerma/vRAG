"""Tests for SemanticChunker using an injected fake embed_fn (no real model needed) — asserts the
strategy actually splits at similarity troughs and fails loudly without an embedder."""

import pytest

from vrag.chunking.base import Document
from vrag.chunking.strategies.semantic import SemanticChunker

# Four sentences: S0/S1 are "close" (topic A), S2/S3 are "close" (topic B), with a clear gap
# between the two topics — the fake embedder encodes exactly that as 2D vectors.
_FAKE_EMBEDDINGS = {
    "टॉपिक ए वाक्य एक।": [1.0, 0.0],
    "टॉपिक ए वाक्य दो।": [0.95, 0.05],
    "टॉपिक बी वाक्य एक।": [0.0, 1.0],
    "टॉपिक बी वाक्य दो।": [0.05, 0.95],
}


def _fake_embed(sentences: list[str]) -> list[list[float]]:
    return [_FAKE_EMBEDDINGS[s] for s in sentences]


def test_splits_at_the_similarity_trough() -> None:
    text = " ".join(_FAKE_EMBEDDINGS.keys())
    doc = Document(doc_id="d1", text=text, language="hi", source_lang="en")
    chunker = SemanticChunker(percentile_threshold=50, embed_fn=_fake_embed)
    chunks = chunker.chunk(doc)
    assert len(chunks) == 2
    assert "टॉपिक ए" in chunks[0].text and "टॉपिक बी" not in chunks[0].text
    assert "टॉपिक बी" in chunks[1].text and "टॉपिक ए" not in chunks[1].text


def test_raises_clearly_without_embed_fn() -> None:
    doc = Document(
        doc_id="d1", text="वाक्य एक। वाक्य दो।", language="hi", source_lang="en"
    )
    with pytest.raises(RuntimeError, match="embed_fn"):
        SemanticChunker().chunk(doc)


def test_single_sentence_doc_short_circuits_without_needing_embedder() -> None:
    doc = Document(doc_id="d1", text="एक ही वाक्य है।", language="hi", source_lang="en")
    chunks = SemanticChunker().chunk(doc)  # no embed_fn passed — must not raise
    assert len(chunks) == 1


def test_empty_doc_yields_no_chunks() -> None:
    doc = Document(doc_id="d1", text="", language="hi", source_lang="en")
    assert SemanticChunker().chunk(doc) == []


def test_invalid_percentile_rejected() -> None:
    with pytest.raises(ValueError, match="percentile_threshold"):
        SemanticChunker(percentile_threshold=0)
    with pytest.raises(ValueError, match="percentile_threshold"):
        SemanticChunker(percentile_threshold=100)
