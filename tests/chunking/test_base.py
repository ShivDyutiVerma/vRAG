"""Tests for the Document/Chunk models — mainly that defaults are sane, since every strategy
implementation depends on these shapes being right."""

from vrag.chunking.base import Chunk, Document


def test_document_defaults() -> None:
    doc = Document(doc_id="d1", text="कुछ पाठ", language="hi", source_lang="en")
    assert doc.is_selected is False
    assert doc.query_id is None


def test_chunk_defaults_have_no_parent_and_empty_metadata() -> None:
    chunk = Chunk(chunk_id="c1", doc_id="d1", text="कुछ पाठ")
    assert chunk.parent_chunk_id is None
    assert chunk.metadata == {}


def test_chunk_metadata_is_independent_per_instance() -> None:
    """Pydantic default {} must not be a shared mutable — a classic Python footgun."""
    c1 = Chunk(chunk_id="c1", doc_id="d1", text="a")
    c2 = Chunk(chunk_id="c2", doc_id="d1", text="b")
    c1.metadata["language"] = "hi"
    assert c2.metadata == {}
