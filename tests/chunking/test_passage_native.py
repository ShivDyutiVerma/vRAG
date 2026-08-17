"""Boundary-behaviour tests for PassageNativeChunker."""

from vrag.chunking.base import Document
from vrag.chunking.strategies.passage_native import PassageNativeChunker


def test_one_passage_yields_exactly_one_chunk_with_full_text() -> None:
    doc = Document(doc_id="d1", text="कुछ पाठ यहाँ है।", language="hi", source_lang="en")
    chunks = PassageNativeChunker().chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text == doc.text
    assert chunks[0].doc_id == doc.doc_id


def test_empty_or_whitespace_only_doc_yields_no_chunks() -> None:
    chunker = PassageNativeChunker()
    assert chunker.chunk(Document(doc_id="d1", text="", language="hi", source_lang="en")) == []
    assert chunker.chunk(Document(doc_id="d1", text="   ", language="hi", source_lang="en")) == []


def test_is_selected_flows_into_chunk_metadata() -> None:
    doc = Document(
        doc_id="d1", text="पाठ", language="hi", source_lang="en", is_selected=True
    )
    chunks = PassageNativeChunker().chunk(doc)
    assert chunks[0].metadata["is_selected"] is True
