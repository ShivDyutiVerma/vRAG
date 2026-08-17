"""Tests for MetadataAwareChunker — mainly that dataset-specific tags actually flow through."""

from vrag.chunking.base import Document
from vrag.chunking.strategies.metadata_aware import MetadataAwareChunker


def test_all_dataset_metadata_fields_present_on_chunk() -> None:
    doc = Document(
        doc_id="d1",
        text="पाठ",
        language="hi",
        source_lang="en",
        query_type="DESCRIPTION",
        is_selected=True,
    )
    chunk = MetadataAwareChunker().chunk(doc)[0]
    assert chunk.metadata["language"] == "hi"
    assert chunk.metadata["source_lang"] == "en"
    assert chunk.metadata["query_type"] == "DESCRIPTION"
    assert chunk.metadata["is_selected"] is True


def test_default_mode_is_boost() -> None:
    doc = Document(doc_id="d1", text="पाठ", language="hi", source_lang="en")
    chunk = MetadataAwareChunker().chunk(doc)[0]
    assert chunk.metadata["metadata_mode"] == "boost"


def test_filter_mode_recorded_in_metadata_and_config() -> None:
    strategy = MetadataAwareChunker(mode="filter")
    doc = Document(doc_id="d1", text="पाठ", language="hi", source_lang="en")
    chunk = strategy.chunk(doc)[0]
    assert chunk.metadata["metadata_mode"] == "filter"
    assert strategy.config()["mode"] == "filter"


def test_empty_doc_yields_no_chunks() -> None:
    doc = Document(doc_id="d1", text="   ", language="hi", source_lang="en")
    assert MetadataAwareChunker().chunk(doc) == []
