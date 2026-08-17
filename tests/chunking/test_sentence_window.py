"""Boundary-behaviour tests for SentenceWindowChunker, including Devanagari danda splitting."""

import pytest

from vrag.chunking.base import Document
from vrag.chunking.strategies.sentence_window import SentenceWindowChunker, split_sentences


def test_split_sentences_handles_devanagari_danda() -> None:
    text = "यह पहला वाक्य है। यह दूसरा वाक्य है। यह तीसरा वाक्य है।"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "यह पहला वाक्य है।"


def test_split_sentences_handles_latin_punctuation() -> None:
    sentences = split_sentences("First sentence. Second sentence! Third one?")
    assert len(sentences) == 3


def test_one_chunk_per_sentence() -> None:
    doc = Document(
        doc_id="d1",
        text="वाक्य एक। वाक्य दो। वाक्य तीन। वाक्य चार। वाक्य पांच।",
        language="hi",
        source_lang="en",
    )
    chunks = SentenceWindowChunker(window=1).chunk(doc)
    assert len(chunks) == 5  # one retrieval unit per sentence


def test_window_includes_neighbours_but_not_beyond() -> None:
    doc = Document(
        doc_id="d1",
        text="S0। S1। S2। S3। S4।",
        language="hi",
        source_lang="en",
    )
    chunks = SentenceWindowChunker(window=1).chunk(doc)
    # middle sentence (index 2) should include S1, S2, S3
    middle = chunks[2]
    assert "S1" in middle.text
    assert "S2" in middle.text
    assert "S3" in middle.text
    assert "S0" not in middle.text
    assert "S4" not in middle.text


def test_edge_sentence_window_does_not_go_out_of_bounds() -> None:
    doc = Document(doc_id="d1", text="S0। S1। S2।", language="hi", source_lang="en")
    chunks = SentenceWindowChunker(window=5).chunk(doc)
    # first chunk's window should clamp to available sentences, not error
    assert "S0" in chunks[0].text
    assert "S2" in chunks[0].text


def test_retrieval_sentence_metadata_is_the_precise_unit() -> None:
    doc = Document(doc_id="d1", text="S0। S1। S2।", language="hi", source_lang="en")
    chunks = SentenceWindowChunker(window=1).chunk(doc)
    assert chunks[1].metadata["retrieval_sentence"] == "S1।"


def test_negative_window_rejected() -> None:
    with pytest.raises(ValueError, match="window"):
        SentenceWindowChunker(window=-1)


def test_empty_doc_yields_no_chunks() -> None:
    doc = Document(doc_id="d1", text="", language="hi", source_lang="en")
    assert SentenceWindowChunker().chunk(doc) == []
