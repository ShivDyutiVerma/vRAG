"""Tests for the sparse (BM25) index. The tokenizer test is required by CLAUDE.md /
AGENT_BUILD_SPEC.md §5.2/§7.1 — a Devanagari sentence must tokenise sensibly, since a naive
`.split()` or an English-only tokenizer silently halves sparse recall on Hindi text without ever
raising an error.
"""

import pytest

from vrag.index.sparse import SparseIndex, tokenize


def test_hindi_sentence_tokenises_into_expected_word_count() -> None:
    # "भारत की राजधानी नई दिल्ली है" = 6 space-separated Devanagari words
    tokens = tokenize("भारत की राजधानी नई दिल्ली है")
    assert len(tokens) == 6


def test_tokenize_is_not_naive_whitespace_split_on_punctuation() -> None:
    # Devanagari danda (।) must not glue onto the preceding word as part of the token
    tokens = tokenize("यह एक वाक्य है।")
    assert "है।" not in tokens
    assert "है" in tokens


def test_tokenize_lowercases_latin_text() -> None:
    assert tokenize("Hindi Query") == ["hindi", "query"]


def test_tokenize_empty_string_yields_no_tokens() -> None:
    assert tokenize("") == []


def test_search_ranks_lexical_overlap_higher() -> None:
    index = SparseIndex()
    index.build(
        ["a", "b"],
        [
            "भारत की राजधानी नई दिल्ली है",
            "मुंबई भारत का सबसे बड़ा शहर है",
        ],
    )
    results = index.search("भारत की राजधानी क्या है", k=2)
    assert results[0][0] == "a"


def test_search_on_unbuilt_index_returns_empty_list() -> None:
    assert SparseIndex().search("कुछ भी", k=5) == []


def test_search_with_no_matching_tokens_does_not_error() -> None:
    index = SparseIndex()
    index.build(["a"], ["भारत की राजधानी"])
    results = index.search("", k=5)
    assert results == []


def test_mismatched_lengths_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        SparseIndex().build(["a", "b"], ["only one text"])
