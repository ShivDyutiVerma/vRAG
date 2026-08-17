"""Written before the model-loading code per docs/BUILD_PLAN.md P1 task 1 ("Write
tests/test_embedder.py FIRST"). Only covers the prefix logic — fast, no network/model download
needed, so this stays in the default `pytest -q` run and in CI without a cached model. The actual
embed_queries/embed_passages calls need the real model and are exercised by scripts/eval_chunking.py
and scripts/build_index.py once those run for real, not by this fast unit suite.
"""

from vrag.index.embedder import (
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    format_passage,
    format_query,
)


def test_query_prefix_is_applied() -> None:
    assert format_query("भारत की राजधानी क्या है?") == f"{QUERY_PREFIX}भारत की राजधानी क्या है?"


def test_passage_prefix_is_applied() -> None:
    assert format_passage("नई दिल्ली भारत की राजधानी है।") == (
        f"{PASSAGE_PREFIX}नई दिल्ली भारत की राजधानी है।"
    )


def test_query_and_passage_prefixes_are_different() -> None:
    """The whole point of E5's prefix scheme is asymmetry between query and passage encoding —
    if these two ever matched, retrieval quality would silently degrade with no error raised."""
    assert QUERY_PREFIX != PASSAGE_PREFIX


def test_empty_text_still_gets_prefixed() -> None:
    assert format_query("") == QUERY_PREFIX
    assert format_passage("") == PASSAGE_PREFIX
