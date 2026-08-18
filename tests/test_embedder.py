"""Written before the model-loading code per docs/BUILD_PLAN.md P1 task 1 ("Write
tests/test_embedder.py FIRST"). Only covers the prefix logic — fast, no network/model download
needed, so this stays in the default `pytest -q` run and in CI without a cached model. The actual
embed_queries/embed_passages calls need the real model and are exercised by scripts/eval_chunking.py
and scripts/build_index.py once those run for real, not by this fast unit suite.
"""

from vrag.index.embedder import (
    EMBEDDER_REGISTRY,
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    BGEM3Embedder,
    E5Embedder,
    Model2VecEmbedder,
    ONNXE5Embedder,
    VyakyarthEmbedder,
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


def test_embedder_registry_has_all_four_a2_candidates_plus_onnx() -> None:
    # multilingual-e5-small-onnx-int8 (docs/DECISIONS_R.md R-019, P6) isn't an A2 candidate --
    # it's a quantised variant of A2's winner, added after A2 concluded.
    assert set(EMBEDDER_REGISTRY.keys()) == {
        "multilingual-e5-small",
        "multilingual-e5-small-onnx-int8",
        "potion-multilingual-128M",
        "bge-m3",
        "vyakyarth",
    }


def test_embedder_registry_maps_to_correct_classes() -> None:
    assert EMBEDDER_REGISTRY["multilingual-e5-small"] is E5Embedder
    assert EMBEDDER_REGISTRY["multilingual-e5-small-onnx-int8"] is ONNXE5Embedder
    assert EMBEDDER_REGISTRY["potion-multilingual-128M"] is Model2VecEmbedder
    assert EMBEDDER_REGISTRY["bge-m3"] is BGEM3Embedder
    assert EMBEDDER_REGISTRY["vyakyarth"] is VyakyarthEmbedder


def test_every_embedder_has_a_name_and_does_not_load_a_model_on_construction() -> None:
    """Constructing any embedder must be cheap (no download/model load) — only the first real
    embed_queries/embed_passages call should trigger that."""
    for cls in EMBEDDER_REGISTRY.values():
        instance = cls()
        assert instance.name
        assert instance._model is None
