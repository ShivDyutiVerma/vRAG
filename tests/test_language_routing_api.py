"""Phase 1 (docs/DECISIONS.md ADR-009) end-to-end language routing, through the real API +
harness, mirroring tests/test_api.py's conventions (TestClient, force-stub via monkeypatch for
determinism where the exact retrieval outcome doesn't matter to what's being tested).
"""

from fastapi.testclient import TestClient

from vrag.api.main import app

client = TestClient(app)


def test_hindi_query_with_detected_language_is_allowed_through_to_retrieval():
    resp = client.post(
        "/ask", json={"query": "भारत की राजधानी क्या है?", "k": 3, "language": "hi-IN"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] != "refused"
    assert "retrieve" in body["timings_ms"]  # RetrieveStage actually ran
    assert body["query_language"] == "hi-IN"


def test_english_query_reaches_retrieval_as_of_phase_3():
    """Phase 1 (ADR-009) refused English (not yet indexed). Phase 3 (ADR-012) indexes it for
    real -- an English query must now reach retrieval, not be refused at G2."""
    resp = client.post(
        "/ask", json={"query": "What is the capital of India?", "k": 3, "language": "en-IN"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] != "refused"
    assert "retrieve" in body["timings_ms"]
    assert body["query_language"] == "en-IN"


def test_unsupported_language_refused_before_retrieval_or_generation():
    resp = client.post(
        "/ask", json={"query": "Bonjour le monde", "k": 3, "language": "fr-FR"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "refused"
    assert "retrieve" not in body["timings_ms"]
    assert "generate" not in body["timings_ms"]


def test_several_additional_indic_languages_reach_retrieval():
    """At least 3 more MSMARCO-XI languages beyond Hindi are allowed through G2 -- they may still
    end up abstained (only Hindi is actually indexed until Phase 2), but must never be refused
    for a language reason."""
    for code in ("bn-IN", "ta-IN", "mr-IN"):
        resp = client.post("/ask", json={"query": "test query text", "k": 3, "language": code})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] != "refused", f"{code} should not be refused: {body}"
        assert "retrieve" in body["timings_ms"], f"{code} should reach retrieval"


def test_query_language_and_retrieved_language_are_tracked_separately(monkeypatch):
    """Forces the Day-0 stub (same convention as test_api.py's stub test) for a deterministic,
    machine-independent check: query_language is the Sarvam BCP-47 code the caller supplied,
    language is the retrieved evidence's own language tag (the stub's fixed "hi") -- two
    different code spaces, two different values, never aliased onto one key."""
    import vrag.retrieval.interface as interface

    monkeypatch.setattr(interface, "_get_real_retriever", lambda: None)

    resp = client.post(
        "/ask", json={"query": "भारत में सबसे ऊँचा पर्वत कौन सा है?", "k": 3, "language": "hi-IN"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["query_language"] == "hi-IN"
    assert body["language"] == "hi"  # the stub chunk's language tag, a different code space
    assert body["query_language"] != body["language"]


def test_no_language_signal_preserves_pre_phase1_behavior():
    """A direct /ask call with no language hint at all must behave exactly as before Phase 1 --
    G2 falls back to its script heuristic, query_language is None throughout."""
    resp = client.post("/ask", json={"query": "भारत की राजधानी क्या है?", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] != "refused"
    assert body["query_language"] is None


def test_existing_hindi_answer_response_shape_unchanged():
    """Frontend compatibility: every pre-existing field is still present with the same meaning;
    query_language is purely additive."""
    resp = client.post("/ask", json={"query": "भारत की राजधानी क्या है?", "k": 3})
    body = resp.json()
    for field in (
        "status",
        "answer",
        "track",
        "citations",
        "confidence",
        "refusal_reason",
        "language",
        "stages_skipped",
        "trace_id",
        "timings_ms",
    ):
        assert field in body
    assert "query_language" in body  # new, additive
