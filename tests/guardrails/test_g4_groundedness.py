from vrag.guardrails import g4_groundedness
from vrag.retrieval.interface import RetrievedChunk

_CHUNK = RetrievedChunk(
    chunk_id="c1",
    passage_id="p1",
    text="कंचनजंगा भारत में स्थित सबसे ऊँचा पर्वत है, जिसकी ऊँचाई लगभग 8,586 मीटर है।",
    score=0.9,
    language="hi",
)


def test_no_citations_fails():
    verdict = g4_groundedness.check("some answer", [], {"c1"})
    assert not verdict.passed


def test_invented_chunk_id_fails():
    verdict = g4_groundedness.check(_CHUNK.text, [_CHUNK], set())
    assert not verdict.passed
    assert "never retrieved" in (verdict.reason or "")


def test_grounded_answer_passes():
    verdict = g4_groundedness.check(_CHUNK.text, [_CHUNK], {"c1"})
    assert verdict.passed


def test_ungrounded_answer_fails_lexical_overlap():
    verdict = g4_groundedness.check(
        "यह पूरी तरह से असंबंधित एक बना हुआ वाक्य है जिसका कोई संबंध नहीं है", [_CHUNK], {"c1"}
    )
    assert not verdict.passed
    assert "overlap" in (verdict.reason or "").lower()


def test_empty_answer_fails():
    verdict = g4_groundedness.check("", [_CHUNK], {"c1"})
    assert not verdict.passed
