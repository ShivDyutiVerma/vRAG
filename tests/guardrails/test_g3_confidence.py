from vrag.guardrails import g3_confidence
from vrag.retrieval.interface import RetrievedChunk


def _chunk(score: float, chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, passage_id="p1", text="some text", score=score, language="hi"
    )


def test_empty_retrieval_fails():
    verdict = g3_confidence.check([])
    assert not verdict.passed


def test_high_confidence_clear_winner_passes():
    verdict = g3_confidence.check([_chunk(0.91, "c1"), _chunk(0.74, "c2"), _chunk(0.52, "c3")])
    assert verdict.passed


def test_low_top1_confidence_fails():
    verdict = g3_confidence.check([_chunk(0.10, "c1"), _chunk(0.08, "c2")])
    assert not verdict.passed
    assert "threshold" in (verdict.reason or "").lower()


def test_ambiguous_close_scores_fail_margin_check():
    # top1 clears tau but is barely distinguishable from the rest — should be treated as
    # ambiguous even though no individual score is "low"
    verdict = g3_confidence.check([_chunk(0.90, "c1"), _chunk(0.89, "c2"), _chunk(0.88, "c3")])
    assert not verdict.passed
    assert "ambiguous" in (verdict.reason or "").lower()


def test_single_chunk_above_tau_passes_no_margin_check_possible():
    verdict = g3_confidence.check([_chunk(0.92, "c1")])
    assert verdict.passed
