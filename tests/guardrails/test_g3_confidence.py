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


def test_margin_zero_means_clustered_scores_pass(monkeypatch):
    # MARGIN=0.0 is the current, deliberately-calibrated value at TAU=0.8835 (see module
    # docstring: a fine sweep found no useful non-zero MARGIN at this TAU on this corpus -- even
    # MARGIN=0.01 pushed false-refusal to 28.7%). Confirms the gate is intentionally a no-op here,
    # not silently broken -- clustered-but-all-above-tau scores now pass.
    monkeypatch.setattr(g3_confidence, "MARGIN", 0.0)
    verdict = g3_confidence.check([_chunk(0.90, "c1"), _chunk(0.89, "c2"), _chunk(0.88, "c3")])
    assert verdict.passed


def test_margin_mechanism_still_catches_ambiguous_scores_when_nonzero(monkeypatch):
    # The margin-gate mechanism itself is still correct -- verified independently of the current
    # calibrated MARGIN=0.0, since a future recalibration (e.g. after a retrieval change) could
    # find a nonzero MARGIN useful again at a different TAU.
    monkeypatch.setattr(g3_confidence, "MARGIN", 0.05)
    verdict = g3_confidence.check([_chunk(0.90, "c1"), _chunk(0.89, "c2"), _chunk(0.88, "c3")])
    assert not verdict.passed
    assert "ambiguous" in (verdict.reason or "").lower()


def test_single_chunk_above_tau_passes_no_margin_check_possible():
    verdict = g3_confidence.check([_chunk(0.92, "c1")])
    assert verdict.passed
