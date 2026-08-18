"""G3 — retrieval confidence gate. Hot path, target <1ms. See AGENT_BUILD_SPEC.md §7.3.

Mechanism: `top1_score < tau` OR `(top1 - top5) < margin`.

Calibrated 2026-08-18 against real data: 300 queries (150 in-domain + 150 out-of-domain, drawn
from ai4bharat/MSMARCO-XI past the indexed 10k-row cutoff) scored against the real production
index (dense-only, e5-small, efSearch=64). Full sweep, root-cause analysis and reference-point
table: docs/DECISIONS_R.md R-015, docs/DECISIONS_P.md P-015, docs/assets/g3_calibration.png.

Headline finding: docs/EVAL_PROTOCOL.md's original targets (false-refusal<10% AND
correct-refusal>80%) are **not simultaneously reachable** via top1-cosine TAU gating on this
corpus — MSMARCO-XI passages recur across many query_ids, so even genuinely out-of-index queries
often retrieve a topically-close or coincidentally-correct passage. TAU=0.8835 is the operating
point weighing both EVAL_PROTOCOL.md targets equally (joint P/R decision, see P-015): 19.3%
false-refusal, 75.3% correct-refusal -- AT MARGIN=0.0.

**MARGIN fixed 2026-08-18 (same day), post-merge: the pre-calibration placeholder MARGIN=0.05
does not carry over to this TAU.** Verified live and by direct sweep: at TAU=0.8835, in-domain
top1-vs-top5 gaps are naturally tiny (this operating point sits in a narrow, tightly-clustered part
of the score distribution -- see R-015), so MARGIN=0.05 pushed false-refusal to 88% in practice, not
the 19.3% this docstring's own design target states. A fine sweep (0.0 to 0.05) shows false-refusal
degrades steeply even at MARGIN=0.01 (28.7%) -- there is no useful non-zero MARGIN at this TAU on
this corpus; MARGIN=0.0 (which structurally disables the gate, since top1-weakest can't be negative)
is the empirically-correct pairing, not a shortcut. If TAU is ever recalibrated, MARGIN must be
re-swept at the new value too -- the two are not independent.
"""

from __future__ import annotations

from pydantic import BaseModel

from vrag.retrieval.interface import RetrievedChunk

TAU = 0.8835  # Calibrated 2026-08-18, see P-015 / R-015. Not re-verified after retrieval changes.
MARGIN = 0.0  # Calibrated 2026-08-18 AT this TAU (see docstring) -- 0.05 caused 88% false-refusal.


class GuardrailVerdict(BaseModel):
    passed: bool
    reason: str | None = None


def check(chunks: list[RetrievedChunk]) -> GuardrailVerdict:
    """HOTPATH — no network, no disk I/O."""
    if not chunks:
        return GuardrailVerdict(passed=False, reason="No passages retrieved.")

    top1 = chunks[0].score
    if top1 < TAU:
        return GuardrailVerdict(
            passed=False, reason=f"Top result confidence {top1:.2f} is below threshold {TAU}."
        )

    if len(chunks) >= 2:
        # top5 if we retrieved that many, else the weakest chunk we actually got back
        weakest = chunks[min(4, len(chunks) - 1)].score
        if (top1 - weakest) < MARGIN:
            return GuardrailVerdict(
                passed=False, reason="Ambiguous match: top result doesn't clearly stand out."
            )

    return GuardrailVerdict(passed=True)
