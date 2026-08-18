"""G3 — retrieval confidence gate. Hot path, target <1ms. See AGENT_BUILD_SPEC.md §7.3.

Mechanism: `top1_score < tau` OR `(top1 - top5) < margin`. Both thresholds are **UNCALIBRATED
placeholders** — real calibration (150 in-domain + 150 out-of-domain queries, sweep tau and
margin, pick the operating point from the false-refusal/correct-refusal tradeoff curve) is joint
work with Workstream R per docs/TEAM_SPLIT.md §5, blocked on a calibration set neither track has
built yet. The mechanism here is real and wired into the live request path; the numbers are a
documented guess, not a result.

Literature prior (docs/EVAL_PROTOCOL.md, docs/TECH_MENU.md §S3): query-document cosine similarity
typically runs ~0.30-0.55, systematically lower than query-query similarity — a threshold guessed
at 0.5+ would refuse almost everything. These placeholders are chosen with that prior in mind, not
picked out of thin air, but they are still a guess until swept against real data.
"""

from __future__ import annotations

from pydantic import BaseModel

from vrag.retrieval.interface import RetrievedChunk

TAU = 0.35  # UNCALIBRATED placeholder
MARGIN = 0.05  # UNCALIBRATED placeholder


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
