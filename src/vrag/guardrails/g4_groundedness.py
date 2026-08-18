"""G4 — groundedness, hot-path half only. Hot path, target <5ms. See AGENT_BUILD_SPEC.md §7.3.

Two deterministic checks, run in this order because a deterministic retrieval check must gate
before any softer signal — an LLM judge (not built here) will happily rate an answer "grounded"
even when the retrieval trace is empty or citations are invented, per the trap
docs/TECH_MENU.md §S12 calls out explicitly:

1. Citation-ID validation — every cited chunk_id must exist in what was actually retrieved.
2. N-gram/lexical overlap — the answer text must share enough vocabulary with its cited passages
   to plausibly be grounded in them, not just topically related.

This is the approximation, not the full picture. The offline NLI entailment eval (Bespoke-
MiniCheck or RAGAS faithfulness) that quantifies how good this approximation actually is is joint
work with Workstream R, not built yet (docs/TEAM_SPLIT.md §5) — this file only implements the
cheap, deterministic hot-path half.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from vrag.retrieval.interface import RetrievedChunk

_WORD = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)
MIN_OVERLAP_RATIO = 0.15  # UNCALIBRATED placeholder — see docs/DECISIONS_P.md


class GuardrailVerdict(BaseModel):
    passed: bool
    reason: str | None = None


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)}


def check(
    answer: str, cited_chunks: list[RetrievedChunk], retrieved_chunk_ids: set[str]
) -> GuardrailVerdict:
    """HOTPATH — no network, no disk I/O. `retrieved_chunk_ids` is the full set actually
    retrieved this request, independent of which subset was cited, so an invented chunk_id is
    caught even if it happens to collide with a real one from a different request."""
    if not cited_chunks:
        return GuardrailVerdict(passed=False, reason="Answer has no citations to check.")

    for c in cited_chunks:
        if c.chunk_id not in retrieved_chunk_ids:
            return GuardrailVerdict(
                passed=False, reason=f"Cited chunk_id {c.chunk_id} was never retrieved."
            )

    answer_words = _tokenize(answer)
    if not answer_words:
        return GuardrailVerdict(passed=False, reason="Empty answer text.")

    context_words: set[str] = set()
    for c in cited_chunks:
        context_words |= _tokenize(c.text)

    overlap = len(answer_words & context_words) / len(answer_words)
    if overlap < MIN_OVERLAP_RATIO:
        return GuardrailVerdict(
            passed=False,
            reason=f"Low lexical overlap ({overlap:.0%}) between answer and cited context.",
        )

    return GuardrailVerdict(passed=True)
