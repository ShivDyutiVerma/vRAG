"""Canonical response schema. See AGENT_BUILD_SPEC.md §7.2.

Reasoning must precede the answer field wherever a schema asks the model to produce both — it
gives the model somewhere to "think" before it commits to a final answer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Citation(BaseModel):
    chunk_id: str
    passage_id: str
    score: float
    text_span: str


class AnswerResponse(BaseModel):
    status: Literal["answered", "abstained", "refused", "degraded"]
    answer: str | None
    track: Literal["extractive", "generative"]
    citations: list[Citation]
    confidence: float  # calibrated, 0-1
    refusal_reason: str | None
    language: str  # the answer/content's language — retrieved chunk's language when answered
    # Additive, Phase 1 (docs/DECISIONS.md ADR-009) — the real Sarvam-detected query language
    # (BCP-47, e.g. "hi-IN"), None when no real STT signal exists (e.g. a direct /ask call).
    # Distinct from `language` above on purpose: a query's language and the language of the
    # evidence that answered it are not the same field, see src/vrag/languages.py.
    query_language: str | None = None
    stages_skipped: list[str]  # deadline-shed stages
    trace_id: str
    timings_ms: dict[str, float]  # per-stage, always populated
