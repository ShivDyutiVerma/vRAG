"""Concrete pipeline stages wiring guardrails + retrieval + Track A/B answer generation into the
Stage/PipelineContext/Budget abstraction (stage.py/pipeline.py/budget.py).

Maps onto docs/ARCHITECTURE.md's request-lifecycle diagram: stages 2 (InputGuard/G1), 2
(ScopeGuard/G2 — same numbered stage in the diagram, split into two Stage objects here since
they're independently testable and independently budgeted), 4 (Retrieve), 6 (GroundGate/G3),
7a (ExtractAnswer, Track A), 7b (Generate, Track B — includes G4's citation/overlap check before
accepting the generated answer), 8 (OutputGuard/G5), 9 (Assemble). Stages 0/1 (AudioIngest,
Transcribe) happen before the pipeline runs at all — the t_pipeline clock
(docs/EVAL_PROTOCOL.md) starts once a transcript is available, which is exactly when
`run_pipeline` is invoked.

G3 and G4's *mechanisms* are implemented and live on the request path — G3's threshold/margin and
G4's overlap ratio are UNCALIBRATED placeholders, though (see
src/vrag/guardrails/g3_confidence.py and g4_groundedness.py). Real calibration (150 in-domain +
150 out-of-domain queries, sweep and pick an operating point) is still joint work with Workstream
R, blocked on a calibration set neither track has built yet.

Guardrail failures short-circuit via `ctx.data["refused"]`/`ctx.data["abstained"]` rather than
raising — every downstream stage checks that flag first and marks itself skipped
("upstream refusal") if set. This is deliberately different from budget-based skipping
(`stage.optional` + `run_pipeline`'s check): a refusal is a content decision, not a time-pressure
decision, and both end up in the same `stages_skipped` list because from the client's perspective
"didn't run" is the fact that matters, regardless of why.

`GenerateStage` (Track B) is the first genuinely optional stage in this pipeline — every other
stage is load-bearing. Under a tight budget, `run_pipeline` sheds it and Track A's
already-computed answer stands as the response: this is the two-track design
(AGENT_BUILD_SPEC.md §3.3) actually operating, not just proven in isolation
(tests/harness/test_degradation.py).
"""

from __future__ import annotations

import asyncio
import logging

from vrag.generation.sarvam_llm import generate as generate_track_b
from vrag.guardrails import (
    g1_input_safety,
    g2_scope_language,
    g3_confidence,
    g4_groundedness,
    g5_output_safety,
)
from vrag.harness.pipeline import PipelineContext
from vrag.harness.stage import Stage, StageResult
from vrag.retrieval.interface import retrieve
from vrag.schemas import AnswerResponse, Citation

logger = logging.getLogger(__name__)


def _upstream_refused(ctx: PipelineContext) -> bool:
    return bool(ctx.data.get("refused") or ctx.data.get("abstained"))


class InputGuardStage(Stage):
    """G1 — input safety. See AGENT_BUILD_SPEC.md §7.3."""

    name = "input_guard"
    min_viable_ms = 2.0
    optional = False

    async def run(self, ctx: PipelineContext) -> StageResult:
        verdict = g1_input_safety.check(ctx.query)
        if not verdict.passed:
            ctx.data["refused"] = True
            ctx.data["refusal_reason"] = verdict.reason
            ctx.data["refusal_layer"] = "G1"
        return StageResult(stage_name=self.name)


class ScopeGuardStage(Stage):
    """G2 — scope & language. See AGENT_BUILD_SPEC.md §7.3."""

    name = "scope_guard"
    min_viable_ms = 1.0
    optional = False

    async def run(self, ctx: PipelineContext) -> StageResult:
        if _upstream_refused(ctx):
            return StageResult(stage_name=self.name, skipped=True, skip_reason="upstream refusal")
        verdict = g2_scope_language.check(ctx.query)
        if not verdict.passed:
            ctx.data["refused"] = True
            ctx.data["refusal_reason"] = verdict.reason
            ctx.data["refusal_layer"] = "G2"
        return StageResult(stage_name=self.name)


class RetrieveStage(Stage):
    """Calls the R/P seam. `retrieve()` never raises by contract (returns [] on internal
    failure), so there is deliberately no retry wrapping here — a raise-based retry policy has
    nothing to catch. See docs/DECISIONS_P.md for why tenacity's retry policy instead targets
    Track B's LLM call once that exists."""

    name = "retrieve"
    min_viable_ms = 50.0  # placeholder until Phase 6 measures the real number
    optional = False

    async def run(self, ctx: PipelineContext) -> StageResult:
        if _upstream_refused(ctx):
            return StageResult(stage_name=self.name, skipped=True, skip_reason="upstream refusal")
        k = ctx.data.get("k", 5)
        try:
            chunks = await retrieve(ctx.query, k=k)
        except Exception:
            logger.exception("retrieve() raised despite its never-raises contract")
            chunks = []
        ctx.data["chunks"] = chunks
        return StageResult(stage_name=self.name)


class GroundGateStage(Stage):
    """G3 — retrieval confidence gate. See AGENT_BUILD_SPEC.md §7.3.

    UNCALIBRATED thresholds — see src/vrag/guardrails/g3_confidence.py and docs/DECISIONS_P.md.
    The mechanism runs for real; the tau/margin numbers are a documented placeholder pending
    joint calibration with Workstream R.
    """

    name = "ground_gate"
    min_viable_ms = 1.0
    optional = False

    async def run(self, ctx: PipelineContext) -> StageResult:
        if _upstream_refused(ctx):
            return StageResult(stage_name=self.name, skipped=True, skip_reason="upstream refusal")
        verdict = g3_confidence.check(ctx.data.get("chunks", []))
        if not verdict.passed:
            ctx.data["abstained"] = True
            ctx.data["refusal_reason"] = verdict.reason
        return StageResult(stage_name=self.name)


class ExtractAnswerStage(Stage):
    """Track A — select the best-supporting span. See AGENT_BUILD_SPEC.md §3.3.

    G3 (GroundGateStage) already ran before this — an empty/failed-confidence retrieval never
    reaches here, so this stage can assume `chunks` is non-empty whenever it actually runs.
    """

    name = "extract_answer"
    min_viable_ms = 10.0
    optional = False

    async def run(self, ctx: PipelineContext) -> StageResult:
        if _upstream_refused(ctx):
            return StageResult(stage_name=self.name, skipped=True, skip_reason="upstream refusal")
        chunks = ctx.data.get("chunks", [])
        if not chunks:
            ctx.data["abstained"] = True
            ctx.data["refusal_reason"] = "No relevant passages found for this query."
        else:
            top = chunks[0]
            ctx.data["answer_text"] = top.text
            ctx.data["citations"] = list(chunks[:2])
            ctx.data["confidence"] = top.score
            ctx.data["language"] = top.language
        return StageResult(stage_name=self.name)


class GenerateStage(Stage):
    """Track B — LLM synthesises a fluent, cited answer over the retrieved context. See
    AGENT_BUILD_SPEC.md §3.3. The first genuinely optional stage in this pipeline: if the budget
    can't afford it, it's shed and Track A's answer (already computed by ExtractAnswerStage)
    stands as the response — never a missing answer, just a less polished one.

    G4 groundedness (src/vrag/guardrails/g4_groundedness.py) gates the generated answer before
    it's accepted: citation-ID validation catches an invented chunk_id, lexical overlap catches
    an answer that's drifted from its cited context. Fails either check → Track A's
    already-computed answer (from ExtractAnswerStage) stands unchanged, per
    AGENT_BUILD_SPEC.md §7.3's G4 failure action ("drop to Track A, or abstained" — this always
    has Track A to drop to, since it runs first).

    `min_viable_ms` is only a pre-flight check (run_pipeline decides whether to *start* this
    stage) — it does not by itself stop the stage from overrunning once started. Found this the
    hard way: with Sarvam's chat endpoint hanging (docs/RISKS.md P-R13), an unguarded call here
    blew a 200ms budget by 10+ real seconds. `asyncio.wait_for` against the *actual* remaining
    budget closes that gap — deadline propagation has to hold during a stage, not just before it.
    """

    name = "generate"
    min_viable_ms = 110.0  # AGENT_BUILD_SPEC.md §4 row 8b: Track B TTFT target
    optional = True

    async def run(self, ctx: PipelineContext) -> StageResult:
        if _upstream_refused(ctx):
            return StageResult(stage_name=self.name, skipped=True, skip_reason="upstream refusal")
        chunks = ctx.data.get("chunks", [])
        if not chunks:
            return StageResult(
                stage_name=self.name, skipped=True, skip_reason="no context to generate over"
            )

        timeout_s = ctx.budget.remaining_ms / 1000
        try:
            result = await asyncio.wait_for(
                generate_track_b(ctx.query, chunks), timeout=max(timeout_s, 0.001)
            )
        except TimeoutError:
            return StageResult(
                stage_name=self.name,
                skipped=True,
                skip_reason=f"generation exceeded remaining budget ({timeout_s:.3f}s)",
            )
        if result is None:
            return StageResult(
                stage_name=self.name, skipped=True, skip_reason="generation failed, kept Track A"
            )

        valid_chunks_by_id = {c.chunk_id: c for c in chunks}
        cited = [
            valid_chunks_by_id[cid]
            for cid in result.cited_chunk_ids
            if cid in valid_chunks_by_id
        ]

        verdict = g4_groundedness.check(result.answer, cited, set(valid_chunks_by_id))
        if not verdict.passed:
            logger.info("Track B answer failed G4 (%s), keeping Track A", verdict.reason)
            return StageResult(
                stage_name=self.name,
                skipped=True,
                skip_reason=f"failed G4 groundedness ({verdict.reason}), kept Track A",
            )

        ctx.data["answer_text"] = result.answer
        ctx.data["citations"] = cited
        ctx.data["track"] = "generative"
        return StageResult(stage_name=self.name)


class OutputGuardStage(Stage):
    """G5 — output safety / PII redaction. See AGENT_BUILD_SPEC.md §7.3.

    G4 groundedness already ran inside GenerateStage (gating whether Track B's answer was
    accepted at all) — this stage only redacts, it doesn't re-check groundedness.
    """

    name = "output_guard"
    min_viable_ms = 2.0
    optional = False

    async def run(self, ctx: PipelineContext) -> StageResult:
        if _upstream_refused(ctx):
            return StageResult(stage_name=self.name, skipped=True, skip_reason="no answer to check")
        result = g5_output_safety.redact(ctx.data.get("answer_text", ""))
        ctx.data["answer_text"] = result.text
        ctx.data["output_redacted"] = result.redacted
        return StageResult(stage_name=self.name)


class AssembleStage(Stage):
    """Builds the final AnswerResponse from whatever the earlier stages left in ctx.data."""

    name = "assemble"
    min_viable_ms = 0.5
    optional = False

    async def run(self, ctx: PipelineContext) -> StageResult:
        if ctx.data.get("refused"):
            response = AnswerResponse(
                status="refused",
                answer=None,
                track="extractive",
                citations=[],
                confidence=0.0,
                refusal_reason=ctx.data.get("refusal_reason"),
                language="hi",
                stages_skipped=ctx.stages_skipped,
                trace_id=ctx.trace_id,
                timings_ms=ctx.timings_ms,
            )
        elif ctx.data.get("abstained"):
            response = AnswerResponse(
                status="abstained",
                answer=None,
                track="extractive",
                citations=[],
                confidence=0.0,
                refusal_reason=ctx.data.get("refusal_reason"),
                language="hi",
                stages_skipped=ctx.stages_skipped,
                trace_id=ctx.trace_id,
                timings_ms=ctx.timings_ms,
            )
        else:
            response = AnswerResponse(
                status="answered",
                answer=ctx.data.get("answer_text"),
                track=ctx.data.get("track", "extractive"),
                citations=[
                    Citation(
                        chunk_id=c.chunk_id,
                        passage_id=c.passage_id,
                        score=c.score,
                        text_span=c.text,
                    )
                    for c in ctx.data.get("citations", [])
                ],
                confidence=ctx.data.get("confidence", 0.0),
                refusal_reason=None,
                language=ctx.data.get("language", "hi"),
                stages_skipped=ctx.stages_skipped,
                trace_id=ctx.trace_id,
                timings_ms=ctx.timings_ms,
            )
        ctx.data["answer_response"] = response
        return StageResult(stage_name=self.name)


def default_stages() -> list[Stage]:
    """The real pipeline: G1 -> G2 -> Retrieve -> G3 -> Track A -> Track B (optional, gated by
    G4) -> G5 -> Assemble. Track B is the only stage that can actually be shed under budget
    pressure."""
    return [
        InputGuardStage(),
        ScopeGuardStage(),
        RetrieveStage(),
        GroundGateStage(),
        ExtractAnswerStage(),
        GenerateStage(),
        OutputGuardStage(),
        AssembleStage(),
    ]
