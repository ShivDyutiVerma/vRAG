"""Concrete pipeline stages wiring guardrails + retrieval + Track A answer selection into the
Stage/PipelineContext/Budget abstraction (stage.py/pipeline.py/budget.py).

Maps onto docs/ARCHITECTURE.md's request-lifecycle diagram: stages 2 (InputGuard/G1), 2
(ScopeGuard/G2 — same numbered stage in the diagram, split into two Stage objects here since
they're independently testable and independently budgeted), 4 (Retrieve), 7a (ExtractAnswer,
Track A), 8 (OutputGuard/G5), 9 (Assemble). Stages 0/1 (AudioIngest, Transcribe) happen before the
pipeline runs at all — the t_pipeline clock (docs/EVAL_PROTOCOL.md) starts once a transcript is
available, which is exactly when `run_pipeline` is invoked. Stage 6 (GroundGate/G3) and stage 8's
groundedness half (G4) aren't implemented yet — joint work with Workstream R, scheduled Day 3 per
docs/TEAM_SPLIT.md §5, since G3 calibration needs real retrieval scores and G4 needs a real answer
to check groundedness against.

Guardrail failures short-circuit via `ctx.data["refused"]`/`ctx.data["abstained"]` rather than
raising — every downstream stage checks that flag first and marks itself skipped
("upstream refusal") if set. This is deliberately different from budget-based skipping
(`stage.optional` + `run_pipeline`'s check): a refusal is a content decision, not a time-pressure
decision, and both end up in the same `stages_skipped` list because from the client's perspective
"didn't run" is the fact that matters, regardless of why.

No stage here is `optional = False` yet — every one of G1/G2/Retrieve/ExtractAnswer/G5/Assemble is
load-bearing; there's nothing to shed. The spec's first genuinely optional stage is Track B
generation (AGENT_BUILD_SPEC.md §4, row 8b) — that lands with generation on Day 3. See
tests/test_harness_degradation.py for why the forced-tight-budget test still matters today even
though nothing sheds yet: it proves the pipeline degrades to `degraded` rather than crashing when
the mechanism has nothing to shed, which is the honest baseline Track B's optionality builds on.
"""

from __future__ import annotations

import logging

from vrag.guardrails import g1_input_safety, g2_scope_language, g5_output_safety
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


class ExtractAnswerStage(Stage):
    """Track A — select the best-supporting span. See AGENT_BUILD_SPEC.md §3.3.

    No G3 confidence gate yet (joint work, Day 3) — today, an empty retrieval result is the only
    abstention trigger. A real G3 threshold-and-margin gate will sit here once calibrated.
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


class OutputGuardStage(Stage):
    """G5 — output safety / PII redaction. See AGENT_BUILD_SPEC.md §7.3.

    No G4 groundedness check yet (joint work, Day 3) — this stage only redacts, it doesn't verify
    citation validity or lexical overlap yet.
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
                track="extractive",
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
    """The real Day-2 pipeline: G1 -> G2 -> Retrieve -> Track A -> G5 -> Assemble."""
    return [
        InputGuardStage(),
        ScopeGuardStage(),
        RetrieveStage(),
        ExtractAnswerStage(),
        OutputGuardStage(),
        AssembleStage(),
    ]
