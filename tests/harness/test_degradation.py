"""The forced-tight-budget degradation test — AGENT_BUILD_SPEC.md §7.2 item 2 / P4 exit criterion:
"force a 50ms budget and assert optional stages are skipped AND the response still returns." This
is the proof of the harness's central idea (deadline propagation, AGENT_BUILD_SPEC.md §3.4): the
system sheds quality instead of blowing the deadline.

Two tests, deliberately separate:
1. `test_optional_stage_skipped_under_tight_budget` — proves the *mechanism* (Stage.optional +
   run_pipeline's budget check) using synthetic dummy stages, independent of any real feature.
2. `test_real_pipeline_survives_near_zero_budget` — proves the *real* production pipeline
   (src/vrag/harness/stages.py::default_stages()) never hangs or crashes under budget pressure.
   Nothing actually sheds in this one yet, because no stage in today's pipeline is optional —
   Track B generation (Day 3) is the first genuinely optional stage. Documented, not hidden: see
   docs/DECISIONS_P.md and the module docstring in stages.py.
"""

from __future__ import annotations

import pytest

from vrag.harness.budget import Budget
from vrag.harness.pipeline import PipelineContext, run_pipeline
from vrag.harness.stage import Stage, StageResult
from vrag.harness.stages import default_stages


class _DummyOptionalStage(Stage):
    name = "dummy_optional"
    min_viable_ms = 100.0
    optional = True

    async def run(self, ctx: PipelineContext) -> StageResult:
        ctx.data["dummy_optional_ran"] = True
        return StageResult(stage_name=self.name)


class _DummyRequiredStage(Stage):
    name = "dummy_required"
    min_viable_ms = 0.001
    optional = False

    async def run(self, ctx: PipelineContext) -> StageResult:
        ctx.data["dummy_required_ran"] = True
        return StageResult(stage_name=self.name)


@pytest.mark.asyncio
async def test_optional_stage_skipped_under_tight_budget():
    # A 50ms budget can never afford a stage whose min_viable_ms is 100ms — it must be skipped,
    # not attempted and left to blow the deadline.
    ctx = PipelineContext(query="test", budget=Budget(total_ms=50))
    await run_pipeline(ctx, [_DummyOptionalStage(), _DummyRequiredStage()])

    assert "dummy_optional" in ctx.stages_skipped
    assert ctx.data.get("dummy_optional_ran") is None
    # the non-optional stage still runs regardless of budget — only optional stages are shed
    assert ctx.data.get("dummy_required_ran") is True


@pytest.mark.asyncio
async def test_optional_stage_runs_when_budget_allows_it():
    ctx = PipelineContext(query="test", budget=Budget(total_ms=5000))
    await run_pipeline(ctx, [_DummyOptionalStage(), _DummyRequiredStage()])

    assert ctx.stages_skipped == []
    assert ctx.data.get("dummy_optional_ran") is True


@pytest.mark.asyncio
async def test_real_pipeline_survives_near_zero_budget():
    ctx = PipelineContext(
        query="भारत में सबसे ऊँचा पर्वत कौन सा है?", budget=Budget(total_ms=0.001)
    )
    await run_pipeline(ctx, default_stages())

    response = ctx.data["answer_response"]
    assert response.status in ("answered", "abstained", "refused", "degraded")
    assert response.trace_id
    assert isinstance(response.timings_ms, dict)
