"""Ordered stage execution over a PipelineContext. See AGENT_BUILD_SPEC.md §7.2 items 1-2.

The runner is deliberately simple today: walk the stages in order, check the budget before each
one, skip optional stages that can't fit, run the rest, record everything. Retries (retry.py) and
the circuit breaker land in Day 2 hardening (docs/TEAM_SPLIT.md §5) — this is the shape they'll
attach to.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from vrag.harness.budget import Budget
from vrag.harness.stage import Stage, StageResult


@dataclass
class PipelineContext:
    """Append-only request-scoped state. Stages read from this and append their own
    results/outputs — never mutate another stage's prior entry."""

    query: str
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    budget: Budget = field(default_factory=lambda: Budget(total_ms=200))
    history: list[StageResult] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)  # stage outputs, keyed by stage name

    def record(self, result: StageResult, output: Any = None) -> None:
        self.history.append(result)
        if output is not None:
            self.data[result.stage_name] = output

    @property
    def stages_skipped(self) -> list[str]:
        return [r.stage_name for r in self.history if r.skipped]

    @property
    def timings_ms(self) -> dict[str, float]:
        return {r.stage_name: r.duration_ns / 1_000_000 for r in self.history if not r.skipped}


async def run_pipeline(ctx: PipelineContext, stages: list[Stage]) -> PipelineContext:
    for stage in stages:
        if stage.optional and not ctx.budget.can_afford(stage.min_viable_ms):
            ctx.record(
                StageResult(
                    stage_name=stage.name,
                    skipped=True,
                    skip_reason=f"budget exhausted: {ctx.budget.remaining_ms:.1f}ms remaining, "
                    f"needed {stage.min_viable_ms}ms",
                )
            )
            continue

        start_ns = time.perf_counter_ns()
        result = await stage.run(ctx)
        result.duration_ns = time.perf_counter_ns() - start_ns
        ctx.record(result)

    return ctx
