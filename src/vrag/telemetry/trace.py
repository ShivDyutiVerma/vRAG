"""Per-request trace records -> traces.jsonl. See AGENT_BUILD_SPEC.md §7.2 item 9.

Every request emits one TraceRecord regardless of outcome (answered/refused/abstained/degraded) —
this is what makes P50/P70/P100 reporting in Phase 6 possible, and what the latency HUD in the
frontend will eventually read from. `traces.jsonl` is gitignored (regenerated, not committed).

Emission must never sit on the hot path: `emit_trace` does a synchronous file append, so callers
fire it via `asyncio.create_task()` *after* the response has already been returned to the client,
never awaited inline before responding. See docs/CONVENTIONS.md's no-disk-I/O-on-the-hot-path rule.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from vrag.harness.pipeline import PipelineContext

_TRACE_FILE = Path(__file__).resolve().parents[3] / "traces.jsonl"


class TraceRecord(BaseModel):
    trace_id: str
    query: str
    status: str
    stages_skipped: list[str]
    timings_ms: dict[str, float]
    budget_total_ms: float
    budget_remaining_ms: float


def build_trace_record(ctx: PipelineContext, budget_total_ms: float) -> TraceRecord:
    response = ctx.data.get("answer_response")
    status = response.status if response is not None else "unknown"
    return TraceRecord(
        trace_id=ctx.trace_id,
        query=ctx.query,
        status=status,
        stages_skipped=ctx.stages_skipped,
        timings_ms=ctx.timings_ms,
        budget_total_ms=budget_total_ms,
        budget_remaining_ms=ctx.budget.remaining_ms,
    )


def emit_trace(record: TraceRecord) -> None:
    """Off the hot path by convention (see module docstring) — appends one JSON line."""
    with _TRACE_FILE.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")
