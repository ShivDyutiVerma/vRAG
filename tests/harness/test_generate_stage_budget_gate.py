"""Proves the R-036 fix: GenerateStage's pre-flight gate (run_pipeline's can_afford() check,
src/vrag/harness/pipeline.py) uses circuit_breaker.MIN_FAIR_TIMEOUT_S (2.0s), not the old
110ms aspirational TTFT target -- so under the real 200ms default budget, Track B is shed
*before* it starts, not attempted-and-timed-out. Track A's already-computed answer is what the
client actually waits on now, not a ~2s doomed wait.

Tests here always go through run_pipeline() + default_stages(), not GenerateStage().run()
directly -- the gate under test lives in run_pipeline's pre-flight check, which a direct
GenerateStage().run() call (as tests/harness/test_generate_stage_circuit_breaker.py's tests do,
deliberately, to test what happens *inside* run()) bypasses entirely. That file is unmodified by
this change and still passes -- proof #5 below.
"""

from __future__ import annotations

import time

import pytest

import vrag.retrieval.interface as interface_module
from vrag.generation import circuit_breaker as cb
from vrag.harness.budget import Budget
from vrag.harness.pipeline import PipelineContext, run_pipeline
from vrag.harness.stages import GenerateStage, default_stages


@pytest.fixture(autouse=True)
def _isolated_breaker(monkeypatch):
    """Same isolation rationale as test_generate_stage_circuit_breaker.py's module docstring --
    don't let these tests read or mutate the real module-level TRACK_B_BREAKER singleton."""
    monkeypatch.setattr(cb, "TRACK_B_BREAKER", cb.CircuitBreaker())


@pytest.fixture(autouse=True)
def _reset_lazy_retriever_singleton():
    """RetrieveStage calls the real retrieve() R/P seam -- forced to the Day-0 stub below so
    these tests exercise real, deterministic, real-timed retrieval scores without depending on
    a local index being present. Same reset pattern as tests/retrieval/test_interface_loading.py."""
    interface_module._retriever = None
    interface_module._retriever_load_attempted = False
    interface_module._warmup_ok = None
    yield
    interface_module._retriever = None
    interface_module._retriever_load_attempted = False
    interface_module._warmup_ok = None


def _force_stub_retrieval(monkeypatch) -> None:
    monkeypatch.setattr(interface_module, "_get_real_retriever", lambda: None)


@pytest.mark.asyncio
async def test_track_b_shed_immediately_under_200ms_budget_not_after_a_2s_wait(monkeypatch):
    """Proof #1. Query text matches a real stub chunk (score 0.91 > G3's 0.8835 TAU) so the
    pipeline reaches GenerateStage rather than aborting earlier at G3."""
    _force_stub_retrieval(monkeypatch)

    called = False

    async def _should_never_be_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("generate_track_b must not be called: budget can't afford it")

    monkeypatch.setattr("vrag.harness.stages.generate_track_b", _should_never_be_called)

    ctx = PipelineContext(
        query="भारत में सबसे ऊँचा पर्वत कौन सा है", budget=Budget(total_ms=200)
    )
    t0 = time.perf_counter()
    await run_pipeline(ctx, default_stages())
    elapsed_s = time.perf_counter() - t0

    assert called is False
    # generous margin over the real stage costs (a few ms) -- old behavior would have blocked
    # for up to ~2s (MIN_FAIR_TIMEOUT_S) waiting on a call that was never even reachable here.
    assert elapsed_s < 1.0, f"pipeline took {elapsed_s:.3f}s -- Track B's old doomed wait is back"


@pytest.mark.asyncio
async def test_track_a_answer_still_returned_when_track_b_is_shed(monkeypatch):
    """Proof #2."""
    _force_stub_retrieval(monkeypatch)
    monkeypatch.setattr(
        "vrag.harness.stages.generate_track_b",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not be called")),
    )

    ctx = PipelineContext(
        query="भारत में सबसे ऊँचा पर्वत कौन सा है", budget=Budget(total_ms=200)
    )
    await run_pipeline(ctx, default_stages())

    response = ctx.data["answer_response"]
    assert response.status == "answered"
    assert response.track == "extractive"
    assert response.answer
    assert len(response.citations) >= 1


@pytest.mark.asyncio
async def test_stages_skipped_and_skip_reason_reflect_the_new_threshold(monkeypatch):
    """Proof #3. skip_reason should cite ~2000ms (MIN_FAIR_TIMEOUT_S*1000), not the old 110ms."""
    _force_stub_retrieval(monkeypatch)

    ctx = PipelineContext(
        query="भारत में सबसे ऊँचा पर्वत कौन सा है", budget=Budget(total_ms=200)
    )
    await run_pipeline(ctx, default_stages())

    assert "generate" in ctx.stages_skipped
    generate_result = next(r for r in ctx.history if r.stage_name == "generate")
    assert generate_result.skipped is True
    assert "budget exhausted" in (generate_result.skip_reason or "")
    assert "2000" in (generate_result.skip_reason or "")


@pytest.mark.asyncio
async def test_track_b_still_attempted_when_budget_is_genuinely_generous(monkeypatch):
    """Proof #4. A caller with budget_ms comfortably above MIN_FAIR_TIMEOUT_S*1000 must still
    reach GenerateStage -- this isn't a disguised removal of Track B, only a gate on when it's
    worth starting."""
    _force_stub_retrieval(monkeypatch)

    called = False

    async def _fake_generate(*args, **kwargs):
        nonlocal called
        called = True
        return None  # fast explicit failure -- only "was it attempted" matters here

    monkeypatch.setattr("vrag.harness.stages.generate_track_b", _fake_generate)

    ctx = PipelineContext(
        query="भारत में सबसे ऊँचा पर्वत कौन सा है", budget=Budget(total_ms=5_000)
    )
    await run_pipeline(ctx, default_stages())

    # _fake_generate simulates a fast explicit provider failure (returns None), so the stage still
    # ends up skipped -- for "generation failed", never reaching the network. What this proves is
    # narrower and more important: it was actually ATTEMPTED (called=True) and the skip, if any,
    # is not attributable to the pre-flight budget gate this fix changed.
    assert called is True
    generate_result = next(r for r in ctx.history if r.stage_name == "generate")
    assert "budget exhausted" not in (generate_result.skip_reason or "")


@pytest.mark.asyncio
async def test_direct_generate_stage_timeout_and_fallback_behavior_is_unchanged():
    """Proof #5, narrow direct check complementing the unmodified
    test_generate_stage_circuit_breaker.py (still 5/5 passing, proof this fix touched nothing
    inside GenerateStage.run() itself). Calls .run() directly, bypassing the pre-flight gate --
    the internal asyncio.wait_for/timeout/fallback path this exercises is untouched by R-036."""
    from vrag.retrieval.interface import RetrievedChunk

    async def _hangs_past_the_timeout(*args, **kwargs):
        import asyncio

        await asyncio.sleep(10)
        raise AssertionError("should have been cancelled by wait_for long before this")

    import vrag.harness.stages as stages_module

    original = stages_module.generate_track_b
    stages_module.generate_track_b = _hangs_past_the_timeout
    try:
        ctx = PipelineContext(query="test", budget=Budget(total_ms=150))
        ctx.data["chunks"] = [
            RetrievedChunk(chunk_id="c1", passage_id="p1", text="context", score=0.9, language="hi")
        ]
        t0 = time.perf_counter()
        result = await GenerateStage().run(ctx)
        elapsed_s = time.perf_counter() - t0
    finally:
        stages_module.generate_track_b = original

    assert result.skipped is True
    assert "exceeded remaining budget" in (result.skip_reason or "")
    assert elapsed_s < 1.0  # cut by the real ~150ms remaining budget, not the 10s sleep
