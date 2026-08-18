"""Proves GenerateStage actually consults the circuit breaker (src/vrag/generation/
circuit_breaker.py), not just that the breaker class works in isolation (see
tests/generation/test_circuit_breaker.py for that).

Each test injects its own fresh CircuitBreaker via monkeypatch rather than touching the real
module-level TRACK_B_BREAKER singleton — that singleton accumulates real state across real /ask
requests (including other tests in this suite that hit the live Sarvam API, see test_api.py's
module docstring), so tests asserting specific breaker transitions need full isolation from it.
"""

from __future__ import annotations

import pytest

from vrag.generation import circuit_breaker as cb
from vrag.harness.budget import Budget
from vrag.harness.pipeline import PipelineContext
from vrag.harness.stages import GenerateStage
from vrag.retrieval.interface import RetrievedChunk


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1", passage_id="p1", text="some real context text", score=0.9, language="hi"
    )


def _ctx(budget_ms: float = 15_000) -> PipelineContext:
    ctx = PipelineContext(query="test query", budget=Budget(total_ms=budget_ms))
    ctx.data["chunks"] = [_chunk()]
    return ctx


@pytest.mark.asyncio
async def test_open_breaker_skips_without_attempting_network_call(monkeypatch):
    fresh = cb.CircuitBreaker(failure_threshold=1, reset_timeout_s=9999.0)
    fresh.record_failure()  # one failure trips it open (threshold=1)
    monkeypatch.setattr(cb, "TRACK_B_BREAKER", fresh)

    called = False

    async def _should_never_be_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("generate_track_b must not be called while the breaker is open")

    monkeypatch.setattr("vrag.harness.stages.generate_track_b", _should_never_be_called)

    result = await GenerateStage().run(_ctx())

    assert called is False
    assert result.skipped is True
    assert "circuit breaker" in (result.skip_reason or "").lower()


@pytest.mark.asyncio
async def test_closed_breaker_attempts_the_call(monkeypatch):
    monkeypatch.setattr(cb, "TRACK_B_BREAKER", cb.CircuitBreaker())

    called = False

    async def _fake_generate(*args, **kwargs):
        nonlocal called
        called = True
        return None  # simulate a fast, explicit provider failure

    monkeypatch.setattr("vrag.harness.stages.generate_track_b", _fake_generate)

    result = await GenerateStage().run(_ctx())

    assert called is True
    assert result.skipped is True
    assert "generation failed" in (result.skip_reason or "")


@pytest.mark.asyncio
async def test_fair_chance_failure_is_recorded_against_the_breaker(monkeypatch):
    fresh = cb.CircuitBreaker(failure_threshold=1, reset_timeout_s=9999.0)
    monkeypatch.setattr(cb, "TRACK_B_BREAKER", fresh)

    async def _fake_generate(*args, **kwargs):
        return None

    monkeypatch.setattr("vrag.harness.stages.generate_track_b", _fake_generate)

    # 15s budget is well above MIN_FAIR_TIMEOUT_S (2.0s) -- a "fair chance" attempt.
    await GenerateStage().run(_ctx(budget_ms=15_000))

    assert fresh.state == cb.CircuitState.OPEN


@pytest.mark.asyncio
async def test_tight_budget_failure_is_not_recorded_against_the_breaker(monkeypatch):
    fresh = cb.CircuitBreaker(failure_threshold=1, reset_timeout_s=9999.0)
    monkeypatch.setattr(cb, "TRACK_B_BREAKER", fresh)

    async def _fake_generate(*args, **kwargs):
        return None

    monkeypatch.setattr("vrag.harness.stages.generate_track_b", _fake_generate)

    # 150ms budget is below MIN_FAIR_TIMEOUT_S (2.0s) -- expected two-track shedding, not a
    # provider-health signal. See circuit_breaker.should_count_as_health_signal's docstring.
    await GenerateStage().run(_ctx(budget_ms=150))

    assert fresh.state == cb.CircuitState.CLOSED


@pytest.mark.asyncio
async def test_success_closes_an_open_breaker(monkeypatch):
    fresh = cb.CircuitBreaker(failure_threshold=1, reset_timeout_s=0.0)  # reset instantly
    fresh.record_failure()
    monkeypatch.setattr(cb, "TRACK_B_BREAKER", fresh)

    # Chunk text and answer share vocabulary on purpose, so G4's lexical-overlap check passes and
    # this test can assert on the stage's actual outcome (not skipped), not just the breaker.
    ctx = _ctx()
    ctx.data["chunks"] = [
        RetrievedChunk(
            chunk_id="c1", passage_id="p1", text="असली उत्तर यहाँ है", score=0.9, language="hi"
        )
    ]

    class _FakeResult:
        answer = "असली उत्तर"
        cited_chunk_ids = ["c1"]

    async def _fake_generate(*args, **kwargs):
        return _FakeResult()

    monkeypatch.setattr("vrag.harness.stages.generate_track_b", _fake_generate)

    result = await GenerateStage().run(ctx)

    assert fresh.state == cb.CircuitState.CLOSED
    assert result.skipped is False
