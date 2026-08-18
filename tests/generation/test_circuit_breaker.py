"""Unit tests for the CircuitBreaker state machine, in isolation from GenerateStage/Sarvam.
Uses an injectable fake clock (not real time.sleep) so state transitions are tested
deterministically and fast — see CircuitBreaker.clock in src/vrag/generation/circuit_breaker.py.

Each test builds its own CircuitBreaker instance rather than touching the module-level
TRACK_B_BREAKER singleton, so these tests never interact with GenerateStage's real state.
"""

from __future__ import annotations

from vrag.generation.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    should_count_as_health_signal,
)


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_starts_closed_and_allows_requests():
    breaker = CircuitBreaker()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.allow_request() is True


def test_stays_closed_below_failure_threshold():
    breaker = CircuitBreaker(failure_threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.allow_request() is True


def test_opens_after_consecutive_failures_reach_threshold():
    breaker = CircuitBreaker(failure_threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN
    assert breaker.allow_request() is False


def test_success_resets_consecutive_failure_count():
    breaker = CircuitBreaker(failure_threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    # Two prior failures don't carry over — takes a fresh run of 3 to open.
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.allow_request() is True


def test_open_rejects_until_reset_timeout_elapses():
    clock = _FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout_s=30.0, clock=clock)
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN

    clock.advance(29.9)
    assert breaker.allow_request() is False
    assert breaker.state == CircuitState.OPEN

    clock.advance(0.2)  # total 30.1s, past the reset timeout
    assert breaker.allow_request() is True
    assert breaker.state == CircuitState.HALF_OPEN


def test_half_open_rejects_concurrent_probes():
    clock = _FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout_s=10.0, clock=clock)
    breaker.record_failure()
    clock.advance(10.0)

    assert breaker.allow_request() is True  # the one probe
    assert breaker.state == CircuitState.HALF_OPEN
    assert breaker.allow_request() is False  # a second concurrent request is rejected


def test_half_open_success_closes_the_breaker():
    clock = _FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout_s=10.0, clock=clock)
    breaker.record_failure()
    clock.advance(10.0)
    breaker.allow_request()  # enters HALF_OPEN

    breaker.record_success()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.allow_request() is True


def test_half_open_failure_reopens_and_restarts_timer():
    clock = _FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout_s=10.0, clock=clock)
    breaker.record_failure()
    clock.advance(10.0)
    breaker.allow_request()  # enters HALF_OPEN

    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN

    clock.advance(9.9)
    assert breaker.allow_request() is False  # timer restarted, not still counting from before
    clock.advance(0.2)
    assert breaker.allow_request() is True


def test_should_count_as_health_signal_below_floor_is_inconclusive():
    assert should_count_as_health_signal(0.05, min_fair_timeout_s=2.0) is False
    assert should_count_as_health_signal(1.99, min_fair_timeout_s=2.0) is False


def test_should_count_as_health_signal_at_or_above_floor_is_conclusive():
    assert should_count_as_health_signal(2.0, min_fair_timeout_s=2.0) is True
    assert should_count_as_health_signal(15.0, min_fair_timeout_s=2.0) is True
