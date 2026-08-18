"""Circuit breaker for Track B's LLM call (src/vrag/generation/sarvam_llm.py).

Motivated directly by the P-R13 outage (docs/RISKS.md): every request paid the full remaining-
budget timeout cost probing an endpoint that was going to fail anyway. This breaker skips straight
to Track A without attempting the network call once recent calls suggest the provider is currently
unhealthy, and periodically re-probes to notice recovery.

Standard three-state design (closed/open/half-open), generic — not GenerateStage-specific.

  CLOSED (normal): every request is attempted. `failure_threshold` consecutive failures -> OPEN.
  OPEN: every request is rejected immediately (no network call). After `reset_timeout_s`, the next
        request is let through as a single probe -> HALF_OPEN.
  HALF_OPEN: exactly one in-flight probe; concurrent requests during the probe window are rejected
             rather than piling onto a provider we're not yet sure has recovered. Probe success ->
             CLOSED. Probe failure -> OPEN again, timer restarts.

Process-wide singleton (`TRACK_B_BREAKER`) since `GenerateStage` is instantiated fresh per request
(`default_stages()` in stages.py) — state has to live outside the Stage object to mean anything
across requests.

What counts as a "failure" is the one genuinely non-obvious design decision here — see
`should_count_as_health_signal()`'s docstring before wiring a new call site's outcome into
`record_failure()`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    reset_timeout_s: float = 30.0
    clock: Callable[[], float] = time.monotonic

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False, repr=False)
    _consecutive_failures: int = field(default=0, init=False, repr=False)
    _opened_at: float | None = field(default=None, init=False, repr=False)

    def allow_request(self) -> bool:
        """HOTPATH-adjacent — called once per GenerateStage attempt, no I/O, no awaits, so this
        is safe to call from a single-threaded asyncio event loop without a lock: the check and
        any state transition it triggers happen atomically between await points."""
        if self._state == CircuitState.CLOSED:
            return True
        if self._state == CircuitState.HALF_OPEN:
            return False
        assert self._opened_at is not None  # only unset when state is CLOSED
        if (self.clock() - self._opened_at) >= self.reset_timeout_s:
            self._state = CircuitState.HALF_OPEN
            return True
        return False

    def record_success(self) -> None:
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        threshold_hit = self._consecutive_failures >= self.failure_threshold
        if self._state == CircuitState.HALF_OPEN or threshold_hit:
            self._state = CircuitState.OPEN
            self._opened_at = self.clock()

    @property
    def state(self) -> CircuitState:
        return self._state


def should_count_as_health_signal(timeout_s: float, min_fair_timeout_s: float) -> bool:
    """Whether an outcome (success or failure) at this `timeout_s` allowance is meaningful
    evidence about *provider* health, as opposed to evidence that *our own budget* was too tight
    for Track B's known non-streaming completion time (~1.4-15s+, docs/DECISIONS_P.md P-014).

    This distinction is the entire reason this function exists, not a stylistic nicety: under the
    default ~200ms request budget, GenerateStage's own `asyncio.wait_for` times out on nearly
    every real attempt even when Sarvam is perfectly healthy — that's the two-track design working
    as intended (docs/PROGRESS_P.md), not a signal to act on. A breaker that counted every such
    timeout as a "failure" would trip open almost immediately in ordinary production traffic and
    then stay open, which would (a) provide no real benefit over the budget/timeout mechanism
    already in place, and (b) actively break generous-budget calls made for real testing (e.g. a
    manual /ask with a large budget_ms) by rejecting them outright during the open window, for a
    reason that has nothing to do with those specific calls.

    `min_fair_timeout_s` should be set comfortably above Sarvam chat's measured P95 TTFT (858ms,
    docs/DECISIONS_P.md P-012) — enough time that the provider had a fair chance to at least start
    responding or fail fast, so an outcome at or above it is attributable to the provider, not to
    us. Below that, the outcome is inconclusive and shouldn't move the breaker either way.
    """
    return timeout_s >= min_fair_timeout_s


# Tuned for the P-R13 outage pattern (fast auth/model errors are already handled before ever
# reaching this breaker — see sarvam_llm.py; what this breaker exists for is the "hangs to 60s+"
# pattern), not for ordinary single-request retries. 3 consecutive fair-chance failures before
# opening; 30s before the next probe, long enough not to hammer a still-down endpoint every
# request but short enough to notice recovery within a normal demo session.
TRACK_B_BREAKER = CircuitBreaker(failure_threshold=3, reset_timeout_s=30.0)

# See should_count_as_health_signal()'s docstring. 2.0s is >2x Sarvam chat's measured P95 TTFT
# (858ms, P-012) — comfortably enough allowance that a failure at or above it is real evidence,
# not budget starvation.
MIN_FAIR_TIMEOUT_S = 2.0
