"""Deadline propagation — the harness's killer feature. See AGENT_BUILD_SPEC.md §3.4.

Every request carries a budget in milliseconds. Each stage checks remaining_ms before running; if
it can't fit and is optional, it's skipped and recorded rather than letting the request blow its
deadline. Full stage-skipping wiring lands with the pipeline runner in Day 2 — this module owns
just the budget bookkeeping so it can be unit-tested in isolation.
"""

from __future__ import annotations

import time


class Budget:
    """Tracks a request's remaining time budget. All internal math is in nanoseconds;
    convert to ms only when reading `remaining_ms` for logging/decisions."""

    def __init__(self, total_ms: float) -> None:
        self._deadline_ns = time.perf_counter_ns() + int(total_ms * 1_000_000)

    @property
    def remaining_ms(self) -> float:
        # HOTPATH
        remaining_ns = self._deadline_ns - time.perf_counter_ns()
        return max(0.0, remaining_ns / 1_000_000)

    def can_afford(self, min_viable_ms: float) -> bool:
        # HOTPATH
        return self.remaining_ms >= min_viable_ms

    def is_exhausted(self) -> bool:
        # HOTPATH
        return self.remaining_ms <= 0
