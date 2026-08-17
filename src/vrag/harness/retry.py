"""Retry policy — stub. Full tenacity wiring + circuit breaker land in Day 2 harness hardening
(docs/TEAM_SPLIT.md §5, AGENT_BUILD_SPEC.md §7.2 items 3-4).

Rules to hold to once implemented: exponential backoff + jitter, capped attempts, only on
idempotent stages, and never allowed to exceed the request's remaining budget (see budget.py).
"""

from __future__ import annotations

from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential_jitter

IDEMPOTENT_STAGE_RETRY = AsyncRetrying(
    stop=stop_after_attempt(2),
    wait=wait_exponential_jitter(initial=0.01, max=0.05),
    reraise=True,
)
