"""Tests the IDEMPOTENT_STAGE_RETRY policy mechanism in isolation.

Not attached to retrieve() — that function never raises by contract (returns [] on internal
failure, see src/vrag/retrieval/interface.py), so a raise-based retry has nothing to catch there.
This proves the policy itself works correctly (retries transient failures, gives up and reraises
after the cap) so it's ready to attach to Track B's LLM call once that exists (Day 3, per
docs/TEAM_SPLIT.md §5) — see docs/DECISIONS_P.md for the full reasoning.
"""

import pytest

from vrag.harness.retry import IDEMPOTENT_STAGE_RETRY


@pytest.mark.asyncio
async def test_retry_succeeds_after_one_transient_failure():
    calls = {"count": 0}

    async def flaky() -> str:
        calls["count"] += 1
        if calls["count"] < 2:
            raise ConnectionError("transient")
        return "ok"

    async for attempt in IDEMPOTENT_STAGE_RETRY:
        with attempt:
            result = await flaky()

    assert result == "ok"
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_retry_gives_up_and_reraises_after_cap():
    calls = {"count": 0}

    async def always_fails() -> None:
        calls["count"] += 1
        raise ConnectionError("permanent")

    with pytest.raises(ConnectionError):
        async for attempt in IDEMPOTENT_STAGE_RETRY:
            with attempt:
                await always_fails()

    # stop_after_attempt(2) — never retries more than the configured cap
    assert calls["count"] == 2
