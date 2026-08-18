"""P6 latency campaign, task 6 (docs/BUILD_PLAN.md): CI regression guard for Track A's true stage
cost, on a small smoke subset -- not the full 500-sample scripts/bench_latency.py run (too slow
for CI, and needs a locally running server).

Asserts on the *stage-sum* from each response's own `timings_ms` (input_guard + scope_guard +
retrieve + ground_gate + extract_answer + output_guard), not wall-clock -- wall-clock on an
"answered" query also includes GenerateStage's real (network-bound, budget-timeout-capped) Track B
attempt, which is real, honest, and NOT what "Track A p50 comfortably < 200ms" (docs/BUILD_PLAN.md
P6 exit criterion) is asking about. See docs/LATENCY_BUDGET.md for the full breakdown and why this
distinction matters -- conflating the two would make this test either pass for the wrong reason or
fail on a >200ms number that isn't actually Track A being slow.

Skips cleanly (not a failure) if the real persisted index isn't present locally -- data/ is
gitignored (AGENT_BUILD_SPEC.md §5.3), so a fresh clone/CI checkout without the index artifact
downloaded has nothing real to measure against; faking a small synthetic index would test the code
path but not the number this file exists to guard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vrag.api.main import app
from vrag.retrieval.interface import _INDEX_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent
QUERIES_PATH = REPO_ROOT / "eval" / "test_queries.json"
N_SMOKE_QUERIES = 20
P50_TARGET_MS = 200.0

pytestmark = pytest.mark.skipif(
    not (_INDEX_DIR / "chunk_lookup.json").exists()
    and not (_INDEX_DIR / "chunk_lookup.sqlite3").exists(),
    reason="real persisted index not present locally (data/ is gitignored) -- nothing real to "
    "measure Track A's stage cost against",
)


def test_track_a_stage_cost_p50_under_200ms() -> None:
    client = TestClient(app)
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    in_domain = [q["query"] for q in queries if q["category"] == "in_domain"][:N_SMOKE_QUERIES]
    assert len(in_domain) == N_SMOKE_QUERIES, "expected at least 20 in-domain queries to sample"

    stage_sums_ms = []
    for query in in_domain:
        resp = client.post("/ask", json={"query": query, "k": 5})
        assert resp.status_code == 200
        timings = resp.json()["timings_ms"]
        stage_sums_ms.append(sum(timings.values()))

    stage_sums_ms.sort()
    p50 = stage_sums_ms[len(stage_sums_ms) // 2]
    assert p50 < P50_TARGET_MS, (
        f"Track A stage-cost P50 ({p50:.1f}ms) exceeded the {P50_TARGET_MS}ms target -- "
        f"a real regression in retrieval/guardrail/extraction speed, not the known Track B "
        f"budget-shedding cost (see docs/LATENCY_BUDGET.md)"
    )
