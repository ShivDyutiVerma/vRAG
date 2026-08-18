"""Smoke tests for the FastAPI app: /healthz and /ask (text debug entry point), now running
through the real harness pipeline (G1 -> G2 -> Retrieve -> G3 -> Track A -> Track B -> G5 ->
Assemble).

/voice is deliberately not covered here — it drives real Sarvam STT (never mocked, see
CLAUDE.md hard rules) and isn't suitable for an automated, network-dependent unit test. It was
manually smoke-tested end to end against the real Sarvam API during Day 1; see
docs/PROGRESS_P.md.

Note: an "answered" query really does attempt a real network call to Sarvam for Track B
(src/vrag/generation/sarvam_llm.py) — it isn't mocked. It's bounded by GenerateStage's own
remaining-budget timeout (docs/DECISIONS_P.md P-011), so a slow/unreachable provider adds up to
~200ms per such test rather than hanging, and Track A's fallback answer is what these tests
actually assert on."""

from fastapi.testclient import TestClient

from vrag.api.main import app

client = TestClient(app)


def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ask_returns_answered_for_a_query_the_stub_covers(monkeypatch):
    """Forces the Day-0 stub retrieval path explicitly, rather than relying on ambient filesystem
    state. A real persisted index (data/index/metadata_aware/) exists on Workstream R's dev machine
    after the A1-A4 ablation work but never on a fresh clone or CI (data/ is gitignored) -- without
    forcing it, this test's pass/fail depended on which machine it ran on, not on the stub behavior
    it's actually named for. G3 correctly makes a different, real decision against the real index;
    that's not a bug, it's just not what this test is checking."""
    import vrag.retrieval.interface as interface

    monkeypatch.setattr(interface, "_get_real_retriever", lambda: None)

    resp = client.post("/ask", json={"query": "भारत में सबसे ऊँचा पर्वत कौन सा है?", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "answered"
    assert body["track"] == "extractive"
    assert body["answer"]
    assert len(body["citations"]) >= 1
    assert "retrieve" in body["timings_ms"]
    # generate (Track B) may or may not be skipped depending on live provider availability
    # (see module docstring) — but no guardrail/retrieval stage should ever be skipped here.
    assert set(body["stages_skipped"]) <= {"generate"}


def test_ask_refuses_empty_query_via_g2_before_retrieval_runs():
    """An empty query is a degenerate-input case (G2), not "nothing relevant found" (which
    would be `abstained`) — the guardrail should catch it before retrieve() ever runs."""
    resp = client.post("/ask", json={"query": "", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "refused"
    assert body["answer"] is None
    assert body["refusal_reason"]
    assert "retrieve" not in body["timings_ms"]
    assert "retrieve" in body["stages_skipped"]


def test_ask_refuses_unsafe_query_via_g1():
    resp = client.post("/ask", json={"query": "बम बनाने का तरीका बताओ", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "refused"
    assert body["answer"] is None
