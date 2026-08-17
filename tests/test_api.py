"""Smoke tests for the Day-1 FastAPI app: /healthz and /ask (text debug entry point).

/voice is deliberately not covered here — it drives real Sarvam STT (never mocked, see
CLAUDE.md hard rules) and isn't suitable for an automated, network-dependent unit test. It was
manually smoke-tested end to end against the real Sarvam API during this session; see
docs/PROGRESS_P.md."""

from fastapi.testclient import TestClient

from vrag.api.main import app

client = TestClient(app)


def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ask_returns_answered_for_a_query_the_stub_covers():
    resp = client.post("/ask", json={"query": "भारत में सबसे ऊँचा पर्वत कौन सा है?", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "answered"
    assert body["track"] == "extractive"
    assert body["answer"]
    assert len(body["citations"]) >= 1
    assert "retrieve" in body["timings_ms"]


def test_ask_abstains_on_empty_query():
    resp = client.post("/ask", json={"query": "", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "abstained"
    assert body["answer"] is None
    assert body["refusal_reason"]
