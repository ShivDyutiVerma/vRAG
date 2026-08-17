# LATENCY_BUDGET.md

Target vs. measured, ms. Targets are hypotheses from `AGENT_BUILD_SPEC.md` §4, to be replaced with
real numbers during Phase 6 (`scripts/bench_latency.py`). **Never report a number here that wasn't
produced by that script** (`CLAUDE.md` hard rule).

| # | Stage | Owner | Target p50 | Min viable | Optional? | Measured |
|---|-------|-------|-----------|-----------|-----------|----------|
| 1 | Input guardrail (local) | P | 2ms | 2ms | no | _TBD — Phase 6_ |
| 2 | Query normalise + embed (ONNX int8, batch=1) | R | 8ms | 8ms | no | _TBD_ |
| 3 | Dense search (HNSW) | R | 3ms | 3ms | no | _TBD_ |
| 4 | Sparse search (BM25) | R | 5ms | — | **yes** | _TBD_ |
| 5 | RRF fusion | R | <1ms | <1ms | no | _TBD_ |
| 6 | Cross-encoder rerank (top-20) | R | 25ms | — | **yes** | _TBD_ |
| 7 | Grounding gate (threshold + margin) | Joint (G3) | 1ms | 1ms | no | _TBD_ |
| 8a | Track A extractive span select | P | 10ms | 10ms | no | _TBD_ |
| 8b | Track B generation TTFT | P | 110ms | — | **yes** | _TBD_ |
| 9 | Output verify (first sentence) | Joint (G4) | 5ms | 5ms | no | _TBD_ |
| | **Total to Track A answer** | | **~30ms** | | | _TBD_ |
| | **Total to Track B first token** | | **~160ms** | | | _TBD_ |

## Hot-path rules that keep this budget honest

- No network calls on the hot path except the LLM (embeddings, BM25, reranker, guardrails all
  in-process — this is why a hosted vector DB was rejected outright, see `TECH_MENU.md` S6).
- No cold starts — every model and the JSON schema are warmed at boot.
- No per-request disk I/O — the index is memory-resident.

## Provider RTT probe

Not yet run — blocked on Sarvam/Groq API keys (see `docs/RISKS.md` R-blocker-1). Once keys are
available, `scripts/probe_latency.py` measures TCP connect / TLS handshake / TTFT / full completion,
N=30 samples, P50/P95/P100 per provider, and the result gets recorded as ADR-003 in the shared
`docs/DECISIONS.md` — not here, so the decision has one canonical home.
