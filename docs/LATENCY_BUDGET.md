# LATENCY_BUDGET.md

Target column is the design contract from `AGENT_BUILD_SPEC.md` §4 — a hypothesis until probed.
Measured column fills in during Phase 6 (`scripts/bench_latency.py`), never hand-typed.

| # | Stage | Target p50 | Min viable | Optional? | Measured |
|---|-------|-----------|-----------|-----------|----------|
| 1 | Input guardrail (local) | 2ms | 2ms | no | _TBD_ |
| 2 | Query normalise + embed (ONNX int8, batch=1) | 8ms | 8ms | no | _TBD_ |
| 3 | Dense search (HNSW) | 3ms | 3ms | no | _TBD_ |
| 4 | Sparse search (BM25) | 5ms | — | yes | _TBD_ |
| 5 | RRF fusion | <1ms | <1ms | no | _TBD_ |
| 6 | Cross-encoder rerank (top-20) | 25ms | — | yes | _TBD_ |
| 7 | Grounding gate (threshold + margin) | 1ms | 1ms | no | _TBD_ |
| 8a | Track A extractive span select | 10ms | 10ms | no | _TBD_ |
| 8b | Track B generation TTFT | 110ms | — | yes | _TBD_ |
| 9 | Output verify (first sentence) | 5ms | 5ms | no | _TBD_ |
| | **Total to Track A answer** | **~30ms** | | | _TBD_ |
| | **Total to Track B first token** | **~160ms** | | | _TBD_ |

Hot-path rules: no network calls except the LLM; no cold starts (all models warmed at boot); no
per-request disk I/O (index memory-resident); no large-payload JSON parsing on the hot path.
