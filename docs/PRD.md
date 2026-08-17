# PRD — vrag

**Status:** living document, reconciled with reality at each phase exit (see `AGENT_BUILD_SPEC.md` §8.2/§8.3).

## Problem

Judges for HH Goa Task 2 want to see a voice-enabled RAG system that (a) actually transcribes real
speech, (b) retrieves from a real corpus with a deliberately-chosen chunking strategy, (c) answers
fast enough server-side to matter for a voice interface, (d) proves that speed with real percentile
measurements instead of a single cherry-picked run, (e) runs inside real orchestration instead of a
bare LLM call, and (f) knows when *not* to answer. Most teams will get a demo working; the constraint
this project is built around (§3 of `AGENT_BUILD_SPEC.md`) is that most teams will silently misreport
the 200ms number when they can't hit it. This project's differentiator is refusing to do that.

## Users & scenario

A judge (or any stranger with a phone) opens the live URL, taps the mic, and asks a question in
Hindi about something in the `ai4bharat/MSMARCO-XI` corpus. They expect: a visible live transcript,
a fast grounded answer with citations to the actual retrieved passage, a visible latency number, and
— for out-of-scope or unsafe questions — a visible refusal instead of a hallucinated answer.

## Functional requirements → constraints

| FR | Requirement | Maps to |
|----|-------------|---------|
| FR-1 | Real Sarvam STT over WebSocket, mic to transcript, no mocking | C1 |
| FR-2 | ≥6 chunking strategies implemented, evaluated, winner justified with numbers | C2 |
| FR-3 | `t_pipeline` (transcript-ready → first answer token, server-side) < 200ms | C3 |
| FR-4 | P50/P70/P100 latency reported over a stratified 100-query set, N=5 reps | C4 |
| FR-5 | Typed pipeline stages, deadline propagation, retries, circuit breaker, tool calling, structured output | C5 |
| FR-6 | G1–G5 guardrail layers, calibrated G3 threshold, demonstrable refusal on command | C6 |
| FR-7 | Public GitHub repo, live HTTPS link, demo + process videos | C7 |

## Non-functional requirements

- Latency: see FR-3/FR-4. Track A (extractive) must clear 200ms comfortably; Track B (generative) is
  reported honestly whether or not it clears it (see `AGENT_BUILD_SPEC.md` §3.3).
- Availability: circuit breaker must keep Track A serving even if the LLM provider is fully down.
- Safety: G1 (unsafe input) and G5 (output PII) must both be demonstrable; no raw exception or
  truncated JSON may ever reach the client.
- Reproducibility: every quality/latency number traces back to a script (`bench_latency.py`,
  `eval_chunking.py`) and a row in `eval/ablation_ledger.csv` or `traces.jsonl`.

## Acceptance criteria per requirement

Each FR above is only "done" when its corresponding exit criteria in the current phase of
`docs/BUILD_PLAN.md` are ticked in `docs/PROGRESS.md` **with evidence** — a number, a file path, or
a screenshot, never an assertion. See `docs/BUILD_PLAN.md` phase-by-phase exit criteria for the exact
bar (e.g. FR-2's bar is "Recall@5 ≥ 0.75 on the winner, ledger committed, noise floor reported").

## Non-goals

- Full multi-GB MSMARCO-XI corpus (target 50k–200k chunks only, per `AGENT_BUILD_SPEC.md` §6.1)
- More than one language before Phase 7 exit (Hindi only for v1 — ADR-002)
- A hosted vector DB (explicitly rejected — network RTT eats the budget, `TECH_MENU.md` S6)
- Semantic response caching (rejected with citation — `TECH_MENU.md` S13)
- Perfect UI polish before the pipeline works end to end

## Evidence artifacts the submission must contain

- `docs/EVAL_RESULTS.md` — chunking/embedder/retrieval ablation tables + charts (R's §1–3),
  generation/guardrail/latency tables + charts (P's §4–6)
- `docs/LATENCY_BUDGET.md` — target vs measured, per stage
- `docs/assets/` — efSearch curve, g3 calibration curve, latency breakdown + CDF charts
- `eval/ablation_ledger.csv` — every experiment run, one row each
- README (Phase 7) — the judged decision trail, per `AGENT_BUILD_SPEC.md` §10.1
- Two videos (demo + process), per `AGENT_BUILD_SPEC.md` §10.2
