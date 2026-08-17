# PRD — vrag

## Problem

Judges/users need to ask a spoken question (Hindi) and get back a grounded, cited answer drawn
from `ai4bharat/MSMARCO-XI`, with the retrieval-and-answer path completing server-side in under
200ms, running inside a real orchestration harness, protected by guardrails that know when to
refuse. This is a hackathon shortlisting task (HH Goa 2026, Task 2) — the deliverable is judged on
working code **and** the evidence artifacts (eval tables, latency distributions, architecture
write-up), which carry equal weight.

## Users & scenario

A judge or demo viewer opens the live URL on a phone or laptop, taps the mic, speaks a Hindi
question whose answer exists somewhere in the MSMARCO-XI Hindi passage set, and expects: a live
transcript, a fast grounded answer with citations back to the source passage, and — when the
question is out of scope, unsafe, or nothing relevant was found — a visible, honest refusal
instead of a confident hallucination.

## Functional requirements

| FR | Requirement | Constraint |
|----|-------------|------------|
| FR-1 | Transcribe spoken Hindi via Sarvam (WebSocket streaming) | C1 |
| FR-2 | Chunk the corpus with ≥2 (shipped: 6 implemented) deliberate strategies, chosen by measured comparison | C2 |
| FR-3 | Retrieve + generate an answer in <200ms server-side (`t_pipeline`, defined in `AGENT_BUILD_SPEC.md` §3.2) | C3 |
| FR-4 | Report P50/P70/P100 latency across a stratified 100-query, N=5-repeat benchmark | C4 |
| FR-5 | Run the request through a typed harness: stages, deadline propagation, retries, circuit breaker, tool calls, structured output | C5 |
| FR-6 | Refuse/abstain via five guardrail layers (G1-G5) with a calibrated confidence gate | C6 |
| FR-7 | Ship as a public GitHub repo + live HTTPS URL + two videos | C7 |

## Non-functional requirements

- Latency: `t_pipeline` p50 comfortably under 200ms for Track A; Track B TTFT reported honestly
  whether or not it clears the budget (see the two-track design, `AGENT_BUILD_SPEC.md` §3.3).
- Availability: circuit breaker prevents one bad LLM provider from taking the whole demo down —
  degrades to Track A only.
- Safety: G1 (input) and G5 (output) guardrails run in-process, no network call, <5ms combined.
- Reproducibility: every quality/latency claim traces back to a row in `eval/ablation_ledger.csv`
  or a `bench_latency.py` run — nothing reported that wasn't measured.

## Acceptance criteria per requirement

Same as the exit criteria per phase in `docs/BUILD_PLAN.md` — see that file for the authoritative,
checkable list per phase. Summary: ≥6 chunking strategies evaluated with a written winner
(`docs/EVAL_RESULTS.md`), hybrid retrieval beats dense-only or the gap is documented, all harness
mechanisms demonstrably working (esp. the forced-50ms-budget degradation test), all three refusal
modes reproducible on demand, P50/P70/P100 reported honestly including the ugly P100, deployed and
reachable from a phone on mobile data.

## Explicit non-goals

- Multi-language support beyond Hindi (`hi`) unless everything else is green by Phase 7 — see ADR-002.
- A hosted vector DB — rejected outright, see `docs/DECISIONS.md`.
- Grid-searching the full 648-config ablation space — staged greedy ablation only (`docs/TECH_MENU.md` §A).
- Perfect hallucination elimination — the two-tier groundedness design (G4) reduces and measures it, doesn't claim zero.

## Evidence artifacts the submission must contain

`docs/EVAL_RESULTS.md` (chunking/embedding/retrieval tables + guardrail calibration curve +
latency charts), `docs/LATENCY_BUDGET.md` (measured column filled in), `eval/ablation_ledger.csv`,
`docs/assets/` charts, README per `AGENT_BUILD_SPEC.md` §10.1, two videos per §10.2.
