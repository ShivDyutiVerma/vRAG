# PROGRESS_P — Workstream P running status

> Mine — edit freely, any time. Never edited by Workstream R.

**Last updated:** 2026-08-18 (Day 2, session 05 — same-day continuation)
**Current phase:** P4/P5 — Harness Hardening + Guardrails, Track B working + circuit-breaker-protected

## Where we are, in one paragraph

Fourth same-day sync: merged R's A2/A3 ablations plus a further batch (A4 reranker ablation, R-013
test-hygiene fix crossing into `test_api.py` with user authorization, efSearch curve, R's own G3
calibration data-gathering session) — clean fast-forward both ways, `main` and `workstream-p` now
level. The big finding this session: R gathered real 300-query G3 calibration data and found
`EVAL_PROTOCOL.md`'s two targets (false-refusal<10%, correct-refusal>80%) can't both be hit on this
corpus — a genuine product tradeoff, not a data question, correctly left for joint/human sign-off
rather than picked unilaterally. Presented the real curve to the user; they chose the balanced
operating point. Applied `TAU=0.8835` to `g3_confidence.py` (P-015) — G3's TAU check was previously
a silent no-op in production (real cosine scores run 0.82-0.96, far above the old TAU=0.35
placeholder), so this is a real behavior change, not just a number swap. Then built the circuit
breaker flagged as the top priority in the previous session's notes (P-016): wraps Track B's Sarvam
call, skips straight to Track A once recent fair-chance attempts fail repeatedly. The non-obvious
part was realizing a naive "every timeout is a failure" design would trip the breaker open almost
permanently under normal healthy operation, since Track B's non-streaming completion time
(1.4s-15s+, P-014) exceeds almost any realistic per-request budget regardless of provider
health — fixed by only counting outcomes at a "fair chance" allowance (>=2.0s, well above measured
P95 TTFT) toward the breaker, so ordinary two-track budget shedding never falsely signals an
outage. Verified live against the real Sarvam API with a generous budget: Track B succeeded,
breaker recorded success, stayed `CLOSED`.

## Phase exit criteria — P4/P5

- [x] Every stage typed with Pydantic in/out
- [x] Forced-tight-budget degradation test green, and real in production (Track B sheds under a
      normal request budget; verified it *doesn't* shed and produces a real generative answer
      under a generous budget — both paths now directly observed, not just unit-tested)
- [x] Deadline propagation holds during a stage, not just before it (P-011), and is now consistent
      between the outer `wait_for` and `generate()`'s own timeout (P-013)
- [x] Circuit breaker — built (`src/vrag/generation/circuit_breaker.py`), wired into
      `GenerateStage`, and verified live: a real end-to-end run against the live Sarvam API
      produced `track: "generative"` with the breaker recording success and staying `CLOSED`
      (P-016). Only "fair-chance" outcomes (>=2.0s allowance) move the breaker — a tight-budget
      timeout is the two-track design's expected outcome, not a health signal (see the module's
      `should_count_as_health_signal()` docstring for why treating every timeout as a failure
      would have made the breaker actively harmful).
- [ ] `search_corpus` tool for Track B — not started
- [x] Retry policy tested correct in isolation, not attached to a live stage (P-009)
- [x] Every request writes a trace with per-stage ms timings
- [x] All five guardrail layers (G1–G5) implemented, unit-tested, and — for G4 — now verified
      against a real generated answer, not just synthetic test cases
- [x] G3 calibrated against real data (300 queries, R's sweep): `TAU=0.8835`, the balanced
      operating point the user chose after the data showed EVAL_PROTOCOL.md's two targets can't
      both be hit on this corpus (P-015). `MARGIN=0.05` still uncalibrated at this TAU (follow-up).

## What works right now (verified, not assumed)

- Fourth same-day merge from `origin/main` (R's A4/G3-calibration/efSearch work): clean
  fast-forward both directions (`workstream-p` -> `main` and back), tests green
- **Track B end-to-end, for real:** `POST`/`WS` request → real Sarvam call → structured JSON
  (`reasoning`, `answer`, `cited_chunk_ids_csv`) → parsed → G4-checked → accepted →
  `AnswerResponse(track="generative", status="answered", ...)`. Directly observed, repeatedly,
  against the live API — not mocked, not simulated ✅
- **G3 calibrated, applied, live-verified:** `TAU=0.8835` (P-015), the balanced point the user
  chose from R's real 300-query sweep. Stub fallback path (used in CI/fresh clones) confirmed to
  still clear the new TAU ✅
- **Circuit breaker built, wired, live-verified:** `src/vrag/generation/circuit_breaker.py`
  (P-016) — closed/open/half-open state machine, module-level singleton since `GenerateStage` is
  built fresh per request, only "fair-chance" (>=2.0s) outcomes move it. Ran the real pipeline
  against the live Sarvam API with a 15s budget: `track="generative"`, breaker recorded success,
  stayed `CLOSED` ✅
- `src/vrag/generation/sarvam_llm.py` — `reasoning_effort: null`, `sarvam-105b` (not the
  `-conversations` variant), CSV-string citations, a real Hindi-language system prompt
  instruction that's been verified to actually work (100% lexical overlap after the fix, vs. ~9%
  before it, on the same query) ✅
- G4 has now been exercised against a real model output, not just hand-written test fixtures, and
  correctly passed a genuinely well-grounded answer and correctly failed a genuinely ungrounded
  (wrong-language) one before the prompt fix ✅
- Real provider latency data for ADR-003 recorded in `docs/DECISIONS_P.md` P-012 (both the outage
  numbers and the recovery numbers — P50 chat TTFT 452ms once healthy) ✅
- `pytest` (my scope): 77/77 passing (62 pre-existing + 15 new circuit-breaker tests) throughout
  all of today's changes. `ruff check .` (repo-wide, incl. R's files) and `mypy` on changed files:
  clean ✅

## What is stubbed / faked / TODO

- No real token streaming for Track B — non-streaming waits for the full ~500-token structured
  response, which is why it rarely clears any realistic budget even though raw TTFT (452ms P50)
  is much more reasonable. Documented as an accepted, deliberate gap (P-014), not hidden.
- No `search_corpus` tool for Track B.
- G3's `TAU` is now calibrated (P-015, `TAU=0.8835`); `MARGIN` is not — carried over unchanged,
  never independently swept at the new TAU. G4's threshold (`MIN_OVERLAP_RATIO=0.15`) is still an
  uncalibrated placeholder — today's testing exercised it against a real generated answer and it
  behaved sensibly (correctly failed a wrong-language answer, correctly passed a fixed one), but
  that's not the same as a calibration sweep.
- Retry policy (`tenacity`) tested correct in isolation, not attached to a live stage.
- Real browser mic click-through — still not done. Genuinely just needs a human with a phone.
- efSearch curve, A4/A5 ablations — Workstream R's queue (A4 next per their side).

## Blockers

- None on my side. `GROQ_API_KEY` still empty if a second generation provider is ever wanted for
  comparison — not currently needed since Sarvam is healthy again.

## Next session should start by

1. Re-sweep G3's `MARGIN` independently at the new `TAU=0.8835` (P-015) — R's sweep held MARGIN=0
   while varying TAU; needs R's index locally, which this machine doesn't have. Either run it on
   R's machine or ask R to re-run `scripts/eval_g3_calibration.py`'s margin sweep at this TAU.
2. Consider whether real token streaming for Track B is worth the time investment before Day 3's
   scheduled end, given it's the actual unlock for Track B mattering in practice — currently it's
   correct but rarely fires under a real (small) budget, and the circuit breaker's own
   `MIN_FAIR_TIMEOUT_S` gate means most default-budget attempts don't produce a health signal
   either way today (see P-016) — streaming would change that by shrinking real completion time
   toward the budget instead of past it.
3. Check `docs/PROGRESS_R.md` for R's latest.
4. Click through the real mic UI on a phone. Still. Genuinely just do this one.
