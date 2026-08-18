# PROGRESS_P — Workstream P running status

> Mine — edit freely, any time. Never edited by Workstream R.

**Last updated:** 2026-08-18 (Day 2, session 06 — same-day continuation)
**Current phase:** P4/P5 — Harness Hardening + Guardrails, Track B streaming + stall-protected

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

Fifth same-day session: picked up real token streaming for Track B (P-017), the item flagged as
"the actual unlock." Before writing client-facing streaming code, probed the live API with the
real schema/prompt first — found this reshaped the whole task. A realistic 5-chunk context (vs.
the earlier 1-2 chunk probes) exposed two real Sarvam-side bugs, not one: (1) our own schema's
unconstrained `reasoning` field can eat the whole token budget before `answer` is ever reached —
`reasoning_effort: null` doesn't touch this, it only disables a *different*, hidden CoT mechanism;
fixed with an explicit "keep it to one sentence" prompt instruction, verified to cut latency on
success from ~2.2-2.7s to ~1.0-1.2s. (2) A second, more consequential bug: even after fixing (1),
the model sometimes completes `reasoning` and `answer` correctly, then fails to continue to the
final field and close the JSON — padding whitespace toward `max_tokens` instead. This is the same
*symptom* as the already-documented P-R15 array-field bug but a *distinct* cause (confirmed:
`cited_chunk_ids_csv` has been a plain string since P-013, and the bug still happens) — P-R15's
original fix addressed A cause, not THE cause. Switched to streaming specifically to catch this
fast: track consecutive whitespace/empty content deltas, abort after 20 in a row (evidence-based
threshold — legitimate JSON formatting whitespace never exceeded 2 in any successful run observed)
instead of waiting out the full `max_tokens` budget. Live-verified: 2 of 3 test runs stalled on
*both* attempts (a genuinely high rate, not a rare edge case) but were each caught and handled in
~2.2-2.6s instead of the 4-60s+ this would have taken before — and Track A correctly covered every
failure, every time. This is a mitigation (bounds the cost of failure), not a fix for the
underlying rate — stated plainly rather than overclaiming, and worth reporting to Sarvam.

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
- [x] Track B switched to streaming with stall detection (P-017) — mitigates (does not fix) a
      newly-found, high-rate Sarvam reliability bug (P-R20). Client-facing token-by-token display
      is a separate, still-undone piece (design tension with the G4 gate needing complete output).

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
- **Track B streaming + stall detection, live-verified:** `_call_once_streaming` in
  `src/vrag/generation/sarvam_llm.py` (P-017) — SSE consumption, 20-consecutive-whitespace-chunk
  stall abort, wired into `generate()`'s existing repair-retry loop. Live: 2/3 test runs stalled on
  both attempts, each caught in ~2.2-2.6s (vs. 4-60s+ before); Track A covered every failure ✅
- `pytest` (my scope): 83/83 passing (77 pre-existing + 6 new streaming/stall tests) throughout
  all of today's changes. `ruff check .` (repo-wide, incl. R's files) and `mypy` on changed files:
  clean ✅

## What is stubbed / faked / TODO

- Server-side streaming now built for Track B (P-017: `stream: true` + stall detection), but this
  is NOT client-facing token-by-token display — the WS `/voice` endpoint still buffers, validates
  via G4, and responds once, same external contract as before. Real reason: G4's groundedness
  check needs the *complete* answer + citations before it's safe to show anything, so streaming
  partial text live to the client has its own separate design tension with the guardrail gate —
  not solved today, flagged as a distinct follow-up. What streaming *did* unlock today: failures
  are now detected in ~a few hundred ms instead of waiting out the full non-streaming completion
  (which could take 4-60s+ to fail). Full non-streaming completion time itself (~1.4s-15s+ on
  success) is unchanged by this — still an accepted, deliberate gap (P-014).
- **A second, more consequential Sarvam reliability bug found this session (P-R20):** distinct
  from P-R15's array-field bug, the model sometimes completes `reasoning`+`answer` correctly then
  fails to continue to the final schema field and close the JSON object, padding whitespace
  instead. Live-observed rate is concerning (2/3 test runs stalled on both attempts under a
  realistic 5-chunk context) — today's fix bounds the *cost* of this failure, it does not fix the
  underlying rate. Worth reporting to Sarvam with the reproduction case.
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

1. Report the P-R20 stall pattern to Sarvam (reproduction: 5-chunk context, `strict: true`,
   3-field all-string schema, model completes 2/3 fields then pads whitespace) — provider-side bug,
   not ours to fix, but worth flagging with real repro steps now that we have them.
2. Consider real client-facing token-by-token display for Track B (the WS `/voice` endpoint
   streaming partial text as it arrives) — separate from today's server-side streaming, and has
   its own design tension: G4's groundedness check needs the *complete* answer + citations before
   it's safe to show anything, so this needs either a redesign (e.g. an incremental/best-effort G4
   check) or accepting a "commit point" partway through the stream. Not scoped yet.
3. Re-sweep G3's `MARGIN` independently at the new `TAU=0.8835` (P-015) — R's sweep held MARGIN=0
   while varying TAU; needs R's index locally, which this machine doesn't have. Either run it on
   R's machine or ask R to re-run `scripts/eval_g3_calibration.py`'s margin sweep at this TAU.
4. Check `docs/PROGRESS_R.md` for R's latest.
5. Click through the real mic UI on a phone. Still. Genuinely just do this one.
