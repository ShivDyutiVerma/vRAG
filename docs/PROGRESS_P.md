# PROGRESS_P — Workstream P running status

> Mine — edit freely, any time. Never edited by Workstream R.

**Last updated:** 2026-08-18 (Day 2, session 08 — same-day continuation)
**Current phase:** P4/P5 done; **critical deployment gap found (R-R21/R4) — live demo has been
running the Day-0 stub this whole session, real retrieval blocked on a cross-team memory fix**

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

Sixth same-day session — the most consequential finding of the day. R checked the actual live URL
directly before starting new work and found something every one of my earlier "verified live"
claims this session had missed: `https://vrag-voice.onrender.com` has been serving the Day-0
retrieval stub the entire time — none of A1-A4, efSearch, or G3 calibration has ever reached the
public demo. Root cause is squarely mine: `Dockerfile` never installs the `retrieval` extra, and
`data/` is gitignored, so the real index was never going to reach the container either way. R
separately measured that even fixing this naively would crash the deploy — the index alone uses
591MB RSS (over Render free tier's 512MB budget before any embedder loads), and the embedder adds
~883MB more regardless of backend (ONNX included — `torch`/`transformers` load either way). Real
pricing checked (not assumed): Render Standard is $25/mo for 2GB RAM. Presented the actual
tradeoff to the user — pay for headroom vs. undesigned free engineering work (shrink the corpus +
drop `torch`/`transformers` at inference) — and caught, before presenting, that shrinking the
corpus alone can't work regardless of size, since the embedder's own framework overhead already
exceeds the 512MB budget by itself. User chose the ambitious free path, with the paid tier as an
explicit, pre-authorized fallback if it doesn't land before Aug 22. Did my half today: found and
fixed a real latent bug while preparing (`_get_real_retriever()` could crash uncaught if the index
was present but unloadable, breaking `retrieve()`'s "never raises" contract), then staged the
`Dockerfile` to download R's published index artifact — built and ran the image locally to verify
it's genuinely inert until R's leaner embedder lands (confirmed: missing-dependency failure is now
caught and logged, falls back to the stub cleanly, current deploy behavior unchanged). R's half
(lean `embedder.py`, corpus resize) is real, undesigned work, not started yet. First deploy attempt
hit a real environment-specific bug (`tar` rejected by Render's build sandbox trying to `chown` to
the artifact's original uid/gid — didn't reproduce locally); fixed with `--no-same-owner`,
redeployed, verified live.

Seventh same-day session: built `search_corpus` (P-019) — the tool-calling capability
AGENT_BUILD_SPEC.md §7.2 item 5 names explicitly, tied to graded requirement C5. Live-probed first
(same methodology as P-017): confirmed Sarvam supports real tool-calling, and confirmed combining
it with strict structured output in one request doesn't work reliably (same whitespace-padding
bug). Designed around that: the existing structured call now also signals `needs_more_context`,
and only then does `generate()` escalate to a real tool-calling round + one final re-answer over
the expanded context — common case unchanged at one round trip. Caught a real G4 integration bug
before shipping (tool-fetched citations would've been wrongly flagged as invented against the
stage's original chunk list) and fixed it via a new `GenerationResult` return type carrying the
full chunk set used. 17 new tests, all green. Live end-to-end testing of the full chain then
surfaced something more important than the feature itself: **12/12 attempts failed via the same
stall bug, and it correlates strongly with "insufficient context" answers specifically** — not a
random rate as P-017 first characterized it. Verified this predates today's schema change (same
failure with the old 3-field schema). This means Track B's own "I don't know" branch — exactly
where a generative answer would matter most — is the case most likely to hit this provider bug.
The mechanism itself is proven correct; live success is currently gated on Sarvam's reliability.

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
- [x] `search_corpus` tool for Track B (P-019, satisfies graded requirement C5's "tool calls") —
      real native OpenAI-style tool-calling, depth capped at 1, correct G4 integration for
      tool-fetched citations, 17 new tests. Live E2E success of the full escalation chain is
      currently blocked by a sharper P-R20 finding: the stall bug correlates strongly with
      "insufficient context" answers specifically (12/12 failures in today's testing) — the
      mechanism is proven correct (mocks + an isolated live tool-calling probe), same "ready the
      moment the provider issue clears" position Track B itself shipped in under earlier this
      session.
- [x] Retry policy tested correct in isolation, not attached to a live stage (P-009)
- [x] Every request writes a trace with per-stage ms timings
- [x] All five guardrail layers (G1–G5) implemented, unit-tested, and — for G4 — now verified
      against a real generated answer, not just synthetic test cases
- [x] G3 fully calibrated against real data (300 queries, R's sweep): `TAU=0.8835`, the balanced
      operating point the user chose (P-015); `MARGIN=0.0`, fixed same day by R (R-017) after
      live-verifying my shipped `MARGIN=0.05` caused 88% false-refusal at this TAU, not the
      intended 19.3% — a real bug, caught by the joint-ownership cross-check on this file working
      exactly as intended.
- [x] Track B switched to streaming with stall detection (P-017) — mitigates (does not fix) a
      newly-found, high-rate Sarvam reliability bug (P-R20). Client-facing token-by-token display
      is a separate, still-undone piece (design tension with the G4 gate needing complete output).

## ⚠️ Deployment: live demo is NOT running real retrieval (R-R21/R4, P-018)

Every "verified live" claim above in this doc was checked against the real public URL and is real
for what it tested — but none of them would have caught this, since the Day-0 stub's output shape
is deliberately identical to the real retriever's. `https://vrag-voice.onrender.com` currently
serves the stub for every query. Fix direction decided (P-018): shrink the corpus (R) + drop
`torch`/`transformers` at inference for a lean `onnxruntime`-only path (R, `embedder.py`) — a
corpus-only fix was checked and confirmed insufficient alone. My half (Dockerfile prep, defensive
fallback fix) is done and deployed live — hit a real environment-specific bug on the first deploy
attempt (Render's build sandbox rejected `tar`'s attempt to `chown` to the artifact's original
uid/gid, didn't reproduce locally; fixed with `--no-same-owner`), redeployed, verified: index
download step is confirmed inert in production, same as local testing. R's half (lean
`embedder.py`, corpus resize) is real, undesigned engineering work, not started.
Paid Render Standard ($25/mo, 2GB RAM) is the pre-authorized fallback if R's half doesn't land with
enough runway before Aug 22.

## What works right now (verified, not assumed)

- Fifth same-day merge from `origin/main` (R's A4/G3-calibration/efSearch work, then R's MARGIN
  fix R-017): clean merges throughout, tests green
- **Track B end-to-end, for real:** `POST`/`WS` request → real Sarvam call → structured JSON
  (`reasoning`, `answer`, `cited_chunk_ids_csv`) → parsed → G4-checked → accepted →
  `AnswerResponse(track="generative", status="answered", ...)`. Directly observed, repeatedly,
  against the live API — not mocked, not simulated ✅
- **G3 fully calibrated, applied, live-verified:** `TAU=0.8835` (P-015) + `MARGIN=0.0` (R-017,
  fixing a real bug in my shipped `MARGIN=0.05` that caused 88% false-refusal at this TAU). Stub
  fallback path (used in CI/fresh clones) confirmed to still clear the new TAU ✅
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
- `pytest` (my scope): 84/84 passing after merging R's MARGIN fix (R-017). `ruff check .`
  (repo-wide, incl. R's files) and `mypy` on my scope's modules: clean ✅

## What is stubbed / faked / TODO

- **The live deployment's retrieval is entirely stubbed** (R-R21/R4, P-018) — see the warning
  section above. This is the top item, not a minor gap.
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
- `search_corpus` tool is built (P-019) but not yet observed succeeding end-to-end live — blocked
  on P-R20's sharpened finding (stalls correlate strongly with insufficient-context answers, the
  exact case that triggers `needs_more_context`). Worth re-testing once P-R20 is reported/improves.
- G3's `TAU`/`MARGIN` are both now calibrated (P-015/R-017). G4's threshold
  (`MIN_OVERLAP_RATIO=0.15`) is still an uncalibrated placeholder — today's testing exercised it
  against a real generated answer and it behaved sensibly (correctly failed a wrong-language
  answer, correctly passed a fixed one), but that's not the same as a calibration sweep.
- Retry policy (`tenacity`) tested correct in isolation, not attached to a live stage.
- Real browser mic click-through — still not done. Genuinely just needs a human with a phone.
- efSearch curve, A4/A5 ablations — Workstream R's queue (A4 next per their side).

## Blockers

- **Real retrieval in production is blocked on R4/R-020's memory fix** — real progress, not
  stalled: R-021 (SQLite chunk_lookup) + R-022 (torch-free `LiteE5Embedder`, byte-identical output
  verified) together cut combined RSS 1,539MB → 741MB (-52%, zero quality cost), still 145% of the
  512MB budget. **User has since given R explicit direction to rule out the paid Render plan for
  now and keep shrinking for free** — the $25/mo Standard-plan fallback documented earlier this
  session is de-prioritized per that direction, not something to execute unilaterally without
  checking first if it still applies as the deadline approaches.
- `GROQ_API_KEY` still empty if a second generation provider is ever wanted for comparison — not
  currently needed since Sarvam is healthy again.

## Next session should start by

1. **Check whether R's memory fix (corpus resize + lean embedder) has landed** — if so, add the
   new leaner extras group to the Dockerfile install line, redeploy, and re-verify with a real
   `/ask` call against the live URL (the same check that found this bug in the first place — never
   trust "should be deployed" without checking the actual URL again).
2. R is close (741MB vs. a 512MB budget, R-021/R-022) and the user has directed R to keep shrinking
   for free rather than pay — respect that unless it's genuinely not going to land in time; check
   with the user again before reviving the paid-plan fallback, don't execute it unilaterally.
3. Report P-R20 to Sarvam — now has a much sharper, highly reproducible repro case (P-019: 12/12
   failures on insufficient-context queries, any chunk count, with or without the extra schema
   field) rather than a vague "sometimes stalls." Worth writing up properly and sending.
4. Re-test `search_corpus`'s full live escalation chain periodically — the mechanism is proven
   correct, just gated on Sarvam's reliability for insufficient-context completions specifically.
5. Consider real client-facing token-by-token display for Track B (the WS `/voice` endpoint
   streaming partial text as it arrives) — separate from today's server-side streaming, and has
   its own design tension: G4's groundedness check needs the *complete* answer + citations before
   it's safe to show anything, so this needs either a redesign (e.g. an incremental/best-effort G4
   check) or accepting a "commit point" partway through the stream. Not scoped yet.
6. Click through the real mic UI on a phone. Still. Genuinely just do this one.
