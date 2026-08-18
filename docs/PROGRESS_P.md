# PROGRESS_P — Workstream P running status

> Mine — edit freely, any time. Never edited by Workstream R.

**Last updated:** 2026-08-18 (Day 2, session 03 — same-day continuation)
**Current phase:** P4/P5 — Harness Hardening + Guardrails (ahead of the Day 3 schedule on G3/G4)

## Where we are, in one paragraph

Second sync of the day: pulled in R's noise-floor validation work (confirmed `metadata_aware`
stands as the chunking winner) via another clean fast-forward merge. Attempted to build Track B
(LLM generation) against Sarvam and hit a real external blocker: Sarvam's `/v1/chat/completions`
hangs indefinitely on every valid request (confirmed via isolated tests — bad key/bad model both
fail fast and correctly, only genuinely valid requests hang) — see `docs/RISKS.md` P-R13. User
decided to retry Sarvam later rather than block on a Groq key. Rather than sit idle, built G3
(confidence gate) and G4 (groundedness) — realized these don't actually need to wait for Day 3's
"joint calibration," since `retrieve()` already returns real scores; only the *threshold tuning*
is joint work, not the mechanism. All 5 guardrail layers are now live on the request path, ahead
of schedule, with G3/G4 thresholds explicitly marked as uncalibrated placeholders. Also built
`GenerateStage` (Track B) itself — even though Sarvam is down, this let me discover and fix a real
architectural gap: a stage's declared `min_viable_ms` is only a pre-flight check, not an active
timeout, so an unguarded network call inside an optional stage can blow the budget by 50x. Fixed
with `asyncio.wait_for` against the actual remaining budget — verified live (10+s → ~204ms).

## Phase exit criteria — P4/P5

- [x] Every stage typed with Pydantic in/out
- [x] Forced-tight-budget degradation test green, **and now real**: `GenerateStage` is the first
      stage that actually sheds under budget pressure in the live pipeline, not just in isolation
- [x] Deadline propagation holds *during* a stage, not just before it (`docs/DECISIONS_P.md` P-011)
- [ ] Circuit breaker — still not started; arguably more urgent now that P-R13 exists (a provider
      that hangs rather than erroring is exactly the failure mode a circuit breaker protects
      against — currently every request pays up to the full remaining budget probing a dead
      endpoint). Candidate for next session.
- [ ] `search_corpus` tool for Track B — not started
- [x] Retry policy tested correct in isolation, not attached to a live stage (P-009)
- [x] Every request writes a trace with per-stage ms timings
- [x] All five guardrail layers (G1/G2/G3/G4/G5) implemented and unit-tested — G3/G4 thresholds
      are documented placeholders pending joint calibration (P-010)

## What works right now (verified, not assumed)

- Second same-day merge from `origin/main` (R's noise-floor validation + metric bug fix): clean
  fast-forward, no conflicts, tests still green ✅
- `src/vrag/harness/stages.py` — full pipeline: G1 → G2 → Retrieve → **G3** → Track A →
  **Track B (G4-gated)** → G5 → Assemble ✅
- G3 (`g3_confidence.py`) — top1<tau OR (top1−top5)<margin, unit-tested including the "clear
  winner passes / ambiguous close scores fail" distinction ✅
- G4 (`g4_groundedness.py`) — citation-ID validation + lexical overlap, unit-tested including an
  invented-chunk-id case and a real-but-ungrounded-answer case ✅
- `GenerateStage` — real Sarvam LLM client (`src/vrag/generation/sarvam_llm.py`) using
  provider-native `response_format: json_schema` (confirmed Sarvam supports this), with one
  repair-attempt on parse failure per spec. **Live path currently blocked by P-R13** (Sarvam
  outage), but the fallback-to-Track-A path is fully verified live, repeatedly, against the actual
  broken endpoint — which is itself a real (if accidental) end-to-end test of the two-track design
  under provider failure ✅
- Deadline propagation now holds *during* GenerateStage, not just as a pre-flight check — verified
  live: unguarded call = 10+ real seconds against a 200ms budget; guarded = ~204ms ✅
- `pytest` (my scope): 47/47 passing. `ruff check` / `mypy src` (my modules): clean ✅

## What is stubbed / faked / TODO

- Track B has never produced a real generated answer — Sarvam's chat endpoint has been down for
  this entire session (P-R13). The code path is real and ready; it's unverified end-to-end because
  the provider is unverified, not because anything here is fake.
- G3/G4 thresholds are placeholders, not calibrated numbers (P-010). Do not report a false-refusal
  rate anywhere until the real 150+150 calibration set exists and gets swept.
- No circuit breaker yet — see exit criteria above, now a more concrete need than before.
- No `search_corpus` tool for Track B.
- Retry policy tested correct in isolation but not attached to a live stage (P-009) —
  `generate()`'s repair-retry is a *different* mechanism (JSON-parse repair), not the tenacity
  policy; worth revisiting whether tenacity should also wrap the whole `generate()` call once
  Sarvam is back, for transient (not permanent) failures.
- Real browser mic click-through — still not done. Fourth time flagging this; genuinely just
  needs someone with a phone to do it, nothing more I can do about it from here.
- efSearch curve, A2/A3/A4 ablations — Workstream R's queue.

## Blockers

- **P-R13 (external, not code):** Sarvam's chat completions endpoint hangs on every valid request.
  User's call: retry later. I'll re-probe periodically but won't block other work on it.
- Real mic click-through needs a human with a phone — not blocking further backend work, but
  genuinely can't be done from here.

## Next session should start by

1. Re-run `scripts/probe_latency.py` (or a quick manual curl) to check if Sarvam's chat endpoint
   has recovered — if so, actually exercise Track B end to end for the first time and record real
   TTFT numbers as ADR-003.
2. Check `docs/PROGRESS_R.md` for R's latest state before touching anything.
3. Build the circuit breaker — P-R13 makes the case for it concrete: N failures/hangs in a rolling
   window should trip it and skip straight to Track A without even attempting the network call,
   rather than paying the timeout cost on every single request.
4. Click through the real mic UI on a phone. Still.
