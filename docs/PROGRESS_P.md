# PROGRESS_P — Workstream P running status

> Mine — edit freely, any time. Never edited by Workstream R.

**Last updated:** 2026-08-18 (Day 2, session 04 — same-day continuation)
**Current phase:** P4/P5 — Harness Hardening + Guardrails, Track B now genuinely working

## Where we are, in one paragraph

Third same-day sync: merged R's A2 (embedder — `e5-small` confirmed) and A3 (retrieval mode —
dense-only beats hybrid on this corpus, a real and slightly surprising finding) ablation work,
clean fast-forward. Sarvam's chat endpoint — down at the start of this session (P-R13) — recovered
mid-session, and Track B is now **verified working end to end for real**: a live query gets a
correct, grounded, Hindi-language generated answer with valid citations, passing G4, becoming the
final response (`track: "generative"`). Getting there took four real, evidence-based fixes: a
genuine Sarvam bug where `array`-typed schema fields make structured output hang/pad forever
(worked around with a CSV string instead), disabling `sarvam-105b`'s reasoning mode (it was
consuming the entire token budget before ever producing an answer), switching off the
"conversations" model variant (broken for structured output specifically), and strengthening the
system prompt after catching the model answering in English to a Hindi question. Also fixed a
budget-consistency bug found while re-testing: `GenerateStage` wasn't passing the real remaining
budget into `generate()`'s own timeout, so its unrelated internal default could cut off calls the
outer enforcement would otherwise have allowed to finish. Confirmed via repeated testing: Track
B's real completion time is highly variable (~1.4s–15s+, non-streaming), so it will rarely fire
within an actual 200ms request budget today — Track A is what ships in practice, exactly as the
two-track design intends, but Track B is now proven correct and ready for when real streaming
lands.

## Phase exit criteria — P4/P5

- [x] Every stage typed with Pydantic in/out
- [x] Forced-tight-budget degradation test green, and real in production (Track B sheds under a
      normal request budget; verified it *doesn't* shed and produces a real generative answer
      under a generous budget — both paths now directly observed, not just unit-tested)
- [x] Deadline propagation holds during a stage, not just before it (P-011), and is now consistent
      between the outer `wait_for` and `generate()`'s own timeout (P-013)
- [ ] Circuit breaker — still not built. More clearly motivated than ever: Sarvam's response time
      varies by 10x+ call to call, which is exactly the instability a circuit breaker exists for.
- [ ] `search_corpus` tool for Track B — not started
- [x] Retry policy tested correct in isolation, not attached to a live stage (P-009)
- [x] Every request writes a trace with per-stage ms timings
- [x] All five guardrail layers (G1–G5) implemented, unit-tested, and — for G4 — now verified
      against a real generated answer, not just synthetic test cases

## What works right now (verified, not assumed)

- Third same-day merge from `origin/main` (R's A2/A3 ablations): clean fast-forward, tests green
- **Track B end-to-end, for real:** `POST`/`WS` request → real Sarvam call → structured JSON
  (`reasoning`, `answer`, `cited_chunk_ids_csv`) → parsed → G4-checked → accepted →
  `AnswerResponse(track="generative", status="answered", ...)`. Directly observed, repeatedly,
  against the live API — not mocked, not simulated ✅
- `src/vrag/generation/sarvam_llm.py` — `reasoning_effort: null`, `sarvam-105b` (not the
  `-conversations` variant), CSV-string citations, a real Hindi-language system prompt
  instruction that's been verified to actually work (100% lexical overlap after the fix, vs. ~9%
  before it, on the same query) ✅
- G3/G4 unaffected by any of today's Track B fixes — G4 in particular has now been exercised
  against a real model output, not just hand-written test fixtures, and correctly passed a
  genuinely well-grounded answer and correctly failed a genuinely ungrounded (wrong-language) one
  before the prompt fix ✅
- Real provider latency data for ADR-003 recorded in `docs/DECISIONS_P.md` P-012 (both the outage
  numbers and the recovery numbers — P50 chat TTFT 452ms once healthy) ✅
- `pytest` (my scope): 47/47 passing throughout all of today's changes. `ruff`/`mypy`: clean ✅

## What is stubbed / faked / TODO

- No circuit breaker yet — now the clearest concrete gap, given directly observed Sarvam latency
  variance (10x+ call to call). Next thing to build.
- No real token streaming for Track B — non-streaming waits for the full ~500-token structured
  response, which is why it rarely clears any realistic budget even though raw TTFT (452ms P50)
  is much more reasonable. Documented as an accepted, deliberate gap (P-014), not hidden.
- No `search_corpus` tool for Track B.
- G3/G4 thresholds are still uncalibrated placeholders (P-010) — today's testing exercised them
  against real data and they behaved sensibly, but that's not the same as a calibration sweep.
- Retry policy (`tenacity`) tested correct in isolation, not attached to a live stage.
- Real browser mic click-through — still not done. Genuinely just needs a human with a phone.
- efSearch curve, A4/A5 ablations — Workstream R's queue (A4 next per their side).

## Blockers

- None on my side. `GROQ_API_KEY` still empty if a second generation provider is ever wanted for
  comparison — not currently needed since Sarvam is healthy again.

## Next session should start by

1. Build the circuit breaker — trip after N failures/slow-timeouts in a rolling window, skip
   straight to Track A without attempting the network call. Directly motivated by today's
   observed Sarvam variance, not hypothetical.
2. Check `docs/PROGRESS_R.md` for R's latest (A4 rerank should land soon).
3. Consider whether real token streaming for Track B is worth the time investment before Day 3's
   scheduled end, given it's the actual unlock for Track B mattering in practice — currently it's
   correct but rarely fires under a real budget.
4. Click through the real mic UI on a phone. Still. Genuinely just do this one.
