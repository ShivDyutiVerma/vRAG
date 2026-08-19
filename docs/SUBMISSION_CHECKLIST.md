# SUBMISSION_CHECKLIST.md

Walked item-by-item against reality (not belief), per `docs/AGENT_BUILD_SPEC.md` §12 (P8 kickoff)
and §10. 13 core items below, restructured into requirement / evidence / status / remaining-action
form so status is auditable, not just checked. Updated 2026-08-20 after the redeploy that shipped
the STT and frontend fixes.

## Build / evidence

| # | Requirement | Evidence / location | Status | Remaining action |
|---|---|---|---|---|
| 1 | All 6+ chunking strategies implemented, evaluated, winner shipped | `docs/EVAL_RESULTS.md` §1, `src/vrag/chunking/strategies/*.py` | ✅ Done | None |
| 2 | Hybrid retrieval + rerank decision made from data | `docs/EVAL_RESULTS.md` §3, `docs/DECISIONS_R.md` R-010/R-012/R-037/R-038 | ✅ Done | None — includes a targeted follow-up experiment, not just the original ablation |
| 3 | Harness: deadline propagation, retries, circuit breaker, tool calling, structured output all working | `src/vrag/harness/{stage,budget,pipeline,retry}.py`, `pytest tests/harness/ tests/generation/` | ✅ Done, with one disclosed gap | `IDEMPOTENT_STAGE_RETRY` is built and unit-tested but not wired into any live stage (no current stage raises in a way retries would help) — either wire it in or leave as a documented, proven-but-currently-inapplicable building block |
| 4 | All 5 guardrail layers demoable on command; G3 calibration curve committed | `src/vrag/guardrails/g{1-5}_*.py`, `docs/assets/g3_calibration.png`, `pytest tests/guardrails/` (31/31) | ✅ Done, with one disclosed gap | G4's `MIN_OVERLAP_RATIO=0.15` is uncalibrated (real check, unvalidated threshold) — optional NLI entailment pass would fix this, not required to ship |
| 5 | P50/P70/P100 latency reported, per stage, from `scripts/bench_latency.py` only | `docs/LATENCY_BUDGET.md` | ✅ Done | LOCAL numbers meet the 200ms target; LIVE RENDER numbers (R-036) do not, disclosed plainly, not hidden |
| 6 | Live HTTPS URL, verified from a phone on mobile data | `https://vrag-voice.onrender.com` — verified via `/healthz`, real `/ask` calls, and real browser automation (2026-08-20) | ⏳ Partial | **Not yet verified from an actual phone on mobile data specifically** — browser-automation verification is real but is not the same test |
| 7 | README is submission-quality (`docs/AGENT_BUILD_SPEC.md` §10.1 structure) | `README.md` (repo root) | ✅ Done | Recommend a final read-through after the videos exist, in case anything in §12/§17 needs a last update |
| 8 | Public GitHub repo, verified from a logged-out browser | `github.com/ShivDyutiVerma/vRAG` | ⏳ Not yet verified | Confirm access from a logged-out browser session specifically |

## Videos

| # | Requirement | Evidence / location | Status | Remaining action |
|---|---|---|---|---|
| 9 | Demo video recorded (script per `docs/AGENT_BUILD_SPEC.md` §10.2) | — | ❌ **Not done** | Record: spoken Hindi question → live transcript; answer + citations + latency HUD; an out-of-scope question aborting via G3; an unsafe input refused via G1; a forced-low-budget request with stages shed; two seconds on the eval tables. **Requires the real human-microphone leg, still pending** |
| 10 | Process video cut (90s, footage from across the week) | — | ❌ **Not done** | Pull together footage captured across the build — this session's real diagnostic/fix work (browser automation screenshots, the R-037 forensic investigation, the R-038 experiment) is real, usable material, not staged |

## Promotion grid — per member × per platform (per `docs/TEAM_SPLIT.md` §7)

Every post tagged `#RAGInGoa`. A shared team post does **not** satisfy either member's
requirement. At least one Instagram account must be public.

| Member | IG (demo) | IG (process) | X (demo) | X (process) | LI (demo) | LI (process) | IG public? |
|--------|-----------|--------------|----------|--------------|-----------|---------------|-----------|
| Shiv (R) | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| Partner (P) | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |

## Final verification (Day 5 / Aug 22)

| # | Requirement | Status |
|---|---|---|
| 11 | Live link works from an incognito window on a different network | ⏳ Not yet verified |
| 12 | Repo public, README renders correctly on GitHub | ⏳ README just created locally — verify rendering after next push |
| 13 | Submission form submitted before 23:59 IST, Aug 22 — **no resubmissions possible** | ❌ Not started (correct at this stage — this is explicitly the last item) |

## Summary

**5/13 done, 3/13 partial/needs a specific re-check, 2/13 explicitly not started (correctly, this
early), 2 videos not recorded, 1 form not submitted.** The two items requiring the most remaining
work are the same two flagged throughout this session's docs: the real human-microphone
verification (blocks the demo video), and the two videos themselves (block the promotion grid and
final submission).
