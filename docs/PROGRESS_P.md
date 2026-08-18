# PROGRESS_P — Workstream P running status

> Mine — edit freely, any time. Never edited by Workstream R.

**Last updated:** 2026-08-18 (Day 2, session 02)
**Current phase:** P4 — Harness Hardening (per `docs/TEAM_SPLIT.md` §5 Day 2 goals)

## Where we are, in one paragraph

Synced with Workstream R's Day 1 work first thing this session: R shipped real chunking (A1
ablation complete, `metadata_aware` chosen), a real `HybridRetriever`, and wired `retrieve()` to
use it — with a clean automatic fallback to the Day-1 stub shape when no index is present locally,
so this machine (no downloaded corpus/model) keeps working unchanged. Merged `origin/main` into
`workstream-p` — fast-forward, no conflicts. Then built out the Day 2 harness: `/ask` and `/voice`
now run through a real pipeline (`src/vrag/harness/stages.py`) — G1 input safety → G2 scope/
language → Retrieve → Track A extraction → G5 output redaction → Assemble — with real deadline
propagation via `Budget`, and every request emits a `TraceRecord` to `traces.jsonl` (fired after
the response is built, never blocking it). All three demo-critical refusal-adjacent paths
(normal answer, G1 unsafe-input refusal, G2 degenerate-input refusal) verified live against the
running app. G3/G4 and Track B are explicitly not today's scope — joint work with R, Day 3.

## Phase exit criteria — P4 (partial; full harness hardening continues Day 3)

- [x] Every stage typed with Pydantic in/out (`StageResult`, `AnswerResponse`)
- [x] Forced-tight-budget degradation test green (`tests/harness/test_degradation.py`) — proves
      the shedding *mechanism* works; nothing sheds in the *real* pipeline yet since no stage is
      optional until Track B lands (Day 3) — see `docs/DECISIONS_P.md`
- [ ] Circuit breaker — not started, needs a real network-calling stage (Track B) to protect
- [ ] `search_corpus` tool for Track B — not started, depends on generation provider decision
- [x] Retry policy tested and proven correct in isolation (`tests/harness/test_retry.py`) — not
      yet attached to a live stage, see `docs/DECISIONS_P.md` P-009 for why
- [x] Every request writes a trace with per-stage ms timings (`src/vrag/telemetry/trace.py`)
- [x] G1/G2/G5 guardrails implemented and unit-tested

## What works right now (verified, not assumed)

- Merge with Workstream R's work: clean fast-forward, `pytest`/`ruff`/`mypy` all green on the
  merged tree within my scope (R's `[retrieval]`-extra tests need `faiss`/`torch`/`bm25s`, not
  installed on this machine by design — CI installs all extras and covers that) ✅
- `src/vrag/harness/stages.py` — real pipeline: `InputGuardStage` (G1), `ScopeGuardStage` (G2),
  `RetrieveStage`, `ExtractAnswerStage` (Track A), `OutputGuardStage` (G5), `AssembleStage` ✅
- `POST /ask` and `WS /voice` both run the real pipeline via `build_answer()` — no more direct
  `retrieve()` calls bypassing the harness ✅
- G1 (`src/vrag/guardrails/g1_input_safety.py`) — regex denylist for unsafe content + a length
  guard, unit-tested against Hindi and English unsafe phrasings + a prompt-injection attempt ✅
- G2 (`src/vrag/guardrails/g2_scope_language.py`) — empty/degenerate/no-recognisable-words
  detection, accepts Devanagari and Latin-script (romanised) queries ✅
- G5 (`src/vrag/guardrails/g5_output_safety.py`) — email/Indian-mobile/card-number redaction ✅
- Live verification of all three paths through the actual running app: normal query → `answered`;
  unsafe query ("बम बनाने का तरीका बताओ") → `refused` via G1; degenerate query ("???") → `refused`
  via G2 ✅
- `TraceRecord` → `traces.jsonl`, one per request, fired as a background task after the response
  is built (confirmed via a live run: both an `answered` and a `refused` trace wrote correctly) ✅
- Forced-tight-budget test (`Budget(total_ms=0.001)`) — the real pipeline still returns a valid
  `AnswerResponse`, never hangs or raises ✅
- `pytest` (my scope): 37/37 passing. `ruff check` / `mypy src` (my modules): clean ✅

## What is stubbed / faked / TODO

- No G3 (retrieval confidence gate) or G4 (groundedness) — joint work with Workstream R, needs
  real retrieval scores and calibration data neither of us has built yet. Day 3 per
  `docs/TEAM_SPLIT.md` §5.
- No Track B (LLM generation) — every answer is still Track A only (best-supporting span,
  verbatim). Generation provider probe (`scripts/probe_latency.py`) hasn't run — blocked on
  deciding Groq vs Sarvam LLM, which needs the Phase 0 latency probe neither track has run yet.
- No circuit breaker — nothing to protect against yet without a real external LLM call on the
  hot path.
- Retry policy (`tenacity`) is tested correct in isolation but not attached to any live stage —
  `retrieve()` never raises by contract, so there's nothing to retry there. Will attach to Track
  B's LLM call.
- Real browser mic click-through still not done — verified via TTS-generated audio (Day 1, P-007)
  and via curl/WS-client smoke tests (today), but never through an actual phone browser tapping
  the mic button. Carrying this forward as the top follow-up again.
- efSearch curve, A2/A3/A4 ablations — Workstream R's queue, not mine.

## Blockers

- None. Generation provider choice (Groq vs Sarvam LLM) blocks starting Track B — needs the
  latency probe, which needs both API keys tested from the deployment region. I have
  `SARVAM_API_KEY`; `GROQ_API_KEY` is still empty in `.env`.

## Next session should start by

1. Click through the real mic UI on an actual phone (still not done — third session in a row this
   is deferred; worth just doing it before starting new feature work).
2. Check `docs/PROGRESS_R.md` for R's latest state before touching anything.
3. If a Groq key is available, run `scripts/probe_latency.py` and record ADR-003 (currently
   blocked on both tracks per the shared `docs/PROGRESS.md`).
4. Start Track A/B split properly: pick a generation provider, wire a real LLM call behind a new
   optional `GenerateStage`, and only then does the forced-budget degradation test get to prove
   something sheds for real.
