# DECISIONS_P — Workstream P's ADR log

> Mine — numbered P-001, P-002... Never touched by Workstream R.

## P-001 — Repo skeleton uses pip + venv, not uv/poetry

**Date:** 2026-08-17 · **Status:** Accepted
**Context:** `docs/BUILD_PLAN.md` P0 suggests uv or poetry; neither is installed on this machine.
System Python is 3.14.6 (spec targets 3.11+).
**Decision:** Standard `venv` + `pip install -e ".[dev]"` against `pyproject.toml`. No new
dependency (uv/poetry) added just for tooling preference.
**Consequences:** Anyone else setting up locally needs `pyproject.toml` + venv, not a lockfile
tool. If R needs a different Python minor version for ML wheel compatibility (faiss-cpu,
onnxruntime), that's a separate venv, not a blocker for this track.

## P-002 — Deploy insurance target: Hugging Face Spaces (Docker SDK) — SUPERSEDED by P-005

**Date:** 2026-08-17 · **Status:** Superseded by P-005
**Context:** `AGENT_BUILD_SPEC.md` §5.3 names HF Spaces as deploy insurance and
Render/Railway/Fly.io as the primary, region-optimised recommendation.
**Decision:** Ship the Day 1 insurance deploy on HF Spaces first. Revisit region-optimised hosting
after the Phase 0 latency probe if Spaces' region measurably hurts `t_pipeline`.
**Rationale:** User has (or can quickly get) an HF account; free HTTPS; simplest path to "live
today," which is the actual Day 1 goal — not the final production host.
**Why superseded:** see P-005 — HF now requires a PRO subscription to host Docker/Gradio Spaces on
free compute; only static Spaces are free, which can't run our FastAPI backend.

## P-005 — Deploy insurance target switched to Render

**Date:** 2026-08-17 · **Status:** Accepted · Supersedes P-002
**Context:** Attempted the HF Spaces deploy (P-002) and hit a real blocker: `create_repo(...,
space_sdk="docker")` returned `402 Payment Required` — "hosting Gradio and Docker Spaces on free
cpu-basic requires a PRO subscription." This is a policy change since `AGENT_BUILD_SPEC.md` was
written (docs assumed free HTTPS on HF Spaces). Confirmed with the user rather than silently
absorbing a cost or working around it with a fake deploy.
**Decision:** Deploy to Render instead — `AGENT_BUILD_SPEC.md` §5.3's own primary recommendation,
not just the fallback. Same Dockerfile, no code changes; added `render.yaml` (Blueprint) at repo
root.
**Consequences:** Region choice for the Phase 0 latency probe now targets Render's available
regions instead of HF's. If HF Spaces access changes later (e.g. PRO subscription obtained), the
Dockerfile still works there unmodified — this wasn't wasted effort.

## P-003 — Sarvam realtime STT: bounded grace period after sending `end`

**Date:** 2026-08-17 · **Status:** Accepted
**Context:** Discovered via a real WS smoke test (silent audio, no API mocking) that Sarvam's
`speech-to-text-realtime` endpoint does not always close its side of the connection promptly after
receiving `{"event": "end"}` — observed hanging indefinitely on silence-only audio with no
transcript to finalize.
**Decision:** `src/vrag/stt/sarvam.py::stream_transcribe` waits up to 3s for a trailing message
after the audio sender completes, then gives up and lets the `async with` block close the
connection, instead of blocking forever on `sarvam_ws.recv()`.
**Consequences:** A request can never hang the `/voice` WebSocket indefinitely because of an idle
STT connection. If Sarvam's realtime API behavior changes (confirmed by re-testing), revisit
whether the grace period is still needed.

## P-006 — Configure Python logging at app startup

**Date:** 2026-08-17 · **Status:** Accepted
**Context:** While diagnosing a live-only `/voice` hang, discovered that `logger.info()`/`.error()`
calls throughout the app were silently dropped — `logging.basicConfig()` was never called, so the
root logger defaulted to `WARNING`. Uvicorn configures its own loggers but not arbitrary module
loggers like `vrag.stt.sarvam`.
**Decision:** Call `logging.basicConfig(level=logging.INFO, ...)` once at the top of
`src/vrag/api/main.py`.
**Consequences:** All app logging is now actually visible in `traces`/Render's log stream. Without
this, the connect-timeout diagnostic added in the same debugging session would have been silently
useless on the hosted deploy where there's no debugger to attach.

## P-004 — Moved `voice-rag-ui-preview.html` into `frontend/reference/`

**Date:** 2026-08-17 · **Status:** Accepted
**Context:** `docs/FRIEND_BRIEF.md` §2 and `docs/TEAM_SPLIT.md` §2 both reference the preview at
`frontend/reference/voice-rag-ui-preview.html`, but it was committed at the repo root.
**Decision:** `git mv voice-rag-ui-preview.html frontend/reference/voice-rag-ui-preview.html` to
match what the docs already assume, rather than updating every doc reference.
**Consequences:** None outside this track's files.

## P-007 — Verified the real STT golden path against the live deploy using Sarvam TTS

**Date:** 2026-08-17 · **Status:** Accepted
**Context:** Needed to confirm real speech → real Sarvam STT → transcript → answer actually works
on the live Render URL (the actual Day 1 exit bar), not just locally, without a physical
microphone available in this environment.
**Decision:** Used Sarvam's own TTS API (`POST /text-to-speech`, `speech_sample_rate: 16000`) to
generate genuine Hindi speech audio for "भारत में सबसे ऊँचा पर्वत कौन सा है", streamed the raw
16kHz PCM16 WAV frames to the live `/voice` WebSocket exactly as a browser would, and confirmed a
correct `transcript_final` ("भारत में सबसे ऊँचा पर्वत कौन सा है?") followed by a correct
`answer_final` — full round trip in ~2.4s.
**Consequences:** This is real audio processed by real Sarvam STT end to end on the actual public
deployment — satisfies the "never mock the STT path" hard rule and the Day 1 exit bar without
needing a physical mic in this session. Worth reusing this technique for Phase 6's TTS'd test
query set (`AGENT_BUILD_SPEC.md` §7.4) rather than reinventing it. Also surfaced P-R12 (see
`docs/RISKS.md`) — a Render connection-teardown quirk that doesn't block this flow.

## P-012 — Real provider latency probe results (feeds ADR-003, not yet merged into shared DECISIONS.md)

**Date:** 2026-08-18 · **Status:** Accepted — recorded here per `docs/TEAM_SPLIT.md` §3 (shared
`DECISIONS.md`/ADR-003 is sync-only; this is the source data for whoever does that merge)
**Context:** Ran `scripts/probe_latency.py --n 30` for real, using the Sarvam key on this machine
(`GROQ_API_KEY` still empty — Groq's chat TTFT column is therefore blank, not zero/bad).

**First run, while Sarvam's chat endpoint was down (see `docs/RISKS.md` P-R13):**

| Measurement | Provider | P50 | P95 | P100 | Failures |
|---|---|---|---|---|---|
| TCP+TLS connect | sarvam (api.sarvam.ai) | 210.9ms | 315.8ms | 368.6ms | 0/30 |
| TCP+TLS connect | groq (api.groq.com) | 173.5ms | 218.3ms | 317.9ms | 0/30 |
| Chat TTFT | sarvam (sarvam-105b) | — | — | — | **30/30 (all timed out)** |
| Chat TTFT | groq | — | — | — | SKIPPED — no `GROQ_API_KEY` |

**Second run, same day, after Sarvam recovered:**

| Measurement | Provider | P50 | P95 | P100 | Failures |
|---|---|---|---|---|---|
| TCP+TLS connect | sarvam (api.sarvam.ai) | 233.0ms | 1697.7ms | 1706.0ms | 0/30 |
| TCP+TLS connect | groq (api.groq.com) | 173.4ms | 293.9ms | 571.8ms | 0/30 |
| Chat TTFT | sarvam (sarvam-105b) | **452.4ms** | **858.1ms** | **903.8ms** | 4/30 |
| Chat TTFT | groq | — | — | — | SKIPPED — no `GROQ_API_KEY` |

**Decision:** The first run's 30/30 Sarvam chat failure was a genuine transient outage, not a
latency data point and not our bug — confirmed by the second run succeeding with reasonable
numbers a few hours later, same code, same key, same day. Streaming TTFT (P50 452ms) is
encouraging but still ~4x the 110ms target; note this is TTFT specifically, not full-completion
time (P-014 below covers why those differ a lot for our non-streaming implementation). Sarvam's
own TCP+TLS P95/P100 got notably worse in the second run (315ms → 1698ms) — network conditions
from this dev machine are themselves noisy; don't over-read either single run. ADR-003 (shared)
still needs a joint sync to actually record this — Groq remains untested (no key).
**Consequences:** These TCP/TLS numbers are real network physics from this specific dev machine's
location, not the eventual deployment region — don't treat them as final either, they're one data
point for the eventual region-choice decision in `docs/ARCHITECTURE.md` §Deploy runbook.

## P-013 — Track B fixes: CSV citations (not array), disable reasoning, switch model, consistent timeout

**Date:** 2026-08-18 · **Status:** Accepted
**Context:** Sarvam's chat endpoint recovered mid-session (P-R13 resolved). Getting Track B to
actually produce a correct, grounded, Hindi-language answer required four separate fixes, each
found by direct testing against the live API, not guessed:
1. **Array-field bug** (P-R15): `response_format: json_schema` with an `array`-typed required
   field (`cited_chunk_ids: list[str]`) makes the model pad whitespace forever instead of closing
   the JSON — confirmed by removing the array field alone and watching `finish_reason` flip from
   `"length"` to `"stop"`. Fixed by changing the schema to `cited_chunk_ids_csv: str`
   (comma-separated), parsed into a list via a `GeneratedAnswer.cited_chunk_ids` property —
   `src/vrag/harness/stages.py`'s `GenerateStage` needed zero changes since it only ever used
   `.cited_chunk_ids`.
2. **Reasoning starves the answer** (P-R16): `sarvam-105b` emits a billed `reasoning_content`
   chain-of-thought before the real content; with `max_tokens=512`, a non-trivial prompt can
   consume the entire budget on reasoning and never produce `content`. Fixed with
   `"reasoning_effort": null` — Sarvam's own documented recommendation for latency-sensitive use.
3. **Wrong model for structured output**: `sarvam-105b-conversations` (chosen in P1 for its
   voice-agent-tuned billing) produces pure whitespace under `response_format: json_schema`
   regardless of schema shape or `reasoning_effort` — switched to plain `sarvam-105b`, which
   works correctly.
4. **Answered in English despite a Hindi question**: with the original system prompt ("respond in
   the same language as the user's question"), the model sometimes ignored this and answered in
   English even for a Hindi query over Hindi context — which also (correctly) tanked G4's lexical
   overlap check to ~9%, since the answer shared almost no tokens with Devanagari context.
   Strengthened the system prompt with an explicit, capitalized instruction naming the script;
   confirmed fixed — overlap on the same query went to 100% after the change.
**Decision:** All four fixes landed together in `src/vrag/generation/{schemas,sarvam_llm}.py`.
**Consequences:** Track B verified working end-to-end for the first time this session: real
Sarvam call → correct Hindi structured answer → correct citation → G4 groundedness pass →
`track: "generative"` in the final `AnswerResponse`. Also fixed a related bug while testing this
(`GenerateStage` was calling `generate_track_b()` without passing the actual remaining budget as
`generate()`'s own `timeout_s`, so its unrelated internal 10s default could cut off a call the
outer `asyncio.wait_for` would otherwise have allowed to finish) — now both timeouts derive from
the same `ctx.budget.remaining_ms` value.

## P-014 — Track B's non-streaming completion time is highly variable; Track A remains the practical default

**Date:** 2026-08-18 · **Status:** Accepted
**Context:** Repeated identical `generate()` calls against the now-healthy Sarvam endpoint took
anywhere from ~1.4s to over 15s to fully complete (non-streaming — waits for the entire
structured JSON response). The probe's streaming TTFT is a real, much better number (P50 452ms,
P95 858ms, `docs/DECISIONS_P.md` P-012) — but that's time-to-*first*-token, not time-to-*full*-
completion, and our non-streaming implementation can't benefit from it.
**Decision:** No code change here — this is a documented, accepted consequence of the earlier
non-streaming design choice, not a bug to fix today. `GenerateStage` already handles it correctly:
under any realistic request budget (200ms, even 2-5s), Track B times out and Track A's
already-computed answer stands, exactly as designed.
**Consequences:** Track B will rarely if ever fire within the actual product's real time budget
until real token streaming is built (AGENT_BUILD_SPEC.md §3.3's "begin emitting on the first
sentence"). Worth being explicit about this in the README's honest-limitations section — Track B
is real, tested, and correct, but not fast enough yet to be more than an occasional/best-effort
upgrade over Track A under the current architecture.

## P-010 — G3/G4 mechanisms implemented ahead of calibration; thresholds are documented placeholders

**Date:** 2026-08-18 · **Status:** Accepted
**Context:** `docs/TEAM_SPLIT.md` §5 schedules G3/G4 as joint work with Workstream R on Day 3,
reasoning that calibration needs real retrieval scores. But `retrieve()` already returns real
scores now (R wired `HybridRetriever` on Day 1) — what's actually still joint/blocked is the
*calibration sweep* (150 in-domain + 150 out-of-domain queries), not the mechanism itself.
**Decision:** Built `g3_confidence.py` (top1<tau OR (top1-top5)<margin) and
`g4_groundedness.py` (citation-ID validation + lexical overlap) with explicit UNCALIBRATED
placeholder thresholds (tau=0.35, margin=0.05, overlap_ratio=0.15), wired live into
`src/vrag/harness/stages.py` (`GroundGateStage` for G3, inline in `GenerateStage` for G4). Chose
placeholders using the literature prior in `docs/TECH_MENU.md` §S3 (query-doc cosine typically
0.30-0.55) rather than guessing blind.
**Consequences:** All 5 guardrail layers are now functionally present and demoable, ahead of the
Day 3 schedule — satisfies `docs/BUILD_PLAN.md` P5's exit criterion structurally, but the
calibration curve, false-refusal-rate measurement, and correct-refusal-rate measurement are still
outstanding and still joint work. Do not report a false-refusal rate or claim these numbers are
calibrated anywhere — they are a documented starting guess, not a result.

## P-011 — Enforce the actual remaining budget during GenerateStage, not just before it

**Date:** 2026-08-18 · **Status:** Accepted
**Context:** Discovered directly while testing GenerateStage against a live outage (P-R13, Sarvam's
chat endpoint hanging): `run_pipeline`'s budget check (`stage.optional and not
ctx.budget.can_afford(min_viable_ms)`) only decides whether to *start* an optional stage — nothing
stopped the stage from running arbitrarily long once started. A 200ms-budget request took 10+ real
seconds because `GenerateStage` awaited `generate_track_b()` unguarded.
**Decision:** `GenerateStage.run()` now wraps its call in `asyncio.wait_for(...,
timeout=ctx.budget.remaining_ms / 1000)` — the *actual* remaining budget at call time, not a
fixed number. A timeout is treated exactly like a generation failure: skip, keep Track A.
**Consequences:** Verified with a live (broken) Sarvam endpoint: request time dropped from 10+s to
~204ms against a 200ms budget. This is a general principle worth remembering for any future
network-calling optional stage, not just this one — a `min_viable_ms` pre-check alone does not
make deadline propagation real; the stage itself must respect the clock once running.

## P-009 — Retry policy not attached to `retrieve()`; tested in isolation instead

**Date:** 2026-08-18 · **Status:** Accepted
**Context:** `docs/TEAM_SPLIT.md` §5 lists "retries" as Day 2 harness-hardening work. But
`retrieve()`'s contract (`src/vrag/retrieval/interface.py`) is explicit: "Never raises. Returns []
on internal failure." A `tenacity` retry policy has nothing to catch if the wrapped call never
raises — wrapping it anyway would be a no-op that looks like real hardening but isn't.
**Decision:** Leave `RetrieveStage` (`src/vrag/harness/stages.py`) unwrapped. Instead,
`tests/harness/test_retry.py` proves `IDEMPOTENT_STAGE_RETRY` (`src/vrag/harness/retry.py`) itself
works correctly in isolation — retries a transient failure once and succeeds, gives up and
reraises after the configured cap — so the policy is verified and ready the moment there's a
stage that can actually raise (Track B's LLM call, Day 3 per `docs/TEAM_SPLIT.md` §5).
**Consequences:** No dishonest "retries: ✓" checkbox — `docs/PROGRESS_P.md` states plainly that
retries aren't exercised on the live request path yet, only proven as a mechanism. Revisit this
note once Track B lands; if the LLM call is the first real attach point, link back here.

## P-008 — Frontend sends `stop` proactively on final transcript

**Date:** 2026-08-17 · **Status:** Accepted
**Context:** Found while reviewing the golden-path test (P-007) that `frontend/index.html` only
sent `{"event": "stop"}` to the server when the user clicked the Stop button mid-listening — not
when a `transcript_final` arrived naturally. Since `stopMicCapture()` tears down the local mic
without telling the server, the browser's audio generator on the server side would sit waiting for
audio that will never come, with no signal to finalize the Sarvam session.
**Decision:** `frontend/index.html` now sends `{"event": "stop"}` immediately alongside
`stopMicCapture()` when a final transcript is received.
**Consequences:** Every real interaction now closes its STT session promptly instead of relying on
the client eventually disconnecting. Interacts with P-R12 (Render close-frame lag) but doesn't
depend on it being fixed.

## P-015 — G3 calibration applied: TAU=0.8835, balanced operating point (joint decision with R)

**Date:** 2026-08-18 · **Status:** Accepted
**Context:** R gathered real G3 calibration data (`docs/DECISIONS_R.md` R-015) — 300 queries scored
against the real production index — and found `docs/EVAL_PROTOCOL.md`'s two targets (false-refusal
<10% in-domain, correct-refusal >80% out-of-domain) are **not simultaneously reachable** via
top1-cosine TAU gating on this corpus: at 10% false-refusal, correct-refusal is only 38%; reaching
75-79% correct-refusal costs 19-30% false-refusal. R deliberately did not pick an operating point
despite `g3_confidence.py` being joint-owned, correctly treating the pick as a value judgment (how
often is refusing a real question acceptable vs. confidently answering with a weak match), not a
data question either track can settle alone (`docs/RISKS.md` R-R19).
**Decision:** Asked the user directly, since this is exactly the kind of product tradeoff neither
track's engineering judgment should silently resolve — presented three real reference points from
R's sweep (favor-answering: TAU=0.8487, 4.7%/13.3%; balanced: TAU≈0.8835, 19.3%/75.3%;
favor-refusing: TAU=0.8918, 30.0%/79.3%) plus the option to leave it uncalibrated. User chose
**balanced**, explicitly weighing both `EVAL_PROTOCOL.md` targets equally rather than favoring
either failure mode — the same tie-break principle R's own `pick_operating_point()` in
`scripts/eval_g3_calibration.py` implements. Applied `TAU=0.8835` to `src/vrag/guardrails/
g3_confidence.py`; `MARGIN` left at `0.05` (not independently re-swept at this TAU — R's sweep held
MARGIN=0 while varying TAU; the interaction between MARGIN and the new TAU is real work still open).
**Verification:** Checked the Day-1 stub's fallback path (`src/vrag/retrieval/interface.py`, used
whenever no real index is present on disk — every fresh clone, CI) still clears the new TAU: stub
top1=0.91 > 0.8835, margin 0.91-0.52=0.39 > 0.05, so `test_api.py` and all other tests that exercise
`/ask` end-to-end against the stub are unaffected. Two `test_g3_confidence.py` cases used score
ranges written against the old TAU=0.35 placeholder and would have silently changed which branch
they exercised (or started failing outright) under the real value — updated both to realistic
0.88-0.92-range scores that still test the same TAU-pass/margin-fail and TAU-pass/single-chunk
paths. 62/62 tests green (my scope; R's `retrieval`/`index`/`chunking` extras not installed
locally, not run). `ruff`/`mypy` clean.
**Consequences:** G3 is no longer a silent no-op on the TAU check in production — previously
TAU=0.35 never fired against real cosine scores (0.82-0.96 range), so only the untested MARGIN
check was doing any real refusing. Now ~1-in-5 real in-domain questions will be wrongly refused,
and ~3-in-4 genuinely out-of-scope questions will be correctly caught — an explicit, chosen
tradeoff, not an accident. `docs/RISKS.md` R-R19 closed as resolved. Re-sweep MARGIN independently
at TAU=0.8835 flagged as follow-up work, not done today (needs R's index locally, which I don't
have — R's sweep script and calibration set are reusable for this without new data collection).

**Update 2026-08-18 (same day):** R completed exactly the follow-up this ADR flagged, same day —
`docs/DECISIONS_R.md` R-017. `MARGIN=0.05` did not carry over to the new `TAU=0.8835`: verified
live it caused 88.0% false-refusal, not the 19.3% this ADR's design target states, because
in-domain top1-vs-top5 gaps are naturally tiny at this operating point. A fine sweep confirmed no
useful non-zero `MARGIN` exists at this `TAU` on this corpus (even `MARGIN=0.01` alone → 28.7%).
Set `MARGIN=0.0`, verified live (2/3 previously-blocked test queries now answer correctly, 1/3
still correctly abstains via a legitimate `TAU` check). Worth recording plainly: this was a real
bug in what I shipped, not a hypothetical — G3 would have wrongly refused roughly 9 in 10 real
in-domain questions in production between this ADR's commit and R's fix, a much worse outcome than
the 19.3% the chosen operating point was supposed to deliver. Caught because `g3_confidence.py` is
joint-owned and R re-verified the shipped result against real data rather than trusting the design
target — exactly what joint ownership is for. Merged into `workstream-p` clean, no conflicts.
84/84 tests green after merge (R split one test into two: one confirming `MARGIN=0.0`'s current
no-op behavior at this operating point, one confirming the margin mechanism itself still works via
`monkeypatch` for any future recalibration).

## P-016 — Circuit breaker for Track B: only "fair-chance" outcomes move it, not every timeout

**Date:** 2026-08-18 · **Status:** Accepted
**Context:** `docs/PROGRESS_P.md`'s own "next session should start by" list flagged the circuit
breaker as the clearest remaining gap, motivated by directly observed Sarvam latency variance
(10x+ call to call, P-R13/P-R17) — every request during the P-R13 outage paid the full
remaining-budget timeout cost probing an endpoint that was going to fail anyway.
**The obvious design has a real bug, caught before writing any code:** a breaker that counts every
`GenerateStage` timeout/failure as a provider-health signal would trip open almost immediately
under completely ordinary, healthy-Sarvam production traffic — not because Sarvam is unhealthy,
but because P-014 already established Track B's real non-streaming completion time (1.4s-15s+)
exceeds almost any realistic per-request budget (~200ms total, often less by the time
`GenerateStage` runs) regardless of provider health. That's the two-track design's normal,
expected shedding behavior, not evidence of an outage. Counting it as a failure would (a) add no
real protection beyond the budget/timeout mechanism already in place (P-011/P-013), since the
breaker would just be permanently open in steady state anyway, and (b) actively break generous-
budget calls made for real testing (e.g. a manual `/ask` with a large `budget_ms`, exactly how
Track B was verified working in the previous session) by rejecting them outright during the open
window, for a reason unrelated to those specific calls.
**Decision:** Built `src/vrag/generation/circuit_breaker.py` — a standard closed/open/half-open
state machine (`CircuitBreaker`), plus `should_count_as_health_signal(timeout_s,
min_fair_timeout_s)`: an outcome only counts toward the breaker if the call was given at least
`MIN_FAIR_TIMEOUT_S=2.0` seconds — comfortably (>2x) above Sarvam chat's measured P95 TTFT (858ms,
P-012), enough that the provider had a fair chance to at least start responding or fail fast.
Below that floor, the outcome is inconclusive (could be either "provider is fine, budget was just
tight" or "provider is actually struggling") and is ignored either way. A completed call (success)
is always recorded regardless of how much time it was given — success is unambiguous evidence of
health no matter the allowance. Wired into `GenerateStage` in `src/vrag/harness/stages.py`:
`allow_request()` gates the network call entirely (skip reason: "circuit breaker open..."),
`record_failure()`/`record_success()` fire on the two existing failure branches (outer
`TimeoutError`, `result is None`) and the success path respectively, all gated by
`should_count_as_health_signal` except the unconditional success recording. A module-level
singleton (`TRACK_B_BREAKER`) holds state, since `GenerateStage` is instantiated fresh per request
via `default_stages()` — a per-instance breaker would reset every request and never accumulate
anything.
**Verification:** 15 new tests: `tests/generation/test_circuit_breaker.py` (10, pure state-machine
tests against an injectable fake clock — no real `time.sleep`, deterministic) and
`tests/harness/test_generate_stage_circuit_breaker.py` (5, proving `GenerateStage` actually
consults the breaker — an open breaker skips without calling `generate_track_b` at all; a
tight-budget failure does *not* move a fresh breaker; a fair-chance failure does; success closes
an open one). Then a real end-to-end run against the live Sarvam API (not mocked) with a 15s
budget: `track="generative"`, `stages_skipped=[]`, breaker state `CLOSED` after — confirms the
closed-state passthrough works in the real async pipeline, not just against mocks. 77/77 tests
green (62 pre-existing + 15 new). `ruff check .` (repo-wide) and `mypy` on changed files clean.
**Consequences:** Given the `MIN_FAIR_TIMEOUT_S=2.0` gate, most default-budget (~200ms) production
requests today don't move the breaker either way — it's real, correct, tested infrastructure that
is currently under-exercised by design, not a limitation to fix. Its practical value today is
protecting repeated generous-budget calls (manual testing, or any future per-request budget
override) from hammering a genuinely down provider; its value grows directly with whatever closes
P-014's gap (real token streaming), since that's what would bring ordinary request budgets close
enough to Track B's real completion time for the breaker's fair-chance window to matter on the
normal request path.

## P-017 — Track B switched to streaming; found and mitigated a second, more common Sarvam bug

**Date:** 2026-08-18 · **Status:** Accepted
**Context:** Started this session's next item — real token streaming for Track B, flagged
repeatedly (P-014, P-016) as the actual unlock for Track B mattering under a real budget. Before
writing any client-facing streaming code, probed the live API with our exact schema/prompt to
measure how long until the `answer` field's content actually starts appearing (not just raw
TTFT), since the schema's reasoning-before-answer field order (a documented invariant, CLAUDE.md)
means the model must write out `reasoning` before any of `answer` streams in.
**What the probe actually found, which changed the scope of this work:** with a single-chunk
context, streaming worked and completed correctly. With a realistic 5-chunk context (closer to a
real k=5 retrieval, forcing the model to discriminate between a correct passage and 4
topically-adjacent distractors), 2 of 3 initial live runs never even reached the `answer` key: the
model wrote a long, unbounded chain-of-thought into our own schema's `reasoning` field (which
`reasoning_effort: null` does NOT constrain — that flag only disables Sarvam's separate hidden CoT
mechanism, not verbosity in a field we defined ourselves) and exhausted `max_tokens=512` before
ever reaching `answer`. Fixed by adding an explicit instruction to `_SYSTEM_PROMPT`: keep
`reasoning` to one short sentence. Verified on retest: reasoning field dropped from ~250-300 chars
to ~30-50 chars, and successful calls dropped from ~2.2-2.7s to ~1.0-1.2s.
**That fix exposed a second, distinct bug, more consequential than the first:** with the brief-
reasoning prompt, several runs still failed — but differently: the model correctly completed
`reasoning` *and* `answer`, then never continued to `cited_chunk_ids_csv` — instead padding pure
whitespace/near-empty tokens toward `max_tokens` (finish_reason: `length`) instead of emitting the
final field and closing the JSON object. This is the same *symptom* as P-R15 (array-field bug)
but confirmed to be a **distinct** cause: `cited_chunk_ids_csv` has been a plain string since
P-013, not an array, and the bug still occurs — P-R15's original diagnosis ("removing the array
field alone flips finish_reason to stop") was a correct but incomplete fix; Sarvam's `strict:
true` json_schema mode has a broader reliability problem with continuing past a completed
substantial field, not one specific to arrays.
**Decision:** Switched `_call_once` to `_call_once_streaming` (`stream: true`, SSE), and added
stall detection: track consecutive whitespace-only/empty content deltas; if
`STALL_THRESHOLD_CHUNKS` (20) is reached, raise `_GenerationStalled(partial_content)` and abort
the connection immediately rather than waiting for `max_tokens` to be exhausted. Chose 20 from
real data: legitimate JSON formatting whitespace (the newline+indent between fields) never
exceeded 2 consecutive whitespace-only chunks in any successful streamed run observed; the padding
bug produces hundreds. `generate()`'s existing repair-retry loop (previously only for JSON parse
failures) now also catches `_GenerationStalled` and retries once with the partial content
included, same "2 attempts total" shape as before — this required no new retry infrastructure,
just widening what the existing loop catches.
**Verification:** 6 new unit tests (`tests/generation/test_sarvam_llm.py`) using
`httpx.MockTransport` to fabricate SSE responses — no live network, deterministic: reconstructs
content correctly from chunked deltas, tolerates a couple of legitimate whitespace chunks without
false-triggering, raises `_GenerationStalled` at the threshold, empty deltas count the same as
whitespace, and `generate()`'s retry-then-succeed / retry-then-give-up paths both work correctly
against a mocked `_call_once_streaming`. Then 3 live runs against the real Sarvam API (5-chunk
context, 15s budget): 2/3 stalled on **both** attempts (confirming the bug is not rare) but were
each detected and handled in ~2.2-2.6s — down from what would have been ~8-14s+ for two full
non-streaming attempts hitting `max_tokens`, or up to 60s+ observed in earlier P-R13-era testing;
the 3rd run hit a genuine network-level hang (zero SSE chunks arrived at all — stall detection
can't help here since there's no content to observe) and was correctly bounded by the existing
outer timeout at exactly the 15s budget. In every case, including both stall failures, the
pipeline correctly fell back to Track A (`status="answered"`, `track="extractive"`) — the
two-track design held throughout. 83/83 tests green (77 pre-existing + 6 new). `ruff`/`mypy` clean.
**Consequences:** This is a **mitigation, not a fix** for the underlying provider reliability
issue — worth stating plainly rather than overclaiming: Track B's success rate under a realistic
multi-chunk context still looks concerning in this small live sample (1/3 succeeded outright, 2/3
failed on both attempts). What changed is the *cost* of a failure: previously slow (4-60s+,
discovered only via the eventual `max_tokens`/read-timeout cutoff), now fast (a few hundred ms to
~2-3s including one repair attempt) — directly improves p95/p99 latency variance (P-R17) and makes
Track A's fallback path meaningfully cheaper to reach when Track B can't deliver. This also
supersedes P-R15's root-cause framing (documented here rather than editing R's own ADR): the array
type was A cause, not THE cause — the deeper issue is `strict: true` json_schema mode's general
reliability continuing generation past a completed field. `docs/RISKS.md` P-R15/P-R16 updated;
P-R20 added for the still-open, broader continuation-reliability question. Real client-facing
token-by-token display (the WS `/voice` endpoint streaming partial text to the browser as it
arrives) is a separate, still-undone piece — this session's streaming work is server-side only
(buffer-then-validate-then-respond, same external contract as before), deliberately, since G4's
groundedness check needs the complete answer and citations before it's safe to show anything to
the user (see P-014's discussion of why client-facing streaming has its own separate design
tension with the G4 gate). Flagged as a distinct follow-up, not done today.

## P-018 — Live deployment has been running the Day-0 stub the whole time; chosen fix direction and status

**Date:** 2026-08-18 · **Status:** In progress — P-side prep done, R-side lean embedder pending
**Context:** R found (`docs/DECISIONS_R.md` R-018) by checking the actual live `/ask` response
before starting new work that `https://vrag-voice.onrender.com` has been serving the Day-0 stub
for retrieval this entire time — none of this session's real retrieval work (A1-A4 ablations,
efSearch, G3 calibration) has ever reached the deployed demo, despite every local test and every
live-verification check I ran this session (including several in my own ADRs today) genuinely
passing. Root cause, squarely mine: `Dockerfile` runs bare `pip install -e .`, never installing
the `retrieval` extra, and `data/` is gitignored so the built index was never going to reach the
container regardless. R separately measured (R-020) that even fixing this naively would likely
crash the deploy: the persisted index alone uses 591MB RSS (over Render free tier's 512MB budget
before any embedder loads), and `sentence-transformers` pulls in the full `torch`/`transformers`
stack regardless of inference backend (ONNX included), pushing total RSS to ~1.5GB either way.
**Decision:** Presented the real tradeoff to the user rather than picking a fix direction
unilaterally — this spans a real cost decision (Render's Standard tier, $25/mo, 2GB RAM,
confirmed via web search rather than assumed pricing) versus real undesigned cross-team
engineering work (shrink R's indexed corpus + replace `sentence-transformers` with a lean
`onnxruntime`-only inference path to drop the ~883MB `torch`/`transformers` baseline). Also caught
and flagged before presenting: shrinking the corpus alone cannot fit under 512MB regardless of
size, since the embedder's own framework overhead (measured: 1,474MB with embedder − 591MB
index-only ≈ 883MB) already exceeds the budget on its own — a corpus-only fix was never a real
standalone option, only useful paired with the leaner-embedder work. User chose the combined free
path (shrink corpus + drop `torch`/`transformers`), with the paid tier as an explicit fallback if
it doesn't land before the deadline.
**What's mine, done today:**
1. **Defensive fix, `src/vrag/retrieval/interface.py`:** `_get_real_retriever()` could previously
   raise uncaught if the index files were present but loading them failed (missing dependency,
   corrupt artifact) — violating `retrieve()`'s own documented "Never raises" contract, which only
   handled "index file missing," not "index present but unloadable." Wrapped the load in
   `try/except Exception`, logs and falls back to the stub. This is a prerequisite for staging the
   fix below without risk: without it, downloading the index artifact ahead of the leaner embedder
   landing would have taken the *working* stub-based demo down to a crash on every query. New
   test (`tests/test_retrieval_interface.py`) reproduces this for real on this dev machine (which
   has no `retrieval` extras installed) rather than mocking the failure.
2. **`Dockerfile`:** added a build-time step downloading R's published index artifact
   (`index-metadata_aware-v1` GitHub Release, verified reachable and its internal tar structure
   inspected before writing the extraction path) to `data/index/`, landing at exactly
   `data/index/metadata_aware/` where `interface.py` looks. Deliberately did **not** add
   `pip install -e ".[retrieval]"` yet — installing today's heavy extras (`torch`,
   `sentence-transformers`) would reproduce R's measured OOM. Built and ran the image locally
   before pushing: confirmed the index files land at the right path (324MB on disk), the app
   starts and serves `/healthz`/`/ask` correctly, and — because of fix (1) — the missing-`faiss`
   import is caught, logged clearly, and falls back to the stub rather than crashing. Verified this
   is a genuinely inert, safe prep step: current deployed behavior is unchanged until the leaner
   embedder dependencies are installed alongside it.
**What's R's, not started here (their file ownership, per `docs/TEAM_SPLIT.md` §2):**
`src/vrag/index/embedder.py`'s torch/transformers-free `onnxruntime`-only inference path, and
`pyproject.toml`'s corresponding leaner runtime dependency group (today's `retrieval` extra bundles
the full ablation-workflow stack, not a minimal production-serving one). Also R's: picking and
rebuilding a smaller working-pool corpus size, re-validated against A1-A4's existing numbers.
**Consequences:** The live demo still serves the Day-0 stub today — this ADR doesn't change that,
only prepares for it safely. If R's lean-embedder work lands, the remaining P-side steps are:
add the new leaner extras group to the Dockerfile install line, redeploy, and re-verify with a
real `/ask` call against the live URL (the same check that found this bug — not trusting "should
be deployed" claims again without checking the actual URL). If it doesn't land with enough runway
before the Aug 22 deadline, the documented fallback is switching Render to the Standard plan
($25/mo) — a same-day, low-risk fix — which the user has already pre-authorized as the safety net.
**What I learned, worth stating plainly:** every "verified live" claim I made earlier this session
(P-015's G3 calibration, P-016's circuit breaker, P-017's streaming) was checked against the real
public URL and was real *for what it tested* — but none of those checks would have caught retrieval
itself being stubbed, because the stub's output shape is deliberately identical to the real path's
(by design, so downstream code never has to know which one answered). "The pipeline behaves
correctly end-to-end" and "the pipeline is running the real retrieval implementation" are different
claims, and I was only directly checking the first. Worth remembering for any future "verified
live" claim: know specifically what a given live check can and cannot distinguish.

**Update, same day — first deploy attempt failed, real environment-specific bug found:**
triggering the Render deploy with the staged `Dockerfile` change failed at the exact download/
extract step, despite building and running correctly on this machine (Docker Desktop, macOS) not
30 minutes earlier. Fetched the real build logs via Render's `/v1/logs` API rather than guessing:
`tar: metadata_aware/chunk_lookup.json: Cannot change ownership to uid 197609, gid 197609: Invalid
argument`. Root cause: R's tarball preserves the original file ownership metadata from their build
machine (uid/gid 197609, consistent with a WSL2/Windows dev environment), and `tar`'s default
behavior tries to `chown` extracted files to match that during extraction — Render's build
environment rejects the `chown` syscall for that UID (likely a sandboxed/user-namespaced builder),
while Docker Desktop's local build VM does not, which is exactly why this didn't surface locally.
**Fix:** added `--no-same-owner` to the `tar` invocation — standard flag for exactly this class of
"container build environment doesn't allow arbitrary chown" problem, tells tar to extract as the
current user rather than the archive's original owner (we only care about file *contents*, never
the original uid/gid). Rebuilt and re-ran locally to confirm the fix doesn't change anything else
(still builds, still serves `/healthz` correctly). Redeployed.
**Consequences:** A concrete reminder that "builds and runs locally" and "builds and runs on the
actual target platform" are different claims when the build environments genuinely differ (sandbox
restrictions, not just OS/arch) — the CI pipeline doesn't build the Docker image at all, only runs
`pytest`/`ruff`/`mypy` directly, so this class of failure is only ever caught by actually deploying,
not by CI passing. Worth remembering going forward for any Docker-level change.
**Redeployed and verified live** after the fix: `/healthz` OK, `/ask` still returns
`stub-chunk-001`/`stub-chunk-002` (unchanged from before this whole change) — confirms the staged
index-download step is genuinely inert in production exactly as designed, not just in local
testing. This P-side prep is done; real retrieval activates the moment R's leaner embedder lands
and the corresponding extras get added to the Dockerfile's install line.

## P-019 — `search_corpus` tool: real native tool-calling, and a sharper P-R20 finding

**Date:** 2026-08-18 · **Status:** Accepted
**Context:** `docs/AGENT_BUILD_SPEC.md` §7.2 item 5 names a real `search_corpus(query, k) ->
list[Passage]` tool as a required harness capability, and `docs/AGENT_BUILD_SPEC.md` §2 explicitly
ties this to graded requirement C5 ("Must run inside a harness: tool calls, retries, structured
I/O, error recovery"). Not started until now.
**Live-probed before designing** (same methodology as P-017): confirmed `sarvam-105b` supports
real OpenAI-style function calling — offered a `search_corpus` tool, the model correctly returned
`finish_reason: "tool_calls"` with well-formed JSON arguments. Then tested the natural next
question — can `tools` and `response_format: json_schema` (strict) be combined in one request, so
the model could choose to answer OR call the tool within the existing structured flow? **No**:
live-tested, the model ignores the tool option entirely and tries to force an answer into the
schema, hitting the same whitespace-padding bug P-017 already found.
**Decision:** Kept the two capabilities in separate requests. Added `needs_more_context: bool` to
`GeneratedAnswer` (positioned after `reasoning`, before `answer` — same "decide before answering"
rationale as the reasoning-before-answer invariant) so the *existing* single structured call
(unchanged cost) signals when a follow-up is needed. Only then does `generate()` escalate: a
tool-decision call (`tool_choice: "required"`, non-streaming — live-observed responses here are
short, ~30 tokens, low stall risk), execute `search_corpus` (a thin wrapper over `retrieve()`, the
R/P seam, with `k` clamped to [1, 10] against an adversarial/runaway value), then one final
structured re-answer over the expanded context. Tool depth capped at 1 by construction — the
follow-up call never offers the tool again. The common case (context already sufficient) costs
exactly one round trip, unchanged from before this feature.
**A genuine G4 integration bug caught during design, before it shipped:** if the tool fetches new
chunks and the model cites one of them, `GenerateStage`'s original code built `valid_chunks_by_id`
from `ctx.data["chunks"]` — RetrieveStage's original list, which wouldn't include anything
`search_corpus` fetched. G4 would have wrongly flagged a genuinely tool-fetched citation as
invented. Fixed by changing `generate()`'s return type to `GenerationResult` (a `NamedTuple` of
`answer` + the full `chunks` set actually used, original ∪ tool-fetched) — `GenerateStage`
validates G4 against that, not the stage's starting list.
**Live end-to-end verification found something more important than a UX detail:** tried to run the
full 3-call escalation live against a genuinely-insufficient-context query. 12/12 attempts across
several context shapes failed — every single one via the *same* whitespace-padding stall P-R20
already documented, but with a much sharper, previously-unknown correlation: **the stall
correlates strongly with "insufficient context" (`needs_more_context: true` / "I don't know"-style)
answers specifically**, not a generic random rate. Verified this predates and is independent of
today's schema change: re-ran the exact same insufficient-context query against the *old* 3-field
schema (no `needs_more_context` field at all) — same result, 4/4 failures, `finish_reason:
"length"` immediately after a correctly-reasoned "the context doesn't contain this" answer, every
time. This means Track A's abstention path and Track B's own "insufficient context" branch —
exactly the case where a nuanced generative answer would matter most — are the *most* likely to
hit this provider bug, not a random subset of calls.
**Verification:** 17 new tests (`tests/generation/test_search_corpus_tool.py`,
`tests/harness/test_generate_stage_search_corpus.py`) covering `search_corpus`'s real integration
with `retrieve()`'s stub path, `k` clamping, tool-decision parsing (well-formed, missing `k`, no
tool call, malformed arguments), and `generate()`'s full orchestration (skip when sufficient,
escalate and expand chunks, three distinct fallback-to-first-answer paths), plus the G4-against-
expanded-chunks integration fix. 97/97 tests green (85 pre-existing + 12 new — 11 in the tool test
file + 1 integration test). `ruff`/`mypy` clean. The happy path (context sufficient,
`needs_more_context: false`) was separately live-verified stable: 5/5 clean successes against a
realistic 5-chunk context right after adding the new field, before this insufficient-context
testing began.
**Consequences:** The `search_corpus` mechanism is built correctly and thoroughly tested — real
native tool-calling (satisfies C5's letter, not a simulated equivalent), correct depth-1 capping,
correct G4 integration for tool-fetched citations. It is genuinely ready to fire the moment
Sarvam's completion succeeds on the triggering call, same "proven correct, not yet reliably
observed end-to-end live" position Track B itself shipped in earlier this session (P-013/P-017).
`docs/RISKS.md` P-R20 sharpened with this finding, not treated as a new separate risk — same root
cause, better characterized. Also strengthens the case for reporting P-R20 to Sarvam with a much
more precise, highly reproducible repro case than was available before today.

## P-020 — Tried R-023's memory fix live; real OOM found on `/ask` specifically; rolled back

**Date:** 2026-08-18 · **Status:** Accepted — tried, real negative result, reverted
**Context:** R-023 wired their memory-fix work into production and specified 3 Dockerfile steps,
but their own isolated measurement (727MB) still exceeded the 512MB budget. Presented the real
tradeoff to the user (try it and roll back if needed, vs. wait for R to shrink the corpus further,
vs. reconsider the now-much-cheaper paid tier). User chose: try deploying, roll back if it crashes.
**Applying the fix surfaced two real issues R's writeup hadn't anticipated:**
1. The `embedder-lite-onnx-v1` release tarball extracts to `multilingual-e5-small-lite/`, not
   `multilingual-e5-small/` (what `LiteE5Embedder`'s `DEFAULT_ONNX_MODEL_DIR` expects) — found by
   downloading and inspecting the actual tarball before writing the extraction path. Fixed with
   `tar --strip-components=1`.
2. `numpy>=2.5` (part of the new `retrieval-lean` extra) requires Python 3.12+; the Dockerfile's
   base image was still `python:3.11-slim` — never surfaced before since retrieval extras were
   never actually installed in the image until this change. Bumped to `python:3.12-slim`.
**Local verification looked genuinely promising:** built and ran the fixed image under Docker's
own `-m 512m` limit (matching Render's stated free-tier constraint) — 446.8MiB steady-state, real
(non-stub) retrieval confirmed working (1072ms `retrieve` timing, a live G3 borderline refusal),
far under R's 727MB isolated measurement. Deployed to Render on that strength.
**First deploy attempt failed at the build stage** — `curl: (56) Recv failure: Connection reset by
peer` mid-download of the 84MB embedder tarball. A transient network failure, not a real bug
(everything else in the build succeeded: `retrieval-lean` installed cleanly, the index downloaded
fine, Python 3.12 worked). Retried — the retry built and deployed successfully.
**The real, live answer, once actually tested:** `/healthz` returned 200 consistently, but every
real `/ask` call returned a fast (<1s) `502 Bad Gateway` with empty content. Checked runtime logs
directly rather than guessing: no `"POST /ask"` access-log line ever appeared before each restart —
the process died mid-request, before FastAPI could even log the completed response, then restarted
(`"Started server process [N]"` a few seconds later). This is the exact signature of an OOM kill
happening during `_get_real_retriever()`'s first real load (the lazy singleton that loads the FAISS
index + SQLite chunk lookup + ONNX embedder) — the process survives serving the cheap `/healthz`
endpoint but cannot survive the actual memory spike of first-use retrieval, even though the
*steady-state* RSS measured comfortably fits locally. Reproduced 3/3 on deliberate, separate `/ask`
calls after confirming `/healthz` was healthy each time — not a fluke.
**Why this doesn't contradict the earlier local test:** the local test measured steady-state RSS
*after* a request had already completed once (loading finishes, memory settles) — it never
specifically captured the *peak* during the singleton's first real initialization, which is
plausibly higher than steady-state (loading a 180MB FAISS index and initializing an ONNX inference
session both have real transient overhead beyond their final resident size). Render's actual OOM
enforcement may also be stricter/less forgiving of a brief overshoot than Docker Desktop's local
limit enforcement. Both are ordinary, plausible explanations — not a sign the local test was
performed wrong, just that "steady-state fits" and "peak-during-first-load fits" are different
claims, and only the live test could distinguish them.
**Decision:** Rolled back immediately via `git revert` (not a force-push or history rewrite) of the
Dockerfile commit, redeployed, and re-verified live: `/healthz` 200, 3/3 `/ask` calls 200 OK,
demo fully restored. Chose `git revert` specifically so a future ordinary push to `main` wouldn't
silently undo the rollback — a straight redeploy-of-an-old-SHA would have left the *branch itself*
still pointing at the broken commit.
**Consequences:** `docs/RISKS.md` R4 stays open — this is a real, live-verified negative result,
not a projection either way anymore. The remaining gap is now specifically characterized as "peak
first-load memory," not just "steady-state RSS," which is new, useful information for whoever
tackles this next (R, if reducing peak-load overhead specifically; or the user, if reconsidering
the paid tier now that even 727MB steady-state — let alone the higher peak — is the real number).
No further deploy attempts against the free tier without either a fix that specifically targets
peak load (e.g., loading the index/embedder in a way that avoids holding multiple copies in memory
during initialization) or a decision to pay for headroom.

## P-021 — Real bug found and fixed in Track B's streaming handler: 0% -> 63% success rate

**Date:** 2026-08-19
**Status:** Accepted — fixed, verified via `pytest tests/generation` (27/27 green) and by direct
re-testing against the live Sarvam API. Found while building the P6 latency campaign's standalone
Track B measurement (`scripts/bench_latency.py`, see the joint P6 entry in `docs/DECISIONS.md`).
**Context:** Single-operator session from 2026-08-19 onward (P's collaborator out of weekly Claude
Code credits) — this entry continues P's numbering since the fix lives in P's module
(`src/vrag/generation/`), same convention as if P's own session had found it.
**What was found:** Calling `generation.sarvam_llm.generate()` directly (bypassing
`GenerateStage`'s budget gate, giving it a real 15s window) against 30 real Sarvam calls: **0/30
succeeded**, every one crashing with `TypeError: can only concatenate str (not "NoneType") to
str` at `_call_once_streaming`'s `accumulated += delta`. Root cause:
`choices[0]["delta"].get("content", "")` — `.get(key, default)`'s default only applies when the
key is *absent*, not when it's present with an explicit `null` value. Sarvam's SSE stream
sometimes sends an explicit `"content": null` (e.g. a trailing or role-only delta chunk), which
`.get()` correctly returns as `None`, not `""` — and nothing downstream guarded against that.
**Fix:** One line — `(choices[0]["delta"].get("content") or "")`. `or ""` normalises both "key
absent" and "key present but null" to the same safe empty string.
**Verified, not assumed:** re-ran the same 30-call standalone measurement after the fix:
**19/30 succeeded (63.3%)**, real coherent Hindi answers, completion latency P50=1976.5ms,
P70=2557.3ms, P100=6429.8ms. The remaining 11/30 failures are the already-documented, genuinely
provider-side stall bug (P-R20 in `docs/RISKS.md`) — distinct from this bug, not fixed by it, and
not something this session can fix (Sarvam-side).
**Consequences:** This was silently masking a large fraction of Track B's real capability — every
previous session's "Track B works, verified live" claims were true for the specific test runs that
happened not to hit a null-content chunk, but the *actual* population success rate was far lower
than anyone had reason to suspect until this systematic 30-call measurement surfaced it. Nothing
else changes: `GenerateStage`'s budget-gated behavior, the circuit breaker, and stall-detection
are all unaffected by this fix — they operate correctly regardless of whether individual calls
crash or complete; this fix just means more of them complete instead of crashing.

## P-022 — Frontend: refused/abstained/degraded no longer render as one hardcoded "Abstained" pill

**Date:** 2026-08-19
**Context:** Same single-operator convention as P-021 — implemented in an `R`-flagged session
(`.workstream` said `R`) since the operator was working both roles that day, then relocated here
from `docs/DECISIONS_R.md` (was numbered R-037) once the module-ownership mismatch was caught and
flagged: `frontend/index.html` is P's module (frontend), not R's (retrieval/ranking). No content
changed in the move, only the number and this context note.
**Status:** Accepted, implemented, verified in a real browser. Backend untouched (no API contract,
schema, retrieval, guardrail, or harness change) -- `frontend/index.html` only.

**Bug:** `answer_final`'s dispatch was a binary branch -- `if(ar.status === 'answered')
goAnswered(ar); else goRefused(ar.refusal_reason);` -- so `refused`, `abstained`, and `degraded`
all fell into `goRefused()`, which hardcoded the pill text `'Abstained · confidence too low'`
regardless of which one actually happened, and always set `body.dataset.state = 'refused'`.

**Fix:** extracted a pure `describeAnswerResponse(ar)` function (`index.html`, no DOM access) with
one explicit branch per canonical status -- `answered`/`degraded`/`abstained`/`refused` each get
their own `state`, headline, pill class, and pill text; an unrecognised status falls through to a
visibly-labeled error rather than silently reusing another status's copy. `renderAnswerResponse`
applies the result to the DOM; the WS `answer_final` handler now just calls it directly. Added one
new CSS pill variant (`.status-pill.blocked`, ink-toned) so `refused` has a 4th genuinely distinct
visual treatment alongside the 3 that already existed (`.ticket` gold-dashed for answered, `.warn`
gold for degraded, base signal-pink for abstained) -- all reusing existing palette tokens, no new
colors invented. Also fixed a real consequential bug the state-split would otherwise have
introduced: `handlePrimaryClick`'s "click primary to start over" check only recognised
`idle`/`answered`/`refused` as terminal states (everything used to collapse into `refused`) --
now explicitly covers all 5 terminal states, or "Try again" on a real abstained/degraded response
would have silently done nothing.

**Verification, real browser (Chrome via automation, not just code review):**
`frontend/test_status_rendering.html` (new) loads the real, unmodified `index.html` in an iframe
so the script runs with its actual DOM, and exercises `window.__describeAnswerResponse` (a small
test-only hook, zero behavior change for real usage) against real captured payloads for
`answered`/`abstained`/`refused` (from the validated `vrag-real:v3-warmup` container, real
retrieval, real Sarvam-backed G1/G2/G3) and one schema-valid `degraded` payload -- `AssembleStage`
(`src/vrag/harness/stages.py`) doesn't emit `degraded` today, out of scope for this fix, so that
one is hand-built to the exact `AnswerResponse`/`Citation` shape and labeled as such, not passed
off as backend-captured. 16/16 assertions pass, run live in a real browser: all 4 statuses map to
distinct `state`/pill-text/pill-class/headline; refused/degraded specifically do not say
"Abstained"; refused's pill is exactly "Refused"; real answer text and citations survive
untouched; an unrecognised status gets its own copy, never a silent reuse. Separately verified
via `getComputedStyle` in the same real browser session that all 4 pill classes produce genuinely
distinct background/border/text colors, not just distinct strings.

No Node/npm/frontend framework exists in this project (single static HTML file, no build step) --
this test needs no new dependency, just a local static server (`python -m http.server`) and the
already-available browser automation tool; documented in the test file's own header comment.

Full backend suite: 232/232 (unaffected, as expected -- no Python files touched).

## P-023 — STT: bounded no-speech timeout + guarded sender shutdown, `src/vrag/stt/sarvam.py`

**Date:** 2026-08-19 / 2026-08-20
**Context:** Same single-operator convention as P-021/P-022 -- implemented in an `R`-flagged
session, then relocated here from `docs/DECISIONS_R.md` (was numbered R-038) once the
module-ownership mismatch was caught and flagged: `src/vrag/stt/` is P's module (STT), not R's.
No content changed in the move, only the number and this context note.
**Status:** Accepted, implemented, tested. `src/vrag/stt/sarvam.py` only -- no retrieval, FAISS,
corpus, embedding, tokenizer, guardrail, Track A/B, latency-budget, Render config, or frontend
change (the frontend's existing `msg.type === 'error'` handling already renders any
`TranscriptEvent(type="error", ...)` correctly; confirmed by re-reading `main.py`'s `/voice`
dispatch and `frontend/index.html`'s error branch, so nothing there needed to change).

**Bug (found via real-browser + real-microphone verification against the deployed
`vrag-voice.onrender.com`, then confirmed with Render's `/v1/logs` API against the live
container):** a silent/no-speech session left the UI stuck in LISTENING indefinitely -- no
transcript, no error, no timeout -- because `stream_transcribe()`'s receive loop waited
*unboundedly* (`asyncio.wait_for(..., timeout=None)`) for Sarvam's first message whenever audio
was still being sent. The only thing that ever unblocked a genuinely silent session was Sarvam's
own ~60s inactivity watchdog (`{"code":"inactivity_timeout","message":"No audio received for
60s.","is_fatal":true,"status_code":408}`) -- confirmed live: a 71.42s real-microphone session (0
utterances) got zero client-visible feedback until that watchdog fired at the 60s mark, which is
indistinguishable from a frozen UI to a real user. Separately, the same traceback surfaced a second,
independent bug: `_sender()`'s `finally` block unconditionally tried to send `{"event":"end"}` to
Sarvam on cleanup, and if Sarvam had already closed the socket first (e.g. via that same watchdog),
this raised `ConnectionClosedError` inside a detached background task (`asyncio.create_task`)
whose exception was never awaited or retrieved -- an "unhandled Task exception" leak, harmless to
the user but a real bug.

**Fix (two, scoped as approved):**
1. New module constant `NO_SPEECH_TIMEOUT_S = 10.0`. The receive loop now tracks
   `received_transcript` (set on `transcript.partial`/`transcript.final`); while sending is still
   in progress *and* nothing has been transcribed yet, the wait is bounded to 10s -- on timeout it
   yields the existing `TranscriptEvent(type="error", text="No speech detected yet. Please try
   again.")` mechanism (no new API contract) and ends the stream. Once a real transcript arrives,
   the wait reverts to unbounded (matching prior behavior exactly) so a normal mid-sentence pause
   is never mistaken for silence. The pre-existing end-of-stream grace period (now
   `END_GRACE_S = 3.0`, promoted to a module constant alongside `NO_SPEECH_TIMEOUT_S` for
   testability, same 3.0s value, unchanged behavior) is completely untouched by this branch.
2. `_sender()`'s cleanup `send({"event": "end"})` is now wrapped in
   `try/except websockets.ConnectionClosed:` (reusing the same exception class already caught at
   the outer level) -- if Sarvam already closed its side, this logs and returns cleanly instead of
   leaking an unretrieved task exception.

**Tests (new, `tests/stt/test_sarvam_stt.py`, 6 tests):** a fake Sarvam WebSocket
(`_FakeSarvamWS`, async context manager + `.send()`/`.recv()` against a scripted
`(delay, message_or_exception)` schedule) stands in for the real network call, monkeypatching
`ws_connect` -- same "fake the network boundary, exercise real control flow" convention as
`tests/generation/test_sarvam_llm.py`'s `httpx.MockTransport`; the *production* STT path itself
is never mocked. Proves: (a) a silent session yields the no-speech error at the configured timeout,
well under a 2s bound, not Sarvam's 60s watchdog; (b) a real transcript arriving inside the
(test-shrunk) no-speech window correctly reverts the wait to unbounded, so a later, longer gap
(0.2s against a 0.05s shrunk timeout) does not truncate the session; (c) a normal
audio-ends-then-final-transcript-then-Sarvam-closes lifecycle ends cleanly, with the sender's own
"end" event confirmed sent; (d) an already-closed connection during the sender's cleanup produces
no unhandled Task exception (verified via a temporary `loop.set_exception_handler` capture, not
just "no crash") while 2 real audio chunks still went out fine beforehand; (e) real Sarvam-origin
`error` events still relay unchanged. One test asserts `NO_SPEECH_TIMEOUT_S == 10.0` directly, the
configured (not separately live-measured) value the user's spec calls for. Sanity-checked the
suite's dependency on the fix itself by reverting `sarvam.py` and re-running -- collection fails
(`ImportError: cannot import name 'NO_SPEECH_TIMEOUT_S'`), confirming the tests are wired to the
real code, not vacuous.

Full suite: 238/238 (232 + 6 new), ruff clean, mypy clean, zero warnings (checked with
`-W error::pytest.PytestUnraisableExceptionWarning`, the class of warning that would have caught
the pre-fix Task-exception leak).

**Not yet done:** redeploy to Render (explicitly deferred by the user pending review).

## P-024 — Phase 1: remove hardcoded `hi-IN`, wire Sarvam's real language signal through G2

**Date:** 2026-08-20. Full decision + rationale: `docs/DECISIONS.md` ADR-008 (shared).
**P-side summary:** `WS /voice`'s `stream_transcribe()` call switched from a hardcoded
`language_code="hi-IN"` to `"auto"` (verified live against Sarvam's real docs: `auto` is a real
adaptive-detection mode, and the *only* mode in which a transcript event's `language` field is
ever populated — a fixed code never got one). `event.language` now flows into `build_answer(...,
language=event.language)` instead of being silently dropped. `AskRequest` gains an optional
`language` field (a caller-supplied hint, since the text debug endpoint has no real STT signal of
its own) purely so G2's language routing is testable without a live Sarvam call. `AnswerResponse`
gains `query_language: str | None = None`, additive — frontend and existing tests unaffected
(verified: all pre-existing tests pass unchanged, including `tests/stt/test_sarvam_stt.py`'s fake-
WebSocket tests, which pass `language_code="hi-IN"` explicitly and are unaffected by the default
change). No Track B/generation change — still Hindi-only output, per Phase 1's explicit scope
("do not enable multilingual generation yet"). 260/260 tests pass (238 pre-existing + 22 new).
