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
