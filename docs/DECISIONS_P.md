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
