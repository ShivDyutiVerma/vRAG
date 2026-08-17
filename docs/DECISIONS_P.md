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
