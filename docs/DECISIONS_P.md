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

## P-004 — Moved `voice-rag-ui-preview.html` into `frontend/reference/`

**Date:** 2026-08-17 · **Status:** Accepted
**Context:** `docs/FRIEND_BRIEF.md` §2 and `docs/TEAM_SPLIT.md` §2 both reference the preview at
`frontend/reference/voice-rag-ui-preview.html`, but it was committed at the repo root.
**Decision:** `git mv voice-rag-ui-preview.html frontend/reference/voice-rag-ui-preview.html` to
match what the docs already assume, rather than updating every doc reference.
**Consequences:** None outside this track's files.
