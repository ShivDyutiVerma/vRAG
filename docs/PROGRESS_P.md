# PROGRESS_P — Workstream P running status

> Mine — edit freely, any time. Never edited by Workstream R.

**Last updated:** 2026-08-17 (Day 1, session 01)
**Current phase:** P1 — Walking Skeleton & First Deploy

## Where we are, in one paragraph

Day 1 walking skeleton is complete and live. **https://vrag-voice.onrender.com** is deployed and
verified with a real, full golden-path test: genuine Hindi speech audio (via Sarvam's own TTS, see
P-007) streamed to the live `/voice` WebSocket, transcribed correctly by real Sarvam STT
("भारत में सबसे ऊँचा पर्वत कौन सा है?"), and answered correctly through the Day-1 `retrieve()`
stub in ~2.4s end to end. No harness orchestration (deadline propagation, retries, guardrails) is
wired through the request path yet — that's explicitly Day 2 scope per `docs/TEAM_SPLIT.md` §5.
Deploy target was switched from Hugging Face Spaces (hit a real paywall — HF's free tier no longer
covers Docker apps) to Render mid-session; see DECISIONS_P P-005.

## What works right now (verified, not assumed)

- `.workstream` = `P`, `workstream-p` branch created and pushed ✅
- `docs/` fully bootstrapped (PRD, ARCHITECTURE, CONVENTIONS, API_CONTRACTS, EVAL_PROTOCOL,
  LATENCY_BUDGET, RISKS, GLOSSARY, EVAL_RESULTS shell, SUBMISSION_CHECKLIST, PROGRESS/DECISIONS
  shared + mine) ✅
- `retrieve()` stub returns valid `RetrievedChunk`s, matches the joint contract exactly ✅ verified
  by `tests/test_retrieval_stub.py`, all passing
- Harness skeleton (`Stage`, `PipelineContext`, `Budget` deadline tracker, retry policy shape)
  imports cleanly, not yet wired into the request path ✅
- Real Sarvam realtime STT (`saaras:v3-realtime`) — full golden path verified against the live
  deployed URL using genuine TTS-generated Hindi speech (not silence, not a mock): correct
  progressive partial transcripts, correct final transcript, correct downstream answer, ~2.4s
  round trip ✅ (see DECISIONS_P P-007)
- `POST /ask` — real end-to-end request through stub `retrieve()` → Track A placeholder answer,
  verified both locally and on the live URL, returns correct Devanagari text + citations ✅
- `WS /voice` — verified end to end on the **live Render deployment** with real speech audio,
  including partial transcripts, final transcript, and the resulting answer ✅. **Not yet tested
  with an actual physical microphone in a real browser** — the server-side path is proven with
  real audio, but browser mic capture (`frontend/index.html`'s `getUserMedia` + PCM conversion)
  hasn't been clicked through in an actual browser this session.
- Frontend (`frontend/index.html`) — real mic capture (ScriptProcessorNode → PCM16) wired to the
  WS, reusing the on-brand design from `frontend/reference/voice-rag-ui-preview.html`. Served
  correctly by the FastAPI app at `/` ✅. Sends `stop` proactively on final transcript (P-008).
- Docker image builds and runs correctly locally (`docker build` + `docker run`, health/frontend/
  ask all verified inside the container) ✅
- Live deployment on Render (`https://vrag-voice.onrender.com`), verified via `/healthz`, `/`,
  `/ask`, and a full real-audio `/voice` round trip ✅
- `pytest`: 7/7 passing ✅

**Known non-blocking issue:** the `/voice` WebSocket doesn't close cleanly on Render — it lingers
~20-25s after the server calls `websocket.close()` and eventually drops with code 1006 instead of
a clean handshake. Confirmed this does **not** affect the actual answer delivery (both
`transcript_final` and `answer_final` arrive correctly within ~2.4s, long before the lingering
window). Documented as P-R12 in `docs/RISKS.md` — worth a look on Day 4 (polish) if there's time,
not worth chasing further today.

## What is stubbed / faked / TODO

- `retrieve()` is the Day-1 stub (hardcoded fake chunks) — swapped for Workstream R's real
  implementation at the Day 2 sync, per the joint contract.
- The "answer" in both `/ask` and `/voice` is Track A only, and a *simplified* Track A at that:
  literally the top retrieved chunk's full text, not a real span-selection algorithm. No G1-G5
  guardrails, no rerank, no grounding gate, no LLM generation (Track B) are wired in yet — every
  `AnswerResponse` honestly lists these in `stages_skipped`.
- The harness (`Stage`/`PipelineContext`/`Budget`) exists as a shape but isn't actually driving
  `/ask` or `/voice` yet — those call `retrieve()` directly. Wiring the real pipeline through them
  is Day 2 hardening work.
- No retries, no circuit breaker, no structured-output repair loop yet.
- Frontend has not been manually tested in a real browser with real mic permission this session —
  only server-side WS logic was verified with simulated audio.
- Deploy: **live on Render, verified end to end including real speech**, see above.

## Blockers

- None. Note for the next session: confirm the Render free-tier instance doesn't spin down in a
  way that breaks the demo (Render free web services sleep after inactivity and cold-start on the
  next request — worth checking response time on first hit before relying on this for the actual
  demo video). Also see the non-blocking WS close-lag issue (P-R12) above.

## Next session should start by

1. Open https://vrag-voice.onrender.com on an actual phone over mobile data and click through the
   real mic UI (not just the server-side audio pipeline, which is already proven) — the one gap
   left from today's verification.
2. Check whether Workstream R has started their session / pushed `docs/PROGRESS_R.md` — if so,
   read it before touching anything.
3. Start Day 2 harness hardening per `docs/TEAM_SPLIT.md` §5: wire `Stage`/`PipelineContext`/
   `Budget` through the actual request path (currently they exist but aren't used by `/ask` or
   `/voice`), add `tenacity` retries, begin G1/G2/G5 guardrails.
