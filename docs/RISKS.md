# RISKS.md — live risk register

Seeded from `AGENT_BUILD_SPEC.md` §13. Update owner/status as the build progresses; append new
risks as they're discovered, don't silently resolve one without a note on how.

| ID | Risk | Impact | Mitigation | Owner | Status |
|----|------|--------|-----------|-------|--------|
| R1 | Provider RTT makes 200ms unreachable | 🔴 high | Phase 0 probe; two-track design; consider local generation | P | open |
| R2 | Deployment/mic fails late | 🔴 high | Deploy in Phase 1, redeploy every phase | P | mitigating (Day 1 HF Spaces deploy in progress) |
| R3 | Rate limits break the benchmark run | 🟡 med | Retries in the bench script; run overnight if needed | P | open |
| R4 | Index too large for host memory | 🟡 med | Cap at 200k chunks; measure RSS in Phase 1 | R | open |
| R5 | MT artifacts in corpus hurt answer quality | 🟡 med | Spot-check in Phase 0; also the justification for G4 | R | open |
| R6 | Team disagreement on the 200ms interpretation resurfaces late | 🟡 med | Settled as ADR-004 (`t_pipeline` definition), see `docs/EVAL_PROTOCOL.md` | joint | mitigated |
| R7 | A member misses a promotion post | 🔴 high | Named grid in `docs/SUBMISSION_CHECKLIST.md`, checked Aug 21 | joint | open |
| R8 | Polish commits break the latency number | 🟡 med | CI latency regression test from Phase 6 | P | open |
| R9 | Scope creep into a second language before core is done | 🟡 med | Hard gate: not before Phase 7 exit | joint | open |
| P-R10 | Python 3.14 on this machine vs spec's 3.11+ target | 🟡 med | `pyproject.toml` targets `>=3.11`; watch for ML-lib wheel availability (faiss/onnxruntime) on 3.14 when R starts installing those — may need a 3.11/3.12 venv instead | P | open, flagged Day 1 |
| P-R11 | `FRIEND_BRIEF.md` referenced `frontend/reference/voice-rag-ui-preview.html` but the file was committed at repo root | 🟢 low | Moved via `git mv` on Day 1, see `docs/DECISIONS_P.md` | P | resolved |
| P-R12 | Render (free tier) doesn't propagate our server-initiated WebSocket close frame promptly — connection lingers ~20-25s and eventually drops with code 1006 (abnormal closure) instead of a clean close | 🟢 low | Verified via a real TTS-generated speech round trip (see `docs/DECISIONS_P.md` P-007) that `transcript_final` + `answer_final` both arrive in ~2.4s, well before the lingering-connection window — so this doesn't block real usage, only delays connection cleanup. Frontend now sends `{"event":"stop"}` proactively on `transcript_final` so the server isn't left waiting on nothing. Revisit if it ever manifests as a user-visible issue (e.g. can't ask a second question without a page refresh) | P | monitoring, not blocking |
