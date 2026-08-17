# RISKS.md — live risk register

Seeded from `AGENT_BUILD_SPEC.md` §13. Owners assigned where the risk maps cleanly to one
workstream; joint where it doesn't. Update this any time a risk changes status — it's meant to be
read at every session start, not just written once. Append new risks as they're discovered; don't
silently resolve one without a note on how.

| ID | Risk | Impact | Owner | Mitigation | Status |
|----|------|--------|-------|-----------|--------|
| R1 | Provider RTT makes 200ms unreachable | 🔴 high | Joint | Phase 0 probe; two-track design; consider local generation | Open — probe not yet run (no API keys on either machine) |
| R2 | Deployment/mic fails late | 🔴 high | P | Deploy in Phase 1, redeploy every phase | Mitigated — live on Render since Day 1, verified with real STT audio |
| R3 | Rate limits break the benchmark run | 🟡 med | P | Retries in the bench script; run overnight if needed | Open |
| R4 | Index too large for host memory | 🟡 med | R | Cap at 200k chunks; measure RSS in Phase 1 | Open |
| R5 | MT artifacts in corpus hurt answer quality | 🟡 med | R | Spot-check in Phase 0; also the justification for G4 | Open — spot-check done, see `docs/DECISIONS_R.md` R-003, real artifacts found |
| R6 | Team disagreement on the 200ms interpretation resurfaces late | 🟡 med | Joint | Settled as ADR-004 (`t_pipeline` definition) | Mitigated — confirmed by both tracks at Day 1 sync |
| R7 | A member misses a promotion post | 🔴 high | Joint | Named grid in `SUBMISSION_CHECKLIST.md`, checked Aug 21 | Open |
| R8 | Polish commits break the latency number | 🟡 med | P | CI latency regression test from Phase 6 | Open |
| R9 | Scope creep into a second language before core is done | 🟡 med | Joint | Hard gate: not before Phase 7 exit | Open |
| P-R10 | Both dev machines run newer Python than the spec's 3.11+ target (R: 3.13, P: 3.14) | 🟡 med | Joint | Each `pyproject.toml`/venv verified real wheels exist for that Python before committing (R-001); watch ML-lib (faiss/onnxruntime/torch) wheel availability as versions climb | Open, monitoring |
| P-R11 | `FRIEND_BRIEF.md` referenced `frontend/reference/voice-rag-ui-preview.html` but the file was committed at repo root | 🟢 low | P | Moved via `git mv` on Day 1, see `docs/DECISIONS_P.md` P-004 | Resolved |
| P-R12 | Render (free tier) doesn't propagate our server-initiated WebSocket close frame promptly — connection lingers ~20-25s and eventually drops with code 1006 instead of a clean close | 🟢 low | P | Verified via a real TTS-generated speech round trip (`docs/DECISIONS_P.md` P-007) that `transcript_final` + `answer_final` both arrive in ~2.4s, well before the lingering-connection window — doesn't block real usage, only delays connection cleanup. Frontend now sends `{"event":"stop"}` proactively on `transcript_final`. Revisit if it ever becomes user-visible | Monitoring, not blocking |

## Blockers found this session (Day 0-1, 2026-08-17)

- **No Sarvam / Groq API keys present on either dev machine.** Blocks `scripts/probe_latency.py`
  (R1/ADR-003) and any real STT/LLM call from R's side. Owner: whoever holds the keys — flagged to
  the user, not guessed around. Everything that doesn't need a live provider call (dataset work,
  chunking, index build, offline eval) proceeds in the meantime. (Note: Workstream P *does* have a
  working Sarvam key, used for the live deploy's real STT — see `docs/DECISIONS_P.md` P-007. R's
  local probe script still needs its own key to run.)
- **IPv6 is broken on this dev machine's network (Workstream R's machine)** (confirmed via `curl`:
  IPv6 to huggingface.co resets the connection; IPv4 works). Not a project risk per se, but worth
  knowing if any HTTP client hangs unexpectedly during dataset download — force IPv4 if so. Further
  wrinkle found: HuggingFace's "Xet" transfer backend for large files does its own networking that
  bypasses a process-wide IPv4 DNS patch — had to be disabled (`HF_HUB_DISABLE_XET=1`) to get a
  reliable download. See `scripts/_netcompat.py`.
