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
| P-R13 | Sarvam's `/v1/chat/completions` endpoint hangs indefinitely (0 bytes back, confirmed to 60s) on every *valid* request — `sarvam-105b` and `sarvam-105b-conversations`, streaming and non-streaming, all hang identically. Isolated with controlled tests: a bad API key returns a fast 403, a deprecated model name returns a fast 400 with a helpful message — only a genuinely valid request hangs. Not an auth/routing/request-shape bug on our side. Blocks Track B (generation) on the only provider key currently available | 🔴 high | P | `scripts/probe_latency.py` records 30/30 Sarvam chat TTFT failures — ADR-003 still pending real numbers. `src/vrag/generation/sarvam_llm.py` is written and ready; its live path is unverified until this responds. Needs a call from the user: retry later (possible transient outage), get a Groq key (`GROQ_API_KEY` empty in `.env`) so Groq can be primary/fallback, or check Sarvam's status/support channel | Open, blocking — flagged 2026-08-18 |
| R-R14 | A3 found hybrid (RRF) retrieval *regresses* quality vs. dense-only on this corpus (`docs/DECISIONS_R.md` R-010, `docs/EVAL_RESULTS.md` §3) — root cause: BM25's weaker top ranks get equal RRF fusion weight against dense's stronger ones, with only a `top_k=10` candidate pool per lane. A larger per-lane candidate pool before fusion (e.g. top-50, truncated to top-10 after) is a standard mitigation, not yet tested — would be a second, separate ablation axis from retrieval mode | 🟢 low | R | Untested idea, not a blocker: dense-only already shipped as A3's winner and unblocks A4. Worth a quick follow-up run only if time remains after A4/A5 land | Open, non-blocking |
| P-R15 | `retrieve()`'s production wiring switched to dense-only (`docs/DECISIONS_R.md` R-010 update), which changes `RetrievedChunk.score`'s scale from RRF-fused (~0.008-0.033, structurally below G3's `TAU=0.35`) to raw cosine similarity (~0.3-0.95) — the scale `g3_confidence.py`'s own docstring says `TAU` was already chosen for. **Confirmed by direct test, not just inferred:** re-ran `test_ask_returns_answered_for_a_query_the_stub_covers`'s real query against the live `/ask` endpoint post-switch — it still abstains, but the `refusal_reason` changed from what would have been a `top1 < TAU` failure (structurally guaranteed under the old RRF scale) to `"Ambiguous match: top result doesn't clearly stand out"` (the *margin* check, `top1 - top5 < MARGIN=0.05`) — i.e. `top1` now clears `TAU` as intended, and G3 is abstaining on a real, if still-uncalibrated, ambiguity signal instead of a structurally-broken one | 🟡 med | P | Not fixed here — `g3_confidence.py` is Workstream P's module, not touched. Flagged for P: the `TAU` check now behaves as its docstring intends; `MARGIN=0.05` is the placeholder most worth scrutinizing first once real G3 calibration (150 in-domain + 150 OOD queries, per the file's own docstring) runs, since it's now the more frequently deciding threshold, at least on this one example | Open, flagged 2026-08-18 |

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
