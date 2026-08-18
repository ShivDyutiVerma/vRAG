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
| P-R13 | Sarvam's `/v1/chat/completions` endpoint hung indefinitely on every valid request for the first part of the session | 🟢 low | P | Confirmed transient — recovered later the same day (user's "retry later" call was correct). Re-probed: 30/30 → 4/30 failures, P50 TTFT 452ms. Full Track B path now verified working end-to-end for real. Kept as a closed record, not deleted — see `docs/DECISIONS_P.md` P-012 for the before/after numbers | Resolved 2026-08-18, same day |
| P-R15 | Sarvam's `response_format: json_schema` mode has a real bug with `array`-typed fields — the model fills preceding fields correctly then pads pure whitespace instead of emitting the array, consuming all of `max_tokens` and never terminating (`finish_reason: "length"`, `content` truncated). Isolated by removing the array field alone and observing `finish_reason` flip to `"stop"`. Also: `sarvam-105b-conversations` (the "voice-agent-tuned" variant) is broken for structured output entirely — produces pure whitespace immediately regardless of schema shape; plain `sarvam-105b` works correctly | 🟡 med | P | Worked around, not fixed upstream: `cited_chunk_ids` is now a comma-separated string in the schema, not a JSON array (`docs/DECISIONS_P.md` P-013). Switched model from `sarvam-105b-conversations` to `sarvam-105b`. If Sarvam fixes this server-side, the workaround can be reverted, but there's no cost to leaving it — a CSV string round-trips fine | Worked around 2026-08-18 |
| P-R16 | `sarvam-105b` is a reasoning model: by default every response includes a `reasoning_content` chain-of-thought that's billed and counted against `max_tokens` *before* the real answer — a short token budget can be entirely consumed by reasoning, leaving `content: null`. Sarvam's own docs recommend `reasoning_effort: null` for latency-sensitive/live-call use | 🟢 low | P | Fixed: `reasoning_effort: null` added to every Track B request (`docs/DECISIONS_P.md` P-013) | Resolved 2026-08-18 |
| P-R17 | Track B's non-streaming full-completion time is highly variable (~1.4s to 15s+ observed across repeated identical calls), well over both the 110ms target and the P50 452ms TTFT the probe measures — because non-streaming mode waits for the entire ~500-token structured response, not just the first token, and generation speed itself varies with provider load. This isn't a bug; it's the direct, expected cost of the "non-streaming for now" design choice (`docs/DECISIONS_P.md` P-013) | 🟡 med | P | Track A already exists precisely for this case and is what a 200ms-budget request gets in practice today — verified live. Real streaming (emit on first sentence, AGENT_BUILD_SPEC.md §3.3) would close most of this gap but is a real feature, not a quick fix; not started | Open, non-blocking (Track A covers it) |
| R-R14 | A3 found hybrid (RRF) retrieval *regresses* quality vs. dense-only on this corpus (`docs/DECISIONS_R.md` R-010, `docs/EVAL_RESULTS.md` §3) — root cause: BM25's weaker top ranks get equal RRF fusion weight against dense's stronger ones, with only a `top_k=10` candidate pool per lane. A larger per-lane candidate pool before fusion (e.g. top-50, truncated to top-10 after) is a standard mitigation, not yet tested — would be a second, separate ablation axis from retrieval mode | 🟢 low | R | Untested idea, not a blocker: dense-only already shipped as A3's winner and unblocks A4. Worth a quick follow-up run only if time remains after A4/A5 land | Open, non-blocking |

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
