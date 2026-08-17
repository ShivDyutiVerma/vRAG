# RISKS.md — live risk register

Seeded from `AGENT_BUILD_SPEC.md` §13. Owners assigned where the risk maps cleanly to one
workstream; joint where it doesn't. Update this any time a risk changes status — it's meant to be
read at every session start, not just written once.

| ID | Risk | Impact | Owner | Mitigation | Status |
|----|------|--------|-------|-----------|--------|
| R1 | Provider RTT makes 200ms unreachable | 🔴 high | Joint | Phase 0 probe; two-track design; consider local generation | Open — probe not yet run (see blocker below) |
| R2 | Deployment/mic fails late | 🔴 high | P | Deploy in Phase 1, redeploy every phase | Open |
| R3 | Rate limits break the benchmark run | 🟡 med | P | Retries in the bench script; run overnight if needed | Open |
| R4 | Index too large for host memory | 🟡 med | R | Cap at 200k chunks; measure RSS in Phase 1 | Open |
| R5 | MT artifacts in corpus hurt answer quality | 🟡 med | R | Spot-check in Phase 0; also the justification for G4 | Open |
| R6 | Team disagreement on the 200ms interpretation resurfaces late | 🟡 med | Joint | Settled as ADR-004 (`t_pipeline` definition), see `docs/DECISIONS.md` | Mitigated — record ADR Day 0 |
| R7 | A member misses a promotion post | 🔴 high | Joint | Named grid in `SUBMISSION_CHECKLIST.md`, checked Aug 21 | Open |
| R8 | Polish commits break the latency number | 🟡 med | P | CI latency regression test from Phase 6 | Open |
| R9 | Scope creep into a second language before core is done | 🟡 med | Joint | Hard gate: not before Phase 7 exit | Open |

## Blockers found this session (Day 0, 2026-08-17)

- **No Sarvam / Groq API keys present** (`.env` doesn't exist). Blocks `scripts/probe_latency.py`
  (R1/ADR-003) and any real STT/LLM call. Owner: whoever holds the keys — flagged to the user, not
  guessed around. Everything that doesn't need a live provider call (dataset work, chunking, index
  build, offline eval) proceeds in the meantime.
- **IPv6 is broken on this dev machine's network** (confirmed via `curl`: IPv6 to huggingface.co
  resets the connection; IPv4 works). Not a project risk per se, but worth knowing if any
  HTTP client hangs unexpectedly during dataset download — force IPv4 if so.
