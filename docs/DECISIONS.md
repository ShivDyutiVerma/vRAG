# Architecture Decision Record — SHARED

> Append-only. Edited only at integration syncs (`docs/TEAM_SPLIT.md` §5) by whoever is merging —
> concatenate `DECISIONS_R.md` + `DECISIONS_P.md`, sort by date. To reverse a decision, add a new ADR
> that supersedes it; never edit history.

## ADR-001 — STT provider: Sarvam

**Date:** 2026-08-17 (pre-decided in `AGENT_BUILD_SPEC.md` §2, transcribed here as the formal record)
**Status:** Accepted
**Context:** Brief requires Sarvam or ElevenLabs (constraint C1). Corpus (`ai4bharat/MSMARCO-XI`) is Indic-language.
**Decision:** Sarvam.
**Rationale:** Indic-native training, code-mixing support, India-hosted → lower RTT from an Indian
deployment, which matters directly against the 200ms constraint. ElevenLabs rejected — strongest in
English/major European/East Asian languages, not the corpus's language family (`TECH_MENU.md` S2).
**Consequences:** STT stack is Indic-coherent end to end. Locked unless a probe disproves the RTT
assumption (see ADR-003, pending). **Confirmed for real** at the Day 1 sync — Workstream P verified
real Sarvam realtime STT end to end against the live deployment (`docs/DECISIONS_P.md` P-007).

## ADR-002 — Corpus scope: Hindi only for v1

**Date:** 2026-08-17 (pre-decided in `AGENT_BUILD_SPEC.md` §6.1)
**Status:** Accepted
**Context:** `ai4bharat/MSMARCO-XI` covers 13 Indian languages. Indexing all of them burns days for no
grading benefit and no team has validated translation quality across all 13.
**Decision:** Index Hindi (`hi`) only, targeting 50k–200k chunks.
**Rationale:** Best downstream tooling support, largest community validation, easiest to spot-check
translation quality by eye. A second language is explicitly gated to Phase 7, only if everything else
is green (`docs/BUILD_PLAN.md` cut list — second language is cut item #1 if behind schedule).
**Consequences:** All chunking/embedding/retrieval ablation work (A1–A4) runs on the Hindi subset only.

## ADR-003 — Provider RTT probe results

**Date:** _pending_
**Status:** **Blocked** — no Sarvam or Groq API keys present in either workstream's dev environment
yet. `scripts/probe_latency.py` (Workstream R) is written and ready to run the moment keys are
available. Owner: whoever gets keys first — infrastructure, not module-owned, either track can run
it. No numbers get recorded here until the real script produces them.

## ADR-004 — `t_pipeline` metric definition

**Date:** 2026-08-17
**Status:** Accepted — confirmed by both workstreams at the Day 1 sync.
**Decision:**
> `t_pipeline` is measured server-side, from the moment the final transcript is available to the
> moment the first grounded answer token is emitted to the client. It excludes: client→server
> network transit, microphone capture, and speech duration. It includes: input guardrails, query
> embedding, hybrid retrieval, fusion, grounding gate, and answer generation up to first token.
> Index construction (chunking + embedding + index build) is a one-time offline cost, reported
> separately and excluded from `t_pipeline`.
**Rationale:** Defensible and standard for voice-RAG systems; separates one-time indexing cost from
per-request latency; `t_e2e_voice` (mic-stop → first audible/visible answer) is reported as a
secondary honest number so STT cost isn't hidden. Settles the metric definition on Day 1 rather than
under deadline pressure later — the exact failure mode risk R6 in `docs/RISKS.md` warns about.
**Consequences:** This exact wording goes in the README verbatim. Full protocol detail in
`docs/EVAL_PROTOCOL.md`.

---

> Further shared ADRs (provider probe results once run, `retrieve()` contract changes if any, joint
> G3/G4 calibration decisions) land here at future integration syncs. Day-to-day, per-track ADRs
> live in `docs/DECISIONS_R.md` and `docs/DECISIONS_P.md` — read both before assuming this file is
> current, since it's only touched at sync points by design.
