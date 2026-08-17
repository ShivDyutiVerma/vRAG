# Architecture Decision Record

> SHARED, append-only. Edited only at integration syncs — concatenate `DECISIONS_R.md` and
> `DECISIONS_P.md`, sort by date. Never edit history; to reverse a decision, add a new ADR marked
> "Supersedes ADR-00N."

## ADR-001 — STT provider: Sarvam

**Date:** 2026-08-14 (pre-existing, from `AGENT_BUILD_SPEC.md`) · **Status:** Accepted
**Context:** Brief requires Sarvam or ElevenLabs. Corpus is Indic-language.
**Decision:** Sarvam.
**Rationale:** Indic-native training; code-mixing support; Indian hosting → lower RTT from an
Indian deployment, which matters given the 200ms constraint.
**Consequences:** Stack is Indic-coherent end to end. Locked unless a probe disproves the RTT
assumption.

## ADR-002 — Corpus scope: Hindi only for v1

**Date:** 2026-08-14 (pre-existing) · **Status:** Accepted
**Context:** `ai4bharat/MSMARCO-XI` covers 13 languages; indexing all of them first is scope creep.
**Decision:** Index Hindi (`hi`) only for v1. Add a second language only after Phase 7 exit.
**Rationale:** Best downstream tooling support, largest community validation, easiest translation
spot-check.

## ADR-003 — Provider RTT probe results

_Pending — Phase 0 task, owner: whoever runs `scripts/probe_latency.py` first (either track can
run this, it's infrastructure, not module-owned)._

## ADR-004 — `t_pipeline` metric definition

**Date:** 2026-08-17 · **Status:** Accepted
**Decision:** `t_pipeline` is measured server-side, from the moment the final transcript is
available to the moment the first grounded answer token is emitted to the client. Excludes
client→server transit, mic capture, speech duration. Includes input guardrails through generation
first-token. Index construction is excluded, reported separately. Full text in
`docs/EVAL_PROTOCOL.md`.
**Rationale:** Prevents the Day 3/Aug 20 argument about what "under 200ms" actually measures —
settled on Day 0/1 per `AGENT_BUILD_SPEC.md` §3.2, which is the exact wording used here.

> This file will pick up further shared ADRs (provider probe results, retrieve() contract changes
> if any, joint G3/G4 calibration decisions) at the Day 2/Day 3 integration syncs. Day-to-day,
> per-track ADRs live in `docs/DECISIONS_R.md` and `docs/DECISIONS_P.md`.
