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

**Date:** 2026-08-19
**Status:** Accepted — real Sarvam key obtained (single-operator session, P's collaborator out of
weekly credits, see the operational note in `docs/PROGRESS.md`); Groq key still empty, that leg
correctly shows as skipped rather than guessed. `scripts/probe_latency.py --n 30`, real output:

**1. Raw TCP + TLS connect time (no API key needed):**

| Provider | Host | P50 (ms) | P95 (ms) | P100 (ms) | Failures |
|----------|------|----------|----------|-----------|----------|
| sarvam | api.sarvam.ai | 98.7 | 119.6 | 121.8 | 0/30 |
| groq | api.groq.com | 67.4 | 88.9 | 92.3 | 0/30 |

**2. Chat completion TTFT:**

| Provider | Model | P50 (ms) | P95 (ms) | P100 (ms) | Failures |
|----------|-------|----------|----------|-----------|----------|
| sarvam | sarvam-105b | 271.3 | 1403.3 | 1965.8 | 0/30 |
| groq | — | — | — | — | SKIPPED — no API key in `.env` |

**Decision:** Sarvam confirmed as the only real candidate (ADR-001's choice stands) — no Groq key
means no real comparison was ever possible, so this isn't a competitive result, just confirmation
that Sarvam's connection physics (~99ms TCP+TLS from this machine) and TTFT (P50=271ms) are real
and measured, not assumed. **Consequence for the 200ms budget:** even Sarvam's P50 TTFT alone
(271ms) exceeds the entire pipeline target before a single retrieval/guardrail stage runs — this is
the quantitative confirmation behind the two-track (Track A always-fast, Track B best-effort)
design, not a new finding but the number that was missing to state it with real evidence. See
`docs/LATENCY_BUDGET.md` for the full P6 latency campaign this probe was run alongside, including a
separate, larger (N=30 real generation calls, not just TTFT-to-first-chunk) Track B measurement
showing P50=1976.5ms full completion.

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

## ADR-005 — P6 latency campaign: real numbers, and Track A's answer is fast but the wire hides it

**Date:** 2026-08-19
**Status:** Accepted. Joint ownership per `docs/TEAM_SPLIT.md` §2 ("Latency benchmark — JOINT"); run
solo (single-operator session, see `docs/PROGRESS.md`'s operational note) so recorded directly here
rather than merged from two per-track logs.
**Context:** `docs/BUILD_PLAN.md` P6 — "never cut" graded core, 0% built before this session.
Built `scripts/make_test_queries.py` (100 real queries, 60 in-domain/20 off-topic/10 unsafe/10
degenerate, every one sourced from already-real, already-vetted content — the frozen held-out set,
G3's calibration out-of-domain pool, and the existing guardrail test suites — not invented),
`scripts/synthesize_test_audio.py` (95 real Sarvam-TTS 16kHz WAVs), `scripts/bench_latency.py`
(500 text samples + a standalone 30-call Track B measurement), `scripts/make_latency_charts.py`,
and `tests/test_latency_regression.py`.
**The headline finding:** Track A's true stage cost (retrieve + all guardrails + extraction,
summed from the harness's own `timings_ms`) is **P50=5.2ms, P100=8.9ms** — comfortably under the
200ms target, exit criterion genuinely met. But the client-observed end-to-end latency for an
answered query is **P50=213-246ms** (three runs) — because `GenerateStage` attempts a real Track B
call on every answerable query with whatever budget remains (~190ms), Sarvam's real latency is
~2s, so the attempt always times out, and the circuit breaker never trips (it only counts >=2.0s
"fair chance" failures, never reached under a 200ms budget) — so every request pays that doomed
wait before Track A's already-ready answer returns. **Decision (user, 2026-08-19): report both
numbers honestly, no harness behavior change** — the shedding design is deliberate and correct;
Sarvam being ~10x slower than the total budget is what makes each attempt futile in practice, not
a flaw in the design itself. Matches `docs/BUILD_PLAN.md`'s own guidance to report Track B's real
cost plainly rather than hide it. Full breakdown, charts, and the response-status mix:
`docs/LATENCY_BUDGET.md`.
**A real bug was found and fixed along the way**, not a pre-existing known issue — see
`docs/DECISIONS_P.md` P-021: Track B's streaming handler crashed on an explicit `null` content
delta from Sarvam's SSE stream, silently masking most of Track B's real success rate (0/30 before
the fix, 19/30 = 63.3% after, in this session's own measurement). Recorded in P's log since the
fix lives in P's module; referenced here since it materially changed this ADR's own Track B numbers
mid-investigation.
**Consequences:** `eval/ablation_ledger.csv` has a new `p6_latency_bench_*` row. `eval/test_queries.json`
and `eval/audio/` are now real, committed, reusable assets for any future latency/voice work — no
need to re-synthesize or re-sample. `tests/test_latency_regression.py` guards Track A's true stage
cost (not the wall-clock-with-Track-B-wait number) against future regression, skipping cleanly on a
fresh checkout without the local index artifact rather than failing or faking a pass.

---

> Further shared ADRs (`retrieve()` contract changes if any, joint G3/G4 calibration decisions) land
> here at future integration syncs. Day-to-day, per-track ADRs live in `docs/DECISIONS_R.md` and
> `docs/DECISIONS_P.md` — read both before assuming this file is current, since it's only touched at
> sync points by design (or, from 2026-08-19, directly by the single-operator session for genuinely
> joint-ownership work — see `docs/PROGRESS.md`).
