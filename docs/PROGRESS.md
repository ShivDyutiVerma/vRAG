# PROGRESS — SHARED

> Edited only at integration syncs (`docs/TEAM_SPLIT.md` §5), by whoever is merging — a hand-written
> combined summary of `PROGRESS_R.md` + `PROGRESS_P.md`. Not a running log; a snapshot.

**Last updated:** 2026-08-17 (Day 0, docs bootstrap — no sync has happened yet)
**Current phase:** P0 — Foundations & Probes
**Days remaining:** 5 (deadline 2026-08-22 23:59 IST)
**Build status:** 🟡 not yet integrated — both workstreams starting independently, no code merged to `main` yet

## Where we are, in one paragraph

Day 0. The repo is initialized and pushed to GitHub with the full docs/spec bundle (initial commit
`a1bc431`). `docs/` has just been bootstrapped by Workstream R per the "First session only" ritual.
Workstream R (this session) is about to start on the Phase 0 tasks that are R's to do: repo skeleton,
dataset reconnaissance, the `retrieve()` interface stub, and starting the chunking ablation. Workstream
P has not started a session yet as of this snapshot. No code exists under `src/vrag/` yet.

## Phase exit criteria — current phase (P0)

- [ ] Probe results committed; provider chosen with evidence, recorded as ADR-003 — **blocked**, no API keys yet
- [x] `t_pipeline` definition agreed and recorded as ADR-004 — recorded, pending Day 0 sync confirmation with P
- [ ] Dataset subset on disk; passage-length distribution known; chunk-count estimate written down
- [ ] 500 held-out pairs frozen and committed
- [ ] `pytest` green, `ruff` clean, CI passing

## What works right now (verified, not assumed)

- Repo on GitHub, `main` branch pushed and matching local `HEAD` — verified by hash comparison ✅ 2026-08-17
- `docs/` fully scaffolded per `CLAUDE.md`'s "First session only" list ✅ 2026-08-17

## What is stubbed / faked / TODO

- Everything under `src/vrag/` — nothing written yet, this is Day 0.
- `scripts/probe_latency.py` — not yet written; blocked on API keys for actually running it.
- `retrieve()` — interface documented, real implementation not started.

## Live numbers

| Metric | Value | Measured on |
|--------|-------|-------------|
| p50 t_pipeline | — | — |
| Recall@5 (prod strategy) | — | — |

## Blockers

- No Sarvam/Groq API keys on this machine — blocks `scripts/probe_latency.py` and ADR-003. Owner: user.

## Next session should start by

1. Whoever merges next reads both `PROGRESS_R.md` and `PROGRESS_P.md` and updates this file for real.
