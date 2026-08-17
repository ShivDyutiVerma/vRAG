# EVAL_RESULTS.md

> Shared file. §1-3 (chunking, embedding, retrieval) are Workstream R's to write. §4-6
> (generation, guardrails, latency) are Workstream P's to write. Different sections of one file
> rarely conflict if everyone stays in their lane — see `docs/TEAM_SPLIT.md` §3.

## §1. Chunking (A1) — Workstream R

_Not yet run._

## §2. Embedding (A2) — Workstream R

_Not yet run._

## §3. Retrieval mode + reranking (A3, A4) — Workstream R

_Not yet run._

## §4. Generation (A5) — Workstream P

_Not yet run. Depends on the Phase 0 provider probe._

## §5. Guardrail calibration (G3/G4) — Workstream P (joint with R for scores)

_Not yet run. Needs a real `retrieve()` implementation with real scores before the G3 threshold
sweep is meaningful — Day 3 per `docs/TEAM_SPLIT.md` §5._

## §6. Latency (`t_pipeline`) — Workstream P

_Not yet run. Requires `scripts/bench_latency.py` (Phase 6) and a stable end-to-end pipeline._
