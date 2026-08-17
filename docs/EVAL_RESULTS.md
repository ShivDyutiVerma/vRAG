# EVAL_RESULTS.md

> Shared file, split by section: §1–3 (chunking / embedding / retrieval) is Workstream R's to write,
> §4–6 (generation / guardrails / latency) is Workstream P's. Different sections of one file rarely
> conflict if everyone stays in their lane (`docs/TEAM_SPLIT.md` §3). **Every number below must trace
> to a row in `eval/ablation_ledger.csv` or a `traces.jsonl` run — no number gets written here first.**

## §1 — Chunking (A1)

_Not run yet. Will report Recall@1/@5/@10, MRR@10, nDCG@10, chunk count, index build time, mean
search latency, P95 chunk length for all 6 strategies, per `docs/EVAL_PROTOCOL.md`._

## §2 — Embedding (A2)

_Not run yet._

## §3 — Retrieval mode + reranking (A3, A4)

_Not run yet. Will include the efSearch recall-vs-latency curve (`docs/assets/efsearch_curve.png`)._

## §4 — Generation (A5)

_Not run yet — Workstream P._

## §5 — Guardrails

_Not run yet — Workstream P + joint G3/G4 calibration. Will include
`docs/assets/g3_calibration.png`._

## §6 — Latency (P50/P70/P100)

_Not run yet — Workstream P, Phase 6. Will include `docs/assets/latency_breakdown.png` and
`docs/assets/latency_cdf.png`._
