# EVAL_PROTOCOL.md

## Chunking / embedding / retrieval (Workstream R's protocol — §1-3 of EVAL_RESULTS.md)

500 held-out query→passage pairs (`eval/heldout_queries.json`), frozen before run 1, never
regenerated. Per strategy × embedder: Recall@1/@5/@10, MRR@10, nDCG@10, chunk count, index build
time, index size, mean search latency, P95 chunk length. Staged greedy ablation per
`docs/TECH_MENU.md` §A — one variable per run, every run appended to `eval/ablation_ledger.csv`.
Winning config run 3× to establish the noise floor before being declared a winner.

## Guardrail calibration (joint — G3/G4)

Calibration set: 150 in-domain + 150 deliberately out-of-domain queries (`eval/calibration.json`).
Sweep τ (retrieval confidence threshold) and the top1−top5 margin; plot false-refusal rate
(in-domain) vs correct-refusal rate (out-of-domain); pick the operating point from the curve, not
intuition. Prior from the literature to sanity-check against: query→document cosine similarity
typically runs ~0.30-0.55, systematically lower than query→query similarity — a threshold guessed
at 0.5+ will refuse almost everything. Chart → `docs/assets/g3_calibration.png`.

G4 offline: run Bespoke-MiniCheck or RAGAS faithfulness over a sample of generated answers to
report a real hallucination rate, quantifying how good the cheap hot-path lexical/citation check
is as an approximation.

## Latency (`t_pipeline`, C3/C4 — my protocol, §4-6 of EVAL_RESULTS.md)

**Metric definition (verbatim, goes in the README too):**
> `t_pipeline` is measured server-side, from the moment the final transcript is available to the
> moment the first grounded answer token is emitted to the client. It excludes: client→server
> network transit, microphone capture, and speech duration. It includes: input guardrails, query
> embedding, hybrid retrieval, fusion, grounding gate, and answer generation up to first token.
> Index construction is a one-time offline cost, reported separately and excluded from `t_pipeline`.

100 queries (60 in-domain / 20 off-topic / 10 unsafe / 10 degenerate), TTS'd via Sarvam for
reproducibility, run N=5 each = 500 samples through `scripts/bench_latency.py`. Discard a warm-up
pass and say so. Report P50/P70/**P100** (not P99 — P100 is the true max, on purpose) for total and
per-stage, separately for Track A and Track B. Caching disabled during any latency run — it would
falsify the number. Strictly sequential execution on an otherwise-idle machine — see
`docs/PARALLEL_EXECUTION.md` §0 and §4; never run alongside a quality-pass ablation.

## The one rule that governs both

If a run doesn't produce a row in `eval/ablation_ledger.csv`, it didn't happen. If two config
columns differ in the same run, neither number is evidence.
