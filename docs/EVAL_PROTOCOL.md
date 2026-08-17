# EVAL_PROTOCOL.md

Exactly how chunking, retrieval, guardrails, and latency get measured — so every number in
`docs/EVAL_RESULTS.md` is reproducible from this document alone.

## Chunking + retrieval quality (Workstream R)

- **Ground truth:** the dataset's own `query → relevant passage(s)` pairs. 500 pairs held out,
  frozen in `eval/heldout_queries.json` before any strategy is evaluated, never regenerated
  (`AGENT_BUILD_SPEC.md` §6.2).
- **Metrics, per strategy × embedder combination:** Recall@1/@5/@10, MRR@10, nDCG@10, chunks
  produced, index build time (s), mean search latency (ms), P95 chunk length (tokens).
- **Staged ablation** — 19 runs, not a 648-config grid (`TECH_MENU.md` §A):
  A1 chunking (6 runs, embedder/mode/rerank frozen) → A2 embedder (4 runs, A1 winner frozen) →
  A3 retrieval mode (3 runs) → A4 rerank (3 runs) → A5 generation (3 runs, Workstream P).
  Then one confirmation pass: top-2 chunkers × top-2 embedders = 4 cross-combination runs.
- **One variable per run.** A run that changes two config columns at once is void and does not get a
  ledger row (or gets one marked void).
- **Noise floor:** the winning config at each stage is run 3× and the spread reported. A "winner"
  whose margin over #2 is smaller than that spread is not a winner — ship the cheaper option and say
  so.
- **Every run appends one row** to `eval/ablation_ledger.csv` with the full column set from
  `TECH_MENU.md` §C — config hash, every param, every metric, wall-clock, git SHA, timestamp. If a
  run doesn't produce a row, it didn't happen.

## Guardrail calibration (G3 — joint, needs R's retrieval scores)

- Calibration set: 150 in-domain + 150 deliberately out-of-domain queries, frozen in
  `eval/calibration.json`.
- Sweep `τ` (top1 retrieval score threshold) and margin (`top1 − top5`) across the full observed
  range — **not** starting from an intuited 0.5+, since query→document cosine typically runs
  0.30–0.55, systematically lower than query→query similarity (`docs/BUILD_PLAN.md` P5 task 3).
- Plot false-refusal rate (in-domain, want low) vs. correct-refusal rate (out-of-domain, want high).
  Chart → `docs/assets/g3_calibration.png`. Pick the operating point from the curve, justify it in
  `docs/DECISIONS.md`.
- Target: false-refusal rate on in-domain < 10%, correct-refusal rate on out-of-domain > 80%
  (`docs/BUILD_PLAN.md` P5 exit criteria).

## Groundedness (G4 — two tiers)

- **Hot path (<10ms, deterministic):** citation-ID validation (every cited `chunk_id` must have
  actually been retrieved) runs *before* any lexical/semantic check — an LLM judge will pass
  invented citations if nobody inspects the retrieval trace first (`TECH_MENU.md` S12). Then n-gram
  overlap between answer and cited spans.
- **Offline:** Bespoke-MiniCheck or RAGAS faithfulness over a sample of answers, to report a real
  hallucination rate and show the hot-path check is a good approximation of it.

## Latency (C3/C4 — the graded numbers)

- Test set: 100 queries, stratified — 60 in-domain / 20 off-topic / 10 unsafe / 10
  degenerate/ambiguous — frozen in `eval/test_queries.json`, spoken versions synthesised with Sarvam
  TTS into `eval/audio/` so the voice benchmark is reproducible without a human talking 100 times.
- `scripts/bench_latency.py` runs all 100 × N=5 repetitions = 500 samples, discards a warm-up pass
  (states so in the report), and **never runs with caching enabled** — caching would falsify the
  number (`CLAUDE.md` hard rule).
- **Strictly sequential** — one config, one process, nothing else running on the machine. This is the
  only measurement in the whole project that is never parallelized; see `docs/PARALLEL_EXECUTION.md`
  §0 for the full reasoning, and §4 for the exact invocation.
- Reports P50/P70/**P100** (not P99 — the brief asked for the literal maximum on purpose, to reveal
  tail-latency understanding) for `t_pipeline` overall and per stage, split by Track A vs Track B, plus
  `t_e2e_voice` with STT cost broken out separately.
- CI regression: `tests/test_latency_regression.py` fails the build if `p50(t_pipeline) > 200ms` on a
  fixed 20-query smoke subset — this exists so a late "polish" commit can't silently destroy the
  headline number.

## What never gets measured together

Per `CLAUDE.md`'s hard rules: never change two variables in one experiment run (that run is void),
never run more than one config at a time during the latency pass, never enable caching during a
latency benchmark. All three are enforced by convention, not tooling — reviewers should check for
violations when reading a ledger row or a bench run before trusting it.
