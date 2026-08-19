# ARCHITECTURE.md

> Reconciled with the code as actually built and actually deployed, per `AGENT_BUILD_SPEC.md` §9
> P7's exit criterion. This describes the final, shipped system — not a Day-1 target. Every claim
> below is backed by a real ADR in `docs/DECISIONS_R.md` / `docs/DECISIONS_P.md`; where the system
> deliberately deviates from `AGENT_BUILD_SPEC.md`'s original assumptions (dense-only vs. hybrid,
> reranking off, Track B's budget gate), the deviation is a measured, documented decision, not an
> oversight — the ADR is cited inline.

## System design

```
┌────────────────────────────────────────────────────────────────┐
│  BROWSER (HTTPS required for mic)                              │
│  getUserMedia → PCM16 chunks → WebSocket (/voice)               │
│  Renders: live transcript, answer + citations, distinct         │
│           answered/abstained/refused/degraded states, latency   │
└───────────────────────────┬────────────────────────────────────┘
                            │ WS: audio in / transcript+answer events out
┌───────────────────────────▼────────────────────────────────────┐
│  FASTAPI ORCHESTRATOR  ("the harness") -- fully wired, live      │
│                                                                │
│  Sarvam STT     saaras:v3-realtime streaming WS, real, never    │
│                 mocked (src/vrag/stt/sarvam.py)                 │
│  ── t_pipeline clock starts once the final transcript lands ──  │
│  G1 InputGuard      safety / degenerate-input check              │
│  G2 ScopeGuard      language / scope check                       │
│  Retrieve           dense-only FAISS HNSW+SQfp16 (A3 winner,     │
│                      R-010) -- NOT hybrid, NOT reranked (A4,     │
│                      R-012/R-038: both measured net-negative     │
│                      on this corpus, not left untested)          │
│  G3 GroundGate      top1-cosine >= TAU(0.8835) -> ABSTAIN?       │
│                      (R-015, real 300-query calibration)         │
│  Track A            extractive span-select, real, ~5ms           │
│  Track B            streaming Sarvam LLM, budget-gated (R-036):  │
│                      skipped outright if remaining budget can't  │
│                      realistically fit a fair attempt             │
│  G4 Groundedness    citation/n-gram overlap check                │
│  G5 OutputGuard     PII redaction                                │
│  Assemble           AnswerResponse + TraceRecord emit             │
└───────────────────────────┬────────────────────────────────────┘
                            │
        ┌───────────────────┼──────────────────┐
        ▼                   ▼                  ▼
  In-process assets    Sarvam APIs        Telemetry sink
  • FAISS HNSW+SQfp16  • STT (WS)         • traces.jsonl
    (dense only)       • LLM (Track B)    • per-stage ns timings
  • LiteE5Embedder
    (SentencePiece +
     ONNX int8, no
     torch/transformers)
  • guardrail checks
```

Full original rationale for the target design is in `docs/AGENT_BUILD_SPEC.md` §5; this file
describes what actually shipped, including every place that differs from that target and why.

## Request lifecycle walkthrough

1. Browser opens `WS /voice`, streams PCM16 audio as the user speaks.
2. Sarvam's streaming STT (`saaras:v3-realtime`) returns interim + final transcripts
   (`src/vrag/stt/sarvam.py`). The receive loop bounds the wait to 10s while no transcript has
   arrived yet (`NO_SPEECH_TIMEOUT_S`, P-023) — a silent/no-speech session gets a clear
   `"No speech detected yet. Please try again."` error instead of hanging until Sarvam's own ~60s
   inactivity watchdog. Once a real transcript arrives, the wait reverts to unbounded so a normal
   mid-sentence pause is never mistaken for silence. `t_pipeline` starts the instant the final
   transcript is available — mic capture and STT itself are reported separately as `t_e2e_voice`,
   never folded into the headline number (§3.2 of the build spec, quoted verbatim in the README).
3. The harness (`src/vrag/harness/pipeline.py`) runs the transcript through typed stages over a
   `PipelineContext`, each stage getting `remaining_ms` from `budget.py`'s deadline propagation —
   **fully wired into both `/ask` and `WS /voice`**, not a parallel shape.
4. `retrieve()` (the R/P seam, `src/vrag/retrieval/interface.py`) calls `HybridRetriever` in
   **`retrieval_mode="dense"`** — the shipped default, not hybrid. The A3 ablation
   (`docs/DECISIONS_R.md` R-010, 500-query held-out set) measured dense-only beating hybrid+RRF on
   this corpus (Recall@5 0.652 vs. 0.604) — BM25 is comparatively weak on this machine-translated
   Hindi text, and naive RRF gives its lower-quality top ranks equal fusion weight against dense's
   stronger ones. `retrieval_mode="hybrid"` remains fully implemented and tested for anyone
   revisiting it, but is not what runs today. Reranking is likewise implemented
   (`src/vrag/retrieval/rerank.py`, FlashRank + cross-encoder) but **off by default** — A4
   (R-012) found `none` wins outright on the full 500-query set, and a follow-up targeted
   experiment (R-038) built specifically to test reranking's strongest hypothesized case — queries
   where dense returns same-template distractors, e.g. "what is India's capital" surfacing
   London/Islamabad/Munich "capital of X" boilerplate — found reranking *still* net-negative there
   too (Recall@5 0.7049→0.3934 on that exact targeted subset, ~100x latency cost, and the
   flagship example gets *worse*, not better). Both are measured, closed findings, not gaps.
5. `GroundGate` (G3) decides answered vs. abstained from a calibrated `TAU=0.8835` top1-cosine
   threshold (`docs/DECISIONS_R.md` R-015 — real 300-query calibration: 150 in-domain + 150
   out-of-domain). The calibration found `docs/EVAL_PROTOCOL.md`'s two targets (false-refusal<10%
   AND correct-refusal>80%) are **not simultaneously reachable** on this corpus via top1-cosine
   gating alone — `TAU=0.8835` is the balanced operating point (19.3% false-refusal / 75.3%
   correct-refusal), a documented product tradeoff, not an oversight.
6. Track A always computes a real extractive answer (span-selection over the top retrieved chunk,
   ~5ms stage cost) — this is not a placeholder, it is what most answers actually ship as. Track B
   streams a generative rewrite over it if the budget realistically allows; its first token
   replaces Track A's answer in the UI when it lands. `GenerateStage`'s pre-flight budget gate
   (`docs/DECISIONS_R.md` R-036) skips attempting Track B entirely — rather than starting it and
   paying a doomed ~2s wait — whenever the remaining budget can't fit a fair Sarvam attempt
   (`min_viable_ms`, reusing the circuit breaker's existing `MIN_FAIR_TIMEOUT_S=2000ms` constant).
7. `VerifyOutput` runs G4 (citation-ID + n-gram grounding check) and G5 (PII redaction) before the
   response is assembled and emitted with `timings_ms` populated for every stage that ran.
8. A `TraceRecord` is appended to `traces.jsonl` for every real request, regardless of outcome
   (answered/abstained/refused/degraded).

## Two-track answer design

Track A (extractive, always runs, real stage cost ~5ms P50 / 8.9ms P100) is emitted reliably
inside the 200ms *stage* budget. Track B (generative, streams in behind it) replaces Track A in
the UI when its first token arrives, if it arrives in time — see `docs/AGENT_BUILD_SPEC.md` §3.3
for the full rationale. **What actually ships:** both tracks are real and working; see
`docs/LATENCY_BUDGET.md` for the honest gap between Track A's stage cost and what a client
actually experiences end-to-end, and why that gap exists.

## STT and frontend, as actually shipped

- **STT** (`src/vrag/stt/sarvam.py`): real `saaras:v3-realtime` WebSocket client, never mocked in
  committed code (repo-wide `tests/` grep for STT mocking returns zero matches). Bounded 10s
  no-speech timeout (P-023) plus a guarded sender-shutdown path (send-after-close no longer leaks
  an unretrieved asyncio Task exception) — both found via a real, live human-browser-and-microphone
  test against the deployed URL, not a hypothetical.
- **Frontend** (`frontend/index.html`): a pure `describeAnswerResponse(ar)` function gives each of
  the four canonical statuses (`answered`/`degraded`/`abstained`/`refused`) its own state, headline,
  pill class, and pill text — an earlier version collapsed all three non-`answered` states into one
  hardcoded "Abstained" pill, which would have made a G1 refusal visually indistinguishable from a
  G3 abstention in the demo video. Fixed and verified live against the deployed page's own
  `window.__describeAnswerResponse` test hook (`docs/DECISIONS_P.md` P-022).

## Module boundaries

See `docs/TEAM_SPLIT.md` §2 for the original ownership table (chunking/embedding/index/retrieval =
Workstream R; harness/guardrails/generation/telemetry/API/frontend/deployment = Workstream P). As
of 2026-08-19 this became effectively a single-operator build (P's collaborator ran out of session
credits) — later P-owned fixes (frontend, STT) were made from the R-flagged session and relabeled
into `docs/DECISIONS_P.md` once the module-ownership mismatch was caught (see P-022/P-023's own
context notes), rather than left mislabeled.

## Memory optimization story

Production retrieval memory dropped from a **1,860MB original GPU-dev-machine baseline** to
**493.8MB steady-state** locally (`docs/DECISIONS_R.md` R-030, a **73.4% cut**) through two real
engineering levers, both verified at zero quality cost:
1. **`LiteE5Embedder`** (R-019/R-022/R-029/R-030): raw `onnxruntime` + Google's `sentencepiece`
   library instead of `sentence-transformers` (which imports `torch` as a hard dependency
   regardless of backend — ~383MB RSS for `import torch` alone) and instead of the HF `tokenizers`
   Rust library (~262MB RSS for this model's 250k-token vocabulary). Verified byte-identical
   output to the original embedder (cosine similarity 1.0) and 100.0000% exact tokenizer-ID
   equivalence against 1,020 real test strings.
2. **FAISS scalar quantization** (`quantization="sqfp16"`, R-033/R-034): `IndexHNSWSQ` +
   `ScalarQuantizer.QT_fp16` stores vectors as 2-byte half-floats instead of 4-byte float32, same
   M/efConstruction/efSearch/metric — saves ~77MB at zero measured Recall@k/MRR@10 regression
   (an int8 alternative was also measured and rejected for a real quality cost this doesn't have).

**Verified under the real target constraint, not just locally:** `docker run -m 512m
--memory-swap 512m` (Render free tier's actual limit) — startup 204.4MiB, peak after 10 real
queries 397.8MiB, **~114MB headroom**, 10/10 queries survived, real citations returned on 6/10
(not `stub-chunk-001`), Recall@10 0.748 vs. 0.750 fp32 baseline (noise-floor difference), MRR@10
0.45550 vs. 0.45627 (~0.17% relative). **This is now confirmed live** — `GET /healthz` on the
deployed URL returns `{"status":"ok","retrieval":"real"}`, verified 2026-08-20 as part of the
redeploy that shipped the STT and frontend fixes.

## Deploy runbook

**Live now:** `https://vrag-voice.onrender.com` — Render (Docker runtime, `render.yaml` Blueprint
at repo root, `Dockerfile` downloads the pre-built index (`index-metadata_aware-v3`, sqfp16) and
embedder (`embedder-lite-onnx-v2`) release assets at build time — **never built inside the
container**, per `docs/AGENT_BUILD_SPEC.md` §5.3 — then serves via `uvicorn vrag.api.main:app`).
Real retrieval confirmed live (not the Day-0 stub), health-checked at `/healthz`.

**Known, honest limitation:** Render's free tier gives 0.1 shared CPU (confirmed via Render's own
documented specs), and a live-Render latency investigation (`docs/DECISIONS_R.md` R-036,
`eval/live_render_latency_results.json`, 40 real sequential queries) found the retrieve stage
alone runs 10-40x slower on live Render than the identical code path measured locally/in Docker
(P50 594.9ms vs. 12-47ms) — predominantly CPU contention on the shared free-tier vCPU, not memory
pressure (Render's own metrics API showed steady ~409-430MB throughout) and not a code
regression. **`t_pipeline` does not hold under 200ms on Render's free tier for an answered query**
— see `docs/LATENCY_BUDGET.md` for the full honest breakdown, both the local/Docker numbers where
the stage-cost budget genuinely is met and the live-Render numbers where CPU contention dominates.

**Why Render, not Hugging Face Spaces (the spec's original insurance target):** attempted HF Spaces
first per `docs/AGENT_BUILD_SPEC.md` §5.3; hit a real paywall (`402 Payment Required` — HF's free
tier no longer covers Docker/Gradio Spaces, policy changed since the spec was written). Switched to
Render — the spec's own primary recommendation, not just a fallback. Full record:
`docs/DECISIONS_P.md` P-002 (superseded) and P-005.

**Known non-blocking issue:** the `/voice` WebSocket lingers ~20-25s before a clean close on Render
free tier (doesn't affect answer delivery — `transcript_final` and `answer_final` both arrive
correctly well before the lingering-connection window). Tracked as P-R12 in `docs/RISKS.md`.
