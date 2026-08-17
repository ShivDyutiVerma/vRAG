# BUILD_PLAN.md
## Phased execution plan — HH Goa Task 2 Voice RAG

> Lives at `docs/BUILD_PLAN.md`. Read the current phase at the start of every session.
> **Phases are gates.** Never begin phase N+1 until every exit criterion of phase N is ticked
> in `docs/PROGRESS.md` with evidence (a number, a file path, or a screenshot — not an assertion).

> **⚠️ DATE CORRECTION (Aug 17):** this file was originally written assuming an Aug 14 start
> and one solo builder. Neither is true — today is **Aug 17**, and the project is now split
> across two people working in parallel (see `docs/TEAM_SPLIT.md`). The phase **content** below
> (tasks, exit criteria) is still exactly what needs to happen — nothing about the work changed.
> Only the calendar mapping is stale. **`docs/TEAM_SPLIT.md` is now the authoritative schedule**;
> map each phase's tasks onto whichever workstream (R or P) owns it there, on the compressed
> 5-day calendar. The table below is left for reference only.

**Original window (stale):** Aug 14 → Aug 22, 23:59 IST · **Deadline (still correct):** Aug 22, 23:59 IST · **No resubmissions.**

**Phase → date map (stale — see `docs/TEAM_SPLIT.md` for the real one)**

| Phase | Name | Dates (as originally planned) | Slack |
|-------|------|-------|-------|
| P0 | Foundations & Probes | Aug 14 | — |
| P1 | Walking Skeleton & First Deploy | Aug 15 | — |
| P2 | Chunking Lab (ablation A1) | Aug 16 | — |
| P3 | Embedding + Retrieval (A2–A4) | Aug 17 | — |
| P4 | Harness Hardening | Aug 18 (AM) | — |
| P5 | Guardrails | Aug 18 (PM) – Aug 19 (AM) | — |
| P6 | Latency Campaign (A5) | Aug 19 (PM) – Aug 20 (AM) | — |
| P7 | Polish, Evidence, README | Aug 20 (PM) – Aug 21 (AM) | — |
| P8 | Freeze, Videos, Submit | Aug 21 (PM) – Aug 22 | ~1 day |

---

# P0 — Foundations & Probes
**Objective: measure the physics and scaffold the project before designing around either.**

### Prerequisites
- Sarvam API key obtained and confirmed working
- At least one LLM provider key (Groq / Sarvam / other)
- Python 3.11+, git, GitHub repo created (public)

### Tasks
1. **Repo skeleton**
   - `pyproject.toml` (uv or poetry), `.gitignore` (with `.env`, `data/`, `*.faiss`), `.env.example`
   - `ruff` + `mypy` + `pre-commit` configured and passing on an empty codebase
   - GitHub Actions: lint + test on push
   - Directory tree per `AGENT_BUILD_SPEC.md` §5, with `__init__.py` and docstrings only
2. **Scaffold `docs/` completely** — every file listed in `CLAUDE.md` §Docs, populated with the templates, not left empty
3. **`scripts/probe_latency.py`** — the most important script in P0:
   - Measures, from your intended deployment region, for each candidate provider: TCP connect, TLS handshake, **TTFT**, full completion, over **N=30 samples**
   - Reports **P50/P95/P100**, not the mean (tail latency is what kills you)
   - Providers: Sarvam STT, Sarvam LLM, Groq, SambaNova/Fireworks if keys available, plus a local `llama.cpp` baseline
   - Also probe from 2 candidate hosting regions if you can (e.g. a free-tier box in `ap-south-1` vs `us-east-1`)
   - **Output → `docs/DECISIONS.md` as ADR-003 with the raw table**
4. **Dataset reconnaissance**
   - Download the Hindi (`hi`) config of `ai4bharat/MSMARCO-XI`; slice a subset targeting 50k–200k chunks
   - `scripts/inspect_dataset.py`: print schema, field types, passage-length distribution, chunk-count estimate
   - **Manually read 20 translated passages.** Note quality impressions in `docs/DECISIONS.md`. If Hindi looks poor, check one other language before committing.
   - Freeze and commit `eval/heldout_queries.json` — 500 query→passage pairs, excluded from indexing
5. **Team alignment (do not skip):** agree and write down the `t_pipeline` metric definition (`AGENT_BUILD_SPEC.md` §3.2) as ADR-004. This prevents the Aug 20 argument.

### Deliverables
`docs/` fully populated · `scripts/probe_latency.py` + results table · `eval/heldout_queries.json` · green CI on an empty project

### Exit criteria
- [ ] Probe results committed; provider chosen with evidence, recorded as ADR-003
- [ ] `t_pipeline` definition agreed and recorded as ADR-004
- [ ] Dataset subset on disk; passage-length distribution known; chunk-count estimate written down
- [ ] 500 held-out pairs frozen and committed
- [ ] `pytest` green, `ruff` clean, CI passing

### If this phase fails
If no provider clears ~150ms TTFT from your region: **do not proceed to P1 as designed.** Escalate to
the user, and prepare the local-model path (`llama.cpp` + Qwen2.5-1.5B) as ADR-005.

---

# P1 — Walking Skeleton & First Deploy
**Objective: a public URL where a stranger speaks Hindi and gets a real answer. Ugly is fine.**

> Highest-risk phase in the project. Deployment failures found today cost an afternoon; found on Aug 21 they cost the submission.

### Tasks
1. **Embedder** (`src/vrag/index/embedder.py`)
   - `multilingual-e5-small`, PyTorch first, ONNX in P6
   - **Write `tests/test_embedder.py` FIRST**, asserting the `"query: "` / `"passage: "` prefixes are applied and that vectors are L2-normalised
2. **Index build** (`scripts/build_index.py`)
   - Naive fixed-size chunking, ~10k passages only (fast iteration)
   - FAISS `IndexHNSWFlat`, `METRIC_INNER_PRODUCT`, M=32, efConstruction=200
   - Persist index + chunk metadata; record an index build hash
3. **Sarvam STT** (`src/vrag/stt/sarvam.py`) — REST first for simplicity; WebSocket streaming lands in P4
4. **Minimal pipeline** — embed → search top-5 → one LLM call → text answer. No harness yet.
5. **FastAPI app** — `POST /ask` (text, for testing) and `WS /voice` (audio)
6. **Minimal frontend** — one HTML page: record button, transcript, answer. No framework needed today.
7. **DEPLOY.** Container → chosen region. Verify HTTPS. **Open the live URL on a phone over mobile data and speak into it.**

### Deliverables
Public HTTPS URL · screenshot in `docs/assets/p1-first-answer.png` · deploy runbook in `docs/ARCHITECTURE.md`

### Exit criteria
- [ ] Live URL works from a phone on mobile data (not just your laptop on wifi)
- [ ] Real Sarvam transcription — **zero mocking anywhere in the STT path**
- [ ] Answer visibly derives from a retrieved passage (log the passage alongside the answer)
- [ ] Embedder prefix test green
- [ ] `docs/PROGRESS.md` updated with what's real vs. stubbed

### If this phase fails
Fall back to Hugging Face Spaces immediately — do not spend more than 3 hours debugging a container
deploy on day 2. A working ugly deploy beats a perfect broken one.

---

# P2 — Chunking Lab (ablation A1)
**Objective: implement ≥6 chunking strategies and prove which wins with numbers.**

### Tasks
1. **`src/vrag/chunking/base.py`** — `ChunkingStrategy` protocol: `name`, `chunk(doc) -> list[Chunk]`, `config() -> dict`
2. **`registry.py`** — self-registration so the eval script enumerates strategies without code changes
3. **Implement strategies 1–6** from `TECH_MENU.md` §S5 (fixed+overlap, passage-native, sentence-window, semantic, metadata-aware, hierarchical)
4. **`scripts/eval_chunking.py`**
   - For each strategy: build index → run 500 held-out queries → compute Recall@1/5/10, MRR@10, nDCG@10, chunk count, index build time, index size, mean search latency, P95 chunk length
   - Append every run to `eval/ablation_ledger.csv`
   - **Run the winner's config 3× to establish the noise floor**
5. **Overlap sub-study** — for strategy 1, sweep overlap ∈ {0, 10%, 20%}. The "always overlap" rule is not universal; find out if it holds here.
6. **Write `docs/EVAL_RESULTS.md`** — markdown table + a bar chart into `docs/assets/`
7. Promote the winner to production config; redeploy

### Deliverables
6 strategies · `eval/ablation_ledger.csv` with ≥9 rows · `docs/EVAL_RESULTS.md` §1 · noise-floor number

### Exit criteria
- [ ] ≥6 strategies implemented, each with a unit test asserting chunk-boundary behaviour
- [ ] All evaluated on the frozen 500-query set; ledger committed
- [ ] Noise floor reported; the declared winner beats #2 by more than it
- [ ] Winner's Recall@5 ≥ 0.75 (if not, that's a P3 retrieval problem — note it, proceed)
- [ ] Production config updated; redeployed

### Guard against
Declaring a winner that's inside the noise band. If strategies are statistically tied, **ship the
cheapest one and say they were tied** — that's a better finding than a fake winner.

---

# P3 — Embedding + Retrieval (ablations A2, A3, A4)
**Objective: fix the embedder, make retrieval hybrid, and decide on reranking with data.**

### Tasks — morning (A2: embedder, 4 runs)
1. Add embedder backends behind one interface: `multilingual-e5-small` (default), `potion-multilingual-128M` (Model2Vec), `BGE-M3`, `Vyakyarth-1-Indic`
2. Re-index with the P2-winning chunker for each; run the eval; log to the ledger
3. **Record both quality AND query-embed latency per model.** A model that wins Recall@5 by 2pts and costs 25ms may still lose.
4. **Cross-check:** take top-2 chunkers × top-2 embedders = 4 confirmation runs (interaction effects)

### Tasks — afternoon (A3: retrieval mode, A4: rerank)
5. **`src/vrag/index/sparse.py`** — `bm25s` index
   - **Unicode-aware tokeniser for Devanagari.** Unit test: a known Hindi sentence yields the expected token count. Do not ship the default English stemmer on Indic text.
6. **`src/vrag/retrieval/hybrid.py`** — dense ∥ sparse via `asyncio.gather`, RRF fusion (k=60)
7. **A3:** evaluate dense-only vs sparse-only vs hybrid — 3 runs
8. **A4:** none vs FlashRank vs ONNX cross-encoder — 3 runs, using the `rerankers` library for a unified API
9. **`efSearch` sweep** — {16, 32, 64, 128, 256}; plot recall vs latency → `docs/assets/efsearch_curve.png`; pick the operating point from the curve
10. **Optional but high-value:** test Sarvam `transcribe` vs `translit` output modes for retrieval quality. Script matching between query and corpus is a real variable almost nobody will test.

### Deliverables
Ledger with ~14 more rows · `docs/EVAL_RESULTS.md` §2–4 · efSearch curve · Devanagari tokeniser test

### Exit criteria
- [ ] 4 embedders evaluated with quality **and** latency
- [ ] Hybrid beats the better of dense-only/sparse-only on Recall@5, or you've documented why it doesn't
- [ ] Rerank decision made from data, with its ms cost recorded
- [ ] efSearch chosen from the measured curve, not guessed
- [ ] Devanagari tokenisation test green
- [ ] Dense and sparse confirmed running concurrently (assert in a test that wall-clock < sum of parts)

---

# P4 — Harness Hardening
**Objective: turn a script into orchestration. This is requirement C5.**

### Tasks
1. **`harness/stage.py`** — `Stage` ABC: `name`, `min_viable_ms`, `optional: bool`, `async run(ctx) -> StageResult`
2. **`harness/pipeline.py`** — ordered stage execution over a `PipelineContext` (append-only history)
3. **`harness/budget.py` — deadline propagation.** Each stage receives `remaining_ms`; if `remaining_ms < min_viable_ms` and the stage is optional, skip it, record it in `stages_skipped`, continue.
   - **Test: force a 50ms budget, assert optional stages are skipped AND a valid response still returns.** This test is the proof of your best idea — make it explicit and name it well.
4. **`harness/retry.py`** — `tenacity` policies, exponential backoff + jitter, idempotent stages only, capped so retries can never exceed the request deadline
5. **Circuit breaker** on the LLM provider — N failures in a rolling window trips it; serve Track A only until half-open
6. **Tool calling** — expose `search_corpus(query, k)` to the generation model; **cap tool-call depth at 1**
7. **Structured output** — provider-native JSON schema or XGrammar; `AnswerResponse` schema per spec §7.2
   - **Put the reasoning field before the answer field** so the model thinks before it commits
   - **Compile/warm the schema at boot** — first-request compilation is 50–200ms and will otherwise hit your first demo user
   - On parse failure: one repair attempt → then Track A fallback. Never surface a raw exception.
8. **Track A / Track B split** — extractive span selection emits immediately; generative streams over it
9. **`telemetry/trace.py`** — `TraceRecord` per request, `perf_counter_ns()` per stage, appended to `traces.jsonl`
10. **Migrate STT to Sarvam WebSocket streaming** so transcript latency overlaps with speech

### Exit criteria
- [ ] Every stage typed with Pydantic in/out
- [ ] Forced-50ms-budget degradation test green
- [ ] Circuit breaker test green (simulate provider 500s)
- [ ] `search_corpus` tool callable by the model; depth cap enforced by test
- [ ] Schema compiled at boot; measured first-request penalty ≈ 0
- [ ] Every request writes a trace with per-stage ns timings
- [ ] Track A answer returns even when the LLM is fully unavailable

---

# P5 — Guardrails
**Objective: prove the system knows when NOT to answer. Requirement C6.**

### Tasks
1. **G1 input safety** — regex/keyword denylist + degeneracy checks, in-process, target <2ms
2. **G2 scope & language** — unsupported language or empty/nonsense query → refuse with a rephrase prompt
3. **G3 retrieval confidence gate — calibrate it, don't guess it**
   - Build a calibration set: **150 in-domain + 150 deliberately out-of-domain queries** (`eval/calibration.json`)
   - **Critical prior from the literature:** query→document cosine similarity typically runs ~0.30–0.55, *systematically lower* than query→query similarity. A threshold intuited at 0.5+ will refuse almost everything. Sweep the full range.
   - Sweep τ and the top1−top5 margin; plot false-refusal vs correct-refusal; pick the operating point
   - Chart → `docs/assets/g3_calibration.png`
4. **G4 groundedness — two tiers**
   - Hot path: citation-ID validation (every cited chunk must have actually been retrieved) **then** n-gram overlap. Deterministic check runs first — an LLM judge will pass invented citations if nobody checks the retrieval trace.
   - Offline: Bespoke-MiniCheck (claim-level, returns spans) or RAGAS faithfulness over a sample → report a real hallucination rate
5. **G5 output safety / PII redaction** — pre-emit pass
6. **`tests/test_guardrails.py`** — adversarial suite: injection attempts, unsafe content, off-topic, empty, single-word, pure noise, mixed-script
7. **Wire refusal states into the UI** so they're demoable on cue

### Exit criteria
- [ ] All five layers implemented; **each measured <10ms on the hot path**
- [ ] Calibration curve committed; τ and margin justified in `DECISIONS.md`
- [ ] False-refusal rate on in-domain queries < 10%
- [ ] Correct-refusal rate on out-of-domain queries > 80%
- [ ] All three demo refusal modes (G1, G3, G4) reproducible **on command** with named test queries written into the demo script
- [ ] Offline hallucination rate reported in `EVAL_RESULTS.md`

---

# P6 — Latency Campaign (ablation A5)
**Objective: earn the headline number honestly.**

### Tasks
1. **Build the 100-query test set** (`scripts/make_test_queries.py`): 60 in-domain / 20 off-topic / 10 unsafe / 10 degenerate
2. **Synthesise spoken versions with Sarvam TTS** → `eval/audio/`. Makes the voice benchmark reproducible instead of depending on someone talking 100 times.
3. **`scripts/bench_latency.py`**
   - 100 queries × N=5 = 500 samples; discard a warm-up pass and say so in the report
   - **Disable any embedding/response cache during benchmarking** — caching would falsify the numbers
   - Emit P50 / P70 / **P100** for `t_pipeline`, per stage, and for Track A and Track B separately
   - Report `t_e2e_voice` separately with STT broken out
4. **A5: generation provider** — 3 runs across candidates from the P0 probe
5. **Optimisation pass**, in this order:
   - ONNX + int8 the embedder (**CPU only** — on GPU use FP16; int8 on GPU is 4-5× *slower*)
   - Warm every model at boot; health check returns ready only after a dummy inference
   - Confirm zero cold starts under load
   - Consider Matryoshka dimension truncation if the embedder supports it and search is hot
   - Re-measure after each change; one change at a time
6. **`tests/test_latency_regression.py`** — CI fails if p50 > 200ms on a 20-query smoke subset
7. **Charts** → `docs/assets/latency_breakdown.png`, `latency_cdf.png`

### Exit criteria
- [ ] P50/P70/P100 reported for total **and** per stage, from real measurements
- [ ] Track A p50 comfortably < 200ms
- [ ] Track B TTFT reported honestly, whatever it is
- [ ] `docs/LATENCY_BUDGET.md` "Measured" column fully populated
- [ ] Latency regression test wired into CI and green
- [ ] Charts committed

### Note on honesty
If Track B can't clear 200ms, **say so plainly and show the layered design that does.** A team that
reports "Track A p50 = 34ms, Track B TTFT p50 = 240ms, here's the shedding logic and here's why we
chose it" is more credible than one reporting a suspiciously round 190ms with no stage breakdown.

---

# P7 — Polish, Evidence, README
**Objective: make the work legible. The README is graded.**

### Tasks
1. **Frontend**: live transcript, streaming answer, citations with source text, refusal states, **latency HUD** (you already return `timings_ms` on every response), degradation indicator showing shed stages
2. **README** per `AGENT_BUILD_SPEC.md` §10.1 — including the metric definition verbatim, the ablation ledger summary, the noise floor, and an honest limitations section
3. **Reconcile `docs/ARCHITECTURE.md`** with what was actually built, not what was planned
4. **Full manual QA**: mobile browser, slow network, mid-sentence stop, silence, background noise, very long question, rapid repeat queries
5. **Clear every stub** listed in `PROGRESS.md`
6. **Second language** — only if everything above is green. This is the first thing to cut.

### Exit criteria
- [ ] A stranger could clone, run, and reproduce your numbers from the README alone
- [ ] Zero stubs remaining in `PROGRESS.md`
- [ ] Mobile browser verified
- [ ] All charts and tables committed and linked from the README

---

# P8 — Freeze, Videos, Submit
**CODE FREEZE ~20:00 IST on freeze day — see `docs/TEAM_SPLIT.md` §5 for which calendar day that is.**

After freeze: documentation, video, verification, and submission only. No new features. No "quick fixes."

### Tasks
1. **Demo video** — scripted, in this order: Hindi question → live transcript → answer with citations + latency HUD → off-topic query refused → unsafe query refused → forced-low-budget request showing shed stages → two seconds on the eval tables
2. **Process video (90s)** — cut from footage captured across the whole week. *If you haven't been capturing since P0, start now and be honest about it in the edit rather than staging it.*
3. **Final deploy + verification** from a device on mobile data
4. **Promotion grid** — every member × both videos × Instagram/X/LinkedIn, each tagged `#RAGInGoa`, ≥1 public Instagram. Track it in `docs/SUBMISSION_CHECKLIST.md` with a name against every box.
5. **Submit the form.** Verify the GitHub repo is public and the live link works from a logged-out browser.

### Exit criteria
- [ ] Every box in `docs/SUBMISSION_CHECKLIST.md` ticked with a name and a timestamp
- [ ] Live link verified from an incognito window on a different network
- [ ] Repo public, README renders correctly on GitHub
- [ ] Form submitted

---

# Cut list — in order, if you fall behind

Decide this now, not at 11pm on Aug 21:

1. Second language (P7)
2. Proposition & contextual-retrieval chunking strategies (keep 6, drop 7–9)
3. ColBERT / late-interaction reranking experiments
4. OpenTelemetry / Phoenix observability
5. Frontend polish beyond functional
6. Late chunking (needs a different embedder — expensive to add late)

**Never cut:** the deploy (P1), the deadline-propagation test (P4), G3 calibration (P5), the latency
benchmark (P6), or the promotion posts (P8). Those are the graded core.
