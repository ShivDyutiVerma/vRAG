# CLAUDE.md

> Auto-loaded context for every Claude Code session in this repo.
> Keep this file short. Depth lives in `AGENT_BUILD_SPEC.md`, `docs/BUILD_PLAN.md`, `docs/TECH_MENU.md`.

## Project

`vrag` — voice-enabled RAG over `ai4bharat/MSMARCO-XI`.
Speak a question → Sarvam STT → hybrid retrieval → grounded, cited answer, **server-side pipeline under 200ms**.

Hackathon submission. **Deadline 2026-08-22 23:59 IST. No resubmissions. Code freeze ~Aug 21 20:00 IST (see `docs/TEAM_SPLIT.md` §5 for the exact schedule).**

**⚠️ Two people, two Claude Code sessions, working in parallel.** Before doing anything else, check
`docs/TEAM_SPLIT.md` to find out which workstream you're operating for — **Workstream R** (Retrieval
& Ranking: chunking, embeddings, index, hybrid search, reranking) or **Workstream P** (Pipeline &
Product: STT, harness, generation, guardrails, telemetry, API, frontend, deployment). Stay inside
your workstream's module ownership; the seam between them is the single `retrieve()` function in
`src/vrag/retrieval/interface.py`, agreed jointly and not to be changed unilaterally. Also read
`docs/PARALLEL_EXECUTION.md` before running any ablation stage — quality comparisons parallelize
aggressively (subagents + tmux), the final latency numbers never do (strictly sequential, isolated
machine — see that file §0 for why).

**Note on dates:** `docs/BUILD_PLAN.md`'s calendar dates assume a solo Aug-14 start and are stale —
`docs/TEAM_SPLIT.md` has the corrected 5-day, two-person schedule from Aug 17. The phase *content*
in `BUILD_PLAN.md` (tasks, exit criteria) is unaffected and still the source of truth for *what* to build.

---

## 🎓 Teaching mode — ON by default

The user is building this to **learn**, not just to ship. Narrate your work in plain language as you go.

**As you work, say out loud:**
- **What** you're about to do, in one sentence, before you do it
- **Why** this approach over the alternative you rejected
- **What just happened** when a command runs — especially when the output is cryptic
- **The concept behind it** when you use something non-obvious (HNSW, RRF, deadline propagation, ONNX quantisation), in 2–3 sentences of plain English, with a concrete analogy where it helps

**Calibration:**
- Assume solid general programming ability and real Python fluency; do NOT assume prior knowledge of RAG internals, vector search, or async orchestration
- Explain jargon on first use in each session — sessions don't share memory, and neither does the reader
- Prefer "this is the part that makes retrieval fast, here's the intuition" over "implementing HNSW index"
- When you hit a bug, narrate the diagnosis, not just the fix. The debugging reasoning is the most valuable thing to watch.

**Don't overdo it:** one short explanation per meaningful step, not a paragraph per line. If the user
says "less explanation" or "just build," respect that for the rest of the session — but keep flagging
genuinely surprising findings and anything that changes a decision.

**At the end of every session, add a "What I learned" section to the session log** — 3–5 bullets in
plain language covering concepts encountered, not tasks completed.

---

## Step 0 — before ANYTHING else, every single session: which workstream is this?

Both collaborators clone this exact same repo with this exact same `CLAUDE.md`. Nothing in the
shared docs says which person you're talking to right now — so check, every time, before touching
any code:

1. Look for a file named `.workstream` at the repo root (it's gitignored — local to this machine only).
2. **If it exists:** read it. It contains exactly one letter, `R` or `P`. For the rest of this
   session, you may only work inside that workstream's ownership — see the table in
   `docs/TEAM_SPLIT.md` §2. If a task would touch a file outside that ownership (e.g. you're `R`
   and asked to edit anything under `src/vrag/harness/`), **stop and say so** rather than doing it —
   that's the other person's module, even if the request sounds reasonable in isolation.
3. **If it does NOT exist:** do not guess from context, do not infer it from what files look
   recently edited, do not assume based on the conversation so far. Ask directly: *"Which
   workstream are you working as — R (Retrieval & Ranking) or P (Pipeline & Product)?"* Once
   answered, create `.workstream` at the repo root containing just that letter (`R` or `P`, nothing
   else), then proceed.

This is the actual fix for a real failure mode: two people, one repo, one shared instruction file —
without this file, Claude Code has no way to know which half of the project it's currently allowed
to touch, across sessions, days, or which laptop it's running on.

## Start-of-session ritual — every time, in order

1. Read `AGENT_BUILD_SPEC.md` (§0 + the current phase)
2. Read `docs/PROGRESS.md` ← **where we actually are**
3. Read `docs/DECISIONS.md` ← settled calls; never silently reverse one
4. Read the current phase in `docs/BUILD_PLAN.md` — including its **exit criteria**
5. Run `pytest`. **Red suite ⇒ fixing it is task #1**, before any new work.
6. State back to the user: current phase · this session's goal · the exit criteria you're targeting · anything you think should be cut

## End-of-session ritual

1. Update `docs/PROGRESS.md` — even if the session went badly
2. Append `docs/SESSION_LOG/YYYY-MM-DD-session-NN.md` (include "What I learned")
3. Append new ADRs to `docs/DECISIONS.md`
4. Append any experiment runs to `eval/ablation_ledger.csv`
5. Update `docs/LATENCY_BUDGET.md` if anything was measured
6. Commit as `[P<phase>] <what changed>`; push

---

## First session only — bootstrap `docs/`

Create every file below, populated from the templates in `AGENT_BUILD_SPEC.md` §8.3. **Do not leave
placeholders** — a doc that says "TODO" is worse than no doc, because future-you will trust it.

```
docs/
├── PRD.md                    problem · users · FR-1..N mapped to constraints C1–C7 · acceptance criteria · non-goals
├── ARCHITECTURE.md           system design · request lifecycle · module boundaries · deploy runbook
├── BUILD_PLAN.md             the 8 phases, tasks, exit criteria   (provided — copy it in; dates are stale, see TEAM_SPLIT.md)
├── TECH_MENU.md              candidate methods per stage + ablation design   (provided — copy it in)
├── TEAM_SPLIT.md             ⭐ who builds what, the retrieve() contract, the real schedule   (provided — copy it in)
├── PARALLEL_EXECUTION.md     ⭐ subagents/tmux for ablation runs; why latency stays sequential   (provided — copy it in)
├── PROGRESS.md               SHARED — edited only at integration syncs, by whoever's merging
├── PROGRESS_R.md             Workstream R's running status — edit freely, any time, never edited by P
├── PROGRESS_P.md             Workstream P's running status — edit freely, any time, never edited by R
├── DECISIONS.md              SHARED ADR log — same rule: only touched at syncs, merging both logs below
├── DECISIONS_R.md            Workstream R's ADRs, numbered R-001, R-002...
├── DECISIONS_P.md            Workstream P's ADRs, numbered P-001, P-002...
├── CONVENTIONS.md            code style · error handling · timing units · hot-path rules
├── API_CONTRACTS.md          stage interfaces · HTTP/WS schemas · AnswerResponse
├── EVAL_PROTOCOL.md          exactly how chunking, guardrails, and latency are measured
├── EVAL_RESULTS.md           ⭐ generated tables + charts — a graded artifact (§1-3 = R's to write, §4-6 = P's)
├── LATENCY_BUDGET.md         the ms allocation table, target vs measured
├── RISKS.md                  live risk register with owners
├── GLOSSARY.md               ⭐ every term explained plainly — grows as teaching mode encounters concepts
├── SUBMISSION_CHECKLIST.md   ⭐ incl. the per-member × per-platform promotion grid
├── SESSION_LOG/
│   ├── track-r/               Workstream R's session logs
│   └── track-p/               Workstream P's session logs
└── assets/                   charts, diagrams, screenshots
```

Also create `eval/ablation_ledger.csv` with the header from `docs/TECH_MENU.md` §C, and
`eval/manifests/` per `docs/PARALLEL_EXECUTION.md` §2 once ablation work starts.

**`GLOSSARY.md` is not optional.** Every time teaching mode explains a term, append it. By P8 it's a
real learning artifact and it makes the README much easier to write. It's shared — either workstream
appends to it, no split needed, term explanations don't collide.

**Never edit the other workstream's `PROGRESS_*.md`, `DECISIONS_*.md`, or `SESSION_LOG/track-*/`
files.** Read them freely; write only your own. The shared `PROGRESS.md`/`DECISIONS.md` get touched
only at the integration syncs listed in `docs/TEAM_SPLIT.md` §5, by whoever is doing that merge.

---

## Hard rules

- **Never mock the STT path** in committed code. Real Sarvam, real audio, always.
- **Never report a latency number** not produced by `scripts/bench_latency.py`.
- **Never put a network call on the hot path** except the LLM. Embeddings, BM25, reranker, and all guardrails run in-process.
- **Never use a hosted vector DB.** Remote vector search costs 50–300ms round trip and eats the entire budget. FAISS in-process, always.
- **Never change two variables in one experiment run.** That run is void.
- **Never add a dependency** without an ADR.
- **Never leave a stub** unlisted in `docs/PROGRESS.md`.
- **Never start a new phase** with failing tests or unmet exit criteria.
- **Never commit secrets.** `.env` gitignored from commit #1.
- **Never enable caching during a latency benchmark.** It falsifies the number.
- **Never run more than one config at a time during the latency pass.** Quality/recall comparisons
  parallelize across subagents and tmux panes freely; the P50/P70/P100 numbers never do — CPU
  contention from a parallel run invalidates them. See `docs/PARALLEL_EXECUTION.md` §0.
- **Never edit the other workstream's `PROGRESS_*.md`, `DECISIONS_*.md`, or module files.** See the
  ownership table in `docs/TEAM_SPLIT.md` §2 — and check `.workstream` (Step 0, above) before every
  session so "the other workstream" is never a guess.

## Hot-path invariants

- Request-path functions carry `# HOTPATH` — no network, no disk, no cold starts
- All timing uses `time.perf_counter_ns()`; convert to ms only at serialisation
- E5 embeddings **require** `"query: "` / `"passage: "` prefixes — there is a test; do not remove it
- Dense and sparse retrieval run concurrently via `asyncio.gather`
- Models **and** the JSON schema are compiled/warmed at boot (schema compilation is 50–200ms on first use, then cached)
- In the structured-output schema, the reasoning field comes **before** the answer field
- ONNX int8 is for **CPU only** — on GPU it is slower than FP32; use FP16 there
- Every request emits a `TraceRecord` with per-stage ns timings

---

## Layout

```
src/vrag/
  harness/     pipeline · stage · budget (deadline propagation) · retry · trace
  chunking/    base protocol + 6 strategies + registry
  index/       embedder (ONNX) · dense (FAISS HNSW) · sparse (bm25s) · fusion (RRF)
  retrieval/   dense ∥ sparse orchestration
  generation/  Track A extractive · Track B streaming LLM + tools
  guardrails/  G1 input · G2 scope · G3 confidence · G4 grounded · G5 output
  telemetry/   trace records → traces.jsonl
  api/         FastAPI routes + WebSocket
scripts/       probe_latency · inspect_dataset · build_index · eval_chunking · bench_latency · make_test_queries
eval/          heldout_queries.json · calibration.json · test_queries.json · audio/ · ablation_ledger.csv
docs/          see bootstrap above
```

## Commands

```bash
make dev          # API + frontend locally
make probe        # provider RTT/TTFT probe (P0)
make index        # build FAISS + BM25 indexes offline
make eval-chunk   # chunking ablation → EVAL_RESULTS.md + ledger
make eval-guard   # guardrail calibration sweep → g3_calibration.png
make bench        # latency benchmark → P50/P70/P100 + charts
make test         # pytest incl. latency regression
```

---

## Experiment discipline

Any run that touches quality or latency **must** append a row to `eval/ablation_ledger.csv` with the
full config. Follow the staged ablation in `docs/TECH_MENU.md` §A — **do not attempt a full grid
search**, it's 648 configs and you have 5 days between two people.

For running the ablation itself — subagents to implement candidates in parallel, tmux/process pools
to execute quality evals in parallel, and why the latency pass stays strictly sequential — see
`docs/PARALLEL_EXECUTION.md` in full before starting any stage.

Before declaring any winner: run the same config 3× and report the spread. **A gap smaller than the
noise floor is not a result** — say the options were tied and ship the cheaper one.

---

## When blocked

Stop. Write the ambiguity into `docs/RISKS.md`. Ask the user — and explain the tradeoff in plain
language so they can make the call themselves.

Never guess on anything touching the six graded requirements (C1–C6 in `AGENT_BUILD_SPEC.md` §2).
