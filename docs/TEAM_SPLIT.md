# TEAM_SPLIT.md
## Two-person parallel build plan — authoritative schedule from Aug 17

> Lives at `docs/TEAM_SPLIT.md`. This supersedes the calendar dates in `docs/BUILD_PLAN.md` (which
> assumed one solo builder starting Aug 14). The phase **content** in `BUILD_PLAN.md` is unchanged
> and still correct — this file says who does which phase content, and when, on the real 5-day clock.

**Today:** Aug 17 · **Internal completion target: Aug 20** · **Actual deadline: Aug 22, 23:59 IST** · **No resubmissions.**

> **Naming note — read this once:** this file uses **Workstream R** (Retrieval & Ranking) and
> **Workstream P** (Pipeline & Product) for the two *people*. This is unrelated to "Track A / Track B"
> in `AGENT_BUILD_SPEC.md` §3.3, which names the extractive-vs-generative *answer paths* inside the
> running system. Same letters, different concept, deliberately renamed here to stop the confusion
> before it starts.

---

## §1. Why a contract-first split, not a phase-first split

The original `BUILD_PLAN.md` was written for one person moving through phases P0→P8 in sequence.
With two people, splitting by *time* (you do days 1-4, he does days 5-8) wastes half your available
parallelism. Splitting by *module* lets both of you work from hour one — but only if you agree on
the **seam** between your modules before either of you writes real code against it.

The seam is one function. Everything else follows from it.

```python
# src/vrag/retrieval/interface.py — AGREE THIS FIRST, TOGETHER, BEFORE ANYTHING ELSE

class RetrievedChunk(BaseModel):
    chunk_id: str
    passage_id: str
    text: str
    score: float          # fused/reranked relevance score, 0-1
    language: str          # ISO code detected/tagged at index time

async def retrieve(query: str, k: int = 5) -> list[RetrievedChunk]:
    """The one function Workstream P's harness calls. Workstream R owns the implementation.
    Must be safe to call concurrently. Must not raise — on internal failure, return []
    and let the harness's grounding gate handle the empty-result case."""
    ...
```

Once this is written down and both of you have read it, you can build in parallel with almost no
conflict risk: Workstream P builds the entire harness against a **stub** implementation of `retrieve()`
that returns canned data; Workstream R builds the **real** implementation in total isolation. Merging
later is swapping an import, not resolving a diff.

**Do this together, live, in the first 30 minutes — a call or a shared doc, not async.** It's the
highest-leverage half hour of the whole project.

### Your own kickoff prompt (Workstream R)

Both of you clone the identical repo with identical instructions — nothing in the shared files says
"you specifically are R." Paste this as your first message so Claude Code knows, and keeps knowing
across every future session (see `CLAUDE.md` "Step 0" for the mechanism):

```
Read CLAUDE.md, AGENT_BUILD_SPEC.md, docs/TECH_MENU.md, and docs/TEAM_SPLIT.md fully.

Create a file named .workstream at the repo root containing exactly the single
character R (no other text). This is gitignored and local to my machine only.

I'm Workstream R (Retrieval & Ranking) on this project — I own chunking, embeddings,
the dense and sparse index, hybrid retrieval, and reranking, all behind the single
retrieve() function in src/vrag/retrieval/interface.py. Workstream P (my collaborator,
on their own machine) owns everything downstream of that function — don't touch
src/vrag/harness/, src/vrag/stt/, src/vrag/generation/, src/vrag/guardrails/,
src/vrag/api/, or frontend/.

Bootstrap docs/ per the "First session only" section in CLAUDE.md if it doesn't exist
yet. Create docs/PROGRESS_R.md and docs/DECISIONS_R.md as MY files — never edit
docs/PROGRESS_P.md, docs/DECISIONS_P.md, or anything under docs/SESSION_LOG/track-p/.

Today's goal (Day 0/1 of the compressed 5-day plan in this file): probe provider
latency, pull the dataset subset, agree the retrieve() contract with my collaborator,
then start on the chunking ablation (A1) per docs/TECH_MENU.md.

Teaching mode is on per CLAUDE.md — narrate what you're doing and why in plain
language as you go.
```

---

## §2. Module ownership

| Area | Owner | Files | Ablation stages |
|------|-------|-------|-----------------|
| Chunking | **Workstream R** (you) | `src/vrag/chunking/` | A1 |
| Embedding | **Workstream R** | `src/vrag/index/embedder.py` | A2 |
| Dense + sparse index | **Workstream R** | `src/vrag/index/{dense,sparse}.py` | — |
| Hybrid retrieval + fusion | **Workstream R** | `src/vrag/retrieval/` | A3 |
| Reranking | **Workstream R** | `src/vrag/retrieval/rerank.py` | A4 |
| Offline retrieval eval | **Workstream R** | `scripts/eval_chunking.py`, `eval/heldout_queries.json` | — |
| **`retrieve()` interface** | **JOINT — agree first** | `src/vrag/retrieval/interface.py` | — |
| STT | **Workstream P** (friend) | `src/vrag/stt/` | — |
| Harness / orchestration | **Workstream P** | `src/vrag/harness/` | — |
| Deadline propagation | **Workstream P** | `src/vrag/harness/budget.py` | — |
| Generation (Track A/B split) | **Workstream P** | `src/vrag/generation/` | A5 |
| Structured output | **Workstream P** | `src/vrag/schemas.py` | — |
| Guardrails G1/G2/G5 | **Workstream P** | `src/vrag/guardrails/` | — |
| Guardrails G3/G4 (needs retrieval scores) | **JOINT — calibration needs both** | `src/vrag/guardrails/` | — |
| Telemetry / tracing | **Workstream P** | `src/vrag/telemetry/` | — |
| FastAPI + WebSocket | **Workstream P** | `src/vrag/api/` | — |
| Frontend | **Workstream P** | `frontend/` — build against `frontend/reference/voice-rag-ui-preview.html` (the styled prototype: idle/listening/working/answered/refused states, on-brand with the HH Goa site) | — |
| Deployment | **Workstream P** (owns it; both verify) | infra/deploy config | — |
| Latency benchmark (sequential, §4 of `PARALLEL_EXECUTION.md`) | **JOINT — run on whichever machine is quietest at the time** | `scripts/bench_latency.py` | — |
| README / architecture writeup | **JOINT — Workstream P assembles, Workstream R contributes §3-4** | `README.md`, `docs/ARCHITECTURE.md` | — |
| Videos + promotion | **JOINT — both required, see §7** | — | — |

If you disagree with this split (e.g. your friend actually wants the retrieval side, or has more
STT/infra experience than you), swap it — the module boundaries matter more than who's on which side.

---

## §3. Docs conflict-avoidance — split what would collide, share what won't

Two people editing the same growing file from two branches is where hackathons lose hours to merge
conflicts on files that don't even matter. Split anything append-heavy; share anything edited rarely.

```
docs/
├── DECISIONS.md              SHARED — but only edited at integration syncs (§5), by whoever merges
├── DECISIONS_R.md            Workstream R's ADRs, numbered R-001, R-002...  never touched by P
├── DECISIONS_P.md            Workstream P's ADRs, numbered P-001, P-002...  never touched by R
├── PROGRESS.md               SHARED — same rule: edited only at syncs
├── PROGRESS_R.md             Workstream R's running status — edit freely, any time
├── PROGRESS_P.md             Workstream P's running status — edit freely, any time
├── SESSION_LOG/
│   ├── track-r/              Workstream R's session logs
│   └── track-p/              Workstream P's session logs
├── EVAL_RESULTS.md           SHARED, but §1-3 (chunking/embed/retrieval) is R's to write,
│                              §4-6 (generation/guardrails/latency) is P's to write — different
│                              sections of one file rarely conflict if you stay in your lane
└── (everything else: AGENT_BUILD_SPEC.md, TECH_MENU.md, BUILD_PLAN.md, TEAM_SPLIT.md,
     PARALLEL_EXECUTION.md — read-only reference, nobody edits these day-to-day)
```

At each integration sync, whoever is doing the merge reads both `PROGRESS_R.md` and `PROGRESS_P.md`
and hand-writes a short combined summary into the shared `docs/PROGRESS.md`. Same pattern for
`DECISIONS.md` — concatenate both logs, sort by date, done. This is 5 minutes of manual work that
prevents every other kind of docs merge conflict.

---

## §4. Git workflow

```bash
# Repo owner (you) creates the repo, then:
git checkout -b workstream-r
# friend, after cloning:
git checkout -b workstream-p
```

- `main` only receives **merged, integrated, working** code — it should be deployable at every point in the timeline, never left broken overnight.
- Each of you pushes your branch freely, as often as you like — it's your sandbox.
- Integration happens via **pull request into `main`**, at the scheduled sync points in §5. PRs here
  are for visibility and a sanity check, not gatekeeping — you have 5 days, don't build ceremony.
- Whoever's not merging does a 5-minute read of the diff before it lands, mainly to catch "wait, that
  changes the interface contract" — which is the one thing worth blocking on.
- **If the interface contract (`retrieve()`) ever needs to change after Day 0**, that's a joint
  decision, recorded as a new ADR in the shared `DECISIONS.md` immediately, not discovered at a merge.

### GitHub setup (do this in the first 15 minutes)

1. Repo already exists / create it: **must be public** (submission requirement).
2. Add your friend as a collaborator:
   - **UI (simplest):** repo → Settings → Collaborators and teams → Add people → enter his GitHub
     username or email → he accepts the emailed invite.
   - **CLI, if you prefer the terminal:**
     ```bash
     gh api --method PUT -H "Accept: application/vnd.github+json" \
       /repos/<your-username>/<repo>/collaborators/<his-username> -f permission='push'
     ```
3. Both of you `git clone` the repo locally, each in your own environment.
4. Both of you drop `CLAUDE.md`, `AGENT_BUILD_SPEC.md`, `docs/BUILD_PLAN.md`, `docs/TECH_MENU.md`,
   `docs/TEAM_SPLIT.md`, `docs/PARALLEL_EXECUTION.md` into the repo root/`docs/` **before** either
   Claude Code session starts — both of you need the same shared context on disk from the first prompt.
5. Branch protection / required reviews: **skip it.** Five days, two people, not worth the friction.

---

## §5. Integration schedule

> **Target: full pipeline complete and integrated by end of Aug 20** (Day 3) — not the Aug 22
> deadline. That's a deliberate extra day of buffer on top of what's needed, given "no
> resubmissions." Aug 21 is dedicated entirely to latency measurement, guardrail calibration, and
> the README — not new features. Aug 22 is submission only.

| When | Sync | What happens |
|------|------|--------------|
| **Day 0 — Aug 17 (kickoff, this evening)** | Joint, live, 30-45 min | Agree the `retrieve()` contract · agree Sarvam/tech choices already decided in `TECH_MENU.md` · GitHub set up, both cloned · both branches created · both `.workstream` files created |
| **Day 1 — Aug 18** | Async only | R: implement + parallel-eval all 6 chunking strategies (A1). P: build harness skeleton + STT hello-world against the **stub** `retrieve()`; get an ugly version **deployed live today** — insurance deploy, per the original P1 principle, now owned by P |
| **Day 2 — Aug 19** | **Sync #1, end of day** | R: finish A2 (embedder) + A3 (hybrid retrieval), real `retrieve()` stabilising. P: harness hardening (deadline propagation, retries, structured output), guardrails G1/G2/G5. **At sync: merge R's real `retrieve()` into P's harness, replacing the stub. First true end-to-end test — do this together, watch it run.** |
| **Day 3 — Aug 20** | **Sync #2, end of day — the "COMPLETE" milestone** | R: finish A4 (rerank) + support G3/G4 calibration (needs retrieval scores). P: A5 (generation provider), guardrail G3/G4 calibration (joint), Track A/B answer-path split wired in. **At sync: full integration — merge everything to `main`, run the adversarial guardrail test suite together, confirm the checklist below.** |
| **Day 4 — Aug 21** | Joint, no new features | **Sequential latency campaign only** (§4 of `PARALLEL_EXECUTION.md` — one machine, one config at a time, whichever laptop is quietest) · finalize the G3 calibration curve · README assembly (P drafts, R contributes retrieval sections) · frontend polish · start assembling video footage · **CODE FREEZE end of day (~20:00 IST)** |
| **Day 5 — Aug 22** | Joint | Record/finish videos, final live-link verification from a phone on mobile data, promotion posts (both of you, both videos, three platforms each, `#RAGInGoa`), submit form before 23:59 |

### The Day 3 / Aug 20 "complete" checklist

This is what "done" means at the Sync #2 milestone — check every box together before calling it complete:

- [ ] Real voice → real Sarvam STT → real retrieval → real generation → real guardrails, wired end to end, no mocked stages anywhere in the path
- [ ] All 6 chunking strategies evaluated; production strategy chosen and shipped
- [ ] Hybrid retrieval live; rerank decision made (even if "none" was the answer, from data)
- [ ] Harness: deadline propagation, retries, and structured output all working
- [ ] All 5 guardrail layers functionally present and each demoable on command (calibration curve can still be rough — that's Day 4's job, not Day 3's)
- [ ] Deployed and reachable on a public HTTPS URL

**Explicitly NOT required by Aug 20:** final P50/P70/P100 numbers, a polished README, or finished
videos. Those are Day 4 and Day 5's entire job — don't let them creep into Day 3 and squeeze out the
integration work above, and don't let "let's also just quickly measure latency today" turn Day 3 into
a parallel-contaminated latency run either (§0 of `PARALLEL_EXECUTION.md`).

**If either workstream is visibly behind at Sync #1 (end of Day 2):** cut scope immediately using the
cut list in `docs/BUILD_PLAN.md` — don't wait for Sync #2 to notice. With the completion target now
three build days instead of four, there's even less slack to discover a problem late.

**Resilience built into this split:** because P builds against a stub from hour one, the project has
*something* demoable and deployed from Day 1 onward regardless of how R's ablation work goes. If R's
retrieval work runs long, P's harness, guardrails, and frontend are still fully testable and mergeable
on schedule — worst case, ship with a slightly less-optimised retrieval config, not a broken pipeline.

---

## §6. Daily async check-in (keep it to 5 minutes)

Neither of you needs a call every day — but drop one message a day (Slack/WhatsApp/whatever) with:
1. What's actually working right now (not "80% done" — a fact: "retrieve() returns real results for Hindi queries, Recall@5 = 0.81")
2. What's blocked
3. Anything that might affect the other person's work

This is exactly what `PROGRESS_R.md` / `PROGRESS_P.md` already capture — just paste the top of yours
into chat.

---

## §7. Submission logistics that are joint regardless of the split

Both of you, individually, on Instagram + X + LinkedIn, both videos, `#RAGInGoa` on every post, at
least one public Instagram between you. Neither workstream "owns" this — it's per-person by
requirement, not per-module. See `FRIEND_BRIEF.md` §5 for his copy of this checklist, and
`docs/SUBMISSION_CHECKLIST.md` for the shared tracking grid.
