# Voice RAG — Your Half of the Build (Workstream P: Pipeline & Product)

Hey — thanks for jumping on this. Here's everything you need: what we're building, what your half
is, and what to paste into Claude Code to get started. Should take you ~15 minutes to read and set up.

**Deadline: Aug 22, 23:59 IST. No resubmissions — so once we submit, that's it.** Today is Aug 17.
**We're targeting a full working pipeline by Aug 20** — a day earlier than we strictly need to,
on purpose, so Aug 21-22 are pure buffer for latency numbers, polish, videos, and submission instead
of a scramble.

---

## 1. The 30-second version

It's a hackathon shortlisting task (HH Goa 2026): build a voice-enabled RAG system. Someone speaks a
question in Hindi, we transcribe it, retrieve relevant passages from a dataset, and generate a
grounded answer — with the retrieval-and-answer part running **under 200ms**, wrapped in a real
orchestration harness, with guardrails that know when to refuse.

Six things are actually graded: STT provider choice, chunking sophistication, the 200ms latency
target, P50/P70/P100 latency reporting, a real orchestration harness (not a raw prompt call), and
guardrails. Full detail is in `AGENT_BUILD_SPEC.md` in the repo — read §1-2 there when you get a
chance, but you don't need it to start.

## 2. The split — you own the pipeline, I own retrieval

We're splitting the codebase by module so we can both build in parallel starting today, not
sequentially. I'm calling these **Workstream R** (me — chunking, embeddings, vector search, ranking)
and **Workstream P** (you — everything the request touches on its way through the system).

**You own:**
- **Speech-to-text** — Sarvam's API, wired to real microphone input over WebSocket
- **The harness** — the orchestration layer: typed pipeline stages, retries, a circuit breaker, and
  critically, **deadline propagation** (every request carries a ms budget; stages that can't fit get
  skipped rather than blowing the budget — this is genuinely the best idea in the whole architecture,
  full spec in `AGENT_BUILD_SPEC.md` §3.4)
- **Generation** — calling the LLM, streaming the response, structured JSON output
- **Guardrails** (most of them) — unsafe-input filtering, scope detection, output safety
- **Telemetry** — timing every stage so we can report real P50/P70/P100 numbers
- **The API + frontend** — FastAPI, WebSocket, the actual UI people talk to. There's a styled
  prototype already built (`frontend/reference/voice-rag-ui-preview.html`) — open it in a browser,
  click the mic, it cycles through idle/listening/working/answered/refused states. It's on-brand
  with the actual HH Goa site (same palette, same hanging-tag motif for pipeline stages, same
  dashed-ticket treatment on the answer moment). Build the real thing against it — it's not final,
  but it's a real starting point, not a mood board.
- **Deployment** — getting this live on a public HTTPS URL (mic access requires HTTPS)

**I own:** chunking strategies, embeddings, the vector index, hybrid search, reranking — basically,
everything that decides *what* gets retrieved. You never need to touch that code.

## 3. The one thing to read before writing anything: the contract

Here's the function your harness calls to get retrieved context. I'm building the real
implementation; you build against this signature from hour one, using a stub that returns fake data:

```python
# src/vrag/retrieval/interface.py

class RetrievedChunk(BaseModel):
    chunk_id: str
    passage_id: str
    text: str
    score: float          # 0-1, fused/reranked relevance
    language: str

async def retrieve(query: str, k: int = 5) -> list[RetrievedChunk]:
    """Never raises. Returns [] on internal failure — your guardrail layer should
    treat an empty list as 'nothing relevant found' and route to abstention."""
    ...
```

**Build a stub of this today** that returns 2-3 fake `RetrievedChunk` objects with made-up text, so
your entire pipeline is testable end to end immediately, without waiting on my retrieval code. On
Day 2 (Aug 19), we swap your stub for my real implementation — should be a one-line import change if
we both stick to the signature above.

If you think this signature should be different (extra fields, different return type, whatever) —
say so *today*, before either of us builds much against it. Changing it later is expensive; changing
it now costs nothing.

## 4. Setup

1. **Clone the repo:** `git clone <repo-url>` (I'll send the URL / add you as a collaborator — check
   your GitHub notifications for the invite).
2. **Create your branch:** `git checkout -b workstream-p`
3. **Read these in order (~12 min total):** `CLAUDE.md` → `AGENT_BUILD_SPEC.md` (skim §0-3, they
   matter most) → `docs/TECH_MENU.md` §S2-S3, S10-S15 (your stages) → `docs/TEAM_SPLIT.md` (the full
   split + our sync schedule) → open `frontend/reference/voice-rag-ui-preview.html` in a browser and
   click around it.
4. **Open Claude Code in the repo root** and paste the kickoff prompt below.

## 5. Kickoff prompt — paste this into Claude Code

```
Read CLAUDE.md, AGENT_BUILD_SPEC.md, docs/TECH_MENU.md, and docs/TEAM_SPLIT.md fully.

Create a file named .workstream at the repo root containing exactly the single
character P (no other text). This is gitignored and local to my machine only — it's
how you'll know which half of the project you're allowed to touch in every future
session, without me having to repeat this every time.

I'm Workstream P (Pipeline & Product) on this project — I own STT, the harness, generation,
most of the guardrails, telemetry, the API, frontend, and deployment. Workstream R (my
teammate) owns everything upstream of the retrieve() function in
src/vrag/retrieval/interface.py — don't touch their modules (chunking/, index/, retrieval
internals beyond the interface file).

Bootstrap docs/ per the "First session only" section in CLAUDE.md if it doesn't exist yet.
Create docs/PROGRESS_P.md and docs/DECISIONS_P.md as MY files — never edit
docs/PROGRESS_R.md, docs/DECISIONS_R.md, or anything under docs/SESSION_LOG/track-r/.

Today's goal (Day 1 of the compressed 5-day plan in docs/TEAM_SPLIT.md): 
1. Write a stub implementation of retrieve() in src/vrag/retrieval/interface.py that returns
   2-3 fake RetrievedChunk objects, so I have something to build the whole pipeline against
   immediately.
2. Get real Sarvam STT working end to end: browser mic -> WebSocket -> Sarvam -> transcript.
   No mocking the STT path, ever, even during early development.
3. Scaffold the harness skeleton and FastAPI app per AGENT_BUILD_SPEC.md §5, §7.2.
4. Get an ugly version deployed to a public HTTPS URL today, even with the stubbed retrieval.
   This is insurance against last-minute deploy problems.

Teaching mode is on per CLAUDE.md - narrate what you're doing and why in plain language as
you go, I want to actually learn this, not just watch it get built.
```

## 6. Timeline — when we sync

| Day | Date | You're doing | We sync |
|-----|------|--------------|---------|
| 0 | Aug 17 (tonight) | — | **30-45 min call, live** — lock the `retrieve()` contract, confirm GitHub access, both branches created, both `.workstream` files made |
| 1 | Aug 18 | Harness skeleton + real STT against the stub; ugly deploy live | async only |
| 2 | Aug 19 | Harness hardening (deadline propagation, retries, structured output), guardrails G1/G2/G5 | **end-of-day sync — merge my real `retrieve()` into your harness, replace the stub, watch it run end to end together** |
| 3 | **Aug 20 — target: COMPLETE** | Generation provider comparison, guardrail confidence calibration (joint — needs my retrieval scores), Track A/B answer-path split wired in | **end-of-day sync — full integration to `main`, adversarial guardrail tests run together, checklist below confirmed** |
| 4 | Aug 21 — no new features | README + frontend polish + video footage prep | **joint — sequential latency benchmark** (one machine, one config, isolated — see note below), finalize G3 calibration curve, code freeze ~20:00 IST |
| 5 | Aug 22 | Videos, promotion, submit | joint |

**What "complete" means at the Aug 20 sync** — check together before calling it done:
- Real voice → real STT → real retrieval → real generation → real guardrails, wired end to end, nothing mocked
- All 6 chunking strategies evaluated, winner shipped · hybrid retrieval + rerank decision made
- Harness: deadline propagation, retries, structured output all working
- All 5 guardrail layers functionally present and demoable on command (the calibration *curve* can
  still be rough — that's explicitly Day 4's job, not Day 3's)
- Deployed and reachable on a public HTTPS URL

Final P50/P70/P100 numbers, a polished README, and finished videos are **not** expected by Aug 20 —
those are what Day 4 and Day 5 are for. Don't let them creep into Day 3 and squeeze out the
integration work above.

**One technical note that matters:** when we run the final latency benchmark on Day 4, it has to run
on **one machine, one config, nothing else running** — not split across our two laptops
simultaneously, and not run alongside anything else. CPU contention from parallel runs would give us
fake (worse) numbers. Full reasoning in `docs/PARALLEL_EXECUTION.md` §0 if you want it — short
version: we parallelize while testing *quality* (recall, accuracy), never while measuring *latency*.

## 7. What's on you personally, separate from the code

The submission has a promotion requirement that's per-person, not per-team:

- Both videos (90s process video + demo video), posted by **you individually** on **Instagram, X,
  and LinkedIn** — a shared team post from me doesn't count for you
- Every post, every platform, includes `#RAGInGoa`
- At least one of our Instagram accounts needs to be public (check now if yours already is)

Since there's no resubmission, this is worth doing early rather than at 11pm on the 22nd.

## 8. Questions

If anything above is ambiguous or you'd split it differently, say so before you start building —
changing the plan on Day 0 costs nothing, changing it on Day 3 costs a day we don't have. Otherwise,
see you at the Day 0 sync tonight.
