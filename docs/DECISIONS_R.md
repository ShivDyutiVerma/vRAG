# Architecture Decision Record — Workstream R

> Mine only. Never edited by Workstream P. Numbered `R-001`, `R-002`, ... Append-only.

## R-001 — Dev environment: system Python 3.13.7, no uv/poetry installed

**Date:** 2026-08-17
**Status:** Accepted
**Context:** `AGENT_BUILD_SPEC.md` §5.2 specifies "Python 3.11+ (3.11 for perf)" but doesn't mandate an
exact minor version. This machine has Python 3.13.7 as the only interpreter, no `uv` or `poetry`
installed.
**Decision:** Build R's `.venv` against system Python 3.13.7 using stdlib `venv` + `pip`, rather than
installing `uv`/`poetry` or downgrading to 3.11.
**Rationale:** 3.13 satisfies "3.11+." Adding a new tool (`uv`/`poetry`) is itself a dependency change
that would need its own ADR and setup time neither of us has spare today; `pip`+`venv` is zero-install
and sufficient for a 5-day project. Verified before committing: `pip install --dry-run` resolved real
cp313-win_amd64 wheels for every planned dependency (faiss-cpu, torch, onnxruntime, transformers,
sentence-transformers, bm25s, datasets) — not assumed.
**Consequences:** If any dependency later turns out to lack a 3.13 wheel and requires a source build
that's too slow/flaky, this ADR gets superseded by one pinning a 3.11/3.12 venv instead — cheap to
reverse, so not over-thought now.

## R-002 — Held-out eval set: heldout queries drawn FROM the indexed pool, not excluded from it

**Date:** 2026-08-17
**Status:** Accepted
**Context:** `docs/BUILD_PLAN.md` P0 task 4 says to freeze `eval/heldout_queries.json` as "500
query→passage pairs, excluded from indexing." Read literally, that would make Recall@k undefined —
if a query's gold passage is never in the search index, it can never be retrieved, so Recall@1/@5/@10
would be zero by construction, not a real quality signal. This contradicts `AGENT_BUILD_SPEC.md`
§7.1's eval protocol, which reports exactly those metrics on the held-out set.
**Decision:** Interpret "excluded" as *excluded from informing chunking-strategy design decisions*
(no eyeballing heldout queries while picking hyperparameters), not *excluded from the physical
index*. Mechanism: draw a working pool of ~20,500 Hindi rows from the corpus; randomly select 500
(fixed seed) as the frozen heldout eval set; index passages from the **full** pool (all 20,500 rows,
heldout included) so every heldout query's gold passage is a valid retrieval candidate.
**Rationale:** This is standard IR practice (a held-out *query* split, not a held-out *document*
split) and is the only reading under which the required metrics are computable at all. Flagged here
rather than silently assumed, since it's a real ambiguity in the spec — not one of the C1-C7 graded
constraints, so proceeding without asking rather than blocking, but documenting the call so it's
easy to revisit if the interpretation turns out wrong.
**Consequences:** `scripts/build_dataset_subset.py` builds both files from one pass over the same
pool. If this interpretation is wrong, both files need regenerating — cheap, since nothing downstream
has been built against the wrong shape yet.
