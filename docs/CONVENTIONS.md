# CONVENTIONS.md

## Code style

- Type hints mandatory on all public functions; `mypy` clean before any commit that touches `src/`.
- `ruff` for lint + format; configured in `pyproject.toml`, run in pre-commit and CI.
- No bare `except:` — every caught exception is logged with `trace_id` where one exists.
- No secrets in code. `.env` + `pydantic-settings`; `.env.example` committed with every key name and
  a placeholder value, never a real one.
- New dependency ⇒ new ADR in your own `DECISIONS_{R,P}.md` (hard rule, `CLAUDE.md`).
- Every new module ⇒ a test file. A module with no test file is a stub, and stubs must be listed in
  `docs/PROGRESS_{R,P}.md`.

## Error handling

- Stages never raise past their own boundary on the hot path — a stage that fails degrades
  (`stages_skipped`) rather than crashing the request. See `harness/budget.py` (Workstream P).
- Stages are pure with respect to `PipelineContext` — they read and append, never mutate history.
- Retries (`tenacity`) only on stages that are idempotent, capped so a retry storm can never exceed
  the request deadline.

## Timing units

- **All timings in nanoseconds internally** — `time.perf_counter_ns()`. Convert to ms only at
  serialisation (response assembly / trace write). Float ms accumulates rounding error across nine
  stages; this is why the rule exists, not style preference.
- `timings_ms` in `AnswerResponse` is the one place floats appear, and only after conversion.

## Hot-path rules

- Functions on the request path carry a `# HOTPATH` comment.
- `# HOTPATH` functions may not perform network I/O (except the LLM call itself) or disk I/O.
- No cold starts: every model (embedder, reranker, guardrail classifiers) and the structured-output
  JSON schema are loaded/compiled and warmed with a dummy inference at boot, before the health check
  returns ready.
- Dense and sparse retrieval run concurrently via `asyncio.gather`, never sequentially — there's a
  test asserting wall-clock < sum of the two stage times.
- E5 embeddings require `"query: "` / `"passage: "` prefixes. There is a test for this
  (`tests/test_embedder.py`). Do not remove it, do not "simplify" it away — Workstream R's module,
  but if you're ever near it, don't touch the prefix logic.
- ONNX int8 is CPU-only — on GPU it's slower than FP32; use FP16 there.
- Structured LLM output: **reasoning field before the answer field** in the schema, so the model
  thinks before it commits.

## Workstream discipline (repo-specific, not generic style)

- Never edit the other workstream's `PROGRESS_*.md`, `DECISIONS_*.md`, `SESSION_LOG/track-*/`, or
  module files (`docs/TEAM_SPLIT.md` §2 has the exact table). Read freely, write only your own.
- Every ADR is dated, numbered per-workstream (`R-001`, `P-001`), and append-only. To reverse one,
  write a new ADR that supersedes it — never edit history.
- Commit messages reference the phase: `[P<phase>] <what changed>` or `[R<phase>] <what changed>`
  depending on which track's work the commit contains.
