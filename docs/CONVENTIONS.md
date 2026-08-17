# CONVENTIONS.md

- Type hints mandatory on all public functions; `mypy` clean.
- All timings in **nanoseconds internally** (`time.perf_counter_ns()`), converted to ms only at
  serialisation. Float ms accumulates error across nine stages — don't do intermediate ms math.
- No bare `except:`; every caught exception is logged with `trace_id`.
- No secrets in code — `.env` + `pydantic-settings`, `.env.example` committed, `.env` never is.
- New dependency ⇒ new ADR in the relevant `DECISIONS_*.md`.
- Every new module ⇒ a test file.
- Hot-path functions carry a `# HOTPATH` comment and may not perform network or disk I/O (except
  the LLM call itself).
- Stages are pure with respect to `PipelineContext` — they read and append, never mutate history.
- Retries (`tenacity`) only on idempotent stages, capped so they can never exceed the request
  deadline.
- Structured LLM output: **reasoning field before the answer field** in the schema, so the model
  thinks before it commits.
- E5 embeddings require `"query: "` / `"passage: "` prefixes — there is a test; do not remove it
  (Workstream R's module, but if you're ever near it, don't touch the prefix logic).
- ONNX int8 is CPU-only — on GPU it's slower than FP32; use FP16 there.
- Commit messages reference the phase: `[P<phase>] <what changed>` (my track) — teammate uses
  `[R<phase>] ...` or their own convention on their commits.
