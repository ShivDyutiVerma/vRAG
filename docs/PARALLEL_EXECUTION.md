# PARALLEL_EXECUTION.md
## Subagents + multi-terminal execution for the ablation runs

> Lives at `docs/PARALLEL_EXECUTION.md`. Extends `docs/TECH_MENU.md` §A (the staged ablation
> design). Read this before running any ablation stage (A1–A5).

---

## §0. The one rule that overrides everything else here

**Parallelism is for correctness/quality testing. It is NEVER used for the final latency numbers.**

Recall@k, MRR, nDCG, faithfulness — these don't care if the CPU is shared with five other processes.
Wall-clock milliseconds absolutely do. Running P50/P70/P100 benchmarks while five other configs hammer
the same CPU cores produces numbers that are actively wrong — inflated, noisy, and non-reproducible —
and reporting them would undermine the one part of the submission that's supposed to be the most
rigorously measured.

So the pipeline has two completely different execution modes:

| Mode | Used for | Parallel? | Machine state |
|------|----------|-----------|----------------|
| **Quality pass** | Recall/MRR/nDCG/faithfulness for every candidate | **Yes — parallelize aggressively** | Contention doesn't matter |
| **Latency pass** | P50/P70/P100 (the graded C3/C4 numbers) | **No — strictly sequential** | One config at a time, machine otherwise idle, no other heavy process running |

Any timing numbers that fall out of a parallel quality-pass run (e.g. "index build time" logged
alongside recall) are **indicative only** — label them as such in the ledger and never let them
substitute for the real `bench_latency.py` numbers.

---

## §1. Two different kinds of parallel work — use the right tool for each

### 1a. Implementing candidates → Claude Code subagents (the Task tool)

Writing the code for 6 chunking strategies, 4 embedder wrappers, 3 rerankers, etc. is **independent,
creative, agentic work** — exactly what subagents are for. Each subagent gets a self-contained brief
and writes one file, following the shared interface (`ChunkingStrategy` protocol, embedder interface,
etc. — see `AGENT_BUILD_SPEC.md` §7.1 and `docs/TECH_MENU.md`).

**How to invoke this in Claude Code:** ask it directly, e.g.

> "Launch 6 parallel subagents to implement the six chunking strategies from `docs/TECH_MENU.md` §S5.
> Each subagent implements exactly one strategy in its own file under `src/vrag/chunking/strategies/`,
> conforming to the `ChunkingStrategy` protocol in `src/vrag/chunking/base.py`, with a unit test in
> `tests/chunking/`. Do not let subagents touch each other's files or the shared registry — after all
> subagents report back, you (the orchestrator) do the registry wiring yourself in one pass."

**Why the orchestrator does the registry wiring, not the subagents:** if 6 subagents all try to add
a line to the same `registry.py`, that's a guaranteed conflict even within one session. One shared
file = one writer. Subagents write isolated files; the parent session does any shared-file edits
serially, after subagents finish.

**Isolation for git safety:** if subagents run with actual shell/git access, put each on its own
worktree (`git worktree add ../vrag-wt-chunk-semantic feature/chunk-semantic`) so a bad `git checkout`
in one can't clobber another's uncommitted work. For a single Claude Code session with subagents
sharing one working directory (the more common case), isolated *file* ownership is enough — no two
subagents write the same path.

### 1b. Running the benchmarks → parallel processes, not more agents

Once the code exists, executing it against the frozen eval set is **deterministic, non-agentic
work.** Spending LLM agent turns just to run `pytest`-style scripts is slower and more expensive than
it needs to be. Use real OS-level parallelism instead:

- **If you want to literally watch it happen (recommended — also great demo footage):** a `tmux`
  session with one pane per config, described in §2.
- **If you just want it done fast, no need to watch:** a Python driver using
  `concurrent.futures.ProcessPoolExecutor`, capped at `os.cpu_count() - 1` workers.

Either way, each parallel unit runs the **same command shape**: build/load what it needs, evaluate
against `eval/heldout_queries.json`, write its own result file. Nothing shared, nothing to lock.

---

## §2. The tmux runner (quality pass)

### Config manifest

Each ablation stage gets a manifest listing its independent runs. Example for A1:

`eval/manifests/A1_chunking.json`:
```json
[
  {"run_id": "A1_fixed_overlap20", "module": "vrag.chunking.strategies.fixed", "params": {"size": 256, "overlap": 0.2}},
  {"run_id": "A1_passage_native",  "module": "vrag.chunking.strategies.passage_native", "params": {}},
  {"run_id": "A1_sentence_window", "module": "vrag.chunking.strategies.sentence_window", "params": {"window": 2}},
  {"run_id": "A1_semantic",        "module": "vrag.chunking.strategies.semantic", "params": {"percentile": 90}},
  {"run_id": "A1_metadata_aware",  "module": "vrag.chunking.strategies.metadata_aware", "params": {}},
  {"run_id": "A1_hierarchical",    "module": "vrag.chunking.strategies.hierarchical", "params": {"child": 128, "parent": 512}}
]
```
Every other stage (A2 embedders, A3 retrieval modes, A4 rerankers, A5 generation providers) gets its
own manifest in the same shape. `run_single_experiment.py` is generic across all of them — it
receives one manifest entry, does the work, writes `eval/results/<stage>/<run_id>.json`.

### `scripts/run_ablation_stage.sh` — what Claude Code should build

```bash
#!/usr/bin/env bash
# Usage: ./scripts/run_ablation_stage.sh A1_chunking
set -euo pipefail
STAGE="$1"
MANIFEST="eval/manifests/${STAGE}.json"
SESSION="ablation-${STAGE}"
MAX_PARALLEL=$(( $(nproc) - 1 ))   # never saturate every core — see §3

tmux new-session -d -s "$SESSION" -n control

RUN_IDS=$(jq -r '.[].run_id' "$MANIFEST")
i=0
for RUN_ID in $RUN_IDS; do
  if (( i % MAX_PARALLEL == 0 )); then
    tmux new-window -t "$SESSION" -n "wave-$((i / MAX_PARALLEL))"
  else
    tmux split-window -t "$SESSION"
    tmux select-layout -t "$SESSION" tiled
  fi
  tmux send-keys -t "$SESSION" \
    "python scripts/run_single_experiment.py --stage $STAGE --run-id $RUN_ID --manifest $MANIFEST; touch eval/results/${STAGE}/${RUN_ID}.done" \
    C-m
  ((i++))
done

echo "Launched in tmux session '$SESSION'. Attach with: tmux attach -t $SESSION"
echo "Waiting for all runs to finish..."
python scripts/wait_for_results.py --stage "$STAGE" --manifest "$MANIFEST"
python scripts/aggregate_ablation.py --stage "$STAGE"
```

`wait_for_results.py` just polls for `.done` marker files (simplest reliable completion signal —
don't try to parse tmux pane state, it's fragile). `aggregate_ablation.py` reads every
`eval/results/<stage>/*.json`, appends rows to `eval/ablation_ledger.csv`, prints a ranked summary
table to the terminal, and computes the noise-floor check (§3 of `TECH_MENU.md`) if any run_id was
repeated 3×.

### Each `run_single_experiment.py` invocation must:
1. Load its manifest entry, resolve the named module/class dynamically
2. Build only what it needs (its own index shard if chunking/embedding differ; reuse the frozen
   `eval/heldout_queries.json` — never touch it)
3. Run the full quality-eval protocol from `TECH_MENU.md` §C
4. Write one self-contained JSON with every ledger column, plus `run_id`, `git_sha`, `timestamp`
5. Never write to a file another run_id also writes to. If two runs need the SAME shared artifact
   (e.g. two rerankers both wanting the same A1+A2-winning index), **build that shared artifact once,
   before the fan-out**, and have both runs open it read-only.

---

## §3. Resource sizing — don't oversubscribe the machine

Before launching a wave, Claude Code should check `os.cpu_count()` and available RAM
(`psutil.virtual_memory().available`), and:

- Cap concurrent workers at `cpu_count - 1` (leave one core for the OS/terminal/you)
- If a stage's manifest has more entries than that cap, run it in **waves**, not all at once
  (the script above already batches into tmux windows per wave)
- Watch memory, not just CPU: 4 concurrent embedding-model loads can exhaust RAM on a laptop long
  before CPU is the bottleneck. If any run OOMs, halve `MAX_PARALLEL` and retry that wave only —
  don't restart the whole stage.
- For A5 (generation providers), the work is mostly waiting on network I/O, not local compute — you
  can safely run all candidates concurrently regardless of core count, since they're not CPU-bound.

---

## §4. The sequential latency pass (P6) — do NOT parallelize this

Runs exactly as specified in `docs/BUILD_PLAN.md` P6 and `AGENT_BUILD_SPEC.md` §3–§4:

```bash
# One config. One process. Nothing else running. This is the only correct way to run this script.
python scripts/bench_latency.py --config eval/winning_config.json --n-repeats 5
```

If you have two machines available (you + your collaborator's laptop), you can run the latency
benchmark for **two different candidate configs simultaneously on two separate machines** — that's
fine, because each machine is still running exactly one config in isolation. What's forbidden is
running two configs on the *same* machine at the *same* time.

If Claude Code ever proposes "let's speed up the latency campaign by running it in the tmux session
too" — that's the one place to say no. Point it back to this section.

---

## §5. What goes in the README

Screenshot or describe the tmux session mid-run for `docs/assets/` — six panes, six chunking
strategies training and evaluating simultaneously, is a genuinely good visual for both the README
and the process video. Caption it with the wall-clock time saved vs. running sequentially, and note
explicitly (one sentence) that the *latency* numbers elsewhere in the README came from the separate,
isolated, sequential pass — so nobody reading closely wonders why the methodology looks different in
two places.
