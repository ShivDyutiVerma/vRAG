"""Phase 0 deliverable: freeze the working corpus subset + the held-out eval set
(AGENT_BUILD_SPEC.md §6.2, docs/BUILD_PLAN.md P0 task 4).

Reads a fixed-size pool of rows from the front of the Hindi train file (order as stored — see
docs/RISKS.md for the caveat that this isn't a random sample of the full 780k-ish Hindi corpus, just
its first N rows), then:

  - eval/heldout_queries.json: 500 query->relevant-passage pairs, randomly selected (fixed seed)
    from the pool. Ground truth only — never used to inform chunking-strategy design decisions.
    See docs/DECISIONS_R.md R-002 for why their passages are still part of the indexed pool
    (needed for Recall@k to be computable at all).
  - data/working_subset.jsonl (gitignored, regenerate-don't-commit): every row in the pool, in the
    shape scripts/build_index.py will consume to build the FAISS/BM25 indexes. Includes the heldout
    rows' passages.

Ground truth passage IDs use the convention f"{query_id}_{passage_index}" — scripts/build_index.py
and every ChunkingStrategy must produce doc_id/chunk_id values traceable back to this convention.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Any

import _dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
HELDOUT_PATH = REPO_ROOT / "eval" / "heldout_queries.json"
WORKING_SUBSET_PATH = REPO_ROOT / "data" / "working_subset.jsonl"
SEED = 20260817  # fixed so the split is reproducible across machines/sessions


def _passage_id(query_id: int, passage_index: int) -> str:
    return f"{query_id}_{passage_index}"


def _relevant_passages(row: dict[str, Any]) -> list[dict[str, Any]]:
    translated = row["passages"].get("Translated_passages", [])
    is_selected = row["passages"].get("is_selected", [])
    return [
        {"passage_id": _passage_id(row["query_id"], i), "text": text}
        for i, (text, selected) in enumerate(zip(translated, is_selected, strict=True))
        if selected
    ]


def build(pool_size: int, heldout_size: int) -> None:
    print(f"Reading {pool_size} rows from the Hindi train file...")
    pool = list(itertools.islice(_dataset.iter_rows(), pool_size))
    if len(pool) < pool_size:
        print(f"WARNING: only {len(pool)} rows available, requested {pool_size}.")

    rng = random.Random(SEED)
    # Sample only from rows that actually have a ground-truth relevant passage — otherwise the
    # heldout count silently falls short (many MSMARCO-XI rows have no is_selected passage at all).
    eligible = [i for i, row in enumerate(pool) if _relevant_passages(row)]
    print(f"{len(eligible)}/{len(pool)} pool rows have at least one relevant passage")
    heldout_indices = set(rng.sample(eligible, min(heldout_size, len(eligible))))

    WORKING_SUBSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with WORKING_SUBSET_PATH.open("w", encoding="utf-8") as f:
        for row in pool:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pool)} rows -> {WORKING_SUBSET_PATH}")

    heldout = []
    skipped_no_relevant = 0
    for i in sorted(heldout_indices):
        row = pool[i]
        relevant = _relevant_passages(row)
        if not relevant:
            skipped_no_relevant += 1
            continue
        heldout.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "query_type": row["query_type"],
                "relevant_passages": relevant,
            }
        )

    HELDOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HELDOUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(heldout, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(heldout)} held-out query->passage pairs -> {HELDOUT_PATH}")
    if skipped_no_relevant:
        print(
            f"({skipped_no_relevant} sampled rows had no is_selected passage and were skipped - "
            f"resample with a larger heldout_size if you need exactly {heldout_size})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-size", type=int, default=10_000)
    parser.add_argument("--heldout-size", type=int, default=500)
    args = parser.parse_args()
    build(args.pool_size, args.heldout_size)
