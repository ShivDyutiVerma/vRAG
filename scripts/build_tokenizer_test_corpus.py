"""Builds the 1,000+ string test corpus for R-029's tokenizer equivalence test, from real project
data only -- no synthetic/generated filler text.

  - 500 real Hindi queries: eval/heldout_queries.json (the frozen held-out set)
  - 500 real English queries: data/working_subset.jsonl's Eng_Query field -- the original English
    MS MARCO queries the Hindi corpus was translated from (same rows, real 1:1 pairing)
  - 20 real mixed-script strings: romanized Hindi + code-mixed examples already used and vetted in
    tests/guardrails/test_g2_scope_language.py, plus English/Hindi concatenations of real strings
    above (a real code-mixing pattern, not synthetic content)

Every string is formatted with the "query: " E5 prefix, matching exactly what production sends to
the tokenizer (format_query() in src/vrag/index/embedder.py) -- an unprefixed test wouldn't be
representative of what's actually tokenized on the hot path.

Usage: python scripts/build_tokenizer_test_corpus.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HELDOUT_PATH = REPO_ROOT / "eval" / "heldout_queries.json"
WORKING_SUBSET_PATH = REPO_ROOT / "data" / "working_subset.jsonl"
OUT_PATH = REPO_ROOT / "eval" / "tokenizer_test_corpus.json"

SEED = 42
N_ENGLISH = 500
N_MIXED = 20

# Real, already-vetted romanized/mixed-script examples from
# tests/guardrails/test_g2_scope_language.py
ROMANIZED_EXAMPLES = [
    "bharat ka sabse uncha parvat kaunsa hai",
]


def build() -> list[dict]:
    rng = random.Random(SEED)

    heldout = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    hindi_queries = [row["query"] for row in heldout]

    eng_queries = []
    with WORKING_SUBSET_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            q = row.get("Eng_Query", "").strip()
            if q:
                eng_queries.append(q)
    english_sample = rng.sample(eng_queries, N_ENGLISH)

    # Mixed: real English + real Hindi concatenated (a real code-mixing pattern), plus the
    # already-vetted romanized examples.
    mixed = list(ROMANIZED_EXAMPLES)
    eng_pool = rng.sample(eng_queries, N_MIXED - len(ROMANIZED_EXAMPLES))
    hindi_pool = rng.sample(hindi_queries, N_MIXED - len(ROMANIZED_EXAMPLES))
    for e, h in zip(eng_pool, hindi_pool, strict=True):
        mixed.append(f"{e} {h}")

    rows = (
        [{"text": q, "category": "hindi"} for q in hindi_queries]
        + [{"text": q, "category": "english"} for q in english_sample]
        + [{"text": q, "category": "mixed"} for q in mixed]
    )
    # E5 prefix applied here, matching format_query() -- this is what's actually tokenized.
    for r in rows:
        r["prefixed_text"] = f"query: {r['text']}"
    return rows


if __name__ == "__main__":
    rows = build()
    OUT_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    print(f"Wrote {len(rows)} strings to {OUT_PATH}")
    print(counts)
