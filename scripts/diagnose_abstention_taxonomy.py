"""Phase 7 (docs/DECISIONS.md ADR-016): classify every abstained query in the 532-query
multilingual held-out set into the requested A-F taxonomy, globally and per language.

Reuses eval/g3_calibration_multilingual_100k_raw.json (Phase 4/ADR-013's real collection via the
production retrieve() path) for relevant_in_top1/5/10/20 -- no new retrieval calls needed for
that part. Adds one cheap real check this file didn't already have: for abstained queries whose
gold passage isn't in the top-20 window, is it anywhere in the full corpus at all (category E) or
genuinely absent from this index (category F)? A single SQLite scan of chunk_lookup.sqlite3's
distinct doc_ids, not a new retrieval pass.

Categories (defined on ABSTAINED queries only -- current_g3_passed == False):
  A. Correct evidence is rank 1 (relevant_in_top1) but G3 rejected it anyway (top1 < TAU) --
     identical to Phase 4's "false_refusal" bucket, cross-checked below.
  B. Correct evidence is in top-5 but not rank 1.
  C. Correct evidence is in top-10 but not top-5.
  D. Correct evidence is in top-20 but not top-10.
  E. Correct evidence exists somewhere in the corpus but outside the top-20 window retrieved.
  F. Correct evidence's doc_id is not present in this index's corpus at all.

Usage: python scripts/diagnose_abstention_taxonomy.py
Output: eval/g3_abstention_taxonomy.json
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = REPO_ROOT / "eval" / "g3_calibration_multilingual_100k_raw.json"
HELDOUT_PATH = REPO_ROOT / "eval" / "heldout_queries_multilingual.json"
INDEX_DIR = REPO_ROOT / "data" / "index" / "multilingual_100k"
OUT_PATH = REPO_ROOT / "eval" / "g3_abstention_taxonomy.json"


def classify(row: dict, gold_doc_ids: set[str], corpus_doc_ids: set[str]) -> str:
    if row["relevant_in_top1"]:
        return "A"
    if row["relevant_in_top5"]:
        return "B"
    if row["relevant_in_top10"]:
        return "C"
    if row["relevant_in_top20"]:
        return "D"
    if gold_doc_ids & corpus_doc_ids:
        return "E"
    return "F"


def main() -> None:
    data = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    rows = data["rows"]
    held = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    held_by_id = {q["query_id"]: q for q in held}

    conn = sqlite3.connect(str(INDEX_DIR / "chunk_lookup.sqlite3"))
    corpus_doc_ids = {r[0] for r in conn.execute("SELECT DISTINCT doc_id FROM chunks")}
    conn.close()
    print(f"corpus_doc_ids: {len(corpus_doc_ids)} distinct passages")

    abstained = [r for r in rows if not r["current_g3_passed"]]
    print(f"abstained queries: {len(abstained)} / {len(rows)}")

    per_query = []
    for r in abstained:
        q = held_by_id[r["query_id"]]
        gold_doc_ids = {p["passage_id"] for p in q["relevant_passages"]}
        cat = classify(r, gold_doc_ids, corpus_doc_ids)
        per_query.append(
            {
                "query_id": r["query_id"],
                "language": r["language"],
                "category": cat,
                "top1_score": r["top1_score"],
                "query": r["query"],
            }
        )

    global_counts = Counter(pq["category"] for pq in per_query)
    per_lang_counts: dict[str, Counter] = defaultdict(Counter)
    for pq in per_query:
        per_lang_counts[pq["language"]][pq["category"]] += 1

    # cross-check: category A count must equal Phase 4's false_refusal count (56)
    n_a = global_counts.get("A", 0)
    print(f"\nCategory A count: {n_a} (Phase 4/ADR-013 false_refusal was 56 -- should match)")

    categories = ["A", "B", "C", "D", "E", "F"]
    labels = {
        "A": "rank1 but G3 rejects",
        "B": "in top5, not rank1",
        "C": "in top10, not top5",
        "D": "in top20, not top10",
        "E": "in corpus, outside top20",
        "F": "not in corpus at all",
    }

    print(f"\n{'Category':10s} {'Label':28s} {'Count':>7s} {'% of abstained':>15s}")
    n_total = len(abstained)
    for cat in categories:
        n = global_counts.get(cat, 0)
        print(f"{cat:10s} {labels[cat]:28s} {n:7d} {n / n_total * 100:14.1f}%")

    print(f"\n{'Language':10s}" + "".join(f"{c:>7s}" for c in categories) + f"{'total':>8s}")
    for lang in sorted(per_lang_counts):
        counts = per_lang_counts[lang]
        row_str = "".join(f"{counts.get(c, 0):7d}" for c in categories)
        total = sum(counts.values())
        print(f"{lang:10s}{row_str}{total:8d}")

    out = {
        "n_queries_total": len(rows),
        "n_abstained": n_total,
        "category_labels": labels,
        "global_counts": dict(global_counts),
        "per_language_counts": {lang: dict(c) for lang, c in per_lang_counts.items()},
        "per_query": per_query,
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote -> {OUT_PATH}")


if __name__ == "__main__":
    main()
