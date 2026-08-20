"""Phase 7 (docs/DECISIONS.md ADR-016), Parts 1 (taxonomy refinement) + 4 (depth sweep): for
every one of the 354 abstained queries, run a REAL retrieve(k=100, language=...) call -- this is
production's actual maximum effective width today, since HybridRetriever's language filter
already searches a raw dense k=100 pool before filtering (_LANGUAGE_FILTER_WIDE_K). The Phase 4
taxonomy only checked up to rank 20; this refines category E ("outside top-20") into real
top-50/top-100 buckets, and separately tests whether searching BEYOND the current WIDE_K=100 raw
pool (raw k=300, bypassing HybridRetriever's hardcoded width) recovers any more of what's left --
directly testing Candidate A ("language filter + larger FAISS candidate retrieval").

Usage: python scripts/depth_recovery_sweep.py
Output: eval/g3_depth_recovery_sweep.json
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ["VRAG_INDEX_DIR"] = str(REPO_ROOT / "data" / "index" / "multilingual_100k")
sys.path.insert(0, str(REPO_ROOT / "src"))

from vrag.index.dense import DenseIndex  # noqa: E402
from vrag.index.embedder import LiteE5Embedder  # noqa: E402
from vrag.index.sqlite_chunk_lookup import SQLiteChunkLookup  # noqa: E402
from vrag.languages import SARVAM_TO_TARGET_LANG  # noqa: E402
from vrag.retrieval.interface import retrieve  # noqa: E402

TARGET_LANG_TO_SARVAM = {v: k for k, v in SARVAM_TO_TARGET_LANG.items()}
INDEX_DIR = REPO_ROOT / "data" / "index" / "multilingual_100k"
RAW_BEYOND_K = 300  # bypasses HybridRetriever's hardcoded WIDE_K=100, real DenseIndex.search()


def classify_at_depths(gold_doc_ids: set[str], doc_ids_ranked: list[str]) -> dict:
    """First-occurrence rank of any gold doc_id in the (already deduped) ranked list."""
    rank = None
    for i, d in enumerate(doc_ids_ranked, start=1):
        if d in gold_doc_ids:
            rank = i
            break
    return {
        "first_gold_rank": rank,
        "in_top10": rank is not None and rank <= 10,
        "in_top20": rank is not None and rank <= 20,
        "in_top50": rank is not None and rank <= 50,
        "in_top100": rank is not None and rank <= len(doc_ids_ranked) and rank <= 100,
    }


async def main() -> None:
    tax = json.loads(
        (REPO_ROOT / "eval" / "g3_abstention_taxonomy.json").read_text(encoding="utf-8")
    )
    held = json.loads(
        (REPO_ROOT / "eval" / "heldout_queries_multilingual.json").read_text(encoding="utf-8")
    )
    held_by_id = {q["query_id"]: q for q in held}
    abstained = tax["per_query"]
    print(f"Re-checking {len(abstained)} abstained queries at k=100 (production's real width)...")

    t0 = time.perf_counter()
    results = []
    for i, pq in enumerate(abstained):
        q = held_by_id[pq["query_id"]]
        gold_doc_ids = {p["passage_id"] for p in q["relevant_passages"]}
        sarvam_code = TARGET_LANG_TO_SARVAM.get(pq["language"])
        chunks = await retrieve(q["query"], k=100, language=sarvam_code)
        doc_ids = list(dict.fromkeys(c.passage_id for c in chunks))  # dedupe, preserve order
        depths = classify_at_depths(gold_doc_ids, doc_ids)
        results.append(
            {
                "query_id": pq["query_id"],
                "language": pq["language"],
                "old_category_top20": pq["category"],
                "n_hits_at_k100_request": len(chunks),
                **depths,
            }
        )
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(abstained)}...")
    t1 = time.perf_counter()
    print(f"k<=100 pass done in {t1 - t0:.1f}s ({(t1 - t0) / len(abstained) * 1000:.1f}ms/query)")

    # For queries still not found within the k=100 filtered results, test genuinely deeper raw
    # dense search (bypasses HybridRetriever's hardcoded WIDE_K=100).
    still_missing = [r for r in results if not r["in_top100"]]
    print(f"\nStill missing after k=100: {len(still_missing)} -- testing raw k={RAW_BEYOND_K}...")

    dense = DenseIndex.load(INDEX_DIR / "dense")
    lookup = SQLiteChunkLookup(INDEX_DIR / "chunk_lookup.sqlite3")
    embedder = LiteE5Embedder()

    from vrag.languages import SARVAM_TO_TARGET_LANG as S2T

    beyond_results = []
    for r in still_missing:
        q = held_by_id[r["query_id"]]
        gold_doc_ids = {p["passage_id"] for p in q["relevant_passages"]}
        target_lang = S2T.get(TARGET_LANG_TO_SARVAM.get(r["language"]))
        vec = embedder.embed_queries([q["query"]])[0]
        raw_hits = dense.search(vec, k=RAW_BEYOND_K)
        same_lang_doc_ids = []
        for chunk_id, _score in raw_hits:
            chunk = lookup.get(chunk_id)
            if (
                chunk is not None
                and chunk.metadata.get("language") == target_lang
                and chunk.doc_id not in same_lang_doc_ids
            ):
                same_lang_doc_ids.append(chunk.doc_id)
        rank = next(
            (i + 1 for i, d in enumerate(same_lang_doc_ids) if d in gold_doc_ids), None
        )
        beyond_results.append(
            {
                "query_id": r["query_id"],
                "language": r["language"],
                "found_in_raw_k300_same_lang_rank": rank,
                "n_same_lang_in_raw_k300": len(same_lang_doc_ids),
            }
        )
    lookup.close()

    recovered_beyond_100 = sum(1 for b in beyond_results if b["found_in_raw_k300_same_lang_rank"])
    print(f"Recovered within raw k={RAW_BEYOND_K} (beyond production's current k=100): "
          f"{recovered_beyond_100}/{len(still_missing)}")

    # aggregate depth-recovery summary, global + per language
    def summarize(rows: list[dict]) -> dict:
        n = len(rows)
        return {
            "n": n,
            "in_top10": sum(r["in_top10"] for r in rows),
            "in_top20": sum(r["in_top20"] for r in rows),
            "in_top50": sum(r["in_top50"] for r in rows),
            "in_top100": sum(r["in_top100"] for r in rows),
        }

    global_summary = summarize(results)
    langs = sorted({r["language"] for r in results})
    per_lang_summary = {
        lang: summarize([r for r in results if r["language"] == lang]) for lang in langs
    }

    print(f"\n=== Global depth recovery (of {len(results)} abstained queries) ===")
    print(json.dumps(global_summary, indent=2))
    print("\n=== Per-language ===")
    for lang in langs:
        print(f"  {lang:10s} {per_lang_summary[lang]}")

    out = {
        "n_abstained": len(results),
        "global_summary": global_summary,
        "per_language_summary": per_lang_summary,
        "per_query_k100": results,
        "beyond_k100_raw300_check": {
            "n_still_missing_after_k100": len(still_missing),
            "n_recovered_in_raw_k300": recovered_beyond_100,
            "details": beyond_results,
        },
    }
    out_path = REPO_ROOT / "eval" / "g3_depth_recovery_sweep.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
