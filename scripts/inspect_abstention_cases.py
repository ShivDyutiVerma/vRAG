"""Phase 7 (docs/DECISIONS.md ADR-016): real, fresh retrieval detail for the curated set of
representative abstention cases (eval/g3_abstention_case_selection.json) -- one rerankable
(category A/B/C/D) and one genuine-miss (category E) case per language where available. Real
production retrieve() path, k=50 (still bounded by HybridRetriever's internal WIDE_K=100 raw
search before language-filtering), so E cases here show whether the correct passage is
recoverable anywhere from rank 21-50 within the pool production already searches -- not yet the
deeper raw-FAISS-k test, which is a separate script (Part 4).

Also computes simple lexical/content-word overlap (same tokenizer as Phase 5, ADR-014 -- the
Mn/Mc-combining-mark-aware one) between the query and (a) the wrong top-1 passage and (b) the
real gold passage, so "does lexical overlap distinguish them" can be read directly off each case
rather than inferred.

Usage: python scripts/inspect_abstention_cases.py
Output: eval/g3_abstention_case_inspection.json
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ["VRAG_INDEX_DIR"] = str(REPO_ROOT / "data" / "index" / "multilingual_100k")
sys.path.insert(0, str(REPO_ROOT / "src"))

from vrag.languages import SARVAM_TO_TARGET_LANG  # noqa: E402
from vrag.retrieval.interface import retrieve  # noqa: E402

TARGET_LANG_TO_SARVAM = {v: k for k, v in SARVAM_TO_TARGET_LANG.items()}


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    cur: list[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "N") or cat in ("Mn", "Mc"):
            cur.append(ch)
        else:
            if cur:
                tokens.append("".join(cur))
                cur = []
    if cur:
        tokens.append("".join(cur))
    return tokens


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


async def inspect_one(case: dict, held_by_id: dict) -> dict:
    q = held_by_id[case["query_id"]]
    target_lang = case["language"]
    sarvam_code = TARGET_LANG_TO_SARVAM.get(target_lang)
    gold = q["relevant_passages"]
    gold_doc_ids = {p["passage_id"] for p in gold}
    query_tokens = set(t.lower() for t in tokenize(q["query"]))

    chunks = await retrieve(q["query"], k=50, language=sarvam_code)
    hits = []
    gold_rank = None
    for rank, c in enumerate(chunks, start=1):
        is_gold = c.passage_id in gold_doc_ids
        if is_gold and gold_rank is None:
            gold_rank = rank
        c_tokens = set(t.lower() for t in tokenize(c.text))
        hits.append(
            {
                "rank": rank,
                "passage_id": c.passage_id,
                "score": c.score,
                "language": c.language,
                "text": c.text[:220],
                "is_gold": is_gold,
                "lexical_overlap_with_query": round(jaccard(query_tokens, c_tokens), 4),
            }
        )

    gold_text = gold[0]["text"] if gold else None
    gold_tokens = set(t.lower() for t in tokenize(gold_text)) if gold_text else set()

    return {
        "query_id": case["query_id"],
        "language": target_lang,
        "category": case["category"],
        "query": q["query"],
        "query_type": q.get("query_type"),
        "gold_passage_id": gold[0]["passage_id"] if gold else None,
        "gold_text_preview": gold_text[:220] if gold_text else None,
        "gold_lexical_overlap_with_query": round(jaccard(query_tokens, gold_tokens), 4),
        "gold_rank_within_50": gold_rank,
        "top10_hits": hits[:10],
    }


async def main() -> None:
    selection = json.loads(
        (REPO_ROOT / "eval" / "g3_abstention_case_selection.json").read_text(encoding="utf-8")
    )
    held = json.loads(
        (REPO_ROOT / "eval" / "heldout_queries_multilingual.json").read_text(encoding="utf-8")
    )
    held_by_id = {q["query_id"]: q for q in held}

    results = []
    for case in selection:
        r = await inspect_one(case, held_by_id)
        results.append(r)
        top1_overlap = (
            r["top10_hits"][0]["lexical_overlap_with_query"] if r["top10_hits"] else None
        )
        print(
            f"[{r['language']:10s} {r['category']}] gold_rank={r['gold_rank_within_50']}"
            f"  gold_overlap={r['gold_lexical_overlap_with_query']}  top1_overlap={top1_overlap}"
        )

    out_path = REPO_ROOT / "eval" / "g3_abstention_case_inspection.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(results)} cases -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
