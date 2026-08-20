"""Phase 5 (docs/DECISIONS.md ADR-014): collect enriched per-query data for the cheap-signal
feature experiment. Phase 4's raw artifact (eval/g3_calibration_multilingual_100k_raw.json) only
kept 100-char text previews for the top 5 hits and no per-hit language tag -- not enough to build
lexical-overlap, cross-passage-redundancy, or same-language-consistency features. This script
re-collects with what those features actually need: full passage text and language tag for the
top 20 hits per query.

Same real production retrieve() path as Phase 4 (VRAG_INDEX_DIR -> data/index/multilingual_100k/),
not a reimplementation. No production code or config touched.

Usage: python scripts/collect_g3_feature_data.py
Output: eval/g3_feature_experiment_raw.json
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ["VRAG_INDEX_DIR"] = str(REPO_ROOT / "data" / "index" / "multilingual_100k")
sys.path.insert(0, str(REPO_ROOT / "src"))

from vrag.guardrails.g3_confidence import MARGIN, TAU  # noqa: E402
from vrag.languages import SARVAM_TO_TARGET_LANG  # noqa: E402
from vrag.retrieval.interface import is_retrieval_real, retrieve  # noqa: E402
from vrag.retrieval.metrics import dedupe_doc_ids  # noqa: E402

TARGET_LANG_TO_SARVAM = {v: k for k, v in SARVAM_TO_TARGET_LANG.items()}
WIDE_K = 20


async def collect_one(q: dict) -> dict:
    target_lang = q["language"]
    sarvam_code = TARGET_LANG_TO_SARVAM.get(target_lang)
    relevant_doc_ids = {p["passage_id"] for p in q["relevant_passages"]}

    chunks = await retrieve(q["query"], k=WIDE_K, language=sarvam_code)
    doc_ids = dedupe_doc_ids([c.passage_id for c in chunks])

    hits = [
        {
            "rank": rank,
            "chunk_id": c.chunk_id,
            "passage_id": c.passage_id,
            "score": c.score,
            "language": c.language,
            "text": c.text,
        }
        for rank, c in enumerate(chunks, start=1)
    ]

    return {
        "query_id": q["query_id"],
        "query": q["query"],
        "query_type": q.get("query_type"),
        "language": target_lang,
        "sarvam_code": sarvam_code,
        "msmarco_lang_code": q.get("msmarco_lang_code"),
        "n_hits": len(chunks),
        "relevant_in_top1": bool(set(doc_ids[:1]) & relevant_doc_ids),
        "relevant_in_top5": bool(set(doc_ids[:5]) & relevant_doc_ids),
        "relevant_in_top10": bool(set(doc_ids[:10]) & relevant_doc_ids),
        "relevant_in_top20": bool(set(doc_ids[:20]) & relevant_doc_ids),
        "hits": hits,
    }


async def main() -> None:
    print(f"TAU={TAU} MARGIN={MARGIN} (unchanged; this script does not touch G3)")
    print(f"VRAG_INDEX_DIR={os.environ['VRAG_INDEX_DIR']}")
    assert is_retrieval_real(), "STUB FALLBACK -- refusing to collect from stub"

    queries = json.loads(
        (REPO_ROOT / "eval" / "heldout_queries_multilingual.json").read_text(encoding="utf-8")
    )
    unmapped = [q for q in queries if q["language"] not in TARGET_LANG_TO_SARVAM]
    assert not unmapped, f"{len(unmapped)} queries have no Sarvam code mapping"

    rows = []
    for i, q in enumerate(queries):
        rows.append(await collect_one(q))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(queries)} queries collected...")

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()

    out = {
        "collected_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "index_dir": os.environ["VRAG_INDEX_DIR"],
        "wide_k": WIDE_K,
        "n_queries": len(rows),
        "source_heldout_set": "eval/heldout_queries_multilingual.json",
        "rows": rows,
    }
    out_path = REPO_ROOT / "eval" / "g3_feature_experiment_raw.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
