"""Phase 4 (docs/DECISIONS.md ADR-013): collect raw per-query calibration data for G3 on the
multilingual candidate (data/index/multilingual_100k/), so the threshold sweep in
calibrate_g3_sweep.py runs against real measurements, not intuition.

Uses the REAL production retrieve() path (src/vrag/retrieval/interface.py), pointed at the
multilingual candidate via VRAG_INDEX_DIR -- same mechanism scripts/smoke_test_multilingual_
candidate.py uses -- not a reimplementation of HybridRetriever's language-filter logic. This
means whatever this script measures is exactly what production would see if VRAG_INDEX_DIR were
ever pointed here for real. No production code or config is modified.

Requests k=20 per query (not the harness's real k=5) so one retrieval call yields everything
needed: G3's real decision only ever looks at the first 5 elements of this list (this script
checks that invariant explicitly with G3_K=5, matching src/vrag/harness/stages.py's
`ctx.data.get("k", 5)` default), while evidence-location analysis (found in top-10? top-20?)
needs the wider window. The ranking is deterministic and stable under truncation, so a k=20 call's
first 5 elements are byte-identical to what a real k=5 call would have returned.

Usage: python scripts/calibrate_g3_collect.py
Output: eval/g3_calibration_multilingual_100k_raw.json
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os_environ_key = "VRAG_INDEX_DIR"
import os  # noqa: E402

os.environ[os_environ_key] = str(REPO_ROOT / "data" / "index" / "multilingual_100k")
sys.path.insert(0, str(REPO_ROOT / "src"))

from vrag.guardrails.g3_confidence import MARGIN, TAU  # noqa: E402
from vrag.languages import SARVAM_TO_TARGET_LANG  # noqa: E402
from vrag.retrieval.interface import is_retrieval_real, retrieve  # noqa: E402
from vrag.retrieval.metrics import (  # noqa: E402
    dedupe_doc_ids,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

TARGET_LANG_TO_SARVAM = {v: k for k, v in SARVAM_TO_TARGET_LANG.items()}
G3_K = 5  # production default (src/vrag/harness/stages.py: ctx.data.get("k", 5))
WIDE_K = 20  # collection width for evidence-location analysis


def g3_decision(top1: float, weakest: float, n: int) -> bool:
    """Byte-identical logic to src/vrag/guardrails/g3_confidence.py's check() -- reads TAU/MARGIN
    directly from that module, never re-guesses the values."""
    if n == 0:
        return False
    if top1 < TAU:
        return False
    return not (n >= 2 and (top1 - weakest) < MARGIN)


async def collect_one(q: dict) -> dict:
    target_lang = q["language"]
    sarvam_code = TARGET_LANG_TO_SARVAM.get(target_lang)
    relevant_doc_ids = {p["passage_id"] for p in q["relevant_passages"]}

    chunks = await retrieve(q["query"], k=WIDE_K, language=sarvam_code)

    doc_ids_raw = [c.passage_id for c in chunks]
    doc_ids = dedupe_doc_ids(doc_ids_raw)

    g3_chunks = chunks[:G3_K]
    scores_g3 = [c.score for c in g3_chunks]
    top1 = scores_g3[0] if scores_g3 else 0.0
    weakest = scores_g3[min(4, len(scores_g3) - 1)] if scores_g3 else 0.0
    n_g3 = len(scores_g3)
    passed = g3_decision(top1, weakest, n_g3)

    top_hits = [
        {
            "rank": rank,
            "chunk_id": c.chunk_id,
            "passage_id": c.passage_id,
            "score": c.score,
            "text_preview": c.text[:100],
        }
        for rank, c in enumerate(chunks[:5], start=1)
    ]

    return {
        "query_id": q["query_id"],
        "query": q["query"],
        "query_type": q.get("query_type"),
        "language": target_lang,
        "sarvam_code": sarvam_code,
        "msmarco_lang_code": q.get("msmarco_lang_code"),
        "n_hits_wide": len(chunks),
        "top1_score": top1,
        "weakest5_score": weakest,
        "all_scores_wide": [c.score for c in chunks],
        "current_g3_passed": passed,
        "relevant_in_top1": bool(set(doc_ids[:1]) & relevant_doc_ids),
        "relevant_in_top5": bool(set(doc_ids[:5]) & relevant_doc_ids),
        "relevant_in_top10": bool(set(doc_ids[:10]) & relevant_doc_ids),
        "relevant_in_top20": bool(set(doc_ids[:20]) & relevant_doc_ids),
        "recall@1": recall_at_k(doc_ids, relevant_doc_ids, k=1),
        "recall@5": recall_at_k(doc_ids, relevant_doc_ids, k=5),
        "recall@10": recall_at_k(doc_ids, relevant_doc_ids, k=10),
        "mrr@10": reciprocal_rank(doc_ids, relevant_doc_ids, k=10),
        "ndcg@10": ndcg_at_k(doc_ids, relevant_doc_ids, k=10),
        "top_hits": top_hits,
    }


async def main() -> None:
    print(f"TAU={TAU} MARGIN={MARGIN} (src/vrag/guardrails/g3_confidence.py, unchanged)")
    print(f"VRAG_INDEX_DIR={os.environ[os_environ_key]}")
    assert is_retrieval_real(), "STUB FALLBACK -- refusing to collect calibration data from stub"

    queries = json.loads(
        (REPO_ROOT / "eval" / "heldout_queries_multilingual.json").read_text(encoding="utf-8")
    )
    unmapped = [q for q in queries if q["language"] not in TARGET_LANG_TO_SARVAM]
    assert not unmapped, f"{len(unmapped)} queries have no Sarvam code mapping: {unmapped[:3]}"

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
        "index_dir": os.environ[os_environ_key],
        "current_tau": TAU,
        "current_margin": MARGIN,
        "g3_k": G3_K,
        "wide_k": WIDE_K,
        "n_queries": len(rows),
        "source_heldout_set": "eval/heldout_queries_multilingual.json",
        "rows": rows,
    }
    out_path = REPO_ROOT / "eval" / "g3_calibration_multilingual_100k_raw.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(rows)} rows -> {out_path}")

    n_correct_top1 = sum(1 for r in rows if r["relevant_in_top1"])
    n_current_pass = sum(1 for r in rows if r["current_g3_passed"])
    print(
        f"Sanity: relevant_in_top1={n_correct_top1}/{len(rows)}, "
        f"current_g3_passed={n_current_pass}/{len(rows)}"
    )


if __name__ == "__main__":
    asyncio.run(main())
