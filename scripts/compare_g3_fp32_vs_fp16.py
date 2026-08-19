"""Compares G3 (`GroundGateStage`, `src/vrag/guardrails/g3_confidence.py`) decisions between the
fp32 baseline index and the fp16-quantized index (docs/DECISIONS_R.md R-033/R-034), on the real
500-query held-out set. Diagnostic only -- does NOT change TAU/MARGIN (0.8835/0.0, calibrated
R-015/P-015), does not touch FAISS/corpus/embedder/tokenizer/deployment config.

Runs the real `HybridRetriever.retrieve()` (unmodified) against two retriever instances that
differ ONLY in which `DenseIndex` they wrap (fp32 vs fp16, both loaded via the same
`DenseIndex.load()`), sharing the same embedder and chunk_lookup -- so any decision difference is
attributable only to the index's own scores, nothing else. Then runs the real, unmodified
`g3_confidence.check()` on each result -- not a reimplementation of the threshold logic.

Usage: python scripts/compare_g3_fp32_vs_fp16.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vrag.guardrails import g3_confidence  # noqa: E402
from vrag.index.dense import DenseIndex  # noqa: E402
from vrag.index.embedder import LiteE5Embedder  # noqa: E402
from vrag.index.sqlite_chunk_lookup import SQLiteChunkLookup  # noqa: E402
from vrag.retrieval.hybrid import HybridRetriever  # noqa: E402

FP32_INDEX_DIR = REPO_ROOT / "data" / "index" / "metadata_aware"
FP16_INDEX_DIR = REPO_ROOT / "data" / "index" / "metadata_aware_sqfp16"
HELDOUT_PATH = REPO_ROOT / "eval" / "heldout_queries.json"


async def run_comparison() -> list[dict]:
    embedder = LiteE5Embedder()
    chunk_lookup = SQLiteChunkLookup(FP32_INDEX_DIR / "chunk_lookup.sqlite3")

    dense_fp32 = DenseIndex.load(FP32_INDEX_DIR / "dense")
    dense_fp16 = DenseIndex.load(FP16_INDEX_DIR / "dense")
    assert dense_fp32._quantization == "none"
    assert dense_fp16._quantization == "sqfp16"

    retriever_fp32 = HybridRetriever(
        dense=dense_fp32, sparse=None, embedder=embedder, chunk_lookup=chunk_lookup
    )
    retriever_fp16 = HybridRetriever(
        dense=dense_fp16, sparse=None, embedder=embedder, chunk_lookup=chunk_lookup
    )

    heldout = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    print(f"Comparing G3 decisions on {len(heldout)} real held-out queries...")

    results = []
    for i, row in enumerate(heldout):
        query = row["query"]
        chunks_fp32 = await retriever_fp32.retrieve(query, k=5)
        chunks_fp16 = await retriever_fp16.retrieve(query, k=5)

        verdict_fp32 = g3_confidence.check(chunks_fp32)
        verdict_fp16 = g3_confidence.check(chunks_fp16)

        top1_fp32 = chunks_fp32[0].score if chunks_fp32 else None
        top1_fp16 = chunks_fp16[0].score if chunks_fp16 else None

        results.append(
            {
                "query_id": row.get("query_id", i),
                "top1_fp32": top1_fp32,
                "top1_fp16": top1_fp16,
                "passed_fp32": verdict_fp32.passed,
                "passed_fp16": verdict_fp16.passed,
            }
        )
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(heldout)}...")

    chunk_lookup.close()
    return results


def summarize(results: list[dict]) -> None:
    n = len(results)
    fp32_answered = sum(r["passed_fp32"] for r in results)
    fp16_answered = sum(r["passed_fp16"] for r in results)
    flipped = [r for r in results if r["passed_fp32"] != r["passed_fp16"]]

    score_diffs = [
        r["top1_fp16"] - r["top1_fp32"]
        for r in results
        if r["top1_fp32"] is not None and r["top1_fp16"] is not None
    ]

    print("\n===== G3 decision comparison, FP32 vs FP16, real g3_confidence.check() =====")
    print(f"TAU={g3_confidence.TAU}  MARGIN={g3_confidence.MARGIN}  n_queries={n}")
    print(f"\nFP32: answered={fp32_answered}  abstained={n - fp32_answered}")
    print(f"FP16: answered={fp16_answered}  abstained={n - fp16_answered}")
    print(f"\nDecisions changed: {len(flipped)} / {n} ({100 * len(flipped) / n:.1f}%)")

    if score_diffs:
        mean_d = statistics.mean(score_diffs)
        stdev_d = statistics.pstdev(score_diffs)
        print("\nTop1 confidence-score differences (fp16 - fp32):")
        print(f"  mean={mean_d:+.5f}  stdev={stdev_d:.5f}")
        print(f"  max abs diff={max(abs(d) for d in score_diffs):.5f}")
        print(f"  min={min(score_diffs):+.5f}  max={max(score_diffs):+.5f}")

    tau = g3_confidence.TAU
    if flipped:
        print("\nFlipped-decision detail (query_id, fp32, fp16, dist to TAU each side):")
        dists_flipped = []
        for r in flipped:
            d32 = abs(r["top1_fp32"] - tau) if r["top1_fp32"] is not None else None
            d16 = abs(r["top1_fp16"] - tau) if r["top1_fp16"] is not None else None
            dists_flipped.append(min(x for x in (d32, d16) if x is not None))
            print(
                f"  {r['query_id']}: fp32={r['top1_fp32']:.4f} (passed={r['passed_fp32']})  "
                f"fp16={r['top1_fp16']:.4f} (passed={r['passed_fp16']})  "
                f"dist_to_tau(fp32)={d32:.4f}  dist_to_tau(fp16)={d16:.4f}"
            )

        all_dists_to_tau = [
            min(abs(r["top1_fp32"] - tau), abs(r["top1_fp16"] - tau))
            for r in results
            if r["top1_fp32"] is not None and r["top1_fp16"] is not None
        ]
        mean_flipped = statistics.mean(dists_flipped)
        mean_all = statistics.mean(all_dists_to_tau)
        print(f"\nMean min-distance-to-TAU, flipped queries: {mean_flipped:.5f}")
        print(f"Mean min-distance-to-TAU, ALL queries:      {mean_all:.5f}")
        print(
            "-> flipped decisions are concentrated near TAU"
            if mean_flipped < mean_all / 3
            else "-> flipped decisions are NOT obviously concentrated near TAU"
        )
    else:
        print("\nNo decisions changed -- nothing to check for threshold concentration.")

    out_path = REPO_ROOT / "eval" / "g3_fp32_vs_fp16_comparison.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote per-query detail to {out_path}")


if __name__ == "__main__":
    results = asyncio.run(run_comparison())
    summarize(results)
