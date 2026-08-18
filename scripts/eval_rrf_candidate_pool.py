"""Follow-up to A3 (docs/RISKS.md R-R14) — A3 found hybrid+RRF regresses vs. dense-only
(docs/DECISIONS_R.md R-010) because each lane only contributes its top-`top_k` (10) candidates
before fusion, so BM25's weaker top ranks get equal RRF weight against dense's stronger ones. The
standard mitigation: fetch a larger per-lane candidate pool before fusion, truncate to `top_k`
after. This was flagged as untested, worth a quick follow-up if time remains — this script is that
follow-up. Chunking/embedder/retrieval_mode="hybrid"/fusion_k all held fixed at their existing
values; only the per-lane candidate pool size varies, one axis at a time per CLAUDE.md's ablation
discipline.

Usage: python scripts/eval_rrf_candidate_pool.py --index-dir data/index/metadata_aware
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import build_index
from eval_chunking import LEDGER_COLUMNS, LEDGER_PATH, _git_sha, _percentile

from vrag.index.embedder import E5Embedder
from vrag.index.fusion import DEFAULT_K, reciprocal_rank_fusion
from vrag.retrieval.metrics import score_hits

REPO_ROOT = Path(__file__).resolve().parent.parent
HELDOUT_PATH = REPO_ROOT / "eval" / "heldout_queries.json"

CHUNK_STRATEGY = "metadata_aware"  # A1 winner
EMBEDDER_NAME = E5Embedder.name  # A2 winner


def evaluate_pool_size(
    candidate_pool: int,
    dense,
    sparse,
    chunk_lookup: dict,
    embedder: E5Embedder,
    heldout: list[dict],
    top_k: int = 10,
    fusion_k: int = DEFAULT_K,
) -> dict:
    chunk_to_doc_id = {chunk_id: c.doc_id for chunk_id, c in chunk_lookup.items()}

    recalls_1, recalls_5, recalls_10, mrrs, ndcgs = [], [], [], [], []
    search_latencies_ms: list[float] = []
    embed_latencies_ms: list[float] = []
    embed_dim = 0

    for query_row in heldout:
        relevant_doc_ids = {p["passage_id"] for p in query_row["relevant_passages"]}
        query = query_row["query"]

        t_embed0 = time.perf_counter()
        query_vec = embedder.embed_queries([query])[0]
        embed_latencies_ms.append((time.perf_counter() - t_embed0) * 1000)
        embed_dim = len(query_vec)

        t0 = time.perf_counter()
        dense_hits = dense.search(query_vec, k=candidate_pool)
        sparse_hits = sparse.search(query, k=candidate_pool)
        hits = reciprocal_rank_fusion([dense_hits, sparse_hits], k=fusion_k)[:top_k]
        search_latencies_ms.append((time.perf_counter() - t0) * 1000)

        scores = score_hits(hits, chunk_to_doc_id, relevant_doc_ids)
        recalls_1.append(scores["recall@1"])
        recalls_5.append(scores["recall@5"])
        recalls_10.append(scores["recall@10"])
        mrrs.append(scores["mrr@10"])
        ndcgs.append(scores["ndcg@10"])

    return {
        "chunk_strategy": CHUNK_STRATEGY,
        "chunk_params": "{}",
        "embedder": EMBEDDER_NAME,
        "embed_backend": "sentence-transformers",
        "embed_dim": embed_dim,
        "index_type": "HNSW32",
        "retrieval_mode": "hybrid",
        "fusion_k": fusion_k,
        "top_k": top_k,
        "candidate_pool": candidate_pool,
        "recall@1": statistics.mean(recalls_1),
        "recall@5": statistics.mean(recalls_5),
        "recall@10": statistics.mean(recalls_10),
        "mrr@10": statistics.mean(mrrs),
        "ndcg@10": statistics.mean(ndcgs),
        "p50_embed_ms": _percentile(embed_latencies_ms, 50),
        "p50_search_ms": _percentile(search_latencies_ms, 50),
    }


def append_to_ledger(result: dict) -> None:
    import csv

    row = dict.fromkeys(LEDGER_COLUMNS, "")
    row.update(
        {
            "run_id": f"r-r14_pool{result['candidate_pool']}_{int(time.time())}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_sha": _git_sha(),
            "config_hash": str(hash(("hybrid", result["candidate_pool"]))),
            "chunk_strategy": result["chunk_strategy"],
            "chunk_params": result["chunk_params"],
            "embedder": result["embedder"],
            "embed_backend": result["embed_backend"],
            "embed_dim": result["embed_dim"],
            "index_type": result["index_type"],
            "retrieval_mode": result["retrieval_mode"],
            "fusion_k": result["fusion_k"],
            "top_k": result["top_k"],
            "recall@1": f"{result['recall@1']:.4f}",
            "recall@5": f"{result['recall@5']:.4f}",
            "recall@10": f"{result['recall@10']:.4f}",
            "mrr@10": f"{result['mrr@10']:.4f}",
            "ndcg@10": f"{result['ndcg@10']:.4f}",
            "p50_embed_ms": f"{result['p50_embed_ms']:.3f}",
            "p50_search_ms": f"{result['p50_search_ms']:.3f}",
            "notes": (
                f"R-R14 follow-up: hybrid+RRF with candidate_pool={result['candidate_pool']} "
                f"per lane before fusion (vs. A3's baseline top_k={result['top_k']})"
            ),
        }
    )
    file_exists = LEDGER_PATH.exists() and LEDGER_PATH.stat().st_size > 0
    with LEDGER_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default="data/index/metadata_aware")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--pools", type=int, nargs="+", default=[10, 30, 50, 100],
        help="per-lane candidate pool sizes to try before fusion (10 = A3's original baseline)",
    )
    args = parser.parse_args()

    heldout = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    dense, sparse, chunk_lookup = build_index.load(args.index_dir)
    embedder = E5Embedder()

    results = {}
    for pool in args.pools:
        print(f"\n--- evaluating candidate_pool={pool} ---")
        result = evaluate_pool_size(
            pool, dense, sparse, chunk_lookup, embedder, heldout, top_k=args.top_k
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        append_to_ledger(result)
        results[pool] = result

    print(f"\nAppended {len(args.pools)} rows to {LEDGER_PATH}")
    print("\n=== summary (A3's dense-only baseline: recall@5=0.652, ndcg@10=0.516) ===")
    for pool, r in results.items():
        print(
            f"pool={pool:4d}  recall@5={r['recall@5']:.4f}  mrr@10={r['mrr@10']:.4f}  "
            f"ndcg@10={r['ndcg@10']:.4f}  p50_search_ms={r['p50_search_ms']:.3f}"
        )
