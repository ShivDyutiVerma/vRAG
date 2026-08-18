"""A3 retrieval-mode ablation (docs/TECH_MENU.md §A row A3, docs/EVAL_PROTOCOL.md). Chunking and
embedder are held fixed at the A1/A2 winners (metadata_aware / multilingual-e5-small) — only the
retrieval mode varies: dense-only, sparse-only, or hybrid (RRF over both). Reuses the already-built
`data/index/metadata_aware/` index rather than re-embedding, since neither chunking nor the embedder
changes for this stage.

Each mode fetches top_k candidates per lane (matching HybridRetriever's actual production shape —
src/vrag/retrieval/hybrid.py fetches k from each of dense/sparse before fusing, not a larger
candidate pool) then scores against the frozen held-out set via the shared score_hits() path, so the
R-006 dedup fix is never skipped. Appends one ledger row per mode.

Usage: python scripts/eval_retrieval_mode.py --index-dir data/index/metadata_aware
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
EMBEDDER_NAME = E5Embedder.name  # A2 winner (multilingual-e5-small)


def evaluate_mode(
    mode: str,
    dense,
    sparse,
    chunk_lookup: dict,
    embedder: E5Embedder,
    heldout: list[dict],
    top_k: int = 10,
    fusion_k: int = DEFAULT_K,
) -> dict:
    if mode not in ("dense", "sparse", "hybrid"):
        raise ValueError(f"unknown retrieval mode: {mode}")

    chunk_to_doc_id = {chunk_id: c.doc_id for chunk_id, c in chunk_lookup.items()}

    recalls_1, recalls_5, recalls_10, mrrs, ndcgs = [], [], [], [], []
    search_latencies_ms: list[float] = []
    embed_latencies_ms: list[float] = []
    embed_dim = 0

    for query_row in heldout:
        relevant_doc_ids = {p["passage_id"] for p in query_row["relevant_passages"]}
        query = query_row["query"]

        query_vec = None
        if mode in ("dense", "hybrid"):
            t_embed0 = time.perf_counter()
            query_vec = embedder.embed_queries([query])[0]
            embed_latencies_ms.append((time.perf_counter() - t_embed0) * 1000)
            embed_dim = len(query_vec)

        t0 = time.perf_counter()
        if mode == "dense":
            hits = dense.search(query_vec, k=top_k)
        elif mode == "sparse":
            hits = sparse.search(query, k=top_k)
        else:  # hybrid — same shape as HybridRetriever.retrieve: k candidates from each lane
            dense_hits = dense.search(query_vec, k=top_k)
            sparse_hits = sparse.search(query, k=top_k)
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
        "retrieval_mode": mode,
        "fusion_k": fusion_k if mode == "hybrid" else "",
        "top_k": top_k,
        "recall@1": statistics.mean(recalls_1),
        "recall@5": statistics.mean(recalls_5),
        "recall@10": statistics.mean(recalls_10),
        "mrr@10": statistics.mean(mrrs),
        "ndcg@10": statistics.mean(ndcgs),
        "p50_embed_ms": _percentile(embed_latencies_ms, 50) if embed_latencies_ms else 0.0,
        "p50_search_ms": _percentile(search_latencies_ms, 50),
    }


def append_to_ledger(result: dict) -> None:
    import csv

    row = dict.fromkeys(LEDGER_COLUMNS, "")
    row.update(
        {
            "run_id": f"a3_{result['retrieval_mode']}_{int(time.time())}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_sha": _git_sha(),
            "config_hash": str(hash(result["retrieval_mode"])),
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
            "notes": "A3 retrieval-mode ablation, chunking+embedder fixed at A1/A2 winners",
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
    args = parser.parse_args()

    heldout = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    dense, sparse, chunk_lookup = build_index.load(args.index_dir)
    embedder = E5Embedder()

    results = {}
    for mode in ("dense", "sparse", "hybrid"):
        print(f"\n--- evaluating mode={mode} ---")
        result = evaluate_mode(
            mode, dense, sparse, chunk_lookup, embedder, heldout, top_k=args.top_k
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        append_to_ledger(result)
        results[mode] = result

    print(f"\nAppended 3 rows to {LEDGER_PATH}")
    print("\n=== summary ===")
    for mode, r in results.items():
        print(
            f"{mode:8s}  recall@5={r['recall@5']:.4f}  mrr@10={r['mrr@10']:.4f}  "
            f"ndcg@10={r['ndcg@10']:.4f}  p50_search_ms={r['p50_search_ms']:.3f}"
        )
