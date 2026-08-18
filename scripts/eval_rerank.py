"""A4 reranker ablation (docs/TECH_MENU.md §A row A4 / §S9, docs/EVAL_PROTOCOL.md). Chunking,
embedder, and retrieval mode are held fixed at the A1-A3 winners (metadata_aware /
multilingual-e5-small / dense-only, docs/DECISIONS_R.md R-004/R-009/R-010) — only the reranker
varies: none / FlashRank / cross-encoder (`src/vrag/retrieval/rerank.py`).

Rerank candidate pool: dense search fetches `candidate_k` (default 50, per docs/TECH_MENU.md §S9's
own FlashRank benchmark shape — "sub-20ms for 50 candidates") hits; the reranker narrows that down
to `top_k` (default 10), which is what actually gets scored against the held-out set. The `none` row
skips reranking and scores dense's native top-`top_k` directly, matching A3's dense row exactly, so
the "none" baseline in this table is directly comparable to A3's number.

Measures quality AND rerank latency in the same run (docs/TECH_MENU.md §A rule 5 — "a config that
wins Recall@5 by 1pt and costs 40ms loses").

Usage: python scripts/eval_rerank.py --index-dir data/index/metadata_aware
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import build_index
from eval_chunking import LEDGER_COLUMNS, LEDGER_PATH, _git_sha, _percentile

from vrag.index.embedder import E5Embedder
from vrag.retrieval.metrics import score_hits
from vrag.retrieval.rerank import CrossEncoderReranker, FlashRankReranker, NoOpReranker

REPO_ROOT = Path(__file__).resolve().parent.parent
HELDOUT_PATH = REPO_ROOT / "eval" / "heldout_queries.json"

CHUNK_STRATEGY = "metadata_aware"  # A1 winner
EMBEDDER_NAME = E5Embedder.name  # A2 winner
RETRIEVAL_MODE = "dense"  # A3 winner


def evaluate_reranker(
    reranker,
    dense,
    chunk_lookup: dict,
    embedder: E5Embedder,
    heldout: list[dict],
    top_k: int = 10,
    candidate_k: int = 50,
    sample_n: int | None = None,
) -> dict:
    """`sample_n` runs on a query prefix instead of the full held-out set — for a reranker whose
    per-query latency alone already disqualifies it from hot-path use (measured, not assumed; see
    docs/DECISIONS_R.md), a full 500-query precision run isn't worth the wall-clock cost. `None`
    (default) uses the full set, matching every other ablation stage."""
    if sample_n is not None:
        heldout = heldout[:sample_n]
    chunk_to_doc_id = {chunk_id: c.doc_id for chunk_id, c in chunk_lookup.items()}
    use_rerank = reranker.name != "none"

    recalls_1, recalls_5, recalls_10, mrrs, ndcgs = [], [], [], [], []
    search_latencies_ms: list[float] = []
    rerank_latencies_ms: list[float] = []

    for query_row in heldout:
        relevant_doc_ids = {p["passage_id"] for p in query_row["relevant_passages"]}
        query = query_row["query"]

        query_vec = embedder.embed_queries([query])[0]

        t0 = time.perf_counter()
        pool_k = candidate_k if use_rerank else top_k
        dense_hits = dense.search(query_vec, k=pool_k)
        search_latencies_ms.append((time.perf_counter() - t0) * 1000)

        if use_rerank:
            candidates = [
                (chunk_id, chunk_lookup[chunk_id].text)
                for chunk_id, _score in dense_hits
                if chunk_id in chunk_lookup
            ]
            t_rerank0 = time.perf_counter()
            hits = reranker.rerank(query, candidates, k=top_k)
            rerank_latencies_ms.append((time.perf_counter() - t_rerank0) * 1000)
        else:
            hits = dense_hits[:top_k]

        scores = score_hits(hits, chunk_to_doc_id, relevant_doc_ids)
        recalls_1.append(scores["recall@1"])
        recalls_5.append(scores["recall@5"])
        recalls_10.append(scores["recall@10"])
        mrrs.append(scores["mrr@10"])
        ndcgs.append(scores["ndcg@10"])

    return {
        "chunk_strategy": CHUNK_STRATEGY,
        "embedder": EMBEDDER_NAME,
        "retrieval_mode": RETRIEVAL_MODE,
        "reranker": reranker.name,
        "top_k": top_k,
        "n_queries": len(heldout),
        "recall@1": statistics.mean(recalls_1),
        "recall@5": statistics.mean(recalls_5),
        "recall@10": statistics.mean(recalls_10),
        "mrr@10": statistics.mean(mrrs),
        "ndcg@10": statistics.mean(ndcgs),
        "p50_search_ms": _percentile(search_latencies_ms, 50),
        "p50_rerank_ms": _percentile(rerank_latencies_ms, 50) if rerank_latencies_ms else 0.0,
    }


def append_to_ledger(result: dict) -> None:
    row = dict.fromkeys(LEDGER_COLUMNS, "")
    row.update(
        {
            "run_id": f"a4_{result['reranker']}_{int(time.time())}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_sha": _git_sha(),
            "config_hash": str(hash(result["reranker"])),
            "chunk_strategy": result["chunk_strategy"],
            "embedder": result["embedder"],
            "retrieval_mode": result["retrieval_mode"],
            "reranker": result["reranker"],
            "top_k": result["top_k"],
            "recall@1": f"{result['recall@1']:.4f}",
            "recall@5": f"{result['recall@5']:.4f}",
            "recall@10": f"{result['recall@10']:.4f}",
            "mrr@10": f"{result['mrr@10']:.4f}",
            "ndcg@10": f"{result['ndcg@10']:.4f}",
            "p50_search_ms": f"{result['p50_search_ms']:.3f}",
            "p50_rerank_ms": f"{result['p50_rerank_ms']:.3f}",
            "notes": (
                "A4 reranker ablation, chunking+embedder+retrieval_mode fixed at A1-A3 winners, "
                f"n_queries={result['n_queries']}"
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
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument(
        "--flashrank-sample-n",
        type=int,
        default=30,
        help=(
            "FlashRank's multilingual model (ms-marco-MultiBERT-L-12) measured at 9-14s/query on "
            "real corpus text on this CPU-only onnxruntime install (docs/DECISIONS_R.md) -- a "
            "latency that alone disqualifies it from hot-path use, so a full 500-query precision "
            "run isn't worth the ~2hr wall-clock cost. Pass 0 to force the full held-out set."
        ),
    )
    args = parser.parse_args()

    heldout = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    dense, _sparse, chunk_lookup = build_index.load(args.index_dir)
    embedder = E5Embedder()

    rerankers = {
        "none": NoOpReranker(),
        "flashrank": FlashRankReranker(),
        "cross-encoder": CrossEncoderReranker(),
    }
    sample_sizes = {
        "none": None,
        "flashrank": args.flashrank_sample_n or None,
        "cross-encoder": None,
    }

    results = {}
    for name, reranker in rerankers.items():
        print(f"\n--- evaluating reranker={name} (sample_n={sample_sizes[name]}) ---")
        result = evaluate_reranker(
            reranker,
            dense,
            chunk_lookup,
            embedder,
            heldout,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            sample_n=sample_sizes[name],
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        append_to_ledger(result)
        results[name] = result

    print(f"\nAppended {len(rerankers)} rows to {LEDGER_PATH}")
    print("\n=== summary ===")
    for name, r in results.items():
        print(
            f"{name:14s}  recall@5={r['recall@5']:.4f}  mrr@10={r['mrr@10']:.4f}  "
            f"ndcg@10={r['ndcg@10']:.4f}  p50_rerank_ms={r['p50_rerank_ms']:.3f}"
        )
