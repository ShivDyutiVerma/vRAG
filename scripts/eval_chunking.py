"""A1 chunking ablation (docs/TECH_MENU.md §A, docs/EVAL_PROTOCOL.md). For one chunking strategy,
builds an index (embedder + retrieval mode held fixed at e5-small / dense-only / no-rerank per the
staged-ablation design — varying more than one axis in a single run makes the row void) and scores
it against the frozen held-out query set, appending one row to eval/ablation_ledger.csv.

Usage: python scripts/eval_chunking.py --strategy passage_native
       python scripts/eval_chunking.py --strategy fixed_overlap --overlap 0.2 --size 256
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import time
from pathlib import Path

import build_index

from vrag.retrieval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank

REPO_ROOT = Path(__file__).resolve().parent.parent
HELDOUT_PATH = REPO_ROOT / "eval" / "heldout_queries.json"
LEDGER_PATH = REPO_ROOT / "eval" / "ablation_ledger.csv"

LEDGER_COLUMNS = [
    "run_id",
    "timestamp",
    "git_sha",
    "config_hash",
    "chunk_strategy",
    "chunk_params",
    "embedder",
    "embed_backend",
    "embed_dim",
    "index_type",
    "ef_search",
    "retrieval_mode",
    "fusion_k",
    "reranker",
    "top_k",
    "generator",
    "prompt_version",
    "recall@1",
    "recall@5",
    "recall@10",
    "mrr@10",
    "ndcg@10",
    "faithfulness",
    "abstention_rate_in_domain",
    "abstention_rate_ood",
    "p50_ms",
    "p70_ms",
    "p95_ms",
    "p100_ms",
    "p50_embed_ms",
    "p50_search_ms",
    "p50_rerank_ms",
    "p50_ttft_ms",
    "index_build_s",
    "index_size_mb",
    "rss_mb",
    "notes",
]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(len(s) * pct / 100), len(s) - 1)]


def evaluate(strategy_name: str, strategy_kwargs: dict, top_k: int = 10) -> dict:
    heldout = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    built = build_index.build(strategy_name, strategy_kwargs)

    from vrag.index.embedder import E5Embedder

    embedder = E5Embedder()

    recalls_1, recalls_5, recalls_10, mrrs, ndcgs = [], [], [], [], []
    search_latencies_ms: list[float] = []
    embed_dim = 0

    for query_row in heldout:
        relevant_doc_ids = {p["passage_id"] for p in query_row["relevant_passages"]}
        query_vec = embedder.embed_queries([query_row["query"]])[0]
        embed_dim = len(query_vec)

        t0 = time.perf_counter()
        hits = built.dense.search(query_vec, k=top_k)
        search_latencies_ms.append((time.perf_counter() - t0) * 1000)

        retrieved_doc_ids = [
            built.chunk_lookup[chunk_id].doc_id
            for chunk_id, _score in hits
            if chunk_id in built.chunk_lookup
        ]

        recalls_1.append(recall_at_k(retrieved_doc_ids, relevant_doc_ids, k=1))
        recalls_5.append(recall_at_k(retrieved_doc_ids, relevant_doc_ids, k=5))
        recalls_10.append(recall_at_k(retrieved_doc_ids, relevant_doc_ids, k=10))
        mrrs.append(reciprocal_rank(retrieved_doc_ids, relevant_doc_ids, k=10))
        ndcgs.append(ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, k=10))

    chunk_lengths = [len(c.text.split()) for c in built.chunk_lookup.values()]

    return {
        "chunk_strategy": strategy_name,
        "chunk_params": json.dumps(strategy_kwargs),
        "embedder": "multilingual-e5-small",
        "embed_backend": "sentence-transformers",
        "embed_dim": embed_dim,
        "index_type": "HNSW32",
        "retrieval_mode": "dense",
        "top_k": top_k,
        "recall@1": statistics.mean(recalls_1),
        "recall@5": statistics.mean(recalls_5),
        "recall@10": statistics.mean(recalls_10),
        "mrr@10": statistics.mean(mrrs),
        "ndcg@10": statistics.mean(ndcgs),
        "p50_search_ms": _percentile(search_latencies_ms, 50),
        "index_build_s": built.build_seconds,
        "n_chunks": built.n_chunks,
        "p95_chunk_length_tokens": _percentile([float(x) for x in chunk_lengths], 95),
    }


def append_to_ledger(result: dict, strategy_kwargs: dict) -> None:
    row = dict.fromkeys(LEDGER_COLUMNS, "")
    row.update(
        {
            "run_id": f"{result['chunk_strategy']}_{int(time.time())}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_sha": _git_sha(),
            "config_hash": str(hash(json.dumps(strategy_kwargs, sort_keys=True))),
            "chunk_strategy": result["chunk_strategy"],
            "chunk_params": result["chunk_params"],
            "embedder": result["embedder"],
            "embed_backend": result["embed_backend"],
            "embed_dim": result["embed_dim"],
            "index_type": result["index_type"],
            "retrieval_mode": result["retrieval_mode"],
            "top_k": result["top_k"],
            "recall@1": f"{result['recall@1']:.4f}",
            "recall@5": f"{result['recall@5']:.4f}",
            "recall@10": f"{result['recall@10']:.4f}",
            "mrr@10": f"{result['mrr@10']:.4f}",
            "ndcg@10": f"{result['ndcg@10']:.4f}",
            "p50_search_ms": f"{result['p50_search_ms']:.3f}",
            "index_build_s": f"{result['index_build_s']:.2f}",
            "notes": (
                f"n_chunks={result['n_chunks']}, "
                f"p95_chunk_len={result['p95_chunk_length_tokens']:.0f}"
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
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--percentile-threshold", type=int, default=None)
    parser.add_argument("--child", type=int, default=None)
    parser.add_argument("--parent", type=int, default=None)
    args = parser.parse_args()

    kwargs = {}
    for key in ("size", "overlap", "window", "percentile_threshold", "child", "parent"):
        value = getattr(args, key)
        if value is not None:
            kwargs[key] = value

    result = evaluate(args.strategy, kwargs)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    append_to_ledger(result, kwargs)
    print(f"\nAppended to {LEDGER_PATH}")
