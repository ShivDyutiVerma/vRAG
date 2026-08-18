"""efSearch recall-vs-latency curve (docs/BUILD_PLAN.md P3 task 9, docs/TECH_MENU.md §S6 —
"Tune efSearch, don't guess it. Produce a recall-vs-latency curve and pick the operating point from
the data"). Chunking/embedder/retrieval mode held at the A1-A3 winners (metadata_aware /
multilingual-e5-small / dense-only). Sweeps efSearch in {16, 32, 64, 128, 256} against the frozen
500-query held-out set, reusing the persisted index — `DenseIndex.set_ef_search()` mutates the
already-built HNSW graph's search-time parameter in place, so this needs zero rebuilds and zero
re-embedding (query vectors are computed once and reused across all five efSearch values).

Usage: python scripts/eval_efsearch.py --index-dir data/index/metadata_aware
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import build_index
import matplotlib
from eval_chunking import LEDGER_COLUMNS, LEDGER_PATH, _git_sha, _percentile

from vrag.index.embedder import E5Embedder
from vrag.retrieval.metrics import score_hits

matplotlib.use("Agg")  # headless backend -- must be set before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
HELDOUT_PATH = REPO_ROOT / "eval" / "heldout_queries.json"
ASSETS_DIR = REPO_ROOT / "docs" / "assets"

CHUNK_STRATEGY = "metadata_aware"
EMBEDDER_NAME = E5Embedder.name
EF_SEARCH_VALUES = [16, 32, 64, 128, 256]


def evaluate_ef_search(
    ef_search: int,
    dense,
    chunk_lookup: dict,
    query_vecs: list[list[float]],
    heldout: list[dict],
    top_k: int = 10,
) -> dict:
    dense.set_ef_search(ef_search)
    chunk_to_doc_id = {chunk_id: c.doc_id for chunk_id, c in chunk_lookup.items()}

    recalls_1, recalls_5, recalls_10, mrrs, ndcgs = [], [], [], [], []
    search_latencies_ms: list[float] = []

    for query_row, query_vec in zip(heldout, query_vecs, strict=True):
        relevant_doc_ids = {p["passage_id"] for p in query_row["relevant_passages"]}

        t0 = time.perf_counter()
        hits = dense.search(query_vec, k=top_k)
        search_latencies_ms.append((time.perf_counter() - t0) * 1000)

        scores = score_hits(hits, chunk_to_doc_id, relevant_doc_ids)
        recalls_1.append(scores["recall@1"])
        recalls_5.append(scores["recall@5"])
        recalls_10.append(scores["recall@10"])
        mrrs.append(scores["mrr@10"])
        ndcgs.append(scores["ndcg@10"])

    return {
        "ef_search": ef_search,
        "chunk_strategy": CHUNK_STRATEGY,
        "embedder": EMBEDDER_NAME,
        "recall@1": statistics.mean(recalls_1),
        "recall@5": statistics.mean(recalls_5),
        "recall@10": statistics.mean(recalls_10),
        "mrr@10": statistics.mean(mrrs),
        "ndcg@10": statistics.mean(ndcgs),
        "p50_search_ms": _percentile(search_latencies_ms, 50),
        "p95_search_ms": _percentile(search_latencies_ms, 95),
        "p100_search_ms": _percentile(search_latencies_ms, 100),
    }


def append_to_ledger(result: dict) -> None:
    row = dict.fromkeys(LEDGER_COLUMNS, "")
    row.update(
        {
            "run_id": f"efsearch_{result['ef_search']}_{int(time.time())}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_sha": _git_sha(),
            "config_hash": str(hash(result["ef_search"])),
            "chunk_strategy": result["chunk_strategy"],
            "embedder": result["embedder"],
            "retrieval_mode": "dense",
            "ef_search": result["ef_search"],
            "top_k": 10,
            "recall@1": f"{result['recall@1']:.4f}",
            "recall@5": f"{result['recall@5']:.4f}",
            "recall@10": f"{result['recall@10']:.4f}",
            "mrr@10": f"{result['mrr@10']:.4f}",
            "ndcg@10": f"{result['ndcg@10']:.4f}",
            "p50_search_ms": f"{result['p50_search_ms']:.3f}",
            "notes": (
                f"efSearch sweep, p95_search_ms={result['p95_search_ms']:.3f}, "
                f"p100_search_ms={result['p100_search_ms']:.3f}"
            ),
        }
    )
    file_exists = LEDGER_PATH.exists() and LEDGER_PATH.stat().st_size > 0
    with LEDGER_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def plot_curve(results: list[dict], out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 5))

    ef_values = [r["ef_search"] for r in results]
    recalls = [r["recall@5"] for r in results]
    latencies = [r["p50_search_ms"] for r in results]

    color1 = "tab:blue"
    ax1.set_xlabel("efSearch")
    ax1.set_ylabel("Recall@5", color=color1)
    ax1.plot(ef_values, recalls, marker="o", color=color1, label="Recall@5")
    ax1.set_xscale("log", base=2)
    ax1.tick_params(axis="y", labelcolor=color1)
    for x, y in zip(ef_values, recalls, strict=True):
        ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("p50 search latency (ms)", color=color2)
    ax2.plot(ef_values, latencies, marker="s", color=color2, label="p50 latency")
    ax2.tick_params(axis="y", labelcolor=color2)
    for x, y in zip(ef_values, latencies, strict=True):
        ax2.annotate(f"{y:.3f}ms", (x, y), textcoords="offset points", xytext=(0, -14), fontsize=8)

    plt.title("efSearch: Recall@5 vs. p50 search latency (dense-only, metadata_aware/e5-small)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default="data/index/metadata_aware")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    heldout = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    dense, _sparse, chunk_lookup = build_index.load(args.index_dir)
    embedder = E5Embedder()

    print(f"Embedding {len(heldout)} held-out queries once, reused across every efSearch value...")
    query_vecs = embedder.embed_queries([q["query"] for q in heldout])

    results = []
    for ef_search in EF_SEARCH_VALUES:
        print(f"\n--- evaluating efSearch={ef_search} ---")
        result = evaluate_ef_search(
            ef_search, dense, chunk_lookup, query_vecs, heldout, top_k=args.top_k
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        append_to_ledger(result)
        results.append(result)

    print(f"\nAppended {len(results)} rows to {LEDGER_PATH}")

    curve_path = ASSETS_DIR / "efsearch_curve.png"
    plot_curve(results, curve_path)
    print(f"Saved curve to {curve_path}")

    print("\n=== summary ===")
    for r in results:
        print(
            f"ef={r['ef_search']:4d}  recall@5={r['recall@5']:.4f}  "
            f"p50={r['p50_search_ms']:.3f}ms  p95={r['p95_search_ms']:.3f}ms"
        )
