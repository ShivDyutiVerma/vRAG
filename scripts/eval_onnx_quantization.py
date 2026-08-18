"""ONNX int8 embedder validation (docs/BUILD_PLAN.md P6 task 5, docs/DECISIONS_R.md R-019).

Tests the realistic production shape, not a full re-index: passages are embedded once, offline,
where build time doesn't matter (already FP32, already built — `data/index/metadata_aware/`).
Query embedding is the actual hot-path cost (docs/AGENT_BUILD_SPEC.md §3.2's 200ms budget), so this
quantizes only the *query-time* embedder and searches the existing FP32-built index with int8-
embedded queries — cross-precision compatibility is exactly what would ship, not a hypothetical
same-precision rebuild.

Two things measured together, per docs/TECH_MENU.md §A rule 5 ("latency and quality in the same
run"):
  1. Quality: Recall@1/5/10/MRR/nDCG on the frozen 500-query held-out set, int8 queries vs. the
     existing FP32-query baseline (docs/EVAL_RESULTS.md §2, Recall@5=0.653).
  2. Latency: real single-query CPU embedding time, FP32-on-CPU vs. int8-ONNX-on-CPU — both forced
     onto CPU for a fair comparison (this dev machine's GPU is not the production target;
     CLAUDE.md's hot-path invariant is explicit that int8 quantization is a CPU-only optimisation).

Usage: python scripts/eval_onnx_quantization.py --index-dir data/index/metadata_aware
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import build_index

from vrag.index.embedder import ONNXE5Embedder, format_query
from vrag.retrieval.metrics import score_hits

REPO_ROOT = Path(__file__).resolve().parent.parent
HELDOUT_PATH = REPO_ROOT / "eval" / "heldout_queries.json"


def get_fp32_cpu_embedder():
    """E5Embedder forced onto CPU -- this dev machine's GPU is not the production target, and
    comparing GPU-FP32 vs CPU-int8 would misrepresent the quantisation's actual speedup."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")

    class _CPUEmbedder:
        def embed_queries(self, texts: list[str]) -> list[list[float]]:
            prefixed = [format_query(t) for t in texts]
            return model.encode(prefixed, normalize_embeddings=True).tolist()

    return _CPUEmbedder()


def evaluate_quality(
    embedder, dense, chunk_lookup: dict, heldout: list[dict], top_k: int = 10
) -> dict:
    chunk_to_doc_id = {chunk_id: c.doc_id for chunk_id, c in chunk_lookup.items()}
    recalls_1, recalls_5, recalls_10, mrrs, ndcgs = [], [], [], [], []

    for query_row in heldout:
        relevant_doc_ids = {p["passage_id"] for p in query_row["relevant_passages"]}
        query_vec = embedder.embed_queries([query_row["query"]])[0]
        hits = dense.search(query_vec, k=top_k)
        scores = score_hits(hits, chunk_to_doc_id, relevant_doc_ids)
        recalls_1.append(scores["recall@1"])
        recalls_5.append(scores["recall@5"])
        recalls_10.append(scores["recall@10"])
        mrrs.append(scores["mrr@10"])
        ndcgs.append(scores["ndcg@10"])

    return {
        "recall@1": statistics.mean(recalls_1),
        "recall@5": statistics.mean(recalls_5),
        "recall@10": statistics.mean(recalls_10),
        "mrr@10": statistics.mean(mrrs),
        "ndcg@10": statistics.mean(ndcgs),
    }


def measure_latency(embedder, queries: list[str], n_warmup: int = 5) -> dict:
    for q in queries[:n_warmup]:
        embedder.embed_queries([q])

    latencies_ms = []
    for q in queries:
        t0 = time.perf_counter()
        embedder.embed_queries([q])
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    latencies_ms.sort()
    n = len(latencies_ms)
    return {
        "p50_ms": latencies_ms[n // 2],
        "p95_ms": latencies_ms[int(n * 0.95)],
        "p100_ms": latencies_ms[-1],
        "mean_ms": statistics.mean(latencies_ms),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default="data/index/metadata_aware")
    parser.add_argument("--n-latency-queries", type=int, default=100)
    args = parser.parse_args()

    heldout = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    dense, _sparse, chunk_lookup = build_index.load(args.index_dir)
    latency_queries = [q["query"] for q in heldout[: args.n_latency_queries]]

    print("=== Quality: int8 ONNX queries against the existing FP32-built index ===")
    onnx_embedder = ONNXE5Embedder()
    onnx_quality = evaluate_quality(onnx_embedder, dense, chunk_lookup, heldout)
    print(json.dumps(onnx_quality, indent=2))

    print("\n=== Latency: FP32-on-CPU vs. int8-ONNX-on-CPU (single-query, hot-path shape) ===")
    fp32_cpu_embedder = get_fp32_cpu_embedder()
    print(f"Warming up + timing {len(latency_queries)} single-query FP32-CPU embed calls...")
    fp32_latency = measure_latency(fp32_cpu_embedder, latency_queries)
    print(f"FP32-CPU: {json.dumps(fp32_latency, indent=2)}")

    print(f"Warming up + timing {len(latency_queries)} single-query int8-ONNX embed calls...")
    onnx_latency = measure_latency(onnx_embedder, latency_queries)
    print(f"int8-ONNX: {json.dumps(onnx_latency, indent=2)}")

    print("\n=== Summary ===")
    print(
        f"Quality (int8 query vs FP32-baseline Recall@5=0.653): "
        f"{onnx_quality['recall@5']:.4f} ({(onnx_quality['recall@5'] - 0.653) * 100:+.2f}pp)"
    )
    speedup = fp32_latency["p50_ms"] / onnx_latency["p50_ms"] if onnx_latency["p50_ms"] > 0 else 0
    print(
        f"Latency: FP32-CPU p50={fp32_latency['p50_ms']:.2f}ms, "
        f"int8-ONNX p50={onnx_latency['p50_ms']:.2f}ms ({speedup:.1f}x speedup)"
    )
