"""Phase 2 (docs/DECISIONS.md ADR-010): language-aware retrieval comparison.

For each multilingual index size (100k/150k/200k), evaluates three query-time strategies against
`eval/heldout_queries_multilingual.json`:

  A. no_filter   -- plain dense search, exactly today's production behaviour
  B. filter      -- search wide (k=100), keep only chunks whose `language` metadata matches the
                    query's real language, take the top 10 of what's left. Falls back to the
                    unfiltered top-k when the filtered set is empty (a language-filter must not
                    manufacture a zero-result failure merely because this particular corpus size
                    happens to have no same-language hit in the search window -- G3 downstream is
                    what should decide "not enough evidence", not the filter).
  C. boost       -- search wide (k=100), multiply same-language candidates' scores by BOOST_FACTOR
                    (documented, not exhaustively tuned -- this compares one representative boost
                    against no-filter/hard-filter, not a boost-strength sweep), re-sort, take
                    top 10.

This is entirely a standalone, offline evaluation -- no production code (src/vrag/retrieval/) is
imported or modified. Whether any of A/B/C gets wired into HybridRetriever is a decision made
AFTER seeing these real numbers, not before (docs/DECISIONS.md ADR-008's `language` param on
retrieve() exists for exactly this, still inert in production).

Usage: python scripts/eval_multilingual_retrieval.py --size 100k --size 150k --size 200k
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vrag.index.dense import DenseIndex  # noqa: E402
from vrag.index.embedder import E5Embedder  # noqa: E402
from vrag.retrieval.metrics import dedupe_doc_ids, score_hits  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
EVAL_DIR = REPO_ROOT / "eval"
HELDOUT_PATH = EVAL_DIR / "heldout_queries_multilingual.json"
WIDE_K = 100  # candidate pool width for filter/boost, before narrowing to top 10
FINAL_K = 10
BOOST_FACTOR = 1.10  # documented, single representative value -- see module docstring


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(len(s) * pct / 100), len(s) - 1)]


def _load_chunk_lookup(index_dir: Path) -> dict[str, dict]:
    raw = json.loads((index_dir / "chunk_lookup.json").read_text(encoding="utf-8"))
    return raw  # chunk_id -> {"chunk_id", "doc_id", "text", "metadata": {"language": ...}, ...}


def _mode_no_filter(hits: list[tuple[str, float]], lookup: dict, target_lang: str) -> list[tuple[str, float]]:
    return hits[:FINAL_K]


def _mode_filter(hits: list[tuple[str, float]], lookup: dict, target_lang: str) -> list[tuple[str, float]]:
    same_lang = [
        (cid, s) for cid, s in hits if lookup.get(cid, {}).get("metadata", {}).get("language") == target_lang
    ]
    if same_lang:
        return same_lang[:FINAL_K]
    return hits[:FINAL_K]  # documented fallback -- never manufacture a zero-result failure


def _mode_boost(hits: list[tuple[str, float]], lookup: dict, target_lang: str) -> list[tuple[str, float]]:
    boosted = [
        (cid, s * BOOST_FACTOR if lookup.get(cid, {}).get("metadata", {}).get("language") == target_lang else s)
        for cid, s in hits
    ]
    boosted.sort(key=lambda pair: pair[1], reverse=True)
    return boosted[:FINAL_K]


MODES = {"no_filter": _mode_no_filter, "filter": _mode_filter, "boost": _mode_boost}


def evaluate_size(size_name: str, query_vecs: dict[int, list[float]], heldout: list[dict]) -> dict:
    index_dir = DATA_DIR / "index" / f"multilingual_{size_name}"
    dense = DenseIndex.load(index_dir / "dense")
    lookup = _load_chunk_lookup(index_dir)

    fallback_triggered = 0
    per_mode: dict[str, dict] = {}
    for mode_name, mode_fn in MODES.items():
        recalls_1, recalls_5, recalls_10, mrrs, ndcgs = [], [], [], [], []
        latencies_ms = []
        for row in heldout:
            relevant_doc_ids = {p["passage_id"] for p in row["relevant_passages"]}
            target_lang = row["language"]
            vec = query_vecs[row["query_id"]]

            t0 = time.perf_counter()
            hits = dense.search(vec, k=WIDE_K)
            selected = mode_fn(hits, lookup, target_lang)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

            if mode_name == "filter":
                same_lang_present = any(
                    lookup.get(cid, {}).get("metadata", {}).get("language") == target_lang
                    for cid, _s in hits
                )
                if not same_lang_present:
                    fallback_triggered += 1

            chunk_to_doc_id = {cid: lookup[cid]["doc_id"] for cid, _s in selected if cid in lookup}
            scores = score_hits(selected, chunk_to_doc_id, relevant_doc_ids)
            recalls_1.append(scores["recall@1"])
            recalls_5.append(scores["recall@5"])
            recalls_10.append(scores["recall@10"])
            mrrs.append(scores["mrr@10"])
            ndcgs.append(scores["ndcg@10"])

        per_mode[mode_name] = {
            "recall@1": statistics.mean(recalls_1),
            "recall@5": statistics.mean(recalls_5),
            "recall@10": statistics.mean(recalls_10),
            "mrr@10": statistics.mean(mrrs),
            "ndcg@10": statistics.mean(ndcgs),
            "latency_p50_ms": _percentile(latencies_ms, 50),
            "latency_p95_ms": _percentile(latencies_ms, 95),
            "latency_p100_ms": _percentile(latencies_ms, 100),
        }

    per_mode["filter"]["fallback_rate"] = fallback_triggered / len(heldout) if heldout else 0.0

    dense_size_bytes = (index_dir / "dense" / "faiss.index").stat().st_size
    lookup_size_bytes = (index_dir / "chunk_lookup.json").stat().st_size

    return {
        "size_name": size_name,
        "n_chunks": len(dense),
        "n_heldout_queries": len(heldout),
        "modes": per_mode,
        "dense_index_disk_bytes": dense_size_bytes,
        "chunk_lookup_disk_bytes": lookup_size_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", action="append", required=True, choices=["100k", "150k", "200k"])
    args = parser.parse_args()

    heldout = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(heldout)} multilingual held-out queries")

    print("Embedding all held-out queries once (reused across every size/mode combination)...")
    embedder = E5Embedder()
    texts = [row["query"] for row in heldout]
    vecs = embedder.embed_queries(texts)
    query_vecs = {row["query_id"]: vec for row, vec in zip(heldout, vecs, strict=True)}

    results = {}
    for size_name in args.size:
        print(f"\n=== Evaluating {size_name} ===")
        result = evaluate_size(size_name, query_vecs, heldout)
        results[size_name] = result
        for mode_name, m in result["modes"].items():
            print(
                f"  [{mode_name:10s}] R@1={m['recall@1']:.4f} R@5={m['recall@5']:.4f} "
                f"R@10={m['recall@10']:.4f} MRR@10={m['mrr@10']:.4f} nDCG@10={m['ndcg@10']:.4f} "
                f"p50={m['latency_p50_ms']:.3f}ms p100={m['latency_p100_ms']:.3f}ms"
                + (f" fallback_rate={m['fallback_rate']:.3f}" if "fallback_rate" in m else "")
            )

    out_path = EVAL_DIR / "multilingual_retrieval_eval_results.json"
    existing = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    existing.update(results)
    out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote results -> {out_path}")


if __name__ == "__main__":
    main()
