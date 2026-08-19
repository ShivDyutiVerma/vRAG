"""Offline FAISS index-type ablation for R4 (docs/DECISIONS_R.md R-032 found FAISS+SQLite load
costs ~298MB, the largest of the two components that together exceed the real 512MB Docker limit).
Investigates whether a lower-memory FAISS index configuration can save the ~60MB R-032's gap
analysis implies is needed, while keeping retrieval quality and search latency acceptable.

Same methodology as scripts/eval_corpus_size.py (R-027): reuses the real, already-computed
production vectors via `faiss.Index.reconstruct()` -- no re-embedding, so every candidate searches
the exact same embeddings the production index does, not a re-derived approximation. Corpus size
(99,767 chunks), the embedding model, the tokenizer, and the 500-query held-out evaluation set
(eval/heldout_queries.json) are all unchanged -- the ONE variable under test is the FAISS index
type/construction config, one axis of change per candidate relative to the baseline (CLAUDE.md:
"Never change two variables in one experiment run"):

  baseline        IndexHNSWFlat, M=32, efConstruction=200, efSearch=64, METRIC_INNER_PRODUCT --
                  the exact production config (src/vrag/index/dense.py's own defaults).
  hnsw_m16        IndexHNSWFlat, M=16 (all else identical to baseline) -- fewer graph links.
  hnsw_sq8        IndexHNSWSQ, ScalarQuantizer.QT_8bit, M=32 (all else identical to baseline) --
                  vectors stored as 1 byte/dim instead of 4.
  hnsw_sqfp16     IndexHNSWSQ, ScalarQuantizer.QT_fp16, M=32 (all else identical to baseline) --
                  vectors stored as 2 bytes/dim (IEEE fp16) instead of 4.

For every candidate: resident RSS after load (isolated subprocess, matching R-020/R-027's
methodology so one candidate's measurement can't pollute another's) and RSS after running all 500
real search queries in that same subprocess (detects search-time growth); disk size; build
(train+add) wall time; Recall@1/5/10 and MRR@10 via the shared score_hits() path (identical to
every other ablation in this project, so dedupe_doc_ids is never skipped by accident); search
latency P50/P95/P100 over the real 500-query held-out set (matching R-014's efSearch curve and
R-028's ONNX settings sweep -- this project's established percentile convention for
component-level search latency, distinct from the full pipeline's P50/P70/P100 convention in
docs/LATENCY_BUDGET.md).

Query vectors are embedded ONCE (real LiteE5Embedder, real "query: " prefix) and reused across all
four candidates -- the embedder is unchanged by design, so precomputing eliminates embedder-run
variance as a confound between candidates, and is strictly more rigorous than re-embedding per
candidate, not less.

Usage: python scripts/eval_faiss_index_variants.py
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import faiss

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vrag.chunking.base import Chunk  # noqa: E402
from vrag.index.dense import (  # noqa: E402
    DEFAULT_EF_CONSTRUCTION,
    DEFAULT_EF_SEARCH,
    DEFAULT_M,
    DenseIndex,
)
from vrag.index.embedder import LiteE5Embedder  # noqa: E402
from vrag.index.persistence import save_built_index  # noqa: E402
from vrag.index.sparse import SparseIndex  # noqa: E402
from vrag.index.sqlite_chunk_lookup import SQLiteChunkLookup  # noqa: E402
from vrag.retrieval.metrics import score_hits  # noqa: E402

INDEX_DIR = REPO_ROOT / "data" / "index" / "metadata_aware"
HELDOUT_PATH = REPO_ROOT / "eval" / "heldout_queries.json"
WORK_DIR = REPO_ROOT / "eval" / "faiss_variant_tmp"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(len(s) * pct / 100), len(s) - 1)]


def load_real_vectors() -> tuple[list[str], int, list[list[float]]]:
    """Reconstructs every real production vector from the current baseline index -- no
    re-embedding. Returns (chunk_ids, dim, vectors) in the same order."""
    meta = json.loads((INDEX_DIR / "dense" / "meta.json").read_text(encoding="utf-8"))
    chunk_ids = meta["chunk_ids"]
    dim = meta["dim"]
    real_index = faiss.read_index(str(INDEX_DIR / "dense" / "faiss.index"))
    print(f"Reconstructing {len(chunk_ids)} real vectors (dim={dim}) from the production index...")
    vectors = [real_index.reconstruct(i).tolist() for i in range(len(chunk_ids))]
    return chunk_ids, dim, vectors


def build_candidate(
    name: str,
    chunk_ids: list[str],
    dim: int,
    vectors: list[list[float]],
    make_index: Callable[[int], faiss.Index],
    out_dir: Path,
) -> float:
    """`make_index(dim)` returns an untrained, empty faiss.Index of the candidate's type/config.
    Trains (if the index needs it) and adds all real vectors, times both, saves via the same
    DenseIndex/save_built_index path production/every other ablation script uses -- not a
    bespoke format. Returns build wall-time in seconds."""
    import numpy as np

    raw_index = make_index(dim)
    arr = np.asarray(vectors, dtype=np.float32)

    t0 = time.perf_counter()
    if not raw_index.is_trained:
        raw_index.train(arr)
    raw_index.add(arr)
    build_s = time.perf_counter() - t0

    dense = DenseIndex(dim=dim)
    dense._index = raw_index  # swap in the real candidate index; DenseIndex.__init__ always
    # creates a throwaway baseline IndexHNSWFlat, discarded here without ever being added to.
    dense._chunk_ids = list(chunk_ids)
    dense.set_ef_search(DEFAULT_EF_SEARCH)

    lookup = SQLiteChunkLookup(INDEX_DIR / "chunk_lookup.sqlite3")
    chunks: dict[str, Chunk] = {cid: lookup[cid] for cid in chunk_ids}
    lookup.close()

    # Sparse index unused by every dense-only eval in this project (ADR-007) but
    # save_built_index() requires one -- an empty-but-valid SparseIndex, never queried, same
    # convention as scripts/eval_corpus_size.py.
    sparse = SparseIndex()
    sparse.build(chunk_ids, [chunks[cid].text for cid in chunk_ids])

    out_dir.mkdir(parents=True, exist_ok=True)
    save_built_index(dense, sparse, chunks, out_dir)
    print(f"  [{name}] build={build_s:.1f}s ntotal={raw_index.ntotal}")
    return build_s


def measure_rss_and_search_growth(index_dir: Path, query_vectors: list[list[float]]) -> dict:
    """Isolated subprocess: load the candidate's DenseIndex, record RSS, run all 500 real
    searches, record RSS again -- detects whether search itself needs meaningfully more RAM on
    top of the resident index, not just whether the index loads under budget."""
    script = f"""
import json, sys, psutil
sys.path.insert(0, r"{REPO_ROOT / "src"}")
from vrag.index.dense import DenseIndex
proc = psutil.Process()
d = DenseIndex.load(r"{index_dir / "dense"}")
rss_after_load = proc.memory_info().rss
query_vectors = json.loads(sys.stdin.read())
for qv in query_vectors:
    d.search(qv, k=10)
rss_after_search = proc.memory_info().rss
print(json.dumps({{
    "rss_after_load_bytes": rss_after_load,
    "rss_after_search_bytes": rss_after_search,
    "n_vectors": len(d),
}}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps(query_vectors),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def evaluate_quality_and_latency(
    index_dir: Path, heldout: list[dict], query_vectors: list[list[float]]
) -> dict:
    dense = DenseIndex.load(index_dir / "dense")
    raw_lookup = json.loads((index_dir / "chunk_lookup.json").read_text(encoding="utf-8"))
    lookup_dict = {cid: Chunk(**data) for cid, data in raw_lookup.items()}

    recalls_1, recalls_5, recalls_10, mrrs = [], [], [], []
    search_latencies_ms = []

    for query_row, query_vec in zip(heldout, query_vectors, strict=True):
        relevant_doc_ids = {p["passage_id"] for p in query_row["relevant_passages"]}

        t0 = time.perf_counter()
        hits = dense.search(query_vec, k=10)
        search_latencies_ms.append((time.perf_counter() - t0) * 1000)

        chunk_to_doc_id = {}
        for chunk_id, _score in hits:
            chunk = lookup_dict.get(chunk_id)
            if chunk is not None:
                chunk_to_doc_id[chunk_id] = chunk.doc_id

        scores = score_hits(hits, chunk_to_doc_id, relevant_doc_ids)
        recalls_1.append(scores["recall@1"])
        recalls_5.append(scores["recall@5"])
        recalls_10.append(scores["recall@10"])
        mrrs.append(scores["mrr@10"])

    faiss_file = index_dir / "dense" / "faiss.index"
    return {
        "recall@1": statistics.mean(recalls_1),
        "recall@5": statistics.mean(recalls_5),
        "recall@10": statistics.mean(recalls_10),
        "mrr@10": statistics.mean(mrrs),
        "search_p50_ms": _percentile(search_latencies_ms, 50),
        "search_p95_ms": _percentile(search_latencies_ms, 95),
        "search_p100_ms": _percentile(search_latencies_ms, 100),
        "faiss_disk_size_bytes": faiss_file.stat().st_size if faiss_file.exists() else None,
        "n_queries": len(heldout),
    }


CANDIDATES: dict[str, Callable[[int], faiss.Index]] = {
    "baseline_hnsw32_flat_fp32": lambda dim: faiss.IndexHNSWFlat(
        dim, DEFAULT_M, faiss.METRIC_INNER_PRODUCT
    ),
    "hnsw16_flat_fp32": lambda dim: faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT),
    "hnsw32_sq8": lambda dim: faiss.IndexHNSWSQ(
        dim, faiss.ScalarQuantizer.QT_8bit, DEFAULT_M, faiss.METRIC_INNER_PRODUCT
    ),
    "hnsw32_sqfp16": lambda dim: faiss.IndexHNSWSQ(
        dim, faiss.ScalarQuantizer.QT_fp16, DEFAULT_M, faiss.METRIC_INNER_PRODUCT
    ),
}


def _set_ef_construction(
    make_index: Callable[[int], faiss.Index],
) -> Callable[[int], faiss.Index]:
    def wrapped(dim: int) -> faiss.Index:
        idx = make_index(dim)
        idx.hnsw.efConstruction = DEFAULT_EF_CONSTRUCTION
        return idx

    return wrapped


if __name__ == "__main__":
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    heldout = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    print(f"Held-out queries: {len(heldout)}")

    chunk_ids, dim, vectors = load_real_vectors()

    print("Embedding all held-out queries once (real LiteE5Embedder, shared across candidates)...")
    embedder = LiteE5Embedder()
    query_vectors = [embedder.embed_queries([q["query"]])[0] for q in heldout]

    results = []
    for name, make_index in CANDIDATES.items():
        print(f"\n=== {name} ===")
        idx_dir = WORK_DIR / name
        if not (idx_dir / "dense" / "faiss.index").exists():
            build_s = build_candidate(
                name, chunk_ids, dim, vectors, _set_ef_construction(make_index), idx_dir
            )
        else:
            build_s = None
            print(f"  [{name}] reusing existing build")

        mem = measure_rss_and_search_growth(idx_dir, query_vectors)
        quality = evaluate_quality_and_latency(idx_dir, heldout, query_vectors)

        result = {"name": name, "build_s": build_s, **mem, **quality}
        print(json.dumps(result, indent=2, ensure_ascii=False))
        results.append(result)

    print("\n=== Summary ===")
    header = (
        f"{'candidate':>26} {'rss_mb':>8} {'search_dmb':>10} {'disk_mb':>8} "
        f"{'recall@1':>9} {'recall@5':>9} {'recall@10':>10} {'mrr@10':>8} "
        f"{'p50_ms':>8} {'p95_ms':>8} {'p100_ms':>8}"
    )
    print(header)
    for r in results:
        search_growth_mb = (r["rss_after_search_bytes"] - r["rss_after_load_bytes"]) / 1e6
        print(
            f"{r['name']:>26} "
            f"{r['rss_after_load_bytes'] / 1e6:>8.1f} "
            f"{search_growth_mb:>10.2f} "
            f"{(r['faiss_disk_size_bytes'] or 0) / 1e6:>8.1f} "
            f"{r['recall@1']:>9.4f} {r['recall@5']:>9.4f} {r['recall@10']:>10.4f} "
            f"{r['mrr@10']:>8.4f} "
            f"{r['search_p50_ms']:>8.3f} {r['search_p95_ms']:>8.3f} {r['search_p100_ms']:>8.3f}"
        )

    out_path = WORK_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")
