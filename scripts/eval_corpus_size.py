"""Corpus-size ablation: the quality half of R-024's investigation that was never run. R-024
(docs/DECISIONS_R.md) measured RSS-vs-chunk-count scaling (20k/50k/99,767 chunks) but explicitly
did NOT measure Recall/MRR/latency at those sizes -- it was a memory-only investigation, and running
a real quality ablation was escalated to the user rather than assumed. This script is that missing
half, run on request.

Methodology, stated plainly:
  - Reuses the real, already-computed FAISS vectors from the production index via
    faiss.Index.reconstruct() -- no re-embedding needed, so this is fast and uses the exact same
    embeddings A1-A4 and R-024 used, not a re-derived approximation.
  - Subsamples chunk_ids via `random.Random(SEED).sample()`, NOT a prefix slice -- R-024's original
    RSS-only script (`scripts/subsample_index.py` in a prior session) used `all_chunk_ids[:N]`,
    which risks ordering bias (the corpus was built by iterating documents; a prefix slice could
    systematically over/under-represent certain content). A random sample is the fair, standard
    choice here.
  - A held-out query whose relevant passage was NOT included in the subsample naturally scores 0 at
    every k for that query, via the exact same score_hits() every A1-A4 ablation already routes
    through -- no special-casing. This is also reported separately as "coverage" (the fraction of
    held-out queries whose answer chunk survived the reduction), since conflating "answer removed
    entirely" with "answer present but ranked poorly" would hide which effect dominates.
  - Search latency is measured the same way as scripts/eval_retrieval_mode.py: real dense.search()
    wall-clock over the full 500-query held-out set, P50/P100.
  - Index-only RSS uses the same in-process psutil sampling as scripts/audit_memory.py (ADR-006),
    run as an isolated subprocess per size so measurements don't pollute each other.

Usage: python scripts/eval_corpus_size.py --sizes 20000 50000 99767
(99767 = the full corpus; reuses the existing production index directly rather than rebuilding it.)
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import faiss

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vrag.chunking.base import Chunk  # noqa: E402
from vrag.index.dense import DenseIndex  # noqa: E402
from vrag.index.persistence import save_built_index  # noqa: E402
from vrag.index.sqlite_chunk_lookup import SQLiteChunkLookup  # noqa: E402
from vrag.retrieval.metrics import score_hits  # noqa: E402

INDEX_DIR = REPO_ROOT / "data" / "index" / "metadata_aware"
HELDOUT_PATH = REPO_ROOT / "eval" / "heldout_queries.json"
SEED = 42


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(len(s) * pct / 100), len(s) - 1)]


def build_subsample(n: int, out_dir: Path) -> None:
    """Reuses already-computed vectors via reconstruct() -- no re-embedding."""
    meta = json.loads((INDEX_DIR / "dense" / "meta.json").read_text(encoding="utf-8"))
    all_chunk_ids = meta["chunk_ids"]
    dim = meta["dim"]
    real_index = faiss.read_index(str(INDEX_DIR / "dense" / "faiss.index"))

    rng = random.Random(SEED)
    subset_ids = rng.sample(all_chunk_ids, n) if n < len(all_chunk_ids) else list(all_chunk_ids)
    id_to_pos = {cid: i for i, cid in enumerate(all_chunk_ids)}
    vectors = [real_index.reconstruct(id_to_pos[cid]).tolist() for cid in subset_ids]

    lookup = SQLiteChunkLookup(INDEX_DIR / "chunk_lookup.sqlite3")
    chunks: dict[str, Chunk] = {cid: lookup[cid] for cid in subset_ids}
    lookup.close()

    dense = DenseIndex(dim=dim)
    dense.add(subset_ids, vectors)

    # Sparse index unused by this script (dense-only, matching production, ADR-007) but
    # save_built_index() requires one -- an empty-but-valid SparseIndex is the honest choice
    # here (never queried), not a real cost since ADR-007 means it won't load for dense mode.
    from vrag.index.sparse import SparseIndex

    sparse = SparseIndex()
    sparse.build(subset_ids, [chunks[cid].text for cid in subset_ids])

    save_built_index(dense, sparse, chunks, out_dir)


def evaluate_quality_and_latency(index_dir: Path, n: int) -> dict:
    from vrag.chunking.base import Chunk as _Chunk
    from vrag.index.dense import DenseIndex as _DenseIndex

    heldout = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    dense = _DenseIndex.load(index_dir / "dense")
    # chunk_lookup.json only here (not the SQLite variant) -- this script measures FAISS's own
    # RSS in a separate isolated subprocess that loads only DenseIndex, so the chunk_lookup
    # format used here has no bearing on the memory numbers reported.
    raw_lookup = json.loads((index_dir / "chunk_lookup.json").read_text(encoding="utf-8"))
    lookup_dict = {cid: _Chunk(**data) for cid, data in raw_lookup.items()}

    from vrag.index.embedder import LiteE5Embedder

    embedder = LiteE5Embedder()

    # Precomputed once: every doc_id actually present anywhere in this subsampled corpus, so
    # "coverage" reflects whether the answer survived the reduction at all -- independent of
    # whether the search actually found it in the top-10 (that's what recall@k already measures).
    corpus_doc_ids = {chunk.doc_id for chunk in lookup_dict.values()}

    recalls_1, recalls_5, recalls_10, mrrs = [], [], [], []
    search_latencies_ms = []
    n_answer_present = 0

    for query_row in heldout:
        relevant_doc_ids = {p["passage_id"] for p in query_row["relevant_passages"]}
        query_vec = embedder.embed_queries([query_row["query"]])[0]

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

        if relevant_doc_ids & corpus_doc_ids:
            n_answer_present += 1

    faiss_file = index_dir / "dense" / "faiss.index"
    return {
        "n_chunks": n,
        "recall@1": statistics.mean(recalls_1),
        "recall@5": statistics.mean(recalls_5),
        "recall@10": statistics.mean(recalls_10),
        "mrr@10": statistics.mean(mrrs),
        "search_p50_ms": _percentile(search_latencies_ms, 50),
        "search_p100_ms": _percentile(search_latencies_ms, 100),
        "faiss_disk_size_bytes": faiss_file.stat().st_size if faiss_file.exists() else None,
        "n_queries": len(heldout),
        "coverage_answer_present_in_corpus": n_answer_present / len(heldout),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()

    work_dir = Path(args.work_dir) if args.work_dir else REPO_ROOT / "eval" / "corpus_size_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for n in args.sizes:
        print(f"\n=== n_chunks={n} ===")
        if n >= 99767:
            idx_dir = INDEX_DIR  # full corpus: reuse the real production index directly
        else:
            idx_dir = work_dir / f"idx_{n}"
            if not (idx_dir / "dense" / "faiss.index").exists():
                print(f"Building subsample (n={n})...")
                build_subsample(n, idx_dir)

        # RSS measured as an isolated subprocess so it doesn't pollute this process's own memory.
        rss_script = f"""
import sys, json, psutil
sys.path.insert(0, r"{REPO_ROOT / "src"}")
from vrag.index.dense import DenseIndex
d = DenseIndex.load(r"{idx_dir / "dense"}")
print(json.dumps({{"rss_bytes": psutil.Process().memory_info().rss, "n_vectors": len(d)}}))
"""
        proc = subprocess.run(
            [sys.executable, "-c", rss_script], capture_output=True, text=True, check=True
        )
        rss_info = json.loads(proc.stdout.strip().splitlines()[-1])

        result = evaluate_quality_and_latency(idx_dir, n)
        result["faiss_rss_bytes"] = rss_info["rss_bytes"]
        print(json.dumps(result, indent=2, ensure_ascii=False))
        results.append(result)

    print("\n=== Summary ===")
    print(
        f"{'n_chunks':>10} {'recall@1':>9} {'recall@5':>9} {'recall@10':>10} {'mrr@10':>8} "
        f"{'rss_mb':>8} {'disk_mb':>8} {'p50_ms':>8} {'p100_ms':>8}"
    )
    for r in results:
        print(
            f"{r['n_chunks']:>10} {r['recall@1']:>9.4f} {r['recall@5']:>9.4f} "
            f"{r['recall@10']:>10.4f} {r['mrr@10']:>8.4f} "
            f"{r['faiss_rss_bytes'] / 1e6:>8.1f} "
            f"{(r['faiss_disk_size_bytes'] or 0) / 1e6:>8.1f} "
            f"{r['search_p50_ms']:>8.3f} {r['search_p100_ms']:>8.3f}"
        )
