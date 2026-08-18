"""Runtime memory audit, component by component -- requested before any architectural change or
Render plan upgrade. Every number here is measured, not estimated: each component loads in a fresh,
isolated subprocess (so one measurement's already-resident memory never pollutes another's), with
an in-process background thread sampling RSS via `psutil` throughout the load (and, for the
embedder/full-app cases, through a warmup call too) to capture true peak, not just the final value.

This replaces the earlier external-tasklist-polling methodology (docs/DECISIONS_R.md R-020) with
in-process sampling -- no PID-matching, no cross-tool-call latency, higher sampling resolution.

Usage:
  python scripts/audit_memory.py --component bare
  python scripts/audit_memory.py --component embedder
  python scripts/audit_memory.py --component faiss
  python scripts/audit_memory.py --component bm25
  python scripts/audit_memory.py --component metadata
  python scripts/audit_memory.py --component reranker
  python scripts/audit_memory.py --component full
Each prints one JSON line to stdout. scripts/run_memory_audit.sh (or the equivalent orchestration
in docs/DECISIONS.md) runs all seven as separate subprocesses and aggregates the JSON lines.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import threading
import time
from pathlib import Path

import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = REPO_ROOT / "data" / "index" / "metadata_aware"


class RSSSampler:
    """Background thread sampling this process's own RSS at high frequency. Tracks the running
    max so a transient allocation spike (e.g. mid-deserialisation) isn't missed just because the
    steady-state reading afterward is lower."""

    def __init__(self, interval_s: float = 0.02) -> None:
        self._proc = psutil.Process()
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak_rss = self._proc.memory_info().rss

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss
            if rss > self.peak_rss:
                self.peak_rss = rss
            time.sleep(self._interval_s)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join()
        return self.peak_rss


def current_rss() -> int:
    return psutil.Process().memory_info().rss


def measure_bare() -> dict:
    sampler = RSSSampler()
    sampler.start()
    baseline = current_rss()
    import fastapi  # noqa: F401
    from fastapi import FastAPI

    _app = FastAPI()
    after = current_rss()
    peak = sampler.stop()
    return {
        "component": "bare_python_fastapi",
        "rss_at_interpreter_start_bytes": baseline,
        "rss_after_load_bytes": after,
        "peak_rss_during_load_bytes": peak,
    }


def measure_embedder() -> dict:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sampler = RSSSampler()
    sampler.start()
    baseline = current_rss()
    from vrag.index.embedder import LiteE5Embedder

    embedder = LiteE5Embedder()
    embedder.embed_queries(["वार्म-अप क्वेरी"])  # forces the lazy session/tokenizer load
    after = current_rss()
    peak = sampler.stop()
    torch_loaded = "torch" in sys.modules
    transformers_loaded = "transformers" in sys.modules
    return {
        "component": "embedder_only_(LiteE5Embedder,_production_choice)",
        "rss_at_interpreter_start_bytes": baseline,
        "rss_after_load_bytes": after,
        "peak_rss_during_load_bytes": peak,
        "torch_in_sys_modules": torch_loaded,
        "transformers_in_sys_modules": transformers_loaded,
        "embedding_dim": len(embedder.embed_queries(["x"])[0]),
    }


def measure_faiss() -> dict:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sampler = RSSSampler()
    sampler.start()
    baseline = current_rss()
    from vrag.index.dense import DenseIndex

    dense = DenseIndex.load(INDEX_DIR / "dense")
    after = current_rss()
    peak = sampler.stop()
    faiss_file = INDEX_DIR / "dense" / "faiss.index"
    return {
        "component": "faiss_index_only",
        "rss_at_interpreter_start_bytes": baseline,
        "rss_after_load_bytes": after,
        "peak_rss_during_load_bytes": peak,
        "n_vectors": len(dense),
        "dim": dense.dim,
        "faiss_index_type": type(dense._index).__name__,
        "faiss_hnsw_ef_search": (
            dense._index.hnsw.efSearch if hasattr(dense._index, "hnsw") else None
        ),
        "faiss_index_file_size_bytes": faiss_file.stat().st_size if faiss_file.exists() else None,
        "vector_dtype": "float32 (DenseIndex.add() casts via np.asarray dtype=np.float32)",
    }


def measure_bm25() -> dict:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sampler = RSSSampler()
    sampler.start()
    baseline = current_rss()
    from vrag.index.sparse import SparseIndex

    sparse = SparseIndex.load(INDEX_DIR / "sparse")
    after = current_rss()
    peak = sampler.stop()
    bm25_dir = INDEX_DIR / "sparse" / "bm25s"
    on_disk_bytes = sum(f.stat().st_size for f in bm25_dir.glob("*")) if bm25_dir.exists() else None
    return {
        "component": "bm25_index_only",
        "rss_at_interpreter_start_bytes": baseline,
        "rss_after_load_bytes": after,
        "peak_rss_during_load_bytes": peak,
        "n_chunks": len(sparse),
        "on_disk_bytes": on_disk_bytes,
        "loaded_with_corpus_text": False,  # bm25s.BM25.load(..., load_corpus=False) -- verified
        "note": "SparseIndex.load() explicitly passes load_corpus=False -- BM25 never holds raw "
        "chunk text in memory, only the tokenised term-frequency index + vocabulary",
    }


def measure_metadata() -> dict:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sampler = RSSSampler()
    sampler.start()
    baseline = current_rss()
    from vrag.index.sqlite_chunk_lookup import SQLiteChunkLookup

    lookup = SQLiteChunkLookup(INDEX_DIR / "chunk_lookup.sqlite3")
    after = current_rss()
    peak = sampler.stop()
    sqlite_file = INDEX_DIR / "chunk_lookup.sqlite3"
    json_file = INDEX_DIR / "chunk_lookup.json"
    return {
        "component": "metadata_corpus_only_(SQLiteChunkLookup,_production_choice)",
        "rss_at_interpreter_start_bytes": baseline,
        "rss_after_load_bytes": after,
        "peak_rss_during_load_bytes": peak,
        "n_chunks": len(lookup),
        "sqlite_file_size_bytes": sqlite_file.stat().st_size if sqlite_file.exists() else None,
        "json_alternative_file_size_bytes": (
            json_file.stat().st_size if json_file.exists() else None
        ),
        "note": "Only chunk_id->doc_id kept fully in memory; full Chunk text (incl. corpus text) "
        "is fetched lazily per-row from SQLite, not held resident",
    }


def measure_reranker() -> dict:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sampler = RSSSampler()
    sampler.start()
    baseline = current_rss()
    from vrag.retrieval.rerank import FlashRankReranker

    reranker = FlashRankReranker()
    dummy_candidates = [(f"c{i}", f"dummy passage text number {i}") for i in range(5)]
    reranker.rerank("dummy query", dummy_candidates, k=3)  # forces lazy model load
    after = current_rss()
    peak = sampler.stop()
    return {
        "component": "reranker_only_(FlashRankReranker_--_NOT_in_production,_A4_chose_none)",
        "rss_at_interpreter_start_bytes": baseline,
        "rss_after_load_bytes": after,
        "peak_rss_during_load_bytes": peak,
    }


def measure_full() -> dict:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import asyncio

    sampler = RSSSampler()
    sampler.start()
    baseline = current_rss()
    from vrag.retrieval.interface import retrieve

    async def _run() -> list:
        return await retrieve("वार्म-अप क्वेरी जो असली रिट्रीवल चलाती है", k=5)

    results = asyncio.run(_run())
    after_first_query = current_rss()
    gc.collect()
    after_gc = current_rss()
    peak_during_load_and_first_query = sampler.stop()

    # Second sampler for a genuine "steady-state after warmup" reading, several more queries in.
    sampler2 = RSSSampler()
    sampler2.start()
    for q in ["दूसरी क्वेरी", "तीसरी क्वेरी", "चौथी क्वेरी"]:
        asyncio.run(retrieve(q, k=5))
    steady_state = current_rss()
    peak_after_warmup = sampler2.stop()

    return {
        "component": "full_application (production wiring: LiteE5Embedder + "
        "SQLiteChunkLookup + FAISS + BM25-loaded-but-unused-in-dense-mode)",
        "rss_at_interpreter_start_bytes": baseline,
        "rss_after_first_real_query_bytes": after_first_query,
        "rss_after_gc_collect_bytes": after_gc,
        "peak_rss_during_startup_and_first_query_bytes": peak_during_load_and_first_query,
        "rss_steady_state_after_4_queries_bytes": steady_state,
        "peak_rss_during_warmup_queries_bytes": peak_after_warmup,
        "n_results_first_query": len(results),
        "retrieval_mode": "dense (A3 winner, R-010) -- BM25/sparse index is None on this path "
        "since ADR-006/ADR-007: load_built_index_lean() only loads it for sparse/hybrid modes",
    }


COMPONENTS = {
    "bare": measure_bare,
    "embedder": measure_embedder,
    "faiss": measure_faiss,
    "bm25": measure_bm25,
    "metadata": measure_metadata,
    "reranker": measure_reranker,
    "full": measure_full,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", required=True, choices=sorted(COMPONENTS))
    args = parser.parse_args()
    result = COMPONENTS[args.component]()
    print(json.dumps(result, ensure_ascii=False))
