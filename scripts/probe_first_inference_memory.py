"""Diagnostic-only, not wired into production or the Dockerfile. Isolates the transient memory
spike behind the real Docker `-m 512m` OOM found in docs/DECISIONS_R.md R-031 (container survives
startup at ~277MiB but is killed on the first real `/ask` query, reproduced 2/2).

Instruments the REAL production objects (LiteE5Embedder, DenseIndex, SQLiteChunkLookup, the
`_get_real_retriever()` singleton) at fine-grained boundaries -- calls their existing public/private
methods directly with checkpoints in between, rather than reimplementing any logic. Where a
boundary falls *inside* one existing method (e.g. "immediately before" vs. "immediately after"
ONNX inference, both inside `LiteE5Embedder._embed`), this script inlines that method's body
verbatim (same lines, same order) so the checkpoint can sit between them -- see `_run_embed_inline`
docstring for the exact provenance of each copied line.

Two measurement layers, combined at every checkpoint:
  - In-process RSS sampling via a ~1ms-interval background thread (`psutil`) -- resolves the
    150ms-blind-spot problem the external `docker stats` polling had in R-031 (ADR-006 already
    documented this same class of sampling-resolution caveat for BM25 loading).
  - `tracemalloc` for Python-domain allocations (this also covers numpy array buffers -- numpy
    hooks `PyTraceMalloc_Track`/`Untrack` on its own allocator since ~1.15, so arrays we allocate
    in the pooling/normalisation math below ARE visible to tracemalloc). ONNX Runtime's and FAISS's
    internal C++ allocations (session arenas, HNSW candidate buffers) are never visible to
    tracemalloc -- (RSS delta) - (tracemalloc current delta) between two checkpoints is reported as
    "native_unexplained_mb", isolating exactly that.

Modes:
  full        -- the real production path end to end: tokenize -> ONNX -> normalize -> FAISS ->
                 SQLite lookup -> retrieve(), reusing the actual module-level retriever singleton
                 (avoids double-loading the index/embedder, which would contaminate the baseline).
  embed_only  -- a fresh process, embedder only, no index/FAISS loaded at all.
  faiss_only  -- a fresh process, index/FAISS/SQLite only -- deliberately does NOT import
                 onnxruntime/sentencepiece/LiteE5Embedder, so this process's own baseline RSS
                 carries zero embedder cost. Uses a fixed-seed random unit vector (matching the
                 index's real dim) standing in for a query embedding.

Every checkpoint is written to a JSON-lines log, flushed and fsync'd immediately -- if the OS
OOM-kills this process mid-run (expected under a tight limit), everything up to the last completed
checkpoint survives on disk for inspection after the fact.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import json
import os
import sys
import threading
import time
import tracemalloc
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import psutil  # noqa: E402

# Real held-out queries, drawn verbatim from eval/heldout_queries.json (not copied into the
# Docker image, so hardcoded here) -- real Hindi text at realistic lengths, not synthetic filler.
REAL_QUERIES = [
    "सिरियस एक्स.एम.वी.",
    "पूर्व कसरत कैसे करें",
    "परिभाषित अंतर-एजेंसी समिति",
    "लॉन्गमोंट, को में औसत आवागमन समय",
    "देवी का आरोप क्या है",
    "तीन प्रकार के एंथ्रेक्स संक्रमण",
    "दोहरी काँच वाली खिड़कियों पर संघनन का कारण क्या है",
    "मानक ब्रेसिज़ कितने होते हैं",
]

_PROC = psutil.Process(os.getpid())
_samples: list[tuple[float, int]] = []
_sampling = True


def _sampler_loop() -> None:
    while _sampling:
        with contextlib.suppress(Exception):  # diagnostic thread, never let it kill the run
            _samples.append((time.perf_counter(), _PROC.memory_info().rss))
        time.sleep(0.001)


def start_sampler() -> threading.Thread:
    t = threading.Thread(target=_sampler_loop, daemon=True, name="rss-sampler")
    t.start()
    return t


class Checkpointer:
    """Stateful so each `checkpoint()` call reports deltas since the *previous* checkpoint, not
    since process start -- that's what makes "peak during this one stage" meaningful."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_ts: float | None = None
        self._last_rss: int | None = None
        self._last_py_current: int | None = None
        self.records: list[dict[str, Any]] = []

    def checkpoint(self, label: str, query_idx: int | None = None) -> dict[str, Any]:
        now = time.perf_counter()
        rss = _PROC.memory_info().rss
        py_current, py_peak_since_last = tracemalloc.get_traced_memory()

        window_peak = rss
        if self._last_ts is not None:
            window = [s for (t, s) in _samples if t >= self._last_ts]
            if window:
                window_peak = max(window)

        rss_delta = rss - self._last_rss if self._last_rss is not None else 0
        py_delta = py_current - self._last_py_current if self._last_py_current is not None else 0
        native_unexplained = rss_delta - py_delta

        record = {
            "label": label,
            "query_idx": query_idx,
            "t_perf": now,
            "rss_mb": round(rss / 1e6, 3),
            "rss_delta_mb": round(rss_delta / 1e6, 3),
            "window_peak_rss_mb": round(window_peak / 1e6, 3),
            "py_current_mb": round(py_current / 1e6, 3),
            "py_peak_since_last_mb": round(py_peak_since_last / 1e6, 3),
            "py_delta_mb": round(py_delta / 1e6, 3),
            "native_unexplained_mb": round(native_unexplained / 1e6, 3),
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        qtag = f" q{query_idx}" if query_idx is not None else ""
        print(
            f"[{label}{qtag}] rss={record['rss_mb']:.1f}MB (d{record['rss_delta_mb']:+.1f}) "
            f"window_peak={record['window_peak_rss_mb']:.1f}MB "
            f"py={record['py_current_mb']:.1f}MB "
            f"(peak_since_last={record['py_peak_since_last_mb']:.1f}) "
            f"native_unexplained_d={record['native_unexplained_mb']:+.1f}MB",
            flush=True,
        )

        tracemalloc.reset_peak()
        self._last_ts = now
        self._last_rss = rss
        self._last_py_current = py_current
        self.records.append(record)
        return record


def _run_embed_inline(
    embedder: Any, cp: Checkpointer, text: str, query_idx: int
) -> list[float]:
    """Inlines `LiteE5Embedder._embed`'s body verbatim (src/vrag/index/embedder.py) so a
    checkpoint can sit between tokenize / before-ONNX / after-ONNX / after-normalize -- the real
    `_embed()` method does all of this in one opaque call. Every line below is copied from that
    method unmodified; nothing here changes what gets computed."""
    from vrag.index.embedder import format_query

    session, _sp = embedder._ensure_loaded()

    prefixed = [format_query(text)]
    input_ids, attention_mask = embedder._tokenize_batch(prefixed)
    cp.checkpoint("03_after_tokenize_query", query_idx)

    token_type_ids = np.zeros_like(input_ids)
    cp.checkpoint("04_before_onnx_inference", query_idx)

    (last_hidden_state,) = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )
    # session.run() is synchronous and blocking -- the background sampler was running throughout,
    # so this checkpoint's window_peak_rss_mb IS "peak during ONNX inference" (boundary 5), and the
    # rss/py figures below are boundary 6 ("immediately after").
    cp.checkpoint("06_after_onnx_inference", query_idx)

    mask = attention_mask[..., None].astype(np.float32)
    mean_pooled = (last_hidden_state * mask).sum(axis=1) / mask.sum(axis=1)
    normalized = mean_pooled / np.linalg.norm(mean_pooled, axis=1, keepdims=True)
    vector = normalized.tolist()[0]
    cp.checkpoint("07_after_normalization", query_idx)
    return vector


def run_full(log_path: Path, num_queries: int) -> None:
    tracemalloc.start()
    start_sampler()
    cp = Checkpointer(log_path)
    cp.checkpoint("01_process_startup")

    import vrag.retrieval.interface as interface

    # Real module-level singleton -- same object production uses. Deliberately NOT constructing a
    # second LiteE5Embedder/index here: that would double-load ~277MB and contaminate the baseline
    # this script is trying to measure.
    retriever = interface._get_real_retriever()
    if retriever is None:
        raise RuntimeError(
            "Real retriever failed to load -- check that the retrieval-lean deps and both "
            "release assets (index-metadata_aware-v2, embedder-lite-onnx-v2) are present."
        )
    # `_get_real_retriever()` constructs `LiteE5Embedder()` but does NOT call `_ensure_loaded()`
    # on it -- that's lazy, deferred to first real use (src/vrag/index/embedder.py). This
    # checkpoint is therefore FAISS + SQLite-lookup only; the ONNX session + SentencePieceProcessor
    # do not exist yet. Kept as a separate, explicit step (rather than folding it into the first
    # tokenize call below) so this mode stays directly comparable to embed_only's own checkpoint 02
    # -- an earlier version of this script conflated the two and produced a mislabeled result.
    cp.checkpoint("02a_faiss_and_sqlite_loaded_embedder_not_yet")

    embedder = retriever._embedder
    dense = retriever._dense
    chunk_lookup = retriever._chunk_lookup

    embedder._ensure_loaded()  # first-ever construction of the ONNX session + SentencePiece proc
    cp.checkpoint("02b_onnx_session_and_tokenizer_loaded")

    for i in range(num_queries):
        text = REAL_QUERIES[i % len(REAL_QUERIES)]
        vector = _run_embed_inline(embedder, cp, text, i)

        cp.checkpoint("08_before_faiss_search", i)
        hits = dense.search(vector, k=5)
        # DenseIndex.search() is synchronous/blocking (faiss releases the GIL internally), so this
        # checkpoint's window_peak_rss_mb is "peak during FAISS search" (boundary 9); rss/py below
        # is boundary 10 ("immediately after").
        cp.checkpoint("10_after_faiss_search", i)

        for chunk_id, _score in hits:
            chunk_lookup.get(chunk_id)
        cp.checkpoint("11_after_sqlite_lookup", i)

        # Boundary 12: the real production R/P seam, same singleton, no double-load. Exercises
        # retrieve()'s own embed+search+lookup a second time for this query (interface.retrieve()
        # doesn't reuse the vector we already computed above) -- accepted, since fidelity to the
        # real call path matters more here than avoiding one extra cheap round trip.
        asyncio.run(interface.retrieve(text, k=5))
        cp.checkpoint("12_after_retrieve_roundtrip", i)

        gc.collect()
        cp.checkpoint("13_after_gc_collect", i)

    _sampling_stop()
    _print_summary(cp.records, "full")


def run_embed_only(log_path: Path, num_queries: int) -> None:
    tracemalloc.start()
    start_sampler()
    cp = Checkpointer(log_path)
    cp.checkpoint("01_process_startup")

    from vrag.index.embedder import LiteE5Embedder

    embedder = LiteE5Embedder()
    embedder._ensure_loaded()  # boundary 2: tokenizer + ONNX session, nothing else resident
    cp.checkpoint("02_embedder_loaded_no_index")

    for i in range(num_queries):
        text = REAL_QUERIES[i % len(REAL_QUERIES)]
        _run_embed_inline(embedder, cp, text, i)
        gc.collect()
        cp.checkpoint("13_after_gc_collect", i)

    _sampling_stop()
    _print_summary(cp.records, "embed_only")


def run_faiss_only(log_path: Path, num_queries: int, index_dir: Path) -> None:
    """Deliberately does not import onnxruntime/sentencepiece/LiteE5Embedder anywhere in this
    process -- this process's own baseline RSS should carry zero embedder cost, isolating FAISS +
    SQLite lookup behavior on its own."""
    tracemalloc.start()
    start_sampler()
    cp = Checkpointer(log_path)
    cp.checkpoint("01_process_startup")

    from vrag.index.persistence import load_built_index_lean

    dense, _sparse, chunk_lookup = load_built_index_lean(index_dir, retrieval_mode="dense")
    cp.checkpoint("02_index_loaded_no_embedder")

    rng = np.random.default_rng(seed=42)
    for i in range(num_queries):
        vec = rng.standard_normal(dense.dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        vector = vec.tolist()
        cp.checkpoint("08_before_faiss_search", i)

        hits = dense.search(vector, k=5)
        cp.checkpoint("10_after_faiss_search", i)  # window_peak = boundary 9, "during" search

        for chunk_id, _score in hits:
            chunk_lookup.get(chunk_id)
        cp.checkpoint("11_after_sqlite_lookup", i)

        gc.collect()
        cp.checkpoint("13_after_gc_collect", i)

    _sampling_stop()
    _print_summary(cp.records, "faiss_only")


def _sampling_stop() -> None:
    global _sampling
    _sampling = False
    time.sleep(0.05)


def _print_summary(records: list[dict[str, Any]], mode: str) -> None:
    print(f"\n===== SUMMARY ({mode}) =====", flush=True)
    by_query: dict[int, list[dict[str, Any]]] = {}
    for r in records:
        if r["query_idx"] is not None:
            by_query.setdefault(r["query_idx"], []).append(r)

    for qidx in sorted(by_query):
        recs = by_query[qidx]
        peak = max(r["window_peak_rss_mb"] for r in recs)
        end_rss = recs[-1]["rss_mb"]
        max_native_step = max(recs, key=lambda r: r["native_unexplained_mb"])
        print(
            f"query {qidx}: peak_rss={peak:.1f}MB end_rss={end_rss:.1f}MB "
            f"largest_native_unexplained_step={max_native_step['label']} "
            f"({max_native_step['native_unexplained_mb']:+.1f}MB)",
            flush=True,
        )

    if by_query:
        first_peak = max(r["window_peak_rss_mb"] for r in by_query[0])
        later_peaks = [
            max(r["window_peak_rss_mb"] for r in by_query[q])
            for q in sorted(by_query)
            if q != 0
        ]
        print(f"\nfirst-query peak: {first_peak:.1f}MB", flush=True)
        if later_peaks:
            print(
                f"subsequent-query peaks: {[round(p, 1) for p in later_peaks]} "
                f"(max={max(later_peaks):.1f}MB)",
                flush=True,
            )
            growth = later_peaks[-1] - first_peak if later_peaks else 0.0
            shape = "per-request growth / leak-shaped" if growth > 15 else "one-time, not repeating"
            print(f"last-query peak minus first-query peak: {growth:+.1f}MB ({shape})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["full", "embed_only", "faiss_only"], required=True)
    parser.add_argument("--num-queries", type=int, default=5)
    parser.add_argument(
        "--log-path", type=Path, default=Path("/tmp/probe_log.jsonl"), help="JSON-lines output"
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/index/metadata_aware"),
        help="only used by --mode faiss_only",
    )
    args = parser.parse_args()

    if args.log_path.exists():
        args.log_path.unlink()

    print(f"mode={args.mode} num_queries={args.num_queries} log={args.log_path}", flush=True)
    if args.mode == "full":
        run_full(args.log_path, args.num_queries)
    elif args.mode == "embed_only":
        run_embed_only(args.log_path, args.num_queries)
    else:
        run_faiss_only(args.log_path, args.num_queries, args.index_dir)


if __name__ == "__main__":
    main()
