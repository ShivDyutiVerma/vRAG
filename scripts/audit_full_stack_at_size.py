"""Measures the REAL full-stack RSS (embedder + FAISS + chunk_lookup, dense-only, matching
production wiring post-ADR-007) at a given subsampled corpus size -- built to directly answer
whether any corpus size fits under Render's 512MB free tier, rather than deriving it by combining
separately-measured components (which risks silently mixing numbers from different measurement
contexts/sessions). Points at a directory already built by scripts/eval_corpus_size.py.

Usage: python scripts/audit_full_stack_at_size.py --index-dir eval/corpus_size_tmp/idx_20000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def current_rss() -> int:
    return psutil.Process().memory_info().rss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", required=True)
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    baseline = current_rss()

    from vrag.index.embedder import LiteE5Embedder
    from vrag.index.persistence import load_built_index_lean

    # load_built_index_lean() with retrieval_mode="dense" -- the exact function and mode
    # production uses (interface.py, post-ADR-007) -- prefers chunk_lookup.sqlite3 when present,
    # matching real deployment behavior rather than always eagerly loading the JSON dict.
    dense, _sparse, _chunk_lookup = load_built_index_lean(index_dir, retrieval_mode="dense")
    embedder = LiteE5Embedder()

    query_vec = embedder.embed_queries(["वार्म-अप क्वेरी"])[0]
    _hits = dense.search(query_vec, k=5)
    after_first_query = current_rss()

    for q in ["दूसरी क्वेरी", "तीसरी क्वेरी"]:
        v = embedder.embed_queries([q])[0]
        dense.search(v, k=5)
    steady_state = current_rss()

    print(json.dumps({
        "n_chunks": len(dense),
        "rss_at_interpreter_start_bytes": baseline,
        "rss_after_first_query_bytes": after_first_query,
        "rss_steady_state_bytes": steady_state,
    }))
