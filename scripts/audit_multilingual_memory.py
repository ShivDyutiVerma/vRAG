"""Phase 2 (docs/DECISIONS.md ADR-010): real, staged RSS measurement for one multilingual index
size, run as an isolated subprocess (matching R-032/R-034's established methodology in
scripts/audit_full_stack_at_size.py) so measurements never mix with anything else resident in the
caller's own process.

Reports every stage Phase 2 asked for: interpreter baseline, after index load, after embedder
load, after first real query, steady-state (a few more queries), and peak (Windows `peak_wset`,
the real OS-reported peak working set for this process -- not sampled/estimated).

Usage (run once per size, in its own subprocess -- see scripts/run_multilingual_memory_audit.py
for the isolated-subprocess wrapper):
  python scripts/audit_multilingual_memory.py --index-dir data/index/multilingual_100k
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _mem() -> dict:
    info = psutil.Process().memory_info()
    d = {"rss_bytes": info.rss}
    # Windows-only field -- the real OS-reported peak working set, not a sampled estimate.
    if hasattr(info, "peak_wset"):
        d["peak_wset_bytes"] = info.peak_wset
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", required=True)
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    stages: dict[str, dict] = {"baseline": _mem()}

    from vrag.index.dense import DenseIndex
    from vrag.index.sqlite_chunk_lookup import SQLiteChunkLookup

    sqlite_path = index_dir / "chunk_lookup.sqlite3"
    dense = DenseIndex.load(index_dir / "dense")
    stages["after_index_load"] = _mem()

    if sqlite_path.exists():
        chunk_lookup: object = SQLiteChunkLookup(sqlite_path)
        lookup_backend = "sqlite (lean, matches production)"
    else:
        raw = json.loads((index_dir / "chunk_lookup.json").read_text(encoding="utf-8"))
        chunk_lookup = raw  # not constructing Chunk objects -- only doc_id is needed below
        lookup_backend = "json dict (no sqlite3 conversion run for this index)"
    stages["after_chunk_lookup_load"] = _mem()

    from vrag.index.embedder import LiteE5Embedder

    embedder = LiteE5Embedder()
    stages["after_embedder_construct"] = _mem()

    warmup_vec = embedder.embed_queries(["वार्म-अप क्वेरी"])[0]
    _hits = dense.search(warmup_vec, k=10)
    stages["after_first_real_query"] = _mem()

    for q in ["दूसरी क्वेरी", "तीसरी क्वेरी", "चौथी क्वेरी", "पांचवी क्वेरी"]:
        v = embedder.embed_queries([q])[0]
        dense.search(v, k=10)
    stages["steady_state_after_5_queries"] = _mem()

    print(
        json.dumps(
            {
                "index_dir": str(index_dir),
                "n_chunks": len(dense),
                "chunk_lookup_backend": lookup_backend,
                "stages": stages,
            },
            ensure_ascii=False,
        )
    )
