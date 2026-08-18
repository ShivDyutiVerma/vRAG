"""Prototype for docs/DECISIONS_R.md R-020's "leaner chunk_lookup.json format" option — converts
the existing JSON chunk_lookup into a SQLite database, to measure (not assume) whether this actually
reduces RSS before committing to refactoring every call site.

Rationale for SQLite specifically: stdlib (`sqlite3`), zero new dependency, on-disk B-tree index
means single-chunk lookup by `chunk_id` doesn't require the whole dataset resident in Python object
form — the OS page cache handles hot pages, not one big live dict of ~100k Pydantic `Chunk`
instances (which have real per-instance overhead beyond the raw JSON bytes: validation machinery,
`__fields_set__`, etc., docs/DECISIONS_R.md R-020).

Usage: python scripts/convert_chunk_lookup_sqlite.py --index-dir data/index/metadata_aware
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def convert(index_dir: Path) -> None:
    json_path = index_dir / "chunk_lookup.json"
    sqlite_path = index_dir / "chunk_lookup.sqlite3"

    print(f"Reading {json_path}...")
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"{len(raw)} chunks")

    if sqlite_path.exists():
        sqlite_path.unlink()

    conn = sqlite3.connect(str(sqlite_path))
    conn.execute("""
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            text TEXT NOT NULL,
            parent_chunk_id TEXT,
            metadata TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX idx_doc_id ON chunks(doc_id)")

    rows = [
        (
            chunk_id,
            data["doc_id"],
            data["text"],
            data.get("parent_chunk_id"),
            json.dumps(data.get("metadata", {}), ensure_ascii=False),
        )
        for chunk_id, data in raw.items()
    ]
    conn.executemany(
        "INSERT INTO chunks (chunk_id, doc_id, text, parent_chunk_id, metadata) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.execute("VACUUM")
    conn.close()

    print(f"Wrote {sqlite_path} ({sqlite_path.stat().st_size / 1e6:.1f}MB)")
    print(f"(JSON was {json_path.stat().st_size / 1e6:.1f}MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default="data/index/metadata_aware")
    args = parser.parse_args()
    convert(REPO_ROOT / args.index_dir)
