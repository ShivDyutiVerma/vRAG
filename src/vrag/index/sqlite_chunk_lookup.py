"""SQLite-backed chunk lookup — prototype for docs/DECISIONS_R.md R-020's "leaner chunk_lookup
format" option. Alternative to `persistence.py`'s all-in-memory `dict[str, Chunk]`, not yet the
default (measurement first, wiring-in decision second — see `scripts/measure_chunk_lookup_rss.py`).

Same read interface the rest of the codebase already uses (`__getitem__`, `.get`, `__contains__`,
`__len__`, `.items()`, `.values()` — verified against every real call site before writing this, not
assumed), but backed by on-disk SQLite instead of one big live dict of ~100k Pydantic `Chunk`
instances. `chunk_id -> doc_id` is the one thing kept fully in memory (two short strings per chunk,
~99,767 pairs) since every eval script needs it for every chunk in bulk; full chunk *text* — the
actual memory cost — is fetched lazily, one row at a time, only for chunks actually requested.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from vrag.chunking.base import Chunk


class SQLiteChunkLookup:
    def __init__(self, sqlite_path: str | Path) -> None:
        self._path = str(sqlite_path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        # kept in memory: cheap (two short strings x ~100k rows), needed by every eval script's
        # bulk chunk_id -> doc_id mapping without a per-chunk round trip to SQLite.
        self._doc_ids: dict[str, str] = dict(
            self._conn.execute("SELECT chunk_id, doc_id FROM chunks").fetchall()
        )

    def _row_to_chunk(self, row: tuple) -> Chunk:
        chunk_id, doc_id, text, parent_chunk_id, metadata_json = row
        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            parent_chunk_id=parent_chunk_id,
            metadata=json.loads(metadata_json),
        )

    def get(self, chunk_id: str, default: Chunk | None = None) -> Chunk | None:
        row = self._conn.execute(
            "SELECT chunk_id, doc_id, text, parent_chunk_id, metadata "
            "FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        return self._row_to_chunk(row) if row is not None else default

    def __getitem__(self, chunk_id: str) -> Chunk:
        chunk = self.get(chunk_id)
        if chunk is None:
            raise KeyError(chunk_id)
        return chunk

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._doc_ids

    def __len__(self) -> int:
        return len(self._doc_ids)

    def doc_id_for(self, chunk_id: str) -> str | None:
        """Fast path for the common "just need doc_id" case (every eval script's chunk_to_doc_id
        mapping) — no SQLite round trip, no full Chunk construction."""
        return self._doc_ids.get(chunk_id)

    def items(self) -> Iterator[tuple[str, Chunk]]:
        cursor = self._conn.execute(
            "SELECT chunk_id, doc_id, text, parent_chunk_id, metadata FROM chunks"
        )
        for row in cursor:
            yield row[0], self._row_to_chunk(row)

    def values(self) -> Iterator[Chunk]:
        for _chunk_id, chunk in self.items():
            yield chunk

    def close(self) -> None:
        self._conn.close()
