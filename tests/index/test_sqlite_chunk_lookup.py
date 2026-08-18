"""Tests for SQLiteChunkLookup (docs/DECISIONS_R.md R-020 prototype) against a tiny synthetic
SQLite file — matches the exact schema scripts/convert_chunk_lookup_sqlite.py produces, verified
against every real call site in the codebase (persistence.py's dict interface), not assumed."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from vrag.index.sqlite_chunk_lookup import SQLiteChunkLookup


@pytest.fixture
def lookup(tmp_path: Path) -> SQLiteChunkLookup:
    db_path = tmp_path / "chunks.sqlite3"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, "
        "text TEXT NOT NULL, parent_chunk_id TEXT, metadata TEXT NOT NULL)"
    )
    conn.executemany(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
        [
            ("c1", "p1", "पहला भाग", None, '{"language": "hi"}'),
            ("c2", "p1", "दूसरा भाग", None, '{"language": "hi"}'),
            ("c3", "p2", "third chunk", "parent1", "{}"),
        ],
    )
    conn.commit()
    conn.close()
    result = SQLiteChunkLookup(db_path)
    yield result
    result.close()


def test_getitem_returns_chunk(lookup: SQLiteChunkLookup) -> None:
    chunk = lookup["c1"]
    assert chunk.chunk_id == "c1"
    assert chunk.doc_id == "p1"
    assert chunk.text == "पहला भाग"
    assert chunk.metadata == {"language": "hi"}


def test_getitem_missing_raises_keyerror(lookup: SQLiteChunkLookup) -> None:
    with pytest.raises(KeyError):
        lookup["nonexistent"]


def test_get_returns_none_for_missing_by_default(lookup: SQLiteChunkLookup) -> None:
    assert lookup.get("nonexistent") is None


def test_get_respects_default(lookup: SQLiteChunkLookup) -> None:
    sentinel = object()
    assert lookup.get("nonexistent", sentinel) is sentinel  # type: ignore[arg-type]


def test_contains(lookup: SQLiteChunkLookup) -> None:
    assert "c1" in lookup
    assert "nonexistent" not in lookup


def test_len(lookup: SQLiteChunkLookup) -> None:
    assert len(lookup) == 3


def test_doc_id_for_fast_path(lookup: SQLiteChunkLookup) -> None:
    assert lookup.doc_id_for("c1") == "p1"
    assert lookup.doc_id_for("c3") == "p2"
    assert lookup.doc_id_for("nonexistent") is None


def test_items_yields_all_chunks(lookup: SQLiteChunkLookup) -> None:
    items = dict(lookup.items())
    assert set(items.keys()) == {"c1", "c2", "c3"}
    assert items["c3"].parent_chunk_id == "parent1"


def test_values_yields_all_chunks(lookup: SQLiteChunkLookup) -> None:
    texts = {c.text for c in lookup.values()}
    assert texts == {"पहला भाग", "दूसरा भाग", "third chunk"}


def test_parent_chunk_id_none_round_trips(lookup: SQLiteChunkLookup) -> None:
    assert lookup["c1"].parent_chunk_id is None
