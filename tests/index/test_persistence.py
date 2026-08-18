"""load_built_index_lean() (docs/DECISIONS_R.md R-023) — the two branches: SQLiteChunkLookup when
chunk_lookup.sqlite3 is present, the plain dict fallback when it isn't (artifacts built before
R-021). load_built_index()/save_built_index() themselves are exercised indirectly by every test that
calls save_built_index as a fixture (e.g. tests/retrieval/test_interface_loading.py).
"""

from __future__ import annotations

from pathlib import Path

from vrag.chunking.base import Chunk
from vrag.index.dense import DenseIndex
from vrag.index.persistence import load_built_index_lean, save_built_index
from vrag.index.sparse import SparseIndex
from vrag.index.sqlite_chunk_lookup import SQLiteChunkLookup

_CHUNK = Chunk(
    chunk_id="c1", doc_id="passage-1", text="भारत की राजधानी नई दिल्ली है",
    metadata={"language": "hi"},
)


def _build(path: Path) -> None:
    dense = DenseIndex(dim=2)
    dense.add(["c1"], [[1.0, 0.0]])
    sparse = SparseIndex()
    sparse.build(["c1"], [_CHUNK.text])
    save_built_index(dense, sparse, {"c1": _CHUNK}, path)


def test_falls_back_to_dict_when_no_sqlite_file(tmp_path: Path) -> None:
    _build(tmp_path / "index")
    _, _, chunk_lookup = load_built_index_lean(tmp_path / "index")
    assert isinstance(chunk_lookup, dict)
    assert chunk_lookup["c1"].text == _CHUNK.text


def test_uses_sqlite_lookup_when_present(tmp_path: Path) -> None:
    index_path = tmp_path / "index"
    _build(index_path)

    import json
    import sqlite3

    conn = sqlite3.connect(index_path / "chunk_lookup.sqlite3")
    conn.execute(
        "CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, doc_id TEXT, text TEXT, "
        "parent_chunk_id TEXT, metadata TEXT)"
    )
    conn.execute(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
        ("c1", _CHUNK.doc_id, _CHUNK.text, None, json.dumps(_CHUNK.metadata)),
    )
    conn.commit()
    conn.close()

    _, _, chunk_lookup = load_built_index_lean(index_path)
    assert isinstance(chunk_lookup, SQLiteChunkLookup)
    assert chunk_lookup["c1"].text == _CHUNK.text
