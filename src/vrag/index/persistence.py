"""Save/load a built (dense, sparse, chunk_lookup) triple to/from disk. Lives here (not in
scripts/build_index.py) so both the offline build script and the runtime retrieve() path
(src/vrag/retrieval/interface.py) can import it as a normal package module — scripts/ isn't meant
to be imported from, only run standalone.

AGENT_BUILD_SPEC.md §5.3: never build the FAISS/BM25 index at container start. Build offline
(scripts/build_index.py --save-dir ...), commit the artifact to a release asset or object storage
(NOT to git — data/ is gitignored, these are large binary files), download-and-mmap at boot.
"""

from __future__ import annotations

import json
from pathlib import Path

from vrag.chunking.base import Chunk
from vrag.index.dense import DenseIndex
from vrag.index.sparse import SparseIndex
from vrag.index.sqlite_chunk_lookup import SQLiteChunkLookup


def save_built_index(
    dense: DenseIndex,
    sparse: SparseIndex | None,
    chunk_lookup: dict[str, Chunk],
    path: str | Path,
) -> None:
    """`sparse=None` (docs/DECISIONS.md ADR-011): skips writing a sparse/BM25 artifact at all --
    for a dense-only production build, that's dead disk weight (never read by
    `load_built_index_lean(retrieval_mode="dense")`, ADR-007). Passing a real `SparseIndex`
    still works exactly as before -- this is additive, not a behavior change for any existing
    caller; `SparseIndex`/`HybridRetriever`'s sparse/hybrid modes are untouched."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    dense.save(path / "dense")
    if sparse is not None:
        sparse.save(path / "sparse")
    chunk_lookup_json = {chunk_id: chunk.model_dump() for chunk_id, chunk in chunk_lookup.items()}
    (path / "chunk_lookup.json").write_text(
        json.dumps(chunk_lookup_json, ensure_ascii=False), encoding="utf-8"
    )


def load_built_index(path: str | Path) -> tuple[DenseIndex, SparseIndex, dict[str, Chunk]]:
    """The fast-path counterpart to save_built_index() — no chunking, no embedding, just
    deserialisation. Raises FileNotFoundError (via the underlying file reads) if `path` wasn't
    built with save_built_index() first; callers decide how to handle that (interface.py falls
    back to the stub)."""
    path = Path(path)
    dense = DenseIndex.load(path / "dense")
    sparse = SparseIndex.load(path / "sparse")
    raw_lookup = json.loads((path / "chunk_lookup.json").read_text(encoding="utf-8"))
    chunk_lookup = {chunk_id: Chunk(**data) for chunk_id, data in raw_lookup.items()}
    return dense, sparse, chunk_lookup


def load_built_index_lean(
    path: str | Path,
    retrieval_mode: str = "dense",
) -> tuple[DenseIndex, SparseIndex | None, SQLiteChunkLookup | dict[str, Chunk]]:
    """Runtime-optimised counterpart to load_built_index() (docs/DECISIONS_R.md R-023) — used by
    src/vrag/retrieval/interface.py's real-retriever loader, not by scripts/build_index.py or eval
    scripts (those build the dict fresh in memory anyway, so there's nothing to save by loading it
    lazily).

    `retrieval_mode` controls whether the BM25/sparse index loads at all (docs/DECISIONS.md
    ADR-006): it was previously loaded unconditionally, costing ~105MB RSS, even under the
    production default `retrieval_mode="dense"` (A3 winner, R-010) which never calls
    `sparse.search()`. Only "sparse"/"hybrid" need it — "dense" gets `None` back instead, and
    `HybridRetriever` requires that whoever passes `retrieval_mode="dense"` also skip loading a
    sparse index it will never touch, so this parameter has to match whatever's passed into
    `HybridRetriever(retrieval_mode=...)` at the call site or construction raises immediately.

    Prefers `chunk_lookup.sqlite3` (R-021's SQLiteChunkLookup — chunk_id->doc_id kept in memory,
    full Chunk text fetched lazily per-row) when present in `path`, falling back to the eager
    `chunk_lookup.json` dict otherwise so this stays a drop-in replacement for artifacts built
    before R-021 (e.g. the current index-metadata_aware-v1 release asset, which predates the
    sqlite file)."""
    path = Path(path)
    dense = DenseIndex.load(path / "dense")
    sparse = SparseIndex.load(path / "sparse") if retrieval_mode in ("sparse", "hybrid") else None
    sqlite_path = path / "chunk_lookup.sqlite3"
    if sqlite_path.exists():
        return dense, sparse, SQLiteChunkLookup(sqlite_path)
    raw_lookup = json.loads((path / "chunk_lookup.json").read_text(encoding="utf-8"))
    chunk_lookup = {chunk_id: Chunk(**data) for chunk_id, data in raw_lookup.items()}
    return dense, sparse, chunk_lookup
