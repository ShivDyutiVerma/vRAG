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


def save_built_index(
    dense: DenseIndex,
    sparse: SparseIndex,
    chunk_lookup: dict[str, Chunk],
    path: str | Path,
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    dense.save(path / "dense")
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
