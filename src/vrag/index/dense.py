"""FAISS dense index (AGENT_BUILD_SPEC.md §5.2). `IndexHNSWFlat` with inner-product metric on
L2-normalised vectors, so inner product equals cosine similarity — normalise at embed time
(`embedder.py` already does, via `normalize_embeddings=True`), never re-normalise here.

`efSearch` is intentionally NOT hardcoded to a "reasonable-looking" number — it gets chosen from
the recall-vs-latency curve in Phase 3 (docs/BUILD_PLAN.md P3 task 9), so the default here is a
placeholder clearly marked as such, not a guess dressed up as a decision.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

DEFAULT_M = 32
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_EF_SEARCH = 64  # placeholder — Phase 3 replaces this with a curve-chosen value


class DenseIndex:
    def __init__(
        self,
        dim: int,
        m: int = DEFAULT_M,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
        ef_search: int = DEFAULT_EF_SEARCH,
    ) -> None:
        self.dim = dim
        self._index: faiss.Index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = ef_construction
        self._index.hnsw.efSearch = ef_search
        self._chunk_ids: list[str] = []

    def add(self, chunk_ids: list[str], vectors: list[list[float]]) -> None:
        if len(chunk_ids) != len(vectors):
            raise ValueError(
                f"chunk_ids and vectors must be the same length, "
                f"got {len(chunk_ids)} and {len(vectors)}"
            )
        if not vectors:
            return
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.shape[1] != self.dim:
            raise ValueError(f"expected vectors of dim {self.dim}, got {arr.shape[1]}")
        self._index.add(arr)
        self._chunk_ids.extend(chunk_ids)

    def search(self, query_vector: list[float], k: int) -> list[tuple[str, float]]:
        if not self._chunk_ids:
            return []
        arr = np.asarray([query_vector], dtype=np.float32)
        scores, indices = self._index.search(arr, min(k, len(self._chunk_ids)))
        return [
            (self._chunk_ids[idx], float(score))
            for idx, score in zip(indices[0], scores[0], strict=True)
            if idx != -1
        ]

    def __len__(self) -> int:
        return len(self._chunk_ids)

    def save(self, path: str | Path) -> None:
        """AGENT_BUILD_SPEC.md §5.3: build offline, download-and-mmap at boot, never rebuild at
        container start. `path` is a directory; created if missing."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "faiss.index"))
        metadata = {"dim": self.dim, "chunk_ids": self._chunk_ids}
        (path / "meta.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> DenseIndex:
        path = Path(path)
        metadata = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        instance = cls(dim=metadata["dim"])
        instance._index = faiss.read_index(str(path / "faiss.index"))
        instance._chunk_ids = metadata["chunk_ids"]
        return instance
