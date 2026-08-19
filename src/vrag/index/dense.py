"""FAISS dense index (AGENT_BUILD_SPEC.md §5.2). `IndexHNSWFlat` with inner-product metric on
L2-normalised vectors, so inner product equals cosine similarity — normalise at embed time
(`embedder.py` already does, via `normalize_embeddings=True`), never re-normalise here.

`efSearch=64` is chosen from the measured recall-vs-latency curve (docs/BUILD_PLAN.md P3 task 9,
docs/DECISIONS_R.md R-014, docs/assets/efsearch_curve.png), not guessed: Recall@5 gains flatten
sharply past this point (0.652 at 64 vs. 0.656 at 256 — within A1's ~0.2-0.4pp noise floor) while
p50 latency keeps climbing roughly linearly with efSearch, so 64 is the knee of the curve — 128/256
buy negligible, likely-noise recall for real added cost, and this project's 200ms end-to-end budget
means every stage's slack matters even when a single stage's own cost looks tiny in isolation.

`quantization` (docs/DECISIONS_R.md R-033/R-034): "none" (default, unchanged behaviour — every
existing caller that doesn't pass this argument builds the exact same `IndexHNSWFlat` it always
has) or "sqfp16" (`IndexHNSWSQ` + `ScalarQuantizer.QT_fp16` — vectors stored as 2-byte IEEE
half-floats instead of 4-byte float32, same M/efConstruction/efSearch/metric). R-033's offline
500-query ablation found "sqfp16" saves ~77MB of resident FAISS memory at zero measured Recall@k/
MRR@10 regression versus "none" — an int8 alternative (`QT_8bit`) was also measured and rejected:
it saves more memory but at a real quality cost "sqfp16" doesn't have. Deliberately not a general
multi-quantizer framework — only the one measured, chosen option is implemented.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, get_args

import faiss
import numpy as np

DEFAULT_M = 32
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_EF_SEARCH = 64  # curve-chosen, docs/DECISIONS_R.md R-014 — not a guess

Quantization = Literal["none", "sqfp16"]
_VALID_QUANTIZATIONS = get_args(Quantization)


def _make_index(dim: int, m: int, quantization: Quantization) -> faiss.Index:
    if quantization == "none":
        return faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
    if quantization == "sqfp16":
        # faiss's stubs type the qtype param as ScalarQuantizer rather than the enum-int the
        # SWIG binding actually accepts at runtime (verified directly, not assumed) -- same class
        # of imprecise-stub issue as the .hnsw accesses below, not a real type error.
        return faiss.IndexHNSWSQ(
            dim,
            faiss.ScalarQuantizer.QT_fp16,  # type: ignore[arg-type]
            m,
            faiss.METRIC_INNER_PRODUCT,
        )
    raise ValueError(
        f"unknown quantization: {quantization!r}, expected one of {_VALID_QUANTIZATIONS}"
    )


class DenseIndex:
    def __init__(
        self,
        dim: int,
        m: int = DEFAULT_M,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
        ef_search: int = DEFAULT_EF_SEARCH,
        quantization: Quantization = "none",
    ) -> None:
        self.dim = dim
        self._quantization: Quantization = quantization
        self._index: faiss.Index = _make_index(dim, m, quantization)
        # `.hnsw` isn't part of faiss's declared `Index` stub (same imprecise-stub situation as
        # set_ef_search() below, which already carries this exact ignore) -- true at runtime for
        # both IndexHNSWFlat and IndexHNSWSQ, they're both HNSW-family indexes.
        self._index.hnsw.efConstruction = ef_construction  # type: ignore[attr-defined]
        self._index.hnsw.efSearch = ef_search  # type: ignore[attr-defined]
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
        # Flat storage is always "trained" (nothing to learn); a quantized storage (e.g. sqfp16's
        # ScalarQuantizer) needs its per-dimension codec fit before it can accept vectors — one
        # generic check covers both without the caller needing to know which quantization this
        # index was built with.
        if not self._index.is_trained:
            self._index.train(arr)
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

    def set_ef_search(self, ef_search: int) -> None:
        """`efSearch` only controls search-time graph traversal depth, not the HNSW graph itself —
        safe to change on an already-built index without rebuilding (docs/BUILD_PLAN.md P3 task 9's
        efSearch sweep relies on this to avoid a full re-embed per sweep point)."""
        self._index.hnsw.efSearch = ef_search  # type: ignore[attr-defined]

    def save(self, path: str | Path) -> None:
        """AGENT_BUILD_SPEC.md §5.3: build offline, download-and-mmap at boot, never rebuild at
        container start. `path` is a directory; created if missing."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "faiss.index"))
        metadata = {
            "dim": self.dim,
            "chunk_ids": self._chunk_ids,
            "quantization": self._quantization,
        }
        (path / "meta.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> DenseIndex:
        path = Path(path)
        metadata = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        # "quantization" postdates R-033/R-034 -- absent on any index built before then (e.g. the
        # index-metadata_aware-v1/v2 release assets), always "none" for those, correctly.
        instance = cls(dim=metadata["dim"], quantization=metadata.get("quantization", "none"))
        instance._index = faiss.read_index(str(path / "faiss.index"))
        instance._chunk_ids = metadata["chunk_ids"]
        return instance
