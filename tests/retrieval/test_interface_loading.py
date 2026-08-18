"""Tests for interface.py's real-index-vs-stub fallback logic specifically — not just the
contract shape (that's tests/test_retrieval_interface.py's job). Uses monkeypatching so these are
deterministic regardless of whether the real persisted index happens to exist on this machine when
the suite runs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import vrag.retrieval.interface as interface_module
from vrag.chunking.base import Chunk
from vrag.index.dense import DenseIndex
from vrag.index.persistence import save_built_index
from vrag.index.sparse import SparseIndex


@pytest.fixture(autouse=True)
def _reset_lazy_singleton():
    """The module caches its loaded retriever in globals — without resetting, whichever test
    runs first would determine every later test's behaviour."""
    interface_module._retriever = None
    interface_module._retriever_load_attempted = False
    yield
    interface_module._retriever = None
    interface_module._retriever_load_attempted = False


def test_falls_back_to_stub_when_index_dir_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(interface_module, "_INDEX_DIR", tmp_path / "does-not-exist")
    results = asyncio.run(interface_module.retrieve("test query", k=2))
    assert results == interface_module._STUB_CHUNKS[:2]


def test_uses_real_retriever_when_index_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dense = DenseIndex(dim=2)
    dense.add(["c1"], [[1.0, 0.0]])
    sparse = SparseIndex()
    sparse.build(["c1"], ["भारत की राजधानी नई दिल्ली है"])
    chunk_lookup = {
        "c1": Chunk(
            chunk_id="c1", doc_id="passage-1", text="भारत की राजधानी नई दिल्ली है",
            metadata={"language": "hi"},
        )
    }
    save_built_index(dense, sparse, chunk_lookup, tmp_path / "index")
    monkeypatch.setattr(interface_module, "_INDEX_DIR", tmp_path / "index")

    class _FakeEmbedder:
        def embed_queries(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(
        "vrag.index.embedder.E5Embedder", lambda: _FakeEmbedder()
    )

    results = asyncio.run(interface_module.retrieve("भारत की राजधानी", k=5))
    assert results != interface_module._STUB_CHUNKS[:5]
    assert any(r.chunk_id == "c1" for r in results)


def test_lazy_load_only_attempted_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(interface_module, "_INDEX_DIR", tmp_path / "does-not-exist")
    asyncio.run(interface_module.retrieve("first call", k=1))
    assert interface_module._retriever_load_attempted is True
    # second call must not re-check the filesystem — same cached (None) result
    monkeypatch.setattr(interface_module, "_INDEX_DIR", tmp_path / "would-exist-now-but-ignored")
    results = asyncio.run(interface_module.retrieve("second call", k=1))
    assert results == interface_module._STUB_CHUNKS[:1]
