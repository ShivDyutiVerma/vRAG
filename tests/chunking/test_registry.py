"""Tests for the chunking strategy registry: registration, lookup, and duplicate-name rejection."""

import pytest

from vrag.chunking.base import Chunk, Document
from vrag.chunking.registry import _REGISTRY, all_strategies, get_strategy, register


class _FakeStrategy:
    name = "fake-strategy-for-tests"

    def chunk(self, doc: Document) -> list[Chunk]:
        return [Chunk(chunk_id=f"{doc.doc_id}-0", doc_id=doc.doc_id, text=doc.text)]

    def config(self) -> dict:
        return {"name": self.name}


@pytest.fixture(autouse=True)
def _clean_registry():
    _REGISTRY.pop(_FakeStrategy.name, None)
    yield
    _REGISTRY.pop(_FakeStrategy.name, None)


def test_register_and_get_strategy() -> None:
    strategy = _FakeStrategy()
    register(strategy)
    assert get_strategy(_FakeStrategy.name) is strategy


def test_register_duplicate_name_raises() -> None:
    register(_FakeStrategy())
    with pytest.raises(ValueError, match="already registered"):
        register(_FakeStrategy())


def test_get_unknown_strategy_raises_with_registered_list() -> None:
    with pytest.raises(KeyError, match="Unknown chunking strategy"):
        get_strategy("does-not-exist")


def test_all_strategies_returns_copy_not_live_reference() -> None:
    register(_FakeStrategy())
    snapshot = all_strategies()
    snapshot["mutated"] = "should not leak back"
    assert "mutated" not in _REGISTRY
