"""ChunkingStrategy protocol (AGENT_BUILD_SPEC.md §7.1). Every strategy implements this and
self-registers in registry.py so scripts/eval_chunking.py can enumerate all of them without a
code change.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


class Document(BaseModel):
    """One retrievable unit before chunking — here, one translated passage from MSMARCO-XI."""

    doc_id: str
    text: str
    language: str
    source_lang: str
    query_id: int | None = None
    query_type: str | None = None
    is_selected: bool = False  # was this passage marked relevant to its query in the dataset


class Chunk(BaseModel):
    """One retrievable unit after chunking. `parent_chunk_id` is used by hierarchical
    (small-to-big) chunking: retrieval matches the (small) chunk, generation is given the parent's
    text via a lookup at answer time."""

    chunk_id: str
    doc_id: str
    text: str
    parent_chunk_id: str | None = None
    metadata: dict[str, Any] = {}


@runtime_checkable
class ChunkingStrategy(Protocol):
    """A pluggable chunking strategy. `config()` is serialised into eval results so every ledger
    row in eval/ablation_ledger.csv is reproducible from its config alone."""

    name: str

    def chunk(self, doc: Document) -> list[Chunk]: ...

    def config(self) -> dict[str, Any]: ...
