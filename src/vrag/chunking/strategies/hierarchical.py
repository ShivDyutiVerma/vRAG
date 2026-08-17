"""Strategy 6: hierarchical / small-to-big (TECH_MENU.md S5 #6, AGENT_BUILD_SPEC.md §7.1 #6).
Small child chunks get indexed and matched for precision; each carries `parent_chunk_id` pointing
at a larger parent chunk that is NOT indexed for retrieval but is returned at generation time for
more complete context. Both parent and child Chunk objects come back from `chunk()` — the index
build step (Phase 1/2, `scripts/build_index.py`) is responsible for indexing only children
(`metadata["is_parent"] is False`) while keeping a `chunk_id -> parent text` lookup for generation.
"""

from __future__ import annotations

from typing import Any

from vrag.chunking.base import Chunk, Document
from vrag.chunking.registry import register


class HierarchicalChunker:
    name = "hierarchical"

    def __init__(self, child: int = 128, parent: int = 512) -> None:
        if child <= 0 or parent <= 0:
            raise ValueError(f"child and parent must be positive, got {child=}, {parent=}")
        if child >= parent:
            raise ValueError(f"child ({child}) must be smaller than parent ({parent})")
        self.child = child
        self.parent = parent

    def chunk(self, doc: Document) -> list[Chunk]:
        words = doc.text.split()
        if not words:
            return []

        chunks: list[Chunk] = []
        parent_starts = range(0, len(words), self.parent)
        for parent_idx, parent_start in enumerate(parent_starts):
            parent_words = words[parent_start : parent_start + self.parent]
            parent_id = f"{doc.doc_id}::hierarchical::parent::{parent_idx}"
            chunks.append(
                Chunk(
                    chunk_id=parent_id,
                    doc_id=doc.doc_id,
                    text=" ".join(parent_words),
                    metadata={"language": doc.language, "is_parent": True},
                )
            )

            child_starts = range(0, len(parent_words), self.child)
            for child_idx, child_start in enumerate(child_starts):
                child_words = parent_words[child_start : child_start + self.child]
                chunks.append(
                    Chunk(
                        chunk_id=f"{parent_id}::child::{child_idx}",
                        doc_id=doc.doc_id,
                        text=" ".join(child_words),
                        parent_chunk_id=parent_id,
                        metadata={"language": doc.language, "is_parent": False},
                    )
                )
        return chunks

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "child": self.child, "parent": self.parent}


register(HierarchicalChunker())
