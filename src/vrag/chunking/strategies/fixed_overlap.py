"""Strategy 1: fixed-size + overlap. The baseline/control (TECH_MENU.md S5 #1, AGENT_BUILD_SPEC.md
§7.1 #1). Word-token windows with a configurable stride — the standard starting point is 256-512
words with 10-20% overlap; overlap is deliberately a variable to sweep (BUILD_PLAN.md P2 task 5),
not an assumed-good default (a Jan 2026 study found no measurable overlap benefit with sparse
retrieval — TECH_MENU.md S5).

Word count (whitespace split), not a real subword tokenizer: this is offline chunking, so exact
token-budget precision matters less than being simple, script-agnostic, and reproducible without
pulling in a model-specific tokenizer here. Documented as a real limitation in config(), not hidden.
"""

from __future__ import annotations

from typing import Any

from vrag.chunking.base import Chunk, Document
from vrag.chunking.registry import register


class FixedOverlapChunker:
    name = "fixed_overlap"

    def __init__(self, size: int = 256, overlap: float = 0.2) -> None:
        if not 0.0 <= overlap < 1.0:
            raise ValueError(f"overlap must be in [0, 1), got {overlap}")
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        self.size = size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        words = doc.text.split()
        if not words:
            return []

        stride = max(1, int(self.size * (1 - self.overlap)))
        chunks: list[Chunk] = []
        start = 0
        idx = 0
        while start < len(words):
            window = words[start : start + self.size]
            metadata = {
                "language": doc.language,
                "word_start": start,
                "word_end": start + len(window),
            }
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::fixed_overlap::{idx}",
                    doc_id=doc.doc_id,
                    text=" ".join(window),
                    metadata=metadata,
                )
            )
            if start + self.size >= len(words):
                break
            start += stride
            idx += 1
        return chunks

    def config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size,
            "overlap": self.overlap,
            "tokenizer": "whitespace-word-split (approximation, not model-specific subwords)",
        }


register(FixedOverlapChunker())
