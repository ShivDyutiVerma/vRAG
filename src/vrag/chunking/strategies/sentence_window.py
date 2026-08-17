"""Strategy 3: sentence-window (TECH_MENU.md S5 #3, AGENT_BUILD_SPEC.md §7.1 #3). Retrieval
granularity is a single sentence; generation granularity is that sentence ± `window` neighbours.
Decouples the two — a query can precisely match one sentence while the answer still gets enough
surrounding context to be coherent.

Sentence splitting handles both Latin punctuation (. ! ?) and the Devanagari danda (।, ॥) used to
end Hindi sentences — a plain `.`-split would silently miss most Hindi sentence boundaries.
"""

from __future__ import annotations

import re
from typing import Any

from vrag.chunking.base import Chunk, Document
from vrag.chunking.registry import register

_SENTENCE_BOUNDARY = re.compile(r"(?<=[।॥.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]


class SentenceWindowChunker:
    name = "sentence_window"

    def __init__(self, window: int = 2) -> None:
        if window < 0:
            raise ValueError(f"window must be >= 0, got {window}")
        self.window = window

    def chunk(self, doc: Document) -> list[Chunk]:
        sentences = split_sentences(doc.text)
        chunks: list[Chunk] = []
        for i, sentence in enumerate(sentences):
            lo = max(0, i - self.window)
            hi = min(len(sentences), i + self.window + 1)
            context_text = " ".join(sentences[lo:hi])
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::sentence_window::{i}",
                    doc_id=doc.doc_id,
                    text=context_text,
                    metadata={
                        "language": doc.language,
                        "retrieval_sentence": sentence,  # the precise unit that gets embedded
                        "sentence_index": i,
                    },
                )
            )
        return chunks

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "window": self.window}


register(SentenceWindowChunker())
