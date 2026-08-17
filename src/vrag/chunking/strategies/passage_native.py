"""Strategy 2: passage-native. Use the dataset's own passage boundaries as retrieval units — MS
MARCO passages are already coherent (TECH_MENU.md S5 #2). Zero splitting logic; the "strategy" is
trusting the corpus's own segmentation instead of imposing an arbitrary window on top of it. Flagged
in TECH_MENU.md as the likely winner on this corpus — not assumed here, only found out by running
the eval (docs/EVAL_PROTOCOL.md).
"""

from __future__ import annotations

from typing import Any

from vrag.chunking.base import Chunk, Document
from vrag.chunking.registry import register


class PassageNativeChunker:
    name = "passage_native"

    def chunk(self, doc: Document) -> list[Chunk]:
        if not doc.text.strip():
            return []
        return [
            Chunk(
                chunk_id=f"{doc.doc_id}::passage_native::0",
                doc_id=doc.doc_id,
                text=doc.text,
                metadata={"language": doc.language, "is_selected": doc.is_selected},
            )
        ]

    def config(self) -> dict[str, Any]:
        return {"name": self.name}


register(PassageNativeChunker())
