"""Strategy 5: metadata-aware (TECH_MENU.md S5 #5, AGENT_BUILD_SPEC.md §7.1 #5) — most
dataset-specific strategy, highlighted deliberately: MSMARCO-XI carries `source_lang`,
`target_lang`, and `query_type` per row, which most chunking implementations would throw away.
Tagging every chunk with these enables retrieval to filter or boost by the language Sarvam's STT
actually detected, rather than treating the corpus as an undifferentiated bag of Hindi text.

`mode="filter"` marks metadata for hard filtering at retrieval time (only ever match same-language
chunks); `mode="boost"` marks it for soft re-ranking (prefer same-language, don't exclude others).
Which mode the retrieval layer actually honours is Workstream R's retrieval-stage decision, made
during A3 — this strategy only produces the tags, it doesn't enforce a policy itself.
"""

from __future__ import annotations

from typing import Any, Literal

from vrag.chunking.base import Chunk, Document
from vrag.chunking.registry import register


class MetadataAwareChunker:
    name = "metadata_aware"

    def __init__(self, mode: Literal["filter", "boost"] = "boost") -> None:
        self.mode = mode

    def chunk(self, doc: Document) -> list[Chunk]:
        if not doc.text.strip():
            return []
        return [
            Chunk(
                chunk_id=f"{doc.doc_id}::metadata_aware::0",
                doc_id=doc.doc_id,
                text=doc.text,
                metadata={
                    "language": doc.language,
                    "source_lang": doc.source_lang,
                    "query_type": doc.query_type,
                    "is_selected": doc.is_selected,
                    "metadata_mode": self.mode,
                },
            )
        ]

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "mode": self.mode}


register(MetadataAwareChunker())
