"""Strategy 4: semantic (TECH_MENU.md S5 #4, AGENT_BUILD_SPEC.md §7.1 #4). Splits at
embedding-similarity troughs between adjacent sentences instead of a fixed word count — boundaries
follow where the meaning actually shifts. TECH_MENU.md flags this as ~14x slower to build than
token-based chunking (Chonkie benchmark), which is fine here since chunking is a one-time offline
cost, not hot-path.

Needs a real sentence embedder, which is Workstream R's Phase 1 deliverable
(`src/vrag/index/embedder.py`) and doesn't exist yet as of this commit — `embed_fn` is injected
rather than imported directly, so this module is honest about that dependency instead of silently
importing a module that isn't there yet. `chunk()` raises clearly if no `embed_fn` was supplied.
Once the embedder exists, wire it in at call sites (e.g. `scripts/eval_chunking.py`) via
`SemanticChunker(embed_fn=embed_passages)` — this file does not need to change.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from vrag.chunking.base import Chunk, Document
from vrag.chunking.registry import register
from vrag.chunking.strategies.sentence_window import split_sentences

EmbedFn = Callable[[list[str]], list[list[float]]]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(len(s) * pct / 100), len(s) - 1)
    return s[idx]


class SemanticChunker:
    name = "semantic"

    def __init__(self, percentile_threshold: int = 90, embed_fn: EmbedFn | None = None) -> None:
        if not 0 < percentile_threshold < 100:
            raise ValueError(
                f"percentile_threshold must be in (0, 100), got {percentile_threshold}"
            )
        self.percentile_threshold = percentile_threshold
        self._embed_fn = embed_fn

    def chunk(self, doc: Document) -> list[Chunk]:
        sentences = split_sentences(doc.text)
        if len(sentences) <= 1:
            if not sentences:
                return []
            return [
                Chunk(
                    chunk_id=f"{doc.doc_id}::semantic::0",
                    doc_id=doc.doc_id,
                    text=sentences[0],
                    metadata={"language": doc.language},
                )
            ]

        if self._embed_fn is None:
            raise RuntimeError(
                "SemanticChunker needs a sentence embedder — pass embed_fn= (e.g. the E5 "
                "embedder from vrag.index.embedder once Workstream R's Phase 1 embedder work "
                "lands). Not wired to a default here on purpose; see this module's docstring."
            )

        embeddings = self._embed_fn(sentences)
        # Distance (1 - cosine similarity) between each adjacent sentence pair. A trough in
        # similarity = a spike in distance = a likely topic boundary.
        distances = [
            1.0 - _cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]
        split_threshold = _percentile(distances, self.percentile_threshold)

        chunks: list[Chunk] = []
        current: list[str] = [sentences[0]]
        chunk_idx = 0
        for i, distance in enumerate(distances):
            if distance > split_threshold:
                chunks.append(self._make_chunk(doc, current, chunk_idx))
                chunk_idx += 1
                current = []
            current.append(sentences[i + 1])
        if current:
            chunks.append(self._make_chunk(doc, current, chunk_idx))
        return chunks

    def _make_chunk(self, doc: Document, sentences: list[str], idx: int) -> Chunk:
        return Chunk(
            chunk_id=f"{doc.doc_id}::semantic::{idx}",
            doc_id=doc.doc_id,
            text=" ".join(sentences),
            metadata={"language": doc.language},
        )

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "percentile_threshold": self.percentile_threshold}


register(SemanticChunker())
