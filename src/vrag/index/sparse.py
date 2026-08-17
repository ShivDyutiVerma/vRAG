"""`bm25s` sparse index (TECH_MENU.md S7 — chosen over `rank_bm25`, which is ~500x slower).

The real risk here isn't the library, it's tokenisation: `bm25s`'s default tokenizer is built
around English stemming and whitespace/word-boundary assumptions, and running an English stemmer
over Hindi tokens would be silently wrong, not an error — the same "no error, just worse" bug class
as the E5 prefix issue.

First attempt at `tokenize` used `\\w+` (Python's Unicode-aware "word character" regex class) and
was wrong in a genuinely subtle way, caught by the test below rather than assumed correct: `\\w`
does not include Devanagari's dependent vowel signs (matras — combining characters like "ा", "ी")
in its word-character set, so `\\w+` silently split single Hindi words apart at every matra
("दिल्ली" -> ['द', 'ल', '्ल'], not one token). Fixed by tokenising the opposite way — split on
whitespace and an explicit punctuation set (including the Devanagari danda "।"/"॥") instead of
trying to positively enumerate what counts as a "word character" one codepoint at a time. This
keeps multi-codepoint grapheme clusters (base consonant + matra) intact.
"""

from __future__ import annotations

import re

import bm25s

_SEPARATOR_PATTERN = re.compile(r"[\s।॥.,!?;:\"'()\[\]{}—–\-]+")


def tokenize(text: str) -> list[str]:
    """Split on whitespace + punctuation (incl. Devanagari danda), no stemming. Deliberately does
    NOT use a `\\w`-style positive word-character match — see module docstring for why."""
    return [t for t in _SEPARATOR_PATTERN.split(text.lower()) if t]


class SparseIndex:
    def __init__(self) -> None:
        self._retriever: bm25s.BM25 | None = None
        self._chunk_ids: list[str] = []

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        if len(chunk_ids) != len(texts):
            raise ValueError(
                f"chunk_ids and texts must be the same length, "
                f"got {len(chunk_ids)} and {len(texts)}"
            )
        self._chunk_ids = list(chunk_ids)
        corpus_tokens = [tokenize(t) for t in texts]
        self._retriever = bm25s.BM25()
        self._retriever.index(corpus_tokens)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._retriever is None or not self._chunk_ids:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        k = min(k, len(self._chunk_ids))
        results, scores = self._retriever.retrieve([query_tokens], k=k)
        return [
            (self._chunk_ids[idx], float(score))
            for idx, score in zip(results[0], scores[0], strict=True)
        ]

    def __len__(self) -> int:
        return len(self._chunk_ids)
