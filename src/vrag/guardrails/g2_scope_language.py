"""G2 — Scope & language. Hot path, target <1ms. Refuses queries that are empty/degenerate
(nothing to search for) or clearly outside the supported language set, so the request never wastes
a retrieval call on something that can't produce a real answer.
"""

from __future__ import annotations

import re

from pydantic import BaseModel


class GuardrailVerdict(BaseModel):
    passed: bool
    reason: str | None = None


# Corpus is Hindi-only for v1 (docs/DECISIONS.md ADR-002). Devanagari script is always in scope;
# Latin-script words are also accepted (romanised/Hinglish queries, per docs/TECH_MENU.md §S2's
# note on translit/codemix modes) rather than rejected outright — a coarse language-ID heuristic
# here is meant to catch pure noise, not to be a real language classifier. Refining this to
# actually reject unsupported languages (vs. just noise) is future work once a language-ID
# signal is available (e.g. from Sarvam's detected language on the transcript).
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
_ONLY_PUNCT_DIGITS_OR_SYMBOLS = re.compile(r"^[\W\d_]*$", re.UNICODE)

_MIN_QUERY_CHARS = 2


def check(query: str) -> GuardrailVerdict:
    """HOTPATH — no network, no disk I/O."""
    stripped = query.strip()
    if not stripped:
        return GuardrailVerdict(passed=False, reason="Empty query.")
    if len(stripped) < _MIN_QUERY_CHARS:
        return GuardrailVerdict(passed=False, reason="Query is too short to be a real question.")
    if _ONLY_PUNCT_DIGITS_OR_SYMBOLS.match(stripped):
        return GuardrailVerdict(passed=False, reason="Query has no recognisable words.")
    if not _DEVANAGARI.search(stripped) and not _LATIN_WORD.search(stripped):
        return GuardrailVerdict(passed=False, reason="Query is not in a supported language.")
    return GuardrailVerdict(passed=True)
