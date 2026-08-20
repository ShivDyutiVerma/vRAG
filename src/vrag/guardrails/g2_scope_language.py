"""G2 — Scope & language. Hot path, target <1ms. Refuses queries that are empty/degenerate
(nothing to search for) or clearly outside the supported language set, so the request never wastes
a retrieval call on something that can't produce a real answer.

Phase 1 (docs/DECISIONS.md ADR-009): now uses the real Sarvam-detected language when one is
available, via `vrag.languages.SUPPORTED_LANGUAGES` — the future-work note this docstring used to
carry ("refining this... once a language-ID signal is available") is done. The old script-presence
heuristic (Devanagari or a Latin word) is NOT removed — it's the fallback for callers with no real
language signal at all (the `/ask` text debug endpoint has no STT behind it, so it has nothing to
detect), preserved exactly as before so existing behavior there is unaffected.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from vrag.languages import is_supported


class GuardrailVerdict(BaseModel):
    passed: bool
    reason: str | None = None


# Fallback heuristic only — used when `language` is None (no real Sarvam signal). Devanagari
# script is always in scope; Latin-script words are also accepted (romanised/Hinglish queries,
# per docs/TECH_MENU.md §S2's note on translit/codemix modes) rather than rejected outright — this
# is meant to catch pure noise, not to be a real language classifier. See module docstring: when a
# real language code IS available, `vrag.languages.is_supported` decides, not this.
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
_ONLY_PUNCT_DIGITS_OR_SYMBOLS = re.compile(r"^[\W\d_]*$", re.UNICODE)

_MIN_QUERY_CHARS = 2


def check(query: str, language: str | None = None) -> GuardrailVerdict:
    """HOTPATH — no network, no disk I/O.

    `language` is the real Sarvam-detected BCP-47 code (`ctx.data["query_language"]`) when one
    exists. None means no real signal was available (e.g. a direct `/ask` call) — falls back to
    the old script-presence heuristic, unchanged from before Phase 1."""
    stripped = query.strip()
    if not stripped:
        return GuardrailVerdict(passed=False, reason="Empty query.")
    if len(stripped) < _MIN_QUERY_CHARS:
        return GuardrailVerdict(passed=False, reason="Query is too short to be a real question.")
    if _ONLY_PUNCT_DIGITS_OR_SYMBOLS.match(stripped):
        return GuardrailVerdict(passed=False, reason="Query has no recognisable words.")

    if language is not None:
        if not is_supported(language):
            return GuardrailVerdict(
                passed=False, reason=f"Unsupported language: {language}."
            )
        return GuardrailVerdict(passed=True)

    if not _DEVANAGARI.search(stripped) and not _LATIN_WORD.search(stripped):
        return GuardrailVerdict(passed=False, reason="Query is not in a supported language.")
    return GuardrailVerdict(passed=True)
