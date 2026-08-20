"""Single source of truth for language support (Phase 1, docs/DECISIONS.md ADR-009).

Two separate code spaces are in play here and must never be conflated:

- **Sarvam's STT language codes** — BCP-47 with an `-IN` suffix (e.g. `"hi-IN"`), returned by
  Sarvam's realtime STT endpoint when `language_code="auto"` is used. Verified live against
  docs.sarvam.ai/api-reference/speech-to-text/transcribe/realtime/ws on 2026-08-20 (not assumed):
  the full accepted set is `auto, en-IN, hi-IN, bn-IN, kn-IN, ml-IN, mr-IN, or-IN, pa-IN, ta-IN,
  te-IN, gu-IN, as-IN, ur-IN, ne-IN, kok-IN, ks-IN, sd-IN, sa-IN, sat-IN, mni-IN, brx-IN, mai-IN,
  doi-IN`. The `language` field on a transcript event is present only when the connection used
  `language_code="auto"` — a fixed code (e.g. the old hardcoded `"hi-IN"`) never populates it.
- **MSMARCO-XI's per-language codes** — the `train/{code}train.parquet` file-naming convention
  (`docs/AGENT_BUILD_SPEC.md` Phase 0 audit, 2026-08-20): `hin, ben, guj, kan, mal, mar, nep, ori,
  pan, san, tam, urd, asm` (13 files, 10,080,140 rows total, verified against real parquet
  metadata). `tel` (Telugu) has a *validation*-only file, no train data — not indexable.

`SUPPORTED_LANGUAGES` is the intersection: every Sarvam code that also maps to a real, indexed
language. Through Phase 1 this was exactly the 13 MSMARCO-XI train languages, with English
explicitly excluded (not a MSMARCO-XI *target* language — it's the shared source/pivot text before
translation). **Phase 3 (docs/DECISIONS.md ADR-012) changes this deliberately**: English is now
indexed for real, using the `English_passages` field every MSMARCO-XI row already carries (Phase 0
finding) — 771 rows (the same per-language budget every other language got), not a back-fill or
a translation. `en-IN` moves from excluded to supported as of this change; see ADR-012 for the
real chunk counts.

One Sarvam-recognisable code remains deliberately excluded:

- `te-IN` (Telugu) — Sarvam can transcribe it, but MSMARCO-XI has no Telugu train data to ever
  index it against.

The other 10 Sarvam codes (`kok-IN, ks-IN, sd-IN, sat-IN, mni-IN, brx-IN, mai-IN, doi-IN`, plus
`auto` itself, which is a connection *mode*, not a detected language) have no MSMARCO-XI
counterpart at all.

**Status as of Phase 3:** all 14 languages in `SUPPORTED_LANGUAGES` are genuinely indexed in the
100k multilingual candidate (`data/index/multilingual_100k/`) — `CURRENTLY_INDEXED_LANGUAGES` now
equals `SUPPORTED_LANGUAGES`, not a subset of it, for the first time. G3's existing confidence
threshold remains the honest backstop for any individual query regardless — supported-and-indexed
is not the same promise as "will always pass G3" (see ADR-012's real abstention-rate finding: the
multilingual corpus's cross-language score distribution abstains far more often than the Hindi-only
baseline did, TAU unchanged, not silently lowered).
"""

from __future__ import annotations

# Sarvam BCP-47 code -> MSMARCO-XI 3-letter train-file code (or "eng" for the Phase-3-added
# English slice, which has no MSMARCO-XI train file of its own -- see module docstring).
SARVAM_TO_MSMARCO_XI: dict[str, str] = {
    "hi-IN": "hin",
    "bn-IN": "ben",
    "gu-IN": "guj",
    "kn-IN": "kan",
    "ml-IN": "mal",
    "mr-IN": "mar",
    "or-IN": "ori",
    "pa-IN": "pan",
    "ta-IN": "tam",
    "as-IN": "asm",
    "ur-IN": "urd",
    "ne-IN": "nep",
    "sa-IN": "san",
    "en-IN": "eng",
}

# Sarvam BCP-47 code -> the real FLORES-style tag observed in the corpus data itself (matches
# chunk.metadata["language"] and the doc_id-qualifying prefix, ADR-011). Needed for retrieval-time
# language filtering (Phase 2/3) -- kept here, next to SARVAM_TO_MSMARCO_XI, as the second half of
# the same code-space mapping.
SARVAM_TO_TARGET_LANG: dict[str, str] = {
    "hi-IN": "hin_Deva",
    "bn-IN": "ben_Beng",
    "gu-IN": "guj_Gujr",
    "kn-IN": "kan_Knda",
    "ml-IN": "mal_Mlym",
    "mr-IN": "mar_Deva",
    "or-IN": "ory_Orya",
    "pa-IN": "pan_Guru",
    "ta-IN": "tam_Taml",
    "as-IN": "asm_Beng",
    "ur-IN": "urd_Arab",
    "ne-IN": "npi_Deva",
    "sa-IN": "san_Deva",
    "en-IN": "eng_Latn",
}

# The Phase 3 target: every MSMARCO-XI train language plus English, expressed as Sarvam codes
# since that's the code space query_language actually arrives in.
SUPPORTED_LANGUAGES: frozenset[str] = frozenset(SARVAM_TO_MSMARCO_XI)

# What the production candidate index can actually answer. As of Phase 3 this equals
# SUPPORTED_LANGUAGES -- all 14 are genuinely indexed in data/index/multilingual_100k/ (ADR-012).
CURRENTLY_INDEXED_LANGUAGES: frozenset[str] = frozenset(SARVAM_TO_MSMARCO_XI)


def is_supported(sarvam_language_code: str | None) -> bool:
    """None means "no real language signal" (e.g. the /ask text debug endpoint with no STT
    behind it) — callers should treat that as a distinct case, not as unsupported. This function
    only answers the question for a real, non-None code."""
    return sarvam_language_code in SUPPORTED_LANGUAGES


# Sarvam BCP-47 code -> (English display name, script name) -- Phase 3 (ADR-012), used only to
# build Track B's language-aware system prompt (src/vrag/generation/sarvam_llm.py). Not used for
# any retrieval/routing decision; those all key off SARVAM_TO_TARGET_LANG instead.
LANGUAGE_DISPLAY_NAMES: dict[str, tuple[str, str]] = {
    "hi-IN": ("Hindi", "Devanagari"),
    "bn-IN": ("Bengali", "Bengali"),
    "gu-IN": ("Gujarati", "Gujarati"),
    "kn-IN": ("Kannada", "Kannada"),
    "ml-IN": ("Malayalam", "Malayalam"),
    "mr-IN": ("Marathi", "Devanagari"),
    "or-IN": ("Odia", "Odia"),
    "pa-IN": ("Punjabi", "Gurmukhi"),
    "ta-IN": ("Tamil", "Tamil"),
    "as-IN": ("Assamese", "Bengali"),
    "ur-IN": ("Urdu", "Perso-Arabic"),
    "ne-IN": ("Nepali", "Devanagari"),
    "sa-IN": ("Sanskrit", "Devanagari"),
    "en-IN": ("English", "Latin"),
}


def display_name(sarvam_language_code: str | None) -> tuple[str, str] | None:
    """(language name, script name) for prompt-building, or None for an unmapped/missing code —
    callers decide their own fallback (src/vrag/generation/sarvam_llm.py falls back to Hindi,
    matching this project's pre-Phase-3 default)."""
    if sarvam_language_code is None:
        return None
    return LANGUAGE_DISPLAY_NAMES.get(sarvam_language_code)
