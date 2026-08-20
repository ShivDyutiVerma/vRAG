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

`SUPPORTED_LANGUAGES` is the intersection: every Sarvam code that also maps to a real MSMARCO-XI
train language. This happens to be all 13 MSMARCO-XI train languages — every one of them has a
matching Sarvam STT code. Two Sarvam-recognisable codes are deliberately excluded:

- `en-IN` (English) — not a MSMARCO-XI *target* language. Every MSMARCO-XI row already carries an
  `English_passages` field (the shared source/pivot text before translation, identical across all
  13 files for the same underlying query) — real English content exists in the raw data, but
  indexing it as its own language slice is a Phase 2 decision, not made here. Until then, English
  is NOT supported: a query detected as English must be refused, not routed into the Hindi index.
- `te-IN` (Telugu) — Sarvam can transcribe it, but MSMARCO-XI has no Telugu train data to ever
  index it against.

The other 10 Sarvam codes (`kok-IN, ks-IN, sd-IN, sat-IN, mni-IN, brx-IN, mai-IN, doi-IN`, plus
`auto` itself, which is a connection *mode*, not a detected language) have no MSMARCO-XI
counterpart at all.

**Phase 1 note:** membership in `SUPPORTED_LANGUAGES` means G2 lets the query continue to
retrieval — it does NOT mean the corpus can currently ground an answer in that language. Only
Hindi chunks exist in the production index until Phase 2 ships (see `CURRENTLY_INDEXED_LANGUAGES`
below). For the other 12 supported-but-not-yet-indexed languages, G3's existing confidence
threshold is the honest backstop, exactly as it already is for cross-lingual English queries today
(verified in this session's language-routing diagnostic: both English and Hindi phrasings of the
same question can retrieve topically-relevant Hindi passages via multilingual-e5's cross-lingual
embedding space, but a genuinely no-evidence case still scores low and G3 still abstains) — Phase 1
does not add filtering or boosting; see `src/vrag/retrieval/interface.py`.
"""

from __future__ import annotations

# Sarvam BCP-47 code -> MSMARCO-XI 3-letter train-file code.
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
}

# The eventual Phase 2 target: every MSMARCO-XI train language, expressed as Sarvam codes since
# that's the code space query_language actually arrives in.
SUPPORTED_LANGUAGES: frozenset[str] = frozenset(SARVAM_TO_MSMARCO_XI)

# What the production index can *actually* answer today. Phase 1 only: everything else in
# SUPPORTED_LANGUAGES passes G2 but relies on G3 to honestly abstain rather than hallucinate.
CURRENTLY_INDEXED_LANGUAGES: frozenset[str] = frozenset({"hi-IN"})


def is_supported(sarvam_language_code: str | None) -> bool:
    """None means "no real language signal" (e.g. the /ask text debug endpoint with no STT
    behind it) — callers should treat that as a distinct case, not as unsupported. This function
    only answers the question for a real, non-None code."""
    return sarvam_language_code in SUPPORTED_LANGUAGES
