"""vrag.languages is the single source of truth for which Sarvam-detected languages are in scope.

Phase 1 (docs/DECISIONS.md ADR-009) established the 13 MSMARCO-XI train languages, English
excluded. Phase 3 (ADR-012) adds English as a 14th genuinely-indexed language (using the
English_passages field every row already carries, Phase 0 finding) -- these tests pin down the
Phase 3 membership so a future edit can't silently drift without a test noticing.
"""

from vrag.languages import (
    CURRENTLY_INDEXED_LANGUAGES,
    SARVAM_TO_MSMARCO_XI,
    SARVAM_TO_TARGET_LANG,
    SUPPORTED_LANGUAGES,
    is_supported,
)

_THIRTEEN_MSMARCO_XI = {
    "hi-IN",
    "bn-IN",
    "gu-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "or-IN",
    "pa-IN",
    "ta-IN",
    "as-IN",
    "ur-IN",
    "ne-IN",
    "sa-IN",
}


def test_fourteen_languages_are_supported_the_thirteen_plus_english():
    """Verified live against the real MSMARCO-XI repo file listing (Phase 0 audit) for the 13,
    plus English added deliberately in Phase 3 (ADR-012)."""
    expected = _THIRTEEN_MSMARCO_XI | {"en-IN"}
    assert expected == SUPPORTED_LANGUAGES
    assert len(SUPPORTED_LANGUAGES) == 14


def test_english_is_now_supported():
    """Phase 3 (ADR-012): English is now genuinely indexed (771 rows from English_passages, the
    same per-language budget every other language got) -- no longer excluded."""
    assert is_supported("en-IN")
    assert "en-IN" in SUPPORTED_LANGUAGES


def test_telugu_is_not_supported_despite_sarvam_stt_support():
    """Sarvam can transcribe Telugu, but MSMARCO-XI only has a validation file for it
    (validation/telval.parquet), no train data -- nothing to ever index it against. Unchanged by
    Phase 3 -- Telugu was never a candidate for the English-style English_passages back-fill."""
    assert not is_supported("te-IN")


def test_languages_with_no_msmarco_xi_counterpart_are_not_supported():
    for code in ("kok-IN", "ks-IN", "sd-IN", "sat-IN", "mni-IN", "brx-IN", "mai-IN", "doi-IN"):
        assert not is_supported(code), f"{code} should not be supported"


def test_unknown_language_code_is_not_supported():
    assert not is_supported("fr-FR")
    assert not is_supported("xx-XX")
    assert not is_supported("")


def test_none_language_is_not_treated_as_supported_by_is_supported():
    """is_supported(None) is False by construction -- but callers (G2) must special-case None
    themselves as "no real signal", not as "unsupported". This test only pins is_supported's own
    contract; see test_g2_scope_language.py for the caller-side distinction."""
    assert is_supported(None) is False


def test_sarvam_to_msmarco_xi_mapping_matches_supported_set():
    assert set(SARVAM_TO_MSMARCO_XI) == SUPPORTED_LANGUAGES
    codes = list(SARVAM_TO_MSMARCO_XI.values())
    assert len(codes) == len(set(codes)) == 14


def test_sarvam_to_target_lang_covers_every_supported_language_with_real_flores_tags():
    """These are the real FLORES-style tags observed in the data itself (captured in
    data/multilingual_dataset_manifest.json during Phase 2 sampling), not guessed against a
    reference table -- e.g. Nepali is npi_Deva, not the nep_Deva a naive guess might produce."""
    assert set(SARVAM_TO_TARGET_LANG) == SUPPORTED_LANGUAGES
    assert SARVAM_TO_TARGET_LANG["hi-IN"] == "hin_Deva"
    assert SARVAM_TO_TARGET_LANG["ne-IN"] == "npi_Deva"
    assert SARVAM_TO_TARGET_LANG["en-IN"] == "eng_Latn"
    tags = list(SARVAM_TO_TARGET_LANG.values())
    assert len(tags) == len(set(tags)) == 14


def test_currently_indexed_languages_equals_supported_as_of_phase_3():
    """As of Phase 3 (ADR-012), the 100k multilingual candidate genuinely indexes every supported
    language -- CURRENTLY_INDEXED_LANGUAGES is no longer a strict subset of SUPPORTED_LANGUAGES
    the way it was in Phase 1 (when only Hindi was indexed)."""
    assert CURRENTLY_INDEXED_LANGUAGES == SUPPORTED_LANGUAGES
