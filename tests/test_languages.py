"""Phase 1 (docs/DECISIONS.md ADR-009): vrag.languages is the single source of truth for which
Sarvam-detected languages are in scope. These tests pin down the exact membership decisions made
in Phase 0's audit so a future edit can't silently drift without a test noticing."""

from vrag.languages import (
    CURRENTLY_INDEXED_LANGUAGES,
    SARVAM_TO_MSMARCO_XI,
    SUPPORTED_LANGUAGES,
    is_supported,
)


def test_all_thirteen_msmarco_xi_train_languages_are_supported():
    """Verified live against the real MSMARCO-XI repo file listing (Phase 0 audit) -- exactly
    these 13 train files exist."""
    expected = {
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
    assert expected == SUPPORTED_LANGUAGES
    assert len(SUPPORTED_LANGUAGES) == 13


def test_english_is_not_supported():
    """English is the shared source/pivot text embedded in every MSMARCO-XI row
    (English_passages), not a translated target language -- deliberately excluded until Phase 2
    makes an explicit decision to index it as its own slice."""
    assert not is_supported("en-IN")
    assert "en-IN" not in SUPPORTED_LANGUAGES


def test_telugu_is_not_supported_despite_sarvam_stt_support():
    """Sarvam can transcribe Telugu, but MSMARCO-XI only has a validation file for it
    (validation/telval.parquet), no train data -- nothing to ever index it against."""
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
    # every value is a real 3-letter MSMARCO-XI train-file code, no duplicates
    codes = list(SARVAM_TO_MSMARCO_XI.values())
    assert len(codes) == len(set(codes)) == 13


def test_currently_indexed_languages_is_hindi_only():
    """Phase 1 only -- the production index has not been rebuilt yet. This must stay hi-IN-only
    until Phase 2 actually ships a multilingual index; a test here means that transition can't
    happen by accident."""
    assert frozenset({"hi-IN"}) == CURRENTLY_INDEXED_LANGUAGES
    assert CURRENTLY_INDEXED_LANGUAGES < SUPPORTED_LANGUAGES
