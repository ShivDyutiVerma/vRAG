from vrag.guardrails import g2_scope_language


def test_normal_hindi_query_passes():
    assert g2_scope_language.check("भारत में सबसे ऊँचा पर्वत कौन सा है?").passed


def test_romanised_query_passes():
    assert g2_scope_language.check("bharat ka sabse uncha parvat kaunsa hai").passed


def test_empty_query_refused():
    verdict = g2_scope_language.check("")
    assert not verdict.passed
    assert "empty" in (verdict.reason or "").lower()


def test_whitespace_only_query_refused():
    assert not g2_scope_language.check("   \n\t  ").passed


def test_single_character_query_refused():
    assert not g2_scope_language.check("a").passed


def test_pure_punctuation_refused():
    assert not g2_scope_language.check("??? !!! ...").passed


def test_pure_digits_refused():
    assert not g2_scope_language.check("123456").passed


def test_pure_symbol_noise_refused():
    assert not g2_scope_language.check("~!@#$%^&*()").passed


# --- Phase 1 (docs/DECISIONS.md ADR-009): language-aware routing, when a real Sarvam-detected
# language is passed in. All tests above stay unchanged (language=None -> old script heuristic).


def test_hindi_language_code_passes():
    verdict = g2_scope_language.check("भारत की राजधानी क्या है?", language="hi-IN")
    assert verdict.passed


def test_english_language_code_is_refused_not_routed_to_hindi():
    """English is not in SUPPORTED_LANGUAGES (Phase 0: not a MSMARCO-XI target language) --
    must be refused explicitly at G2, never silently passed through to search the Hindi index."""
    verdict = g2_scope_language.check("What is the capital of India?", language="en-IN")
    assert not verdict.passed
    assert "unsupported" in (verdict.reason or "").lower()
    assert "en-IN" in (verdict.reason or "")


def test_several_additional_indic_languages_pass():
    """At least 3 more MSMARCO-XI languages beyond Hindi, per Phase 1's test requirement."""
    for code in ("bn-IN", "ta-IN", "mr-IN", "gu-IN"):
        verdict = g2_scope_language.check("test query text", language=code)
        assert verdict.passed, f"{code} should be supported"


def test_unsupported_language_code_is_refused_with_clear_reason():
    verdict = g2_scope_language.check("bonjour le monde", language="fr-FR")
    assert not verdict.passed
    assert verdict.reason
    assert "fr-FR" in verdict.reason


def test_telugu_is_refused_despite_being_a_real_sarvam_code():
    """Sharpest edge case: Sarvam recognises Telugu, but it's not in SUPPORTED_LANGUAGES (no
    MSMARCO-XI train data) -- must still be refused."""
    verdict = g2_scope_language.check("తెలుగు ప్రశ్న", language="te-IN")
    assert not verdict.passed


def test_language_aware_path_still_rejects_empty_and_short_queries_first():
    """The empty/too-short/punct-only checks must run before the language branch, regardless of
    what language is passed -- a real language code doesn't bypass basic input validation."""
    assert not g2_scope_language.check("", language="hi-IN").passed
    assert not g2_scope_language.check("a", language="hi-IN").passed
    assert not g2_scope_language.check("!!!", language="hi-IN").passed


def test_none_language_falls_back_to_script_heuristic_unchanged():
    """No real signal (e.g. a direct /ask call) must behave exactly as before Phase 1."""
    assert g2_scope_language.check("भारत", language=None).passed
    assert g2_scope_language.check("bharat mein", language=None).passed
