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
