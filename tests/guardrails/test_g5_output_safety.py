from vrag.guardrails import g5_output_safety


def test_clean_text_not_flagged():
    result = g5_output_safety.redact("कंचनजंगा भारत में स्थित सबसे ऊँचा पर्वत है।")
    assert not result.redacted
    assert result.text == "कंचनजंगा भारत में स्थित सबसे ऊँचा पर्वत है।"


def test_email_redacted():
    result = g5_output_safety.redact("Contact us at support@example.com for more.")
    assert result.redacted
    assert "support@example.com" not in result.text
    assert "[REDACTED EMAIL]" in result.text


def test_indian_mobile_number_redacted():
    result = g5_output_safety.redact("Call 9876543210 for details.")
    assert result.redacted
    assert "9876543210" not in result.text
    assert "[REDACTED PHONE]" in result.text


def test_card_like_number_redacted():
    result = g5_output_safety.redact("Card: 4111111111111111 expires soon.")
    assert result.redacted
    assert "4111111111111111" not in result.text


def test_multiple_pii_types_all_redacted():
    result = g5_output_safety.redact("Email me@test.com or call 9123456789.")
    assert result.redacted
    assert "me@test.com" not in result.text
    assert "9123456789" not in result.text
