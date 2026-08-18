"""G5 — Output safety / PII redaction. Hot path, target <2ms. A pre-emit pass over the final
answer string. Failure action per AGENT_BUILD_SPEC.md §7.3: redact + flag — never blocks the whole
answer for a single PII hit.
"""

from __future__ import annotations

import re

from pydantic import BaseModel


class RedactionResult(BaseModel):
    text: str
    redacted: bool


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_INDIAN_MOBILE = re.compile(r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{9}(?!\d)")
_CARD_NUMBER = re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)")

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (_EMAIL, "[REDACTED EMAIL]"),
    (_INDIAN_MOBILE, "[REDACTED PHONE]"),
    (_CARD_NUMBER, "[REDACTED NUMBER]"),
]


def redact(text: str) -> RedactionResult:
    """HOTPATH — no network, no disk I/O."""
    out = text
    redacted = False
    for pattern, replacement in _PATTERNS:
        out, count = pattern.subn(replacement, out)
        if count:
            redacted = True
    return RedactionResult(text=out, redacted=redacted)
