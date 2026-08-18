"""G1 — Input safety. Hot path, target <2ms. Local regex/keyword denylist only — no classifier,
no LLM judge. Per docs/TECH_MENU.md §S3's design principle: cheap regex first, classifiers second,
LLM judges only on ambiguous cases. A guardrail sitting at 400ms p50 becomes the latency story for
the whole product, so this stays deliberately deterministic and narrow.
"""

from __future__ import annotations

import re

from pydantic import BaseModel


class GuardrailVerdict(BaseModel):
    passed: bool
    reason: str | None = None


# Deliberately narrow and deterministic — this demonstrates the refusal path
# (AGENT_BUILD_SPEC.md §7.3's demo requirement: an unsafe input must visibly refuse), not a
# production-grade safety classifier. Extend with real coverage before anything but a demo.
_UNSAFE_PATTERNS = [
    re.compile(r"बम\s*(बनाने|बनाना)", re.IGNORECASE),
    re.compile(r"\bhow\s+to\s+(make|build)\s+(a\s+)?bomb\b", re.IGNORECASE),
    re.compile(r"\bhow\s+to\s+kill\s+(a\s+)?(person|someone|myself)\b", re.IGNORECASE),
    re.compile(r"आत्महत्या\s*(कैसे|करने)", re.IGNORECASE),
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+(in\s+)?(developer|dan|jailbreak)\s+mode\b", re.IGNORECASE),
]

_MAX_QUERY_CHARS = 500  # degenerate-input guard against absurdly long single queries


def check(query: str) -> GuardrailVerdict:
    """HOTPATH — no network, no disk I/O."""
    if len(query) > _MAX_QUERY_CHARS:
        return GuardrailVerdict(passed=False, reason="Query is too long.")
    for pattern in _UNSAFE_PATTERNS:
        if pattern.search(query):
            return GuardrailVerdict(
                passed=False, reason="Query matched an unsafe-content pattern."
            )
    return GuardrailVerdict(passed=True)
