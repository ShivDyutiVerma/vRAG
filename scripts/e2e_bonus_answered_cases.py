"""Phase 6 addendum: one real 'answered' example per remaining tested language (Tamil, Kannada,
Urdu, Gujarati, Assamese) whose natural-draw case in e2e_demo_readiness_test.py happened to
abstain -- picked from real true-accept rows already measured in
eval/g3_calibration_multilingual_100k_raw.json (current_g3_passed=True, relevant_in_top1=True),
not fabricated. Completes Task 3's language-generation verification with real per-language
evidence rather than relying only on the existing unit tests.

Usage: python scripts/e2e_bonus_answered_cases.py
Output: eval/e2e_bonus_answered_results.json
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ["VRAG_INDEX_DIR"] = str(REPO_ROOT / "data" / "index" / "multilingual_100k")
sys.path.insert(0, str(REPO_ROOT / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e_demo_readiness_test import run_case  # noqa: E402

_HELDOUT = json.loads(
    (REPO_ROOT / "eval" / "heldout_queries_multilingual.json").read_text(encoding="utf-8")
)
_HELDOUT_BY_ID = {q["query_id"]: q for q in _HELDOUT}

LANG_TO_SARVAM = {
    "tam_Taml": "ta-IN",
    "kan_Knda": "kn-IN",
    "urd_Arab": "ur-IN",
    "guj_Gujr": "gu-IN",
    "asm_Beng": "as-IN",
}
PICKS = {
    "tam_Taml": 871988,
    "kan_Knda": 580438,
    "urd_Arab": 996838,
    "guj_Gujr": 774823,
    "asm_Beng": 478449,
}


async def main() -> None:
    results = []
    for lang, qid in PICKS.items():
        q = _HELDOUT_BY_ID[qid]
        result = await run_case(q["query"], LANG_TO_SARVAM[lang])
        result["language"] = lang
        result["query_id"] = qid
        result["query"] = q["query"]
        results.append(result)
        print(f"{lang}: status={result['status']} answer_language={result['answer_language']}")

    out_path = REPO_ROOT / "eval" / "e2e_bonus_answered_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(results)} cases -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
