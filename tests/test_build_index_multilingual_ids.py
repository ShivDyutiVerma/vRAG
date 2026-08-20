"""Regression tests for the cross-language chunk_id collision fix (docs/DECISIONS.md ADR-011).

MSMARCO-XI's `query_id` is a global identifier shared across all 13 language files -- a
multilingual build sampling from more than one language can draw the same query_id twice,
colliding two genuinely different chunks onto one doc_id/chunk_id unless doc_id is qualified by
language. `_rows_to_documents(rows, qualify_doc_id_by_language=True)` is the fix; these tests
pin both the fixed behavior AND that the default (single-language/production Hindi) path is
byte-identical to before -- that's the part that must never silently change.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_index import _rows_to_documents  # noqa: E402


def _row(
    query_id: int, target_lang: str, source_lang: str = "eng_Latn", n_passages: int = 2
) -> dict:
    return {
        "query_id": query_id,
        "target_lang": target_lang,
        "source_lang": source_lang,
        "query_type": "DESCRIPTION",
        "passages": {
            "Translated_passages": [f"passage {i} in {target_lang}" for i in range(n_passages)],
            "is_selected": [1] + [0] * (n_passages - 1),
        },
    }


def test_default_behavior_unchanged_for_single_language_hindi_pipeline():
    """The production Hindi pipeline's doc_id format must stay exactly f'{query_id}_{i}' --
    changing it would silently break eval/heldout_queries.json's passage_id compatibility."""
    rows = [_row(107055, "hin_Deva")]
    docs = _rows_to_documents(rows)  # default: qualify_doc_id_by_language=False
    assert docs[0].doc_id == "107055_0"
    assert docs[1].doc_id == "107055_1"


def test_qualified_mode_prefixes_doc_id_with_target_lang():
    rows = [_row(107055, "hin_Deva")]
    docs = _rows_to_documents(rows, qualify_doc_id_by_language=True)
    assert docs[0].doc_id == "hin_Deva::107055_0"
    assert docs[1].doc_id == "hin_Deva::107055_1"


def test_same_query_id_across_languages_no_longer_collides_when_qualified():
    """The exact real-world case found in ADR-009: query_id=655605 existed as both real Assamese
    and real Malayalam content in the same sample -- unqualified, both produced doc_id
    '655605_0', silently colliding in chunk_lookup. Qualified, they must be distinct."""
    rows = [
        _row(655605, "asm_Beng", n_passages=1),
        _row(655605, "mal_Mlym", n_passages=1),
    ]
    docs = _rows_to_documents(rows, qualify_doc_id_by_language=True)
    doc_ids = [d.doc_id for d in docs]
    assert len(doc_ids) == len(set(doc_ids)), f"collision: {doc_ids}"
    assert doc_ids == ["asm_Beng::655605_0", "mal_Mlym::655605_0"]


def test_same_query_id_across_languages_collides_when_unqualified():
    """Confirms the bug is real and reproducible -- the exact failure ADR-009 documented."""
    rows = [
        _row(655605, "asm_Beng", n_passages=1),
        _row(655605, "mal_Mlym", n_passages=1),
    ]
    docs = _rows_to_documents(rows)  # unqualified -- the old/default behavior
    doc_ids = [d.doc_id for d in docs]
    assert len(doc_ids) != len(set(doc_ids)), "expected the known collision, got none"
    assert doc_ids == ["655605_0", "655605_0"]


def test_qualified_mode_preserves_passage_identity_and_text():
    """Language-qualifying the id must not change which passage's text/metadata a Document
    carries -- only the id used to address it."""
    rows = [_row(42, "tam_Taml", n_passages=1)]
    docs = _rows_to_documents(rows, qualify_doc_id_by_language=True)
    assert docs[0].text == "passage 0 in tam_Taml"
    assert docs[0].language == "tam_Taml"
    assert docs[0].query_id == 42
    assert docs[0].is_selected is True


def test_within_one_language_ids_still_unique_when_qualified():
    """Qualifying by language must not itself introduce new collisions within a single
    language's own rows."""
    rows = [_row(1, "hin_Deva", n_passages=3), _row(2, "hin_Deva", n_passages=3)]
    docs = _rows_to_documents(rows, qualify_doc_id_by_language=True)
    doc_ids = [d.doc_id for d in docs]
    assert len(doc_ids) == len(set(doc_ids)) == 6
