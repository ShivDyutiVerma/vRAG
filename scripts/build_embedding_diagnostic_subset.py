"""Phase 8 (docs/DECISIONS.md ADR-017): builds ONE shared representative corpus subset, used
identically by every embedding model tested in this diagnostic (same passages, same queries, same
gold labels -- only the embedding model changes, per the explicit instructions).

Per language: every gold passage for the real 532-query held-out set (guaranteed present, so
Recall@k is meaningful) UNION a fixed-seed random sample of ~2,000 other same-language chunks
from the real 107,678-chunk multilingual_100k corpus, preserving real distractor density
(including the same-template distractors Phase 7 (ADR-016) identified as the dominant failure
mode -- a subset with no distractors would make every model look perfect and prove nothing).

~2,000/language x 14 languages ~= 28,000 chunks -- a real "representative subset", not the full
107,678, per the explicit instruction not to rebuild the whole corpus per model.

Usage: python scripts/build_embedding_diagnostic_subset.py
Output: eval/embedding_diagnostic_subset.json
"""

from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = REPO_ROOT / "data" / "index" / "multilingual_100k"
HELDOUT_PATH = REPO_ROOT / "eval" / "heldout_queries_multilingual.json"
OUT_PATH = REPO_ROOT / "eval" / "embedding_diagnostic_subset.json"

SEED = 20260821
PER_LANGUAGE_SAMPLE = 2000

# Known same-template distractors from earlier phases (Phase 6/7) -- force-included so the
# critical-case checks (capital-of-India etc.) remain meaningful across every model tested, not
# dependent on whether the fixed-seed random sample happened to catch them.
FORCE_INCLUDE_DOC_IDS = {
    "hin_Deva::1001095_3",  # Bangkok/Thailand capital passage -- the flagship Hindi distractor
    "hin_Deva::1149223_6",  # Lusaka/Zambia capital passage
    "eng_Latn::1012189_7",  # "world's most expensive cities" -- English capital-of-India distractor
    "eng_Latn::498294_9",  # India constitution passage
}


def main() -> None:
    held = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    gold_doc_ids_all: set[str] = set()
    for q in held:
        for p in q["relevant_passages"]:
            gold_doc_ids_all.add(p["passage_id"])
    print(f"532 held-out queries, {len(gold_doc_ids_all)} distinct gold passage ids")

    conn = sqlite3.connect(str(INDEX_DIR / "chunk_lookup.sqlite3"))
    rows = conn.execute("SELECT chunk_id, doc_id, text, metadata FROM chunks").fetchall()
    conn.close()
    print(f"Loaded {len(rows)} total chunks from the real index")

    by_lang: dict[str, list[tuple]] = {}
    for chunk_id, doc_id, text, metadata_json in rows:
        lang = json.loads(metadata_json).get("language")
        by_lang.setdefault(lang, []).append((chunk_id, doc_id, text, lang))

    rng = random.Random(SEED)
    subset: list[dict] = []
    manifest = {}
    for lang, lang_rows in sorted(by_lang.items()):
        gold_rows = [r for r in lang_rows if r[1] in gold_doc_ids_all]
        forced_rows = [
            r for r in lang_rows if r[1] in FORCE_INCLUDE_DOC_IDS and r[1] not in gold_doc_ids_all
        ]
        already_included = {r[0] for r in gold_rows} | {r[0] for r in forced_rows}
        other_rows = [r for r in lang_rows if r[0] not in already_included]
        sample_n = min(PER_LANGUAGE_SAMPLE, len(other_rows))
        sampled = rng.sample(other_rows, sample_n)
        lang_subset = gold_rows + forced_rows + sampled
        for chunk_id, doc_id, text, lang_ in lang_subset:
            subset.append(
                {"chunk_id": chunk_id, "doc_id": doc_id, "text": text, "language": lang_}
            )
        manifest[lang] = {
            "total_in_full_corpus": len(lang_rows),
            "gold_passages_included": len(gold_rows),
            "force_included_distractors": len(forced_rows),
            "random_sample": sample_n,
            "subset_total": len(lang_subset),
        }
        print(
            f"  {lang:10s} full={len(lang_rows):6d} gold={len(gold_rows):4d} "
            f"forced={len(forced_rows):2d} "
            f"sampled={sample_n:5d} subset_total={len(lang_subset):5d}"
        )

    print(f"\nTotal subset size: {len(subset)} chunks (full corpus: {len(rows)})")

    out = {
        "seed": SEED,
        "per_language_sample_target": PER_LANGUAGE_SAMPLE,
        "n_total": len(subset),
        "manifest": manifest,
        "chunks": subset,
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote -> {OUT_PATH}")


if __name__ == "__main__":
    main()
