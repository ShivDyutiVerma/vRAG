"""Pre-Phase-3 G3 re-evaluation on the new 100k multilingual/filter candidate (docs/DECISIONS.md
ADR-011/012).

Runs TWO evaluations, deliberately kept separate, because they answer different questions:

1. The ORIGINAL 500-query Hindi held-out set (eval/heldout_queries.json) against the NEW
   multilingual index, language-filtered to Hindi. Requested verbatim by the user ("rerun the
   full 500-query evaluation"). **Important, checked directly before running this (not assumed):
   the new multilingual index's Hindi slice (771 independently reservoir-sampled rows, seed
   20260820) has ZERO passage_id overlap with the original 500 queries' gold passages (drawn from
   the OLD Hindi pipeline's first-10,000-rows working pool, seed 20260817, 10x larger and
   differently sampled).** That means this rerun's abstention rate reflects corpus-coverage
   mismatch from independent resampling, not a retrieval-quality regression -- reported plainly,
   not hidden, exactly like R-037/R-038's prior forensic discipline on this project.
2. The NEW 494-query multilingual held-out set (eval/heldout_queries_multilingual.json,
   regenerated with qualified passage_ids to match the new index) -- the actually fair,
   apples-to-apples measurement of the new index's real quality, since its gold passages ARE in
   the new index by construction.

Both use k=100 wide search + hard language-filter + top-5 slice for the real G3 decision (TAU/
MARGIN read directly from src/vrag/guardrails/g3_confidence.py, never duplicated/re-guessed).

Usage: python scripts/reeval_g3_on_multilingual_100k.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vrag.guardrails.g3_confidence import MARGIN, TAU  # noqa: E402
from vrag.index.dense import DenseIndex  # noqa: E402
from vrag.index.embedder import LiteE5Embedder  # noqa: E402
from vrag.index.sqlite_chunk_lookup import SQLiteChunkLookup  # noqa: E402
from vrag.retrieval.metrics import dedupe_doc_ids, score_hits  # noqa: E402

INDEX_DIR = REPO_ROOT / "data" / "index" / "multilingual_100k"
WIDE_K = 100
FINAL_K = 10


def g3_refuses(top1: float, weakest: float, n: int) -> bool:
    if n == 0:
        return True
    if top1 < TAU:
        return True
    return bool(n >= 2 and (top1 - weakest) < MARGIN)


def run_eval(
    label: str, queries: list[dict], get_target_lang, embedder, dense, lookup, corpus_doc_ids
):
    rows = []
    for q in queries:
        relevant_doc_ids = {p["passage_id"] for p in q["relevant_passages"]}
        target_lang = get_target_lang(q)
        vec = embedder.embed_queries([q["query"]])[0]
        hits = dense.search(vec, k=WIDE_K)

        chunk_to_doc_id = {cid: lookup.doc_id_for(cid) for cid, _s in hits}
        chunk_to_doc_id = {k_: v for k_, v in chunk_to_doc_id.items() if v is not None}

        same_lang_hits = [
            (cid, s)
            for cid, s in hits
            if lookup.doc_id_for(cid) and lookup.doc_id_for(cid).startswith(f"{target_lang}::")
        ]
        filtered = same_lang_hits if same_lang_hits else hits

        # recall/MRR/nDCG on the filtered ranking's own top-10, matching the sibling eval script
        scores = score_hits(filtered[:FINAL_K], chunk_to_doc_id, relevant_doc_ids)

        # real G3 decision: exactly what production's RetrieveStage->GroundGateStage would see,
        # k=5 slice of the (filtered) ranking, clamped scores
        clamped = [max(0.0, min(1.0, s)) for _cid, s in filtered[:5]]
        top1 = clamped[0] if clamped else 0.0
        weakest = clamped[min(4, len(clamped) - 1)] if clamped else 0.0
        abstained = g3_refuses(top1, weakest, len(clamped))

        in_top10 = scores["recall@10"] == 1.0
        all_doc_ids_20 = dedupe_doc_ids(
            [chunk_to_doc_id[cid] for cid, _s in filtered[:20] if cid in chunk_to_doc_id]
        )
        in_top20 = bool(set(all_doc_ids_20) & relevant_doc_ids)
        in_corpus = bool(relevant_doc_ids & corpus_doc_ids)

        rows.append(
            {
                **scores,
                "top1": top1,
                "abstained": abstained,
                "in_top10": in_top10,
                "in_top20": in_top20,
                "in_corpus": in_corpus,
            }
        )

    n = len(rows)
    n_abstained = sum(1 for r in rows if r["abstained"])
    abstained_rows = [r for r in rows if r["abstained"]]
    result = {
        "label": label,
        "n_queries": n,
        "recall@1": statistics.mean(r["recall@1"] for r in rows),
        "recall@5": statistics.mean(r["recall@5"] for r in rows),
        "recall@10": statistics.mean(r["recall@10"] for r in rows),
        "mrr@10": statistics.mean(r["mrr@10"] for r in rows),
        "answered": n - n_abstained,
        "abstained": n_abstained,
        "abstain_rate": n_abstained / n if n else 0.0,
        "abstained_evidence_location": {
            "found_in_top10": sum(1 for r in abstained_rows if r["in_top10"]),
            "found_in_top11_to_20": sum(
                1 for r in abstained_rows if r["in_top20"] and not r["in_top10"]
            ),
            "in_corpus_but_outside_top20": sum(
                1 for r in abstained_rows if r["in_corpus"] and not r["in_top20"]
            ),
            "not_in_corpus_at_all": sum(1 for r in abstained_rows if not r["in_corpus"]),
        },
        "top1_score_distribution_of_abstentions": {
            "min": min((r["top1"] for r in abstained_rows), default=None),
            "max": max((r["top1"] for r in abstained_rows), default=None),
            "mean": statistics.mean([r["top1"] for r in abstained_rows])
            if abstained_rows
            else None,
        },
    }
    return result


def main():
    print(f"TAU={TAU} MARGIN={MARGIN} (src/vrag/guardrails/g3_confidence.py, unchanged)")
    dense = DenseIndex.load(INDEX_DIR / "dense")
    lookup = SQLiteChunkLookup(INDEX_DIR / "chunk_lookup.sqlite3")
    embedder = LiteE5Embedder()

    import sqlite3

    conn = sqlite3.connect(str(INDEX_DIR / "chunk_lookup.sqlite3"))
    corpus_doc_ids = {row[0] for row in conn.execute("SELECT DISTINCT doc_id FROM chunks")}
    conn.close()

    original_500 = json.loads(
        (REPO_ROOT / "eval" / "heldout_queries.json").read_text(encoding="utf-8")
    )
    multilingual_494 = json.loads(
        (REPO_ROOT / "eval" / "heldout_queries_multilingual.json").read_text(encoding="utf-8")
    )

    print(f"\n=== Eval 1: original {len(original_500)} Hindi queries, filtered to hin_Deva ===")
    r1 = run_eval(
        "original_500_hindi_on_new_index",
        original_500,
        lambda q: "hin_Deva",
        embedder,
        dense,
        lookup,
        corpus_doc_ids,
    )
    print(json.dumps(r1, indent=2, ensure_ascii=False))

    print(
        f"\n=== Eval 2: new {len(multilingual_494)} multilingual queries, own-language filter ==="
    )
    r2 = run_eval(
        "multilingual_494_own_language",
        multilingual_494,
        lambda q: q["language"],
        embedder,
        dense,
        lookup,
        corpus_doc_ids,
    )
    print(json.dumps(r2, indent=2, ensure_ascii=False))

    print(
        "\n=== Regression case: capital-of-India, English + Hindi, "
        "no special-casing beyond logging ==="
    )
    for label, query, lang in [
        ("hindi", "भारत की राजधानी क्या है?", "hin_Deva"),
        ("english", "What is the capital of India?", "hin_Deva"),  # filtered into the Hindi slice
    ]:
        vec = embedder.embed_queries([query])[0]
        hits = dense.search(vec, k=WIDE_K)
        same_lang = [
            (cid, s)
            for cid, s in hits
            if lookup.doc_id_for(cid) and lookup.doc_id_for(cid).startswith(f"{lang}::")
        ]
        filtered = same_lang if same_lang else hits
        top5 = filtered[:5]
        print(f"\n[{label}] query={query!r}")
        for rank, (cid, score) in enumerate(top5, 1):
            chunk = lookup.get(cid)
            clamped = max(0.0, min(1.0, score))
            print(
                f"  #{rank} score={clamped:.4f} doc_id={chunk.doc_id if chunk else None} "
                f"text={(chunk.text[:80] + '...') if chunk else None!r}"
            )
        clamped_scores = [max(0.0, min(1.0, s)) for _c, s in top5]
        top1 = clamped_scores[0] if clamped_scores else 0.0
        weakest = clamped_scores[min(4, len(clamped_scores) - 1)] if clamped_scores else 0.0
        abstained = g3_refuses(top1, weakest, len(clamped_scores))
        print(f"  G3: top1={top1:.4f} TAU={TAU} -> abstained={abstained}")

    lookup.close()

    out = {"original_500_on_new_index": r1, "multilingual_494_on_new_index": r2}
    out_path = REPO_ROOT / "eval" / "g3_reevaluation_multilingual_100k.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote -> {out_path}")


if __name__ == "__main__":
    main()
