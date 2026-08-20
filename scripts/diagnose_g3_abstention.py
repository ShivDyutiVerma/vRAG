"""Diagnostic (read-only): is "not enough evidence" (G3 abstention) primarily caused by
insufficient corpus coverage, poor retrieval ranking, or the TAU threshold itself?

Requested 2026-08-20. Runs the existing 500-query held-out eval set (`eval/heldout_queries.json`)
through the exact production retrieval config (dense-only, LiteE5Embedder, efSearch=64,
metadata_aware index, k=5 as the harness passes -- src/vrag/harness/stages.py:106) and the exact
production G3 decision (`src/vrag/guardrails/g3_confidence.py`: refuse iff top1 < TAU=0.8835,
MARGIN=0.0 structurally disabled).

Does NOT touch corpus size, TAU, FAISS config, reranker, or any production file -- read-only
measurement against the real index already on disk.

One search per query at k=20 (not two): FAISS HNSW's top-k ranking is a prefix of the graph
traversal, so hits[:5] at k=20 is identical to a real k=5 production call (same efSearch=64,
deterministic ANN traversal) -- verified this holds by construction (DenseIndex.search's HNSW
graph/efSearch never depends on the k argument). This single call gives top-5 (for the real G3
decision), top-10 and top-20 (for the coverage breakdown) without three separate searches.

Usage: python scripts/diagnose_g3_abstention.py
"""

from __future__ import annotations

import json
import sqlite3
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

INDEX_DIR = REPO_ROOT / "data" / "index" / "metadata_aware"
HELDOUT_PATH = REPO_ROOT / "eval" / "heldout_queries.json"
CALIBRATION_PATH = REPO_ROOT / "eval" / "calibration.json"
DECISIONS_R_PATH = REPO_ROOT / "docs" / "DECISIONS_R.md"
OUT_PATH = REPO_ROOT / "eval" / "g3_abstention_diagnostic.json"
K = 20  # top-20 is the widest window requested; top-5/10 are slices of the same hits


def g3_refuses(top1: float, weakest: float, n_chunks: int) -> bool:
    """Exact replica of g3_confidence.check()'s decision, operating on precomputed scores."""
    if n_chunks == 0:
        return True
    if top1 < TAU:
        return True
    return bool(n_chunks >= 2 and (top1 - weakest) < MARGIN)


def main() -> None:
    print(f"Production TAU={TAU}, MARGIN={MARGIN} (src/vrag/guardrails/g3_confidence.py)")
    print(f"Index: {INDEX_DIR} (efSearch=64, dense-only, LiteE5Embedder)")

    heldout = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    print(f"Held-out set: {len(heldout)} queries ({HELDOUT_PATH})")

    dense = DenseIndex.load(INDEX_DIR / "dense")
    lookup = SQLiteChunkLookup(INDEX_DIR / "chunk_lookup.sqlite3")
    embedder = LiteE5Embedder()

    # Full-corpus doc_id set, straight from SQLite (fast: one query, ~99,767 rows) -- this is
    # the ground truth for "does the relevant passage exist anywhere in the indexed corpus at
    # all", independent of ranking.
    conn = sqlite3.connect(str(INDEX_DIR / "chunk_lookup.sqlite3"))
    corpus_doc_ids = {row[0] for row in conn.execute("SELECT DISTINCT doc_id FROM chunks")}
    conn.close()
    print(f"Corpus doc_ids (distinct passages indexed): {len(corpus_doc_ids)}")

    rows = []
    for query_row in heldout:
        relevant_doc_ids = {p["passage_id"] for p in query_row["relevant_passages"]}
        query_vec = embedder.embed_queries([query_row["query"]])[0]
        hits = dense.search(query_vec, k=K)  # k=20; hits[:5]/[:10] are identical to real k=5/k=10
        # calls (HNSW top-k is a prefix of the graph traversal at fixed efSearch -- see docstring)

        chunk_to_doc_id = {cid: lookup.doc_id_for(cid) for cid, _s in hits}
        chunk_to_doc_id = {k_: v for k_, v in chunk_to_doc_id.items() if v is not None}

        # recall@1/5/10 + mrr@10: score_hits() on hits[:10], matching every prior ablation
        # script's methodology (scripts/eval_retrieval_mode.py, eval_corpus_size.py) exactly --
        # NOT on the full 20-window, which would let a doc_id first-seen at position 11-20 leak
        # into the "top 10" via score_hits' internal dedupe-before-slice.
        hits10 = hits[:10]
        scores = score_hits(hits10, chunk_to_doc_id, relevant_doc_ids)

        # top-20 coverage window: a genuinely separate, wider check (not a slice of the above)
        raw_doc_ids_20 = [chunk_to_doc_id[cid] for cid, _s in hits if cid in chunk_to_doc_id]
        deduped_doc_ids_20 = dedupe_doc_ids(raw_doc_ids_20)

        clamped = [max(0.0, min(1.0, s)) for _cid, s in hits]
        top1 = clamped[0] if clamped else 0.0
        weakest5 = clamped[min(4, len(clamped) - 1)] if clamped else 0.0
        # exactly what the production harness passes to G3: a real k=5 retrieve() call
        abstained = g3_refuses(top1, weakest5, min(len(clamped), 5))

        in_top10 = scores["recall@10"] == 1.0
        in_top20 = bool(set(deduped_doc_ids_20) & relevant_doc_ids)
        in_corpus = bool(relevant_doc_ids & corpus_doc_ids)

        rows.append(
            {
                "query_id": query_row["query_id"],
                "query_type": query_row.get("query_type"),
                **scores,
                "top1_score": top1,
                "weakest5_score": weakest5,
                "abstained": abstained,
                "in_top10": in_top10,
                "in_top20": in_top20,
                "in_corpus": in_corpus,
            }
        )

    lookup.close()

    # --- 1. Overall retrieval quality ---
    overall = {
        "recall@1": statistics.mean(r["recall@1"] for r in rows),
        "recall@5": statistics.mean(r["recall@5"] for r in rows),
        "recall@10": statistics.mean(r["recall@10"] for r in rows),
        "mrr@10": statistics.mean(r["mrr@10"] for r in rows),
        "n_queries": len(rows),
    }

    # --- 2. G3 answered vs abstained ---
    n_abstained = sum(1 for r in rows if r["abstained"])
    n_answered = len(rows) - n_abstained

    # --- 3. Among abstained queries: where does the evidence actually sit? ---
    abstained_rows = [r for r in rows if r["abstained"]]
    n_abst_top10 = sum(1 for r in abstained_rows if r["in_top10"])
    n_abst_top20 = sum(1 for r in abstained_rows if r["in_top20"] and not r["in_top10"])
    n_abst_in_corpus_not_top20 = sum(
        1 for r in abstained_rows if r["in_corpus"] and not r["in_top20"]
    )
    n_abst_not_in_corpus = sum(1 for r in abstained_rows if not r["in_corpus"])

    # --- 4. Score distribution where evidence exists but G3 abstained anyway ---
    false_refusal_rows = [r for r in abstained_rows if r["in_corpus"]]
    top1_scores_fr = [r["top1_score"] for r in false_refusal_rows]

    # --- 5. False-refusal rate, existing methodology: heldout set is "in-domain" by
    # construction (every query's ground-truth passage was drawn from the indexed corpus,
    # docs/EVAL_PROTOCOL.md); false refusal = abstained despite relevant evidence existing.
    false_refusal_rate_heldout = (
        len(false_refusal_rows) / len(heldout) if heldout else 0.0
    )

    # --- 6. Compare against the original G3 calibration set (eval/calibration.json) ---
    calibration_summary = None
    if CALIBRATION_PATH.exists():
        calib = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        n_in = sum(1 for c in calib if c["label"] == "in_domain")
        n_out = sum(1 for c in calib if c["label"] == "out_domain")
        calibration_summary = {
            "n_in_domain": n_in,
            "n_out_domain": n_out,
            "reported_false_refusal_at_TAU": 0.193,  # docs/DECISIONS_R.md R-015 / g3_confidence.py
            "reported_correct_refusal_at_TAU": 0.753,
            "note": (
                "calibration.json's in-domain false-refusal (19.3%) was measured on 150 queries "
                "drawn from the same distribution but scored differently (single top1/top5 pull, "
                "not full Recall@k) -- this diagnostic's false_refusal_rate_heldout is the "
                "comparable number on the actual 500-query retrieval eval set."
            ),
        }

    # --- 7. Would 200k corpus plausibly help? Use existing R-024/corpus-size measurements. ---
    corpus_size_note = (
        "eval/corpus_size_tmp/ contains prior subsampled-index artifacts (20k, 50k chunks) from "
        "an earlier corpus-size investigation (scripts/eval_corpus_size.py, referenced by "
        "docs/DECISIONS_R.md R-024's follow-up). That prior work measured recall/coverage as "
        "corpus SHRINKS from the current ~99,767-chunk production size -- it does not directly "
        "answer what happens if corpus GROWS to 200k, since MSMARCO-XI's additional rows are a "
        "different, not-yet-indexed slice of the dataset. The only evidence this diagnostic can "
        "honestly offer for the growth question is indirect: see 'in_corpus' vs 'not_in_corpus' "
        "breakdown below -- if abstentions are overwhelmingly cases where the answer already IS "
        "in the current corpus (ranking/threshold problem), more corpus rows would not fix them; "
        "only the 'not_in_corpus' bucket is a candidate for corpus-size being the cause, and even "
        "for those, growing to 200k only helps if the missing passage happens to be in the next "
        "100k rows specifically, which existing measurements cannot confirm (would require "
        "actually indexing those rows -- not done here per instruction)."
    )

    report = {
        "config": {
            "TAU": TAU,
            "MARGIN": MARGIN,
            "index_dir": str(INDEX_DIR),
            "corpus_chunks": len(dense),
            "corpus_distinct_passages": len(corpus_doc_ids),
            "k_searched": K,
            "k_passed_to_g3_in_production": 5,
        },
        "1_overall_retrieval_quality": overall,
        "2_g3_answered_vs_abstained": {
            "answered": n_answered,
            "abstained": n_abstained,
            "abstain_rate": n_abstained / len(rows),
        },
        "3_abstained_evidence_location": {
            "n_abstained": len(abstained_rows),
            "found_in_top10": n_abst_top10,
            "found_in_top11_to_20": n_abst_top20,
            "found_in_corpus_but_outside_top20": n_abst_in_corpus_not_top20,
            "not_in_corpus_at_all": n_abst_not_in_corpus,
        },
        "4_score_distribution_false_refusals": {
            "n_queries_evidence_exists_but_abstained": len(false_refusal_rows),
            "top1_score_min": min(top1_scores_fr) if top1_scores_fr else None,
            "top1_score_max": max(top1_scores_fr) if top1_scores_fr else None,
            "top1_score_mean": statistics.mean(top1_scores_fr) if top1_scores_fr else None,
            "top1_score_median": statistics.median(top1_scores_fr) if top1_scores_fr else None,
            "TAU": TAU,
            "note": (
                "these are queries where the correct passage is somewhere in the corpus "
                "(in_corpus=True) yet G3 abstained (top1 < TAU) -- the score gap between "
                "top1_score_* above and TAU is the size of the threshold's conservatism, in "
                "score units, on real queries with real evidence."
            ),
        },
        "5_false_refusal_rate_heldout_methodology": {
            "false_refusal_rate": false_refusal_rate_heldout,
            "definition": (
                "fraction of the 500 held-out (in-domain-by-construction) queries where G3 "
                "abstained despite the relevant passage existing somewhere in the indexed corpus"
            ),
        },
        "6_comparison_to_prior_g3_calibration": calibration_summary,
        "7_corpus_size_200k_investigation": {
            "not_in_corpus_count": n_abst_not_in_corpus,
            "not_in_corpus_share_of_abstentions": (
                n_abst_not_in_corpus / len(abstained_rows) if abstained_rows else 0.0
            ),
            "note": corpus_size_note,
        },
    }

    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull report written to {OUT_PATH}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
