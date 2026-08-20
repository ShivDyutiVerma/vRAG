"""Phase 7 (docs/DECISIONS.md ADR-016), Part 3, Candidate D: a REAL BM25 (bm25s) index built over
the full multilingual_100k corpus (107,678 chunks -- ADR-010 dropped this artifact from the
shipped candidate as dead weight for the dense-only production path, but building it fresh here
for this experiment is cheap and legitimate). Tested as a language-filtered secondary candidate
generator, fused with the real dense ranks via RRF (src/vrag/index/fusion.py's formula) -- not the
old full hybrid pipeline (R-010 already found that regresses on the Hindi-only corpus; this tests
whether BM25's IDF term-weighting behaves differently from Candidate B/E's raw-Jaccard lexical
overlap, which failed decisively -- BM25 down-weights common template words like "capital" by
document frequency, Jaccard does not).

Reuses the real dense ranks already collected in eval/g3_feature_experiment_raw.json (Phase 5) --
no new dense retrieval calls needed, only new BM25 search calls.

Usage: python scripts/rerank_candidate_d_bm25.py
Output: eval/g3_rerank_candidate_d_bm25_results.json
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from vrag.index.sparse import SparseIndex

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = REPO_ROOT / "data" / "index" / "multilingual_100k"
RAW_PATH = REPO_ROOT / "eval" / "g3_feature_experiment_raw.json"
HELDOUT_PATH = REPO_ROOT / "eval" / "heldout_queries_multilingual.json"
OUT_PATH = REPO_ROOT / "eval" / "g3_rerank_candidate_d_bm25_results.json"

CURRENT_TAU = 0.8835
BM25_WIDE_K = 100


def rrf_fuse(dense_rank: int, other_rank: int, k: int = 60) -> float:
    return 1.0 / (k + dense_rank) + 1.0 / (k + other_rank)


def main() -> None:
    print("Loading all chunks from chunk_lookup.sqlite3...")
    conn = sqlite3.connect(str(INDEX_DIR / "chunk_lookup.sqlite3"))
    rows_db = conn.execute("SELECT chunk_id, doc_id, text, metadata FROM chunks").fetchall()
    conn.close()

    chunk_ids, texts, doc_id_of, lang_of = [], [], {}, {}
    for chunk_id, doc_id, text, metadata_json in rows_db:
        chunk_ids.append(chunk_id)
        texts.append(text)
        doc_id_of[chunk_id] = doc_id
        lang_of[chunk_id] = json.loads(metadata_json).get("language")
    print(f"Loaded {len(chunk_ids)} chunks")

    print("Building real bm25s index over the full multilingual corpus...")
    t0 = time.perf_counter()
    sparse = SparseIndex()
    sparse.build(chunk_ids, texts)
    t1 = time.perf_counter()
    print(f"BM25 build time: {t1 - t0:.1f}s")

    data = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    held = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    held_by_id = {q["query_id"]: q for q in held}
    rows = data["rows"]

    n_correct_dense = n_correct_bm25 = n_correct_rrf = 0
    abstained_flip = 0
    abstained_flip_clears_tau = 0
    regressions = 0
    per_lang: dict[str, dict[str, int]] = {}
    search_times = []

    for row in rows:
        q = held_by_id[row["query_id"]]
        gold_doc_ids = {p["passage_id"] for p in q["relevant_passages"]}
        target_lang = row["language"]

        dense_hits = row["hits"]
        dense_doc_ids_ranked = list(dict.fromkeys(h["passage_id"] for h in dense_hits))
        dense_top1_correct = bool(dense_doc_ids_ranked) and dense_doc_ids_ranked[0] in gold_doc_ids

        t0 = time.perf_counter()
        bm25_hits = sparse.search(q["query"], k=BM25_WIDE_K)
        search_times.append(time.perf_counter() - t0)

        bm25_doc_ids_ranked = []
        seen = set()
        for chunk_id, _score in bm25_hits:
            if lang_of.get(chunk_id) != target_lang:
                continue
            doc_id = doc_id_of[chunk_id]
            if doc_id not in seen:
                seen.add(doc_id)
                bm25_doc_ids_ranked.append(doc_id)
        bm25_top1_correct = bool(bm25_doc_ids_ranked) and bm25_doc_ids_ranked[0] in gold_doc_ids

        # RRF fusion: real dense rank (from the wide dense window already collected) + real BM25
        # rank (language-filtered), over the union of doc_ids either side surfaced.
        dense_rank_of = {d: i + 1 for i, d in enumerate(dense_doc_ids_ranked)}
        bm25_rank_of = {d: i + 1 for i, d in enumerate(bm25_doc_ids_ranked)}
        union_doc_ids = set(dense_rank_of) | set(bm25_rank_of)
        fused = sorted(
            union_doc_ids,
            key=lambda d: -rrf_fuse(dense_rank_of.get(d, 10_000), bm25_rank_of.get(d, 10_000)),
        )
        rrf_top1_correct = bool(fused) and fused[0] in gold_doc_ids
        rrf_top1_dense_score = None
        if fused:
            for h in dense_hits:
                if h["passage_id"] == fused[0]:
                    rrf_top1_dense_score = h["score"]
                    break

        n_correct_dense += dense_top1_correct
        n_correct_bm25 += bm25_top1_correct
        n_correct_rrf += rrf_top1_correct

        if not dense_top1_correct and rrf_top1_correct:
            abstained_flip += 1
            if rrf_top1_dense_score is not None and rrf_top1_dense_score >= CURRENT_TAU:
                abstained_flip_clears_tau += 1
        if dense_top1_correct and not rrf_top1_correct:
            regressions += 1

        lang_bucket = per_lang.setdefault(target_lang, {"dense": 0, "bm25": 0, "rrf": 0, "n": 0})
        lang_bucket["dense"] += dense_top1_correct
        lang_bucket["bm25"] += bm25_top1_correct
        lang_bucket["rrf"] += rrf_top1_correct
        lang_bucket["n"] += 1

    n = len(rows)
    print(f"\nn_queries={n}  avg_bm25_search_ms={sum(search_times) / len(search_times) * 1000:.2f}")
    print("\n=== Recall@1 ===")
    print(f"  dense (baseline)  {n_correct_dense}/{n} = {n_correct_dense / n:.4f}")
    print(f"  bm25 alone        {n_correct_bm25}/{n} = {n_correct_bm25 / n:.4f}")
    print(f"  dense+bm25 RRF    {n_correct_rrf}/{n} = {n_correct_rrf / n:.4f}")
    print(
        f"\nabstained flip to correct: {abstained_flip}"
        f"  (of which clear TAU: {abstained_flip_clears_tau})"
    )
    print(f"regressions (currently correct, demoted): {regressions}")

    print(f"\n{'Language':10s} {'dense':>8s} {'bm25':>8s} {'rrf':>8s} {'n':>5s}")
    for lang in sorted(per_lang):
        b = per_lang[lang]
        print(f"{lang:10s} {b['dense']:8d} {b['bm25']:8d} {b['rrf']:8d} {b['n']:5d}")

    out = {
        "n_queries": n,
        "bm25_build_time_s": t1 - t0,
        "avg_bm25_search_ms": sum(search_times) / len(search_times) * 1000,
        "recall_at_1": {
            "dense": n_correct_dense / n,
            "bm25_alone": n_correct_bm25 / n,
            "dense_bm25_rrf": n_correct_rrf / n,
        },
        "abstained_flip_to_correct": abstained_flip,
        "abstained_flip_and_clears_tau": abstained_flip_clears_tau,
        "regressions_currently_correct_demoted": regressions,
        "per_language": per_lang,
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote -> {OUT_PATH}")


if __name__ == "__main__":
    main()
