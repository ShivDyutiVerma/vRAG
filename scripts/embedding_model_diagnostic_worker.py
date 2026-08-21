"""Phase 8 (docs/DECISIONS.md ADR-017): per-model worker, run as an ISOLATED SUBPROCESS (one
model per process, matching this project's established real-RAM-measurement methodology, e.g.
scripts/audit_multilingual_memory.py) so RAM numbers for each model are never contaminated by a
previously-loaded model still resident in the same process.

Same corpus (eval/embedding_diagnostic_subset.json), same 532 queries, same gold labels, same
language filtering (wide search then filter, same shape as HybridRetriever's real mechanism), same
FAISS methodology (src/vrag/index/dense.py's own DenseIndex, real HNSW32) and same evaluation code
(src/vrag/retrieval/metrics.py) as production/every prior phase. Only the embedding model differs
between runs of this worker.

Candidates (checked what's already available locally first, per the instructions):
BAAI/bge-m3 was already fully cached on this machine (no download needed) -- a modern,
retrieval-focused multilingual model, tests "does a fundamentally stronger/newer architecture
close the gap". google/LaBSE (a translation-alignment-trained model, directly targeting Phase 7's
"translation-induced lexical variance" finding) was considered but its local cache turned out
incomplete (config/tokenizer present, no weights) and a fresh download hung in this environment's
known Hub-connectivity conditions (same class of issue documented in scripts/_netcompat.py) --
swapped for intfloat/multilingual-e5-base instead: same architecture/tokenizer/prefix convention
as the production baseline (isolates "does more capacity within the same family help" as a clean,
independent variable) and, sharing infrastructure already proven reliable in this environment, a
more realistically-usable candidate per the instructions' own criterion.

Usage: python scripts/embedding_model_diagnostic_worker.py --model-key e5small|bgem3|e5base
Output: eval/embedding_diagnostic_{model_key}.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import psutil  # noqa: E402

from vrag.index.dense import DenseIndex  # noqa: E402
from vrag.retrieval.metrics import (  # noqa: E402
    dedupe_doc_ids,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

SUBSET_PATH = REPO_ROOT / "eval" / "embedding_diagnostic_subset.json"
HELDOUT_PATH = REPO_ROOT / "eval" / "heldout_queries_multilingual.json"

MODEL_SPECS = {
    "e5small": {
        "hf_name": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "batch_size": 64,
        "max_seq_length": None,
    },
    "e5base": {
        "hf_name": "intfloat/multilingual-e5-base",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "batch_size": 64,
        "max_seq_length": None,
    },
    "bgem3": {
        "hf_name": "BAAI/bge-m3",
        "query_prefix": "",
        "passage_prefix": "",
        # bge-m3 supports up to 8192-token context (vs e5's ~512) -- a real, previously-hit
        # failure mode: batch_size=64 with unbounded sequence length caused a genuine CPU OOM
        # (RuntimeError: tried to allocate 17,179,869,184 bytes, from attention-mask expansion
        # scaling with the batch's longest sequence). Capped here to what this corpus's real
        # passage lengths need (MSMARCO-XI passages are short, chunked well under 512 tokens) --
        # itself a real, worth-reporting CPU-deployability cost of this model, not a benchmark
        # artifact papered over.
        "batch_size": 16,
        "max_seq_length": 512,
    },
}

WIDE_K = 100
FINAL_K = 20


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def peak_wset_mb() -> float:
    try:
        return psutil.Process().memory_info().peak_wset / 1e6  # type: ignore[attr-defined]
    except AttributeError:
        return rss_mb()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, choices=list(MODEL_SPECS))
    args = parser.parse_args()
    spec = MODEL_SPECS[args.model_key]

    baseline_rss = rss_mb()
    print(f"[{args.model_key}] baseline RSS: {baseline_rss:.1f}MB", flush=True)

    from sentence_transformers import SentenceTransformer

    t0 = time.perf_counter()
    model = SentenceTransformer(spec["hf_name"], device="cpu")
    if spec["max_seq_length"] is not None:
        model.max_seq_length = spec["max_seq_length"]
    load_s = time.perf_counter() - t0
    after_load_rss = rss_mb()
    dim = model.get_sentence_embedding_dimension()
    print(
        f"[{args.model_key}] loaded in {load_s:.1f}s, dim={dim}, RSS={after_load_rss:.1f}MB",
        flush=True,
    )

    subset = json.loads(SUBSET_PATH.read_text(encoding="utf-8"))["chunks"]
    held = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    print(f"[{args.model_key}] embedding {len(subset)} passages...", flush=True)

    passage_texts = [spec["passage_prefix"] + c["text"] for c in subset]
    t0 = time.perf_counter()
    passage_vectors = model.encode(
        passage_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=spec["batch_size"],
    )
    embed_passages_s = time.perf_counter() - t0
    after_embed_rss = rss_mb()
    print(
        f"[{args.model_key}] embedded passages in {embed_passages_s:.1f}s "
        f"({embed_passages_s / len(subset) * 1000:.2f}ms/passage), RSS={after_embed_rss:.1f}MB",
        flush=True,
    )

    dense = DenseIndex(dim=dim)
    chunk_ids = [c["chunk_id"] for c in subset]
    dense.add(chunk_ids, passage_vectors.tolist())
    chunk_to_doc = {c["chunk_id"]: c["doc_id"] for c in subset}
    chunk_to_lang = {c["chunk_id"]: c["language"] for c in subset}

    queries = [q["query"] for q in held]
    query_texts = [spec["query_prefix"] + q for q in queries]
    t0 = time.perf_counter()
    query_vectors = model.encode(
        query_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=spec["batch_size"],
    )
    embed_queries_s = time.perf_counter() - t0
    per_query_embed_ms = embed_queries_s / len(queries) * 1000
    print(
        f"[{args.model_key}] embedded {len(queries)} queries in {embed_queries_s:.1f}s "
        f"({per_query_embed_ms:.2f}ms/query)",
        flush=True,
    )

    # real per-query timed search latency (separate from the batch embed above)
    search_times = []
    per_query_results = []
    for q, qvec in zip(held, query_vectors.tolist(), strict=True):
        target_lang = q["language"]
        gold_doc_ids = {p["passage_id"] for p in q["relevant_passages"]}

        t0 = time.perf_counter()
        hits = dense.search(qvec, k=WIDE_K)
        search_times.append(time.perf_counter() - t0)

        same_lang_doc_ids = []
        seen = set()
        for chunk_id, _score in hits:
            if chunk_to_lang.get(chunk_id) != target_lang:
                continue
            d = chunk_to_doc[chunk_id]
            if d not in seen:
                seen.add(d)
                same_lang_doc_ids.append(d)
        filtered = (
            same_lang_doc_ids
            if same_lang_doc_ids
            else dedupe_doc_ids([chunk_to_doc[c] for c, _s in hits])
        )
        doc_ids = filtered[:FINAL_K]

        per_query_results.append(
            {
                "query_id": q["query_id"],
                "language": target_lang,
                "recall@1": recall_at_k(doc_ids, gold_doc_ids, k=1),
                "recall@5": recall_at_k(doc_ids, gold_doc_ids, k=5),
                "recall@10": recall_at_k(doc_ids, gold_doc_ids, k=10),
                "mrr@10": reciprocal_rank(doc_ids, gold_doc_ids, k=10),
                "ndcg@10": ndcg_at_k(doc_ids, gold_doc_ids, k=10),
                "top1_doc_id": doc_ids[0] if doc_ids else None,
                "top1_correct": bool(doc_ids) and doc_ids[0] in gold_doc_ids,
            }
        )

    avg_search_ms = sum(search_times) / len(search_times) * 1000

    # Critical cases (task-required, not in the 532-query set): capital-of-India in both
    # languages, against the same forced-included known distractors every model sees identically.
    critical_cases = [
        ("hindi_capital_of_india", "भारत की राजधानी क्या है?", "hin_Deva"),
        ("english_capital_of_india", "What is the capital of India?", "eng_Latn"),
    ]
    critical_results = []
    for label, q_text, target_lang in critical_cases:
        qvec = model.encode(
            [spec["query_prefix"] + q_text], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()
        hits = dense.search(qvec, k=WIDE_K)
        same_lang = [(c, s) for c, s in hits if chunk_to_lang.get(c) == target_lang]
        top = same_lang[0] if same_lang else (hits[0] if hits else (None, None))
        top_chunk_id, top_score = top
        critical_results.append(
            {
                "label": label,
                "query": q_text,
                "top1_chunk_id": top_chunk_id,
                "top1_doc_id": chunk_to_doc.get(top_chunk_id) if top_chunk_id else None,
                "top1_score": top_score,
                "top1_is_known_distractor": chunk_to_doc.get(top_chunk_id)
                in {"hin_Deva::1001095_3", "hin_Deva::1149223_6", "eng_Latn::1012189_7"},
            }
        )
        doc = chunk_to_doc.get(top_chunk_id)
        print(f"[{args.model_key}] CRITICAL {label}: top1_score={top_score} doc={doc}")

    final_rss = rss_mb()
    peak_wset = peak_wset_mb()

    import statistics

    def agg(rows: list[dict], metric: str) -> float:
        return statistics.mean(r[metric] for r in rows)

    global_metrics = {
        m: agg(per_query_results, m)
        for m in ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"]
    }
    langs = sorted({r["language"] for r in per_query_results})
    per_lang_metrics = {
        lang: {
            m: agg([r for r in per_query_results if r["language"] == lang], m)
            for m in ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"]
        }
        for lang in langs
    }

    print(f"\n[{args.model_key}] GLOBAL: {global_metrics}")
    for lang in langs:
        print(f"  {lang:10s} {per_lang_metrics[lang]}")

    print(f"\n[{args.model_key}] Focus languages (Sanskrit/Tamil/Urdu/Hindi/English):")
    for lang in ["san_Deva", "tam_Taml", "urd_Arab", "hin_Deva", "eng_Latn"]:
        if lang in per_lang_metrics:
            print(f"  {lang:10s} {per_lang_metrics[lang]}")

    out = {
        "model_key": args.model_key,
        "hf_name": spec["hf_name"],
        "dim": dim,
        "n_passages": len(subset),
        "n_queries": len(queries),
        "load_time_s": load_s,
        "embed_passages_total_s": embed_passages_s,
        "embed_passages_ms_per_item": embed_passages_s / len(subset) * 1000,
        "embed_queries_total_s": embed_queries_s,
        "embed_queries_ms_per_item": per_query_embed_ms,
        "avg_search_ms": avg_search_ms,
        "baseline_rss_mb": baseline_rss,
        "after_load_rss_mb": after_load_rss,
        "after_embed_rss_mb": after_embed_rss,
        "final_rss_mb": final_rss,
        "peak_wset_mb": peak_wset,
        "global_metrics": global_metrics,
        "per_language_metrics": per_lang_metrics,
        "critical_cases": critical_results,
        "per_query_results": per_query_results,
    }
    out_path = REPO_ROOT / "eval" / f"embedding_diagnostic_{args.model_key}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote -> {out_path}")


if __name__ == "__main__":
    main()
