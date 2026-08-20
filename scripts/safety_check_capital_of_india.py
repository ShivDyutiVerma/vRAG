"""Phase 7 (docs/DECISIONS.md ADR-016): explicit safety check for the flagship regression case
against every reranking candidate tested (B lexical, C entity, D BM25+RRF, E dense+lexical RRF) --
required by this phase's SAFETY REQUIREMENT before any candidate could even be considered, not
just at the end.

Usage: python scripts/safety_check_capital_of_india.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ["VRAG_INDEX_DIR"] = str(REPO_ROOT / "data" / "index" / "multilingual_100k")
sys.path.insert(0, str(REPO_ROOT / "src"))

from vrag.index.sparse import SparseIndex  # noqa: E402
from vrag.retrieval.interface import retrieve  # noqa: E402

CURRENT_TAU = 0.8835
INDEX_DIR = REPO_ROOT / "data" / "index" / "multilingual_100k"
_DIGIT_RUN = re.compile(r"\d+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    cur: list[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "N") or cat in ("Mn", "Mc"):
            cur.append(ch)
        else:
            if cur:
                tokens.append("".join(cur))
                cur = []
    if cur:
        tokens.append("".join(cur))
    return tokens


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def entity_score(query_tokens: list[str], hit_tokens: list[str]) -> float:
    q_nums = {t for t in query_tokens if _DIGIT_RUN.fullmatch(t)}
    h_nums = {t for t in hit_tokens if _DIGIT_RUN.fullmatch(t)}
    q_content = {t.lower() for t in query_tokens if len(t) >= 4 and not _DIGIT_RUN.fullmatch(t)}
    h_content = {t.lower() for t in hit_tokens if len(t) >= 4 and not _DIGIT_RUN.fullmatch(t)}
    return 2.0 * jaccard(q_nums, h_nums) + jaccard(q_content, h_content)


def rrf(dense_rank: int, other_rank: int, k: int = 60) -> float:
    return 1.0 / (k + dense_rank) + 1.0 / (k + other_rank)


async def main() -> None:
    print("Building BM25 index for the safety check...")
    conn = sqlite3.connect(str(INDEX_DIR / "chunk_lookup.sqlite3"))
    rows_db = conn.execute("SELECT chunk_id, doc_id, text, metadata FROM chunks").fetchall()
    conn.close()
    chunk_ids, texts, doc_id_of, lang_of = [], [], {}, {}
    for chunk_id, doc_id, text, metadata_json in rows_db:
        chunk_ids.append(chunk_id)
        texts.append(text)
        doc_id_of[chunk_id] = doc_id
        lang_of[chunk_id] = json.loads(metadata_json).get("language")
    sparse = SparseIndex()
    sparse.build(chunk_ids, texts)
    print("BM25 built.\n")

    cases = [
        ("hindi", "भारत की राजधानी क्या है?", "hin_Deva", "hi-IN"),
        ("english", "What is the capital of India?", "eng_Latn", "en-IN"),
    ]

    for label, query, target_lang, sarvam in cases:
        print(f"=== {label}: {query!r} ===")
        chunks = await retrieve(query, k=20, language=sarvam)
        dense_hits = [
            {"passage_id": c.passage_id, "score": c.score, "text": c.text} for c in chunks
        ]
        dense_doc_ids = [h["passage_id"] for h in dense_hits]
        query_tokens = tokenize(query)
        query_set = {t.lower() for t in query_tokens}

        scored = []
        for h in dense_hits:
            h_tokens = tokenize(h["text"])
            h_set = {t.lower() for t in h_tokens}
            scored.append(
                {
                    **h,
                    "lexical": jaccard(query_set, h_set),
                    "entity": entity_score(query_tokens, h_tokens),
                }
            )

        lexical_order = sorted(scored, key=lambda x: -x["lexical"])
        entity_order = sorted(scored, key=lambda x: -x["entity"])

        bm25_hits = sparse.search(query, k=100)
        bm25_doc_ids = []
        seen = set()
        for chunk_id, _score in bm25_hits:
            if lang_of.get(chunk_id) != target_lang:
                continue
            d = doc_id_of[chunk_id]
            if d not in seen:
                seen.add(d)
                bm25_doc_ids.append(d)
        dense_rank_of = {d: i + 1 for i, d in enumerate(dense_doc_ids)}
        bm25_rank_of = {d: i + 1 for i, d in enumerate(bm25_doc_ids)}
        union_d = set(dense_rank_of) | set(bm25_rank_of)
        d_bm25_rrf_order = sorted(
            union_d, key=lambda d: -rrf(dense_rank_of.get(d, 10_000), bm25_rank_of.get(d, 10_000))
        )
        lex_rank_of = {h["passage_id"]: i + 1 for i, h in enumerate(lexical_order)}
        d_lex_rrf_order = sorted(
            scored, key=lambda x: -rrf(dense_rank_of[x["passage_id"]], lex_rank_of[x["passage_id"]])
        )

        def report(name: str, top1_passage_id: str | None, top1_score: float | None):
            accept = top1_score is not None and top1_score >= CURRENT_TAU
            print(
                f"  {name:20s} top1={top1_passage_id}  dense_score={top1_score}"
                f"  -> {'*** ACCEPT (UNSAFE if wrong) ***' if accept else 'abstain (safe)'}"
            )

        report(
            "dense (baseline)",
            dense_doc_ids[0] if dense_doc_ids else None,
            dense_hits[0]["score"] if dense_hits else None,
        )
        report(
            "B lexical rerank",
            lexical_order[0]["passage_id"] if lexical_order else None,
            lexical_order[0]["score"] if lexical_order else None,
        )
        report(
            "C entity rerank",
            entity_order[0]["passage_id"] if entity_order else None,
            entity_order[0]["score"] if entity_order else None,
        )
        if d_bm25_rrf_order:
            top1_id = d_bm25_rrf_order[0]
            score = next((h["score"] for h in dense_hits if h["passage_id"] == top1_id), None)
            report("D bm25+dense RRF", top1_id, score)
        if d_lex_rrf_order:
            top1 = d_lex_rrf_order[0]
            report("E dense+lex RRF", top1["passage_id"], top1["score"])
        print()


if __name__ == "__main__":
    asyncio.run(main())
