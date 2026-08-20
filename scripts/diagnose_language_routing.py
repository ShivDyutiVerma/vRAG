"""Read-only diagnostic (requested 2026-08-20): trace an English text query and a Hindi text
query through the real production retrieval path -- same DenseIndex, same LiteE5Embedder, same
G2/G3 guardrail logic as production -- to see exactly what happens to each, without touching the
mic/STT leg. No production code, corpus, index, or config is modified by this script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vrag.guardrails import g2_scope_language  # noqa: E402
from vrag.guardrails.g3_confidence import MARGIN, TAU  # noqa: E402
from vrag.index.dense import DenseIndex  # noqa: E402
from vrag.index.embedder import LiteE5Embedder  # noqa: E402
from vrag.index.sqlite_chunk_lookup import SQLiteChunkLookup  # noqa: E402

INDEX_DIR = REPO_ROOT / "data" / "index" / "metadata_aware"

QUERIES = [
    ("english", "What is the capital of India?"),
    ("hindi", "भारत की राजधानी क्या है?"),
]


def main() -> None:
    dense = DenseIndex.load(INDEX_DIR / "dense")
    lookup = SQLiteChunkLookup(INDEX_DIR / "chunk_lookup.sqlite3")
    embedder = LiteE5Embedder()

    for label, query in QUERIES:
        print(f"\n{'=' * 70}\n[{label}] raw query: {query!r}")

        g2 = g2_scope_language.check(query)
        print(f"G2 verdict: passed={g2.passed} reason={g2.reason!r}")

        # No normalization stage exists in the pipeline (grepped src/vrag -- no hits) -- the
        # embedder receives ctx.query completely unmodified, prefixed only with "query: " inside
        # embed_queries() (E5's own contract, not a project-level normalization step).
        normalized_query = query
        print(f"normalized query (== raw, no normalization stage exists): {normalized_query!r}")

        vec = embedder.embed_queries([normalized_query])[0]
        hits = dense.search(vec, k=5)  # k=5, exactly what the harness passes to retrieve()

        results = []
        for rank, (chunk_id, score) in enumerate(hits, start=1):
            chunk = lookup.get(chunk_id)
            clamped = max(0.0, min(1.0, score))
            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "doc_id": chunk.doc_id if chunk else None,
                    "score_raw": score,
                    "score_clamped": clamped,
                    "language_metadata": chunk.metadata.get("language") if chunk else None,
                    "text_preview": (chunk.text[:120] + "...") if chunk else None,
                }
            )
        print("top-5 production dense search results:")
        print(json.dumps(results, ensure_ascii=False, indent=2))

        clamped_scores = [r["score_clamped"] for r in results]
        top1 = clamped_scores[0] if clamped_scores else 0.0
        weakest5 = clamped_scores[min(4, len(clamped_scores) - 1)] if clamped_scores else 0.0
        if (
            not clamped_scores
            or top1 < TAU
            or len(clamped_scores) >= 2
            and (top1 - weakest5) < MARGIN
        ):
            abstained = True
        else:
            abstained = False

        print(f"G3: top1={top1:.4f} weakest5={weakest5:.4f} TAU={TAU} -> abstained={abstained}")
        print(
            "embedding path: LiteE5Embedder (multilingual-e5-small, ONNX int8), "
            "'query: ' prefix, same code path regardless of query language -- no "
            "language-conditional branching exists in embedder.py or hybrid.py"
        )

    lookup.close()


if __name__ == "__main__":
    main()
