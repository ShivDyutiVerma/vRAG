"""Phase 2 (docs/DECISIONS.md ADR-010): builds a real FAISS+BM25 index from one of the
multilingual working-subset pools (scripts/build_multilingual_dataset_subset.py), reusing the
exact production build path -- same chunking strategy (metadata_aware), same offline embedder
(E5Embedder), same FAISS config (HNSW32/efConstruction=200/efSearch=64/sqfp16) -- nothing about
the embedding model, tokenizer, or FAISS variant is changed from the shipped Hindi-only index.

Deliberately does NOT touch data/index/metadata_aware/ (the real production/Render-deployed
index) or data/working_subset.jsonl (the real production working pool) -- everything here reads
and writes to size-suffixed paths only.

Usage: python scripts/build_multilingual_index.py --size 100k
       python scripts/build_multilingual_index.py --size 100k --size 150k --size 200k
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent / "src"))

import build_index  # noqa: E402 -- reuses _rows_to_documents, registered chunking strategies
from vrag.chunking.strategies import metadata_aware  # noqa: E402, F401 -- registers the strategy
from vrag.index.dense import DenseIndex  # noqa: E402
from vrag.index.embedder import E5Embedder  # noqa: E402
from vrag.index.persistence import save_built_index  # noqa: E402
from vrag.index.sparse import SparseIndex  # noqa: E402

DATA_DIR = REPO_ROOT.parent / "data"


def build_one(size_name: str) -> dict:
    subset_path = DATA_DIR / f"working_subset_multilingual_{size_name}.jsonl"
    if not subset_path.exists():
        raise FileNotFoundError(
            f"{subset_path} missing -- run scripts/build_multilingual_dataset_subset.py first"
        )
    rows = [json.loads(line) for line in subset_path.open(encoding="utf-8")]
    documents = build_index._rows_to_documents(rows)
    print(f"[{size_name}] {len(rows)} rows -> {len(documents)} documents (passages)")

    embedder = E5Embedder()
    strategy = metadata_aware.MetadataAwareChunker(mode="boost")  # matches production's own tag

    t0 = time.perf_counter()
    all_chunks = []
    for doc in documents:
        all_chunks.extend(strategy.chunk(doc))
    indexable_chunks = [c for c in all_chunks if not c.metadata.get("is_parent", False)]
    chunk_lookup = {c.chunk_id: c for c in all_chunks}
    real_ratio = len(indexable_chunks) / len(rows) if rows else 0.0
    print(f"[{size_name}] chunked -> {len(indexable_chunks)} indexable chunks "
          f"(real ratio {real_ratio:.4f} chunks/row, vs. the 9.9767 planning estimate)")

    vectors = embedder.embed_passages([c.text for c in indexable_chunks])

    # Same FAISS config as production (src/vrag/index/dense.py's own defaults: HNSW32,
    # efConstruction=200, efSearch=64), with sqfp16 quantization explicitly requested -- same
    # opt-in R-034 used for the real production index, not a new/different quantizer.
    dense = DenseIndex(dim=len(vectors[0]) if vectors else 384, quantization="sqfp16")
    dense.add([c.chunk_id for c in indexable_chunks], vectors)

    sparse = SparseIndex()
    sparse.build([c.chunk_id for c in indexable_chunks], [c.text for c in indexable_chunks])

    build_seconds = time.perf_counter() - t0

    out_dir = DATA_DIR / "index" / f"multilingual_{size_name}"
    save_built_index(dense, sparse, chunk_lookup, out_dir)

    # Real observed language distribution of what actually got indexed, not the sampling target.
    from collections import Counter

    lang_counts = Counter(c.metadata.get("language") for c in indexable_chunks)

    result = {
        "size_name": size_name,
        "n_rows": len(rows),
        "n_documents": len(documents),
        "n_chunks": len(indexable_chunks),
        "build_seconds": build_seconds,
        "language_distribution": dict(lang_counts),
        "out_dir": str(out_dir.relative_to(REPO_ROOT.parent)),
    }
    print(f"[{size_name}] built {len(indexable_chunks)} chunks in {build_seconds:.1f}s -> {out_dir}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", action="append", required=True, choices=["100k", "150k", "200k"])
    args = parser.parse_args()

    results = [build_one(size) for size in args.size]
    out_path = DATA_DIR / "multilingual_index_build_results.json"
    existing = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for r in results:
        existing[r["size_name"]] = r
    out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote build results -> {out_path}")
