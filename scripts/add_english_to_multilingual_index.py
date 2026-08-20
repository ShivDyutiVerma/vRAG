"""Phase 3 (docs/DECISIONS.md ADR-012): adds English as a 14th indexed language to the 100k
multilingual candidate, using the `English_passages` field already present in every MSMARCO-XI
row (Phase 0 finding) -- the shared source text before translation, real content, not a
translation-quality-questionable back-fill. 771 rows (the same per-language budget every other
language got at the 100k tier), deduped by query_id and drawn from the ALREADY-sampled 100k pool
(scripts/build_multilingual_dataset_subset.py) -- no new download, no re-sampling.

Appends to the EXISTING dense index (HNSW supports incremental add; the sqfp16 ScalarQuantizer's
per-dimension calibration doesn't need retraining -- E5 embeddings are L2-normalised regardless of
input language, so English vectors sit in the same statistical range the quantizer already
calibrated against) rather than rebuilding from scratch.

Usage: python scripts/add_english_to_multilingual_index.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent / "src"))

from vrag.chunking.base import Document  # noqa: E402
from vrag.chunking.strategies import metadata_aware  # noqa: E402
from vrag.index.dense import DenseIndex  # noqa: E402
from vrag.index.embedder import E5Embedder  # noqa: E402

DATA_DIR = REPO_ROOT.parent / "data"
INDEX_DIR = DATA_DIR / "index" / "multilingual_100k"
SLICE_PATH = DATA_DIR / "working_subset_english_slice_100k.jsonl"


def _rows_to_english_documents(rows: list[dict]) -> list[Document]:
    """Same shape as build_index._rows_to_documents, but reads English_passages (the shared
    source text, Phase 0 finding) instead of Translated_passages, tagged eng_Latn, doc_id
    qualified exactly like every other language (docs/DECISIONS.md ADR-011)."""
    docs: list[Document] = []
    for row in rows:
        english = row["passages"].get("English_passages", [])
        is_selected = row["passages"].get("is_selected", [])
        for i, text in enumerate(english):
            docs.append(
                Document(
                    doc_id=f"eng_Latn::{row['query_id']}_{i}",
                    text=text,
                    language="eng_Latn",
                    source_lang="eng_Latn",
                    query_id=row["query_id"],
                    query_type=row["query_type"],
                    is_selected=bool(is_selected[i]) if i < len(is_selected) else False,
                )
            )
    return docs


def main() -> None:
    rows = [json.loads(line) for line in SLICE_PATH.open(encoding="utf-8")]
    documents = _rows_to_english_documents(rows)
    print(f"{len(rows)} rows -> {len(documents)} English documents (passages)")

    strategy = metadata_aware.MetadataAwareChunker(mode="boost")
    t0 = time.perf_counter()
    all_chunks = []
    for doc in documents:
        all_chunks.extend(strategy.chunk(doc))
    indexable_chunks = [c for c in all_chunks if not c.metadata.get("is_parent", False)]
    print(f"chunked -> {len(indexable_chunks)} English chunks")

    embedder = E5Embedder()
    vectors = embedder.embed_passages([c.text for c in indexable_chunks])

    dense = DenseIndex.load(INDEX_DIR / "dense")
    n_before = len(dense)
    dense.add([c.chunk_id for c in indexable_chunks], vectors)
    n_after = len(dense)
    print(f"dense index: {n_before} -> {n_after} chunks (+{n_after - n_before})")
    dense.save(INDEX_DIR / "dense")

    existing_lookup: dict[str, dict] = json.loads(
        (INDEX_DIR / "chunk_lookup.json").read_text(encoding="utf-8")
    )
    n_lookup_before = len(existing_lookup)
    for c in indexable_chunks:
        existing_lookup[c.chunk_id] = c.model_dump()
    (INDEX_DIR / "chunk_lookup.json").write_text(
        json.dumps(existing_lookup, ensure_ascii=False), encoding="utf-8"
    )
    print(f"chunk_lookup.json: {n_lookup_before} -> {len(existing_lookup)} entries")

    build_seconds = time.perf_counter() - t0
    print(f"Done in {build_seconds:.1f}s. Run scripts/convert_chunk_lookup_sqlite.py next.")


if __name__ == "__main__":
    main()
