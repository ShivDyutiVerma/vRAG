"""Builds a FAISS + BM25 index pair from a chunking strategy applied to the working subset
(data/working_subset.jsonl, produced by build_dataset_subset.py). One index per (strategy, config)
combination — this is what scripts/eval_chunking.py calls once per ablation run (docs/TECH_MENU.md
§A). Hierarchical chunking indexes only child chunks (parents are generation-time lookups, not
retrieval candidates) — see src/vrag/chunking/strategies/hierarchical.py.

Index build is a one-time offline cost (AGENT_BUILD_SPEC.md §3.2) — this script is deliberately not
optimised for speed the way the hot-path retrieval code must be.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from vrag.chunking.base import Chunk, Document
from vrag.chunking.registry import get_strategy
from vrag.chunking.strategies import (  # noqa: F401  — import registers every strategy
    fixed_overlap,
    hierarchical,
    metadata_aware,
    passage_native,
    semantic,
    sentence_window,
)
from vrag.index.dense import DenseIndex
from vrag.index.embedder import EMBEDDER_REGISTRY, E5Embedder
from vrag.index.persistence import load_built_index, save_built_index
from vrag.index.sparse import SparseIndex

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKING_SUBSET_PATH = REPO_ROOT / "data" / "working_subset.jsonl"


@dataclass
class BuiltIndex:
    dense: DenseIndex
    sparse: SparseIndex
    chunk_lookup: dict[str, Chunk]  # chunk_id -> Chunk, needed for hierarchical parent lookups
    build_seconds: float
    n_chunks: int


def _rows_to_documents(
    rows: list[dict], qualify_doc_id_by_language: bool = False
) -> list[Document]:
    """One Document per translated passage (AGENT_BUILD_SPEC.md §6.1 — passage is the natural
    retrievable unit before any chunking strategy is applied).

    `qualify_doc_id_by_language` (default False, docs/DECISIONS.md ADR-009/ADR-011): every
    existing caller (the single-language Hindi pipeline, `eval/heldout_queries.json`'s
    `passage_id` convention) keeps the exact `f"{query_id}_{i}"` doc_id it always has -- changing
    that format would silently break passage_id compatibility with the frozen Hindi held-out set.
    MSMARCO-XI's `query_id` is a *global* identifier shared across all 13 language files (the same
    underlying English query, translated 13 ways) -- a multilingual build that samples rows from
    more than one language file can therefore draw the same query_id twice, colliding two
    genuinely different chunks onto one doc_id/chunk_id (found and quantified in ADR-009: 0.67-
    1.27% of rows). Passing True (the multilingual build path only) qualifies doc_id with the
    row's own `target_lang` tag, which IS unique per language file, guaranteeing global uniqueness
    without changing anything about how a single-language build behaves.
    """
    docs: list[Document] = []
    for row in rows:
        translated = row["passages"].get("Translated_passages", [])
        is_selected = row["passages"].get("is_selected", [])
        for i, text in enumerate(translated):
            doc_id = (
                f"{row['target_lang']}::{row['query_id']}_{i}"
                if qualify_doc_id_by_language
                else f"{row['query_id']}_{i}"
            )
            docs.append(
                Document(
                    doc_id=doc_id,
                    text=text,
                    language=row["target_lang"],
                    source_lang=row["source_lang"],
                    query_id=row["query_id"],
                    query_type=row["query_type"],
                    is_selected=bool(is_selected[i]) if i < len(is_selected) else False,
                )
            )
    return docs


def build(
    strategy_name: str, strategy_kwargs: dict, embedder_name: str = E5Embedder.name
) -> BuiltIndex:
    if not WORKING_SUBSET_PATH.exists():
        raise FileNotFoundError(
            f"{WORKING_SUBSET_PATH} doesn't exist — run scripts/build_dataset_subset.py first"
        )

    rows = [json.loads(line) for line in WORKING_SUBSET_PATH.open(encoding="utf-8")]
    documents = _rows_to_documents(rows)

    try:
        embedder_cls = EMBEDDER_REGISTRY[embedder_name]
    except KeyError:
        raise KeyError(
            f"Unknown embedder '{embedder_name}'. Registered: {sorted(EMBEDDER_REGISTRY)}"
        ) from None
    embedder = embedder_cls()

    registered = get_strategy(strategy_name)
    if strategy_name == "semantic":
        # SemanticChunker needs a real sentence embedder injected — it deliberately has no
        # default (src/vrag/chunking/strategies/semantic.py) so this isn't a silent dependency.
        strategy_kwargs = {**strategy_kwargs, "embed_fn": embedder.embed_passages}
    strategy = registered.__class__(**strategy_kwargs) if strategy_kwargs else registered

    t0 = time.perf_counter()
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(strategy.chunk(doc))

    # Hierarchical chunking emits parent chunks too — index children only, keep parents in the
    # lookup for generation-time context expansion.
    indexable_chunks = [c for c in all_chunks if not c.metadata.get("is_parent", False)]
    chunk_lookup = {c.chunk_id: c for c in all_chunks}

    vectors = embedder.embed_passages([c.text for c in indexable_chunks])

    dense = DenseIndex(dim=len(vectors[0]) if vectors else 384)
    dense.add([c.chunk_id for c in indexable_chunks], vectors)

    sparse = SparseIndex()
    sparse.build([c.chunk_id for c in indexable_chunks], [c.text for c in indexable_chunks])

    build_seconds = time.perf_counter() - t0
    return BuiltIndex(
        dense=dense,
        sparse=sparse,
        chunk_lookup=chunk_lookup,
        build_seconds=build_seconds,
        n_chunks=len(indexable_chunks),
    )


def save(built: BuiltIndex, path: str | Path) -> None:
    """Persists a BuiltIndex to disk so it can be loaded fast at API boot instead of rebuilt —
    AGENT_BUILD_SPEC.md §5.3: never build FAISS at container start."""
    save_built_index(built.dense, built.sparse, built.chunk_lookup, path)


def load(path: str | Path) -> tuple[DenseIndex, SparseIndex, dict[str, Chunk]]:
    """The fast-path counterpart to save() — no chunking, no embedding, just deserialisation."""
    return load_built_index(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="passage_native")
    parser.add_argument("--embedder", default=E5Embedder.name, choices=sorted(EMBEDDER_REGISTRY))
    parser.add_argument(
        "--save-dir",
        default=None,
        help="if set, persist the built index here (e.g. data/index/metadata_aware)",
    )
    args = parser.parse_args()
    result = build(args.strategy, {}, embedder_name=args.embedder)
    print(f"Built '{args.strategy}' index: {result.n_chunks} chunks in {result.build_seconds:.1f}s")
    if args.save_dir:
        save(result, args.save_dir)
        print(f"Saved to {args.save_dir}")
