"""Builds the index-metadata_aware-v3 release asset: the exact same corpus, chunk_ids, sparse
index, and chunk_lookup as v2 -- the ONLY thing that changes is `dense/faiss.index`, rebuilt as
`IndexHNSWSQ` + `ScalarQuantizer.QT_fp16` instead of `IndexHNSWFlat` (docs/DECISIONS_R.md R-033/
R-034). Same M/efConstruction/efSearch/METRIC_INNER_PRODUCT as production.

Reconstructs every vector from the CURRENT production dense index via `faiss.Index.reconstruct()`
-- no re-embedding, no re-chunking, so this is provably the exact same corpus/vectors R-033's
ablation measured, not a re-derived approximation that could silently drift from it (e.g. if
`data/working_subset.jsonl` had changed since v2 was built). `chunk_lookup.json`,
`chunk_lookup.sqlite3`, and `sparse/` are copied byte-for-byte from the current v2 directory,
never rebuilt -- corpus size, SQLite lookup architecture, and BM25 behaviour are all explicitly
out of scope for this change.

Usage: python scripts/build_index_sqfp16.py --out-dir data/index/metadata_aware_sqfp16
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import faiss

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vrag.index.dense import DenseIndex  # noqa: E402

SOURCE_INDEX_DIR = REPO_ROOT / "data" / "index" / "metadata_aware"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=SOURCE_INDEX_DIR,
        help="the current production index directory to reconstruct vectors and copy "
        "chunk_lookup/sparse from (default: data/index/metadata_aware)",
    )
    args = parser.parse_args()

    if not (args.source_dir / "dense" / "faiss.index").exists():
        raise FileNotFoundError(
            f"{args.source_dir / 'dense' / 'faiss.index'} not found -- expected the current "
            "production index (index-metadata_aware-v2) to already be present locally"
        )

    print(f"Reading real vectors + chunk_ids from {args.source_dir / 'dense'}...")
    import json

    meta = json.loads((args.source_dir / "dense" / "meta.json").read_text(encoding="utf-8"))
    chunk_ids = meta["chunk_ids"]
    dim = meta["dim"]
    real_index = faiss.read_index(str(args.source_dir / "dense" / "faiss.index"))
    if real_index.ntotal != len(chunk_ids):
        raise RuntimeError(
            f"source index ntotal={real_index.ntotal} != len(chunk_ids)={len(chunk_ids)} -- "
            "refusing to proceed on an inconsistent source index"
        )
    print(f"  {len(chunk_ids)} vectors, dim={dim}")

    print("Reconstructing all vectors (no re-embedding)...")
    vectors = [real_index.reconstruct(i).tolist() for i in range(len(chunk_ids))]

    print("Building IndexHNSWSQ + ScalarQuantizer.QT_fp16 (M=32, efConstruction=200)...")
    t0 = time.perf_counter()
    new_dense = DenseIndex(dim=dim, quantization="sqfp16")
    new_dense.add(chunk_ids, vectors)
    build_s = time.perf_counter() - t0
    print(f"  build_s={build_s:.1f} ntotal={len(new_dense)}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    new_dense.save(out_dir / "dense")
    print(f"  saved to {out_dir / 'dense'}")

    print("Copying chunk_lookup.json, chunk_lookup.sqlite3, sparse/ verbatim from source (not "
          "rebuilt -- corpus/SQLite/BM25 are all out of scope for this change)...")
    for name in ("chunk_lookup.json", "chunk_lookup.sqlite3"):
        src = args.source_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
            print(f"  copied {name}")
    src_sparse = args.source_dir / "sparse"
    if src_sparse.exists():
        dst_sparse = out_dir / "sparse"
        if dst_sparse.exists():
            shutil.rmtree(dst_sparse)
        shutil.copytree(src_sparse, dst_sparse)
        print("  copied sparse/")

    new_size = (out_dir / "dense" / "faiss.index").stat().st_size
    old_size = (args.source_dir / "dense" / "faiss.index").stat().st_size
    print(
        f"\ndense/faiss.index: {old_size / 1e6:.1f}MB -> {new_size / 1e6:.1f}MB "
        f"({(old_size - new_size) / 1e6:+.1f}MB)"
    )
    print(f"Done. New index at: {out_dir}")


if __name__ == "__main__":
    main()
