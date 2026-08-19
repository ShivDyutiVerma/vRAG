"""Re-verifies the real, built `index-metadata_aware-v3` (sqfp16) artifact against the real
500-query held-out set -- requirement #8/#9 of docs/DECISIONS_R.md R-034: don't declare success
based only on R-033's offline ablation script's own temp copy, confirm the actual release-asset
artifact directly. Reuses eval_faiss_index_variants.py's own evaluation function, not a
reimplementation, so there's exactly one scoring code path to trust.

Usage: python scripts/verify_index_sqfp16.py --index-dir data/index/metadata_aware_sqfp16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_faiss_index_variants import evaluate_quality_and_latency  # noqa: E402

from vrag.index.dense import DenseIndex  # noqa: E402
from vrag.index.embedder import LiteE5Embedder  # noqa: E402

HELDOUT_PATH = REPO_ROOT / "eval" / "heldout_queries.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", required=True, type=Path)
    args = parser.parse_args()

    dense = DenseIndex.load(args.index_dir / "dense")
    print(f"Loaded {args.index_dir / 'dense'}: {len(dense)} vectors, quantization="
          f"{dense._quantization!r}")

    heldout = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    embedder = LiteE5Embedder()
    print(f"Embedding {len(heldout)} real held-out queries...")
    query_vectors = [embedder.embed_queries([q["query"]])[0] for q in heldout]

    result = evaluate_quality_and_latency(args.index_dir, heldout, query_vectors)
    print(json.dumps(result, indent=2, ensure_ascii=False))
