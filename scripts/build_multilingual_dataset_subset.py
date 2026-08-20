"""Phase 2 (docs/DECISIONS.md ADR-010): multilingual counterpart to build_dataset_subset.py.

Replaces the Hindi-only pipeline's "first N rows in file order" bias with real, reproducible
random sampling (Algorithm-R reservoir sampling, fixed seed) across all 13 real MSMARCO-XI train
languages (src/vrag/languages.py's SUPPORTED_LANGUAGES / Phase 0 audit).

**Row-group reality check, done empirically before writing this (not assumed):** every one of
these parquet files has exactly ONE row group (confirmed via pyarrow metadata on the real
hintrain.parquet), which means there is no cheap partial/range-based read available -- any random
sample requires reading through the whole file once. This script therefore reservoir-samples over
a LOCAL, fully-downloaded copy of each language's file (huggingface_hub's own on-disk cache,
already used by scripts/_dataset.py for Hindi) rather than attempting a partial remote read.

**Nested pools, not three independent samples.** The 100k pool is a random subset of the 150k
pool, which is a random subset of the 200k pool -- not three separately-drawn samples. This
guarantees a single held-out eval set (drawn from the smallest, 100k pool) has its gold passages
present in all three corpora, so Recall@k differences across corpus sizes measure a real "more
corpus helps or doesn't" effect, not an artifact of the held-out set's own coverage changing
between sizes.

Row-to-chunk ratio: measured 9.9767 chunks/row on the real Hindi build (99,767 chunks / 10,000
rows, scripts/build_dataset_subset.py's existing default). Used here as the planning ratio for
every language (same MS-MARCO-derived passage-list structure); the ACTUAL resulting chunk count
per language is measured for real at chunking time (scripts/build_multilingual_index.py), not
assumed to hit the target exactly.

Usage: python scripts/build_multilingual_dataset_subset.py
"""

from __future__ import annotations

import itertools
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _netcompat  # noqa: E402, F401 -- forces IPv4 DNS, must import before any network call
from huggingface_hub import hf_hub_download  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from vrag.languages import SARVAM_TO_MSMARCO_XI  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
EVAL_DIR = REPO_ROOT / "eval"

SEED = 20260820  # fixed -- distinct from build_dataset_subset.py's 20260817, documented here once
MSMARCO_LANGS = sorted(set(SARVAM_TO_MSMARCO_XI.values()))  # 13 codes: asm, ben, guj, ...
CHUNKS_PER_ROW = 9.9767  # measured, see module docstring

TARGET_CHUNKS = {"100k": 100_000, "150k": 150_000, "200k": 200_000}
HELDOUT_TOTAL = 500  # matches the existing Hindi-only heldout_queries.json size, for comparability


def _local_parquet_path(lang: str) -> str:
    return hf_hub_download("ai4bharat/MSMARCO-XI", f"train/{lang}train.parquet", repo_type="dataset")


def _reservoir_sample(lang: str, k: int, rng: random.Random) -> list[dict[str, Any]]:
    """Algorithm R, one sequential pass, O(k) memory regardless of file size -- real uniform
    random sampling over the entire file, not a prefix."""
    reservoir: list[dict[str, Any]] = []
    pf = pq.ParquetFile(_local_parquet_path(lang))
    n_seen = 0
    for batch in pf.iter_batches(batch_size=2048):
        for row in batch.to_pylist():
            n_seen += 1
            if len(reservoir) < k:
                reservoir.append(row)
            else:
                j = rng.randint(0, n_seen - 1)
                if j < k:
                    reservoir[j] = row
    return reservoir


def _passage_id(query_id: int, passage_index: int) -> str:
    return f"{query_id}_{passage_index}"


def _relevant_passages(row: dict[str, Any]) -> list[dict[str, Any]]:
    translated = row["passages"].get("Translated_passages", [])
    is_selected = row["passages"].get("is_selected", [])
    return [
        {"passage_id": _passage_id(row["query_id"], i), "text": text}
        for i, (text, selected) in enumerate(zip(translated, is_selected, strict=True))
        if selected
    ]


def build() -> None:
    rng = random.Random(SEED)
    rows_per_lang_200k = round(TARGET_CHUNKS["200k"] / len(MSMARCO_LANGS) / CHUNKS_PER_ROW)
    rows_per_lang_150k = round(TARGET_CHUNKS["150k"] / len(MSMARCO_LANGS) / CHUNKS_PER_ROW)
    rows_per_lang_100k = round(TARGET_CHUNKS["100k"] / len(MSMARCO_LANGS) / CHUNKS_PER_ROW)
    print(
        f"Planned rows/language -> 100k:{rows_per_lang_100k} 150k:{rows_per_lang_150k} "
        f"200k:{rows_per_lang_200k} (x{len(MSMARCO_LANGS)} languages)"
    )

    pools_200k: dict[str, list[dict[str, Any]]] = {}
    pools_150k: dict[str, list[dict[str, Any]]] = {}
    pools_100k: dict[str, list[dict[str, Any]]] = {}
    target_lang_by_code: dict[str, str] = {}  # real FLORES-style tag observed in the data itself

    for lang in MSMARCO_LANGS:
        print(f"Reservoir-sampling {rows_per_lang_200k} rows from {lang}train.parquet ...")
        superset = _reservoir_sample(lang, rows_per_lang_200k, rng)
        pools_200k[lang] = superset
        if superset:
            target_lang_by_code[lang] = superset[0]["target_lang"]
        # Nested subsets -- random draws FROM the superset already drawn, not independent samples.
        idx_150 = rng.sample(range(len(superset)), min(rows_per_lang_150k, len(superset)))
        pools_150k[lang] = [superset[i] for i in idx_150]
        idx_100 = rng.sample(idx_150, min(rows_per_lang_100k, len(idx_150)))
        pools_100k[lang] = [superset[i] for i in idx_100]
        print(f"  {lang}: 200k-pool={len(superset)} 150k-pool={len(pools_150k[lang])} "
              f"100k-pool={len(pools_100k[lang])} target_lang={target_lang_by_code.get(lang)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "seed": SEED,
        "chunks_per_row_ratio_used": CHUNKS_PER_ROW,
        # Real observed target_lang tags (e.g. "hin_Deva") per MSMARCO-XI 3-letter code -- read
        # directly from sampled rows, not guessed against a FLORES table. Used by
        # scripts/eval_multilingual_retrieval.py to map Sarvam's query_language to the chunk
        # metadata language tag for filter/boost experiments.
        "target_lang_by_msmarco_code": target_lang_by_code,
        "sizes": {},
    }

    for size_name, pools in (("100k", pools_100k), ("150k", pools_150k), ("200k", pools_200k)):
        out_path = DATA_DIR / f"working_subset_multilingual_{size_name}.jsonl"
        all_rows = [row for lang_rows in pools.values() for row in lang_rows]
        with out_path.open("w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        per_lang_counts = {lang: len(rows) for lang, rows in pools.items()}
        manifest["sizes"][size_name] = {
            "path": str(out_path.relative_to(REPO_ROOT)),
            "total_rows": len(all_rows),
            "rows_per_language": per_lang_counts,
            "estimated_chunks": round(len(all_rows) * CHUNKS_PER_ROW),
        }
        print(f"Wrote {len(all_rows)} rows ({size_name} pool) -> {out_path}")

    # Held-out set: drawn ONLY from the smallest (100k) pool, so its gold passages are guaranteed
    # present in all three corpora (100k subset of 150k subset of 200k) -- same methodology as
    # build_dataset_subset.py (real is_selected passage required), proportional across languages.
    per_lang_heldout = max(1, HELDOUT_TOTAL // len(MSMARCO_LANGS))
    heldout: list[dict[str, Any]] = []
    for lang, rows in pools_100k.items():
        eligible = [r for r in rows if _relevant_passages(r)]
        chosen = rng.sample(eligible, min(per_lang_heldout, len(eligible)))
        for row in chosen:
            heldout.append(
                {
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "query_type": row["query_type"],
                    "language": row["target_lang"],
                    "msmarco_lang_code": lang,
                    "relevant_passages": _relevant_passages(row),
                }
            )
    heldout_path = EVAL_DIR / "heldout_queries_multilingual.json"
    with heldout_path.open("w", encoding="utf-8") as f:
        json.dump(heldout, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(heldout)} multilingual held-out query->passage pairs -> {heldout_path}")
    manifest["heldout"] = {
        "path": str(heldout_path.relative_to(REPO_ROOT)),
        "total": len(heldout),
        "per_language_target": per_lang_heldout,
        "drawn_from": "100k pool (nested subset of 150k and 200k)",
    }

    manifest_path = DATA_DIR / "multilingual_dataset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    build()
