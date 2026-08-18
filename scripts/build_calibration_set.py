"""G3 calibration set (docs/EVAL_PROTOCOL.md "Guardrail calibration (G3)", docs/BUILD_PLAN.md P5
task 3). Builds `eval/calibration.json`: 150 in-domain + 150 deliberately out-of-domain queries.

**In-domain** = a random sample of `eval/heldout_queries.json` (real corpus questions whose gold
passage *is* in the indexed 10,000-row working pool) — G3 should NOT abstain on these.

**Out-of-domain, specific to G3's actual job** (distinguishing "confidently answer" from "no good
match in this index," not off-topic/unsafe detection — that's G1/G2, upstream of retrieval and
upstream of G3 in the pipeline): real, structurally identical MSMARCO-XI Hindi questions whose gold
passage is genuinely absent from the indexed pool, drawn from the same parquet file, past row 10,000
(`docs/DECISIONS_R.md` R-003's working-pool cutoff). These are well-formed, on-topic-sounding
questions that G1/G2 would happily pass through — the point is that G3 alone has to notice retrieval
came up empty-handed for THIS corpus, without any topic/safety signal to lean on. No LLM call or new
external dataset needed; the already-downloaded, HF-cached parquet is reused for both halves.

Usage: python scripts/build_calibration_set.py
"""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Any

import _dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
HELDOUT_PATH = REPO_ROOT / "eval" / "heldout_queries.json"
CALIBRATION_PATH = REPO_ROOT / "eval" / "calibration.json"

WORKING_POOL_SIZE = 10_000  # must match build_dataset_subset.py's --pool-size default
N_IN_DOMAIN = 150
N_OUT_DOMAIN = 150
SEED = 20260818  # fixed, distinct from build_dataset_subset.py's SEED -- a different sample


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

    heldout = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    in_domain_sample = rng.sample(heldout, N_IN_DOMAIN)
    in_domain = [
        {
            "query_id": row["query_id"],
            "query": row["query"],
            "label": "in_domain",
            "relevant_passages": row["relevant_passages"],
        }
        for row in in_domain_sample
    ]
    print(f"in_domain: sampled {len(in_domain)} from {len(heldout)}-row eval/heldout_queries.json")

    print(f"Reading past row {WORKING_POOL_SIZE} of the Hindi train file for out-of-domain rows...")
    beyond_pool = list(
        itertools.islice(_dataset.iter_rows(), WORKING_POOL_SIZE, WORKING_POOL_SIZE + 20_000)
    )
    eligible = [row for row in beyond_pool if _relevant_passages(row)]
    print(f"{len(eligible)}/{len(beyond_pool)} beyond-pool rows have a real relevant passage")
    out_domain_sample = rng.sample(eligible, N_OUT_DOMAIN)
    out_domain = [
        {
            "query_id": row["query_id"],
            "query": row["query"],
            "label": "out_domain",
            # relevant_passages exist in the FULL dataset but NOT in the indexed working pool --
            # recorded for transparency, not used as a "should retrieve this" target the way
            # in-domain's is; G3's job here is just "notice nothing good was found."
            "relevant_passages": _relevant_passages(row),
        }
        for row in out_domain_sample
    ]
    print(f"out_domain: sampled {len(out_domain)} from {len(eligible)} eligible beyond-pool rows")

    calibration = in_domain + out_domain
    rng.shuffle(calibration)

    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(calibration)} calibration queries -> {CALIBRATION_PATH}")


if __name__ == "__main__":
    build()
