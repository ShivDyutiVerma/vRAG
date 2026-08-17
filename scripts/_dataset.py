"""Shared helper for reading ai4bharat/MSMARCO-XI's Hindi data locally. See inspect_dataset.py's
docstring for why this goes through a direct parquet download rather than `datasets.load_dataset`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import _netcompat  # noqa: F401  — must import before any network call; forces IPv4 DNS on this network
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO_ID = "ai4bharat/MSMARCO-XI"
HINDI_TRAIN_FILE = "train/hintrain.parquet"


def local_parquet_path() -> str:
    """Downloads train/hintrain.parquet on first call (~3.7GB); instant later (HF cache)."""
    return hf_hub_download(REPO_ID, HINDI_TRAIN_FILE, repo_type="dataset")


def iter_rows(batch_size: int = 512) -> Iterator[dict[str, Any]]:
    """Lazily yields every row of the Hindi train file, in file order, without loading it all
    into memory at once. Callers slice with itertools.islice for a bounded subset."""
    pf = pq.ParquetFile(local_parquet_path())
    for batch in pf.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()
