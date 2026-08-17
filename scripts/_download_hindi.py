"""Standalone, retry-wrapped download of train/hintrain.parquet. Meant to be launched as a fully
detached OS process (see the session notes in docs/RISKS.md) so it survives independently of
whatever tool session started it. Writes progress to stdout, which the launcher redirects to a
log file — poll that file's size/mtime to check progress from outside.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import _netcompat  # noqa: E402,F401

os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import hf_hub_download  # noqa: E402

MAX_ATTEMPTS = 10

for attempt in range(1, MAX_ATTEMPTS + 1):
    try:
        print(f"[attempt {attempt}/{MAX_ATTEMPTS}] starting hf_hub_download...", flush=True)
        path = hf_hub_download(
            "ai4bharat/MSMARCO-XI", "train/hintrain.parquet", repo_type="dataset"
        )
        print(f"DONE: {path}", flush=True)
        break
    except Exception as e:  # noqa: BLE001 — this is a standalone retry driver, not library code
        print(f"[attempt {attempt}] failed: {e!r}", flush=True)
        if attempt == MAX_ATTEMPTS:
            print("GAVE UP after max attempts", flush=True)
            raise
        time.sleep(5)
