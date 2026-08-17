"""Standalone, retry-wrapped download of the multilingual-e5-small model — same rationale and
pattern as _download_hindi.py (detached process, HF_HUB_DISABLE_XET, retry loop)."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import _netcompat  # noqa: E402,F401

os.environ["HF_HUB_DISABLE_XET"] = "1"

from sentence_transformers import SentenceTransformer  # noqa: E402

MAX_ATTEMPTS = 10
MODEL_NAME = "intfloat/multilingual-e5-small"

for attempt in range(1, MAX_ATTEMPTS + 1):
    try:
        print(f"[attempt {attempt}/{MAX_ATTEMPTS}] loading {MODEL_NAME}...", flush=True)
        model = SentenceTransformer(MODEL_NAME)
        vec = model.encode(["query: sanity check"], normalize_embeddings=True)
        print(f"DONE: model loaded, sanity embedding dim={vec.shape[1]}", flush=True)
        break
    except Exception as e:  # noqa: BLE001 — standalone retry driver, not library code
        print(f"[attempt {attempt}] failed: {e!r}", flush=True)
        if attempt == MAX_ATTEMPTS:
            print("GAVE UP after max attempts", flush=True)
            raise
        time.sleep(5)
