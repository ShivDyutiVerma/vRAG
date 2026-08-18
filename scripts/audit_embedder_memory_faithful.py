"""Faithful companion to audit_embedder_memory.py -- the first script split "load the ONNX model"
from "create the inference session" via onnx.load() + SerializeToString(), which does NOT match
what production actually does (ort.InferenceSession(path) reads+parses the file directly, in one
C++-side call, without ever materializing a full Python-side ModelProto). This script measures the
REAL production code path with no artificial split, to check whether the staged version's numbers
are trustworthy in absolute terms (they turned out not to be -- see the report).

Usage: python scripts/audit_embedder_memory_faithful.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

MODEL_DIR = REPO_ROOT / "data" / "onnx" / "multilingual-e5-small"
MODEL_FILE = MODEL_DIR / "onnx" / "model_quint8_avx2.onnx"
TOKENIZER_FILE = MODEL_DIR / "tokenizer.json"


def rss() -> int:
    return psutil.Process().memory_info().rss


if __name__ == "__main__":
    steps = {"1_before_import": {"rss_bytes": rss()}}

    from vrag.index import embedder as _embedder_module  # noqa: F401

    steps["2_after_importing_embedder_module"] = {"rss_bytes": rss()}

    import onnxruntime as ort

    steps["3_after_importing_onnxruntime"] = {"rss_bytes": rss()}

    from tokenizers import Tokenizer

    steps["4_after_importing_tokenizers_lib"] = {"rss_bytes": rss()}

    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
    tokenizer.enable_truncation(max_length=512)
    steps["5_after_loading_tokenizer"] = {"rss_bytes": rss()}

    # Exactly what LiteE5Embedder._ensure_loaded() does -- InferenceSession(path), no onnx.load()
    # detour. This single call does both "read the model file" and "build the session" at once;
    # the real API gives no way to observe an intermediate point between them.
    session = ort.InferenceSession(str(MODEL_FILE))
    steps["6_after_creating_inference_session_from_path_directly"] = {"rss_bytes": rss()}

    import numpy as np

    def run_one(text: str) -> None:
        encodings = tokenizer.encode_batch([text])
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)
        session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )

    t0 = time.perf_counter()
    run_one("query: वार्म-अप क्वेरी")
    first_ms = (time.perf_counter() - t0) * 1000
    steps["7_after_first_dummy_inference"] = {
        "rss_bytes": rss(), "first_inference_latency_ms": first_ms
    }

    latencies = []
    for i in range(20):
        t0 = time.perf_counter()
        run_one(f"query: क्वेरी {i}")
        latencies.append((time.perf_counter() - t0) * 1000)
    steps["8_after_20_subsequent_inferences"] = {
        "rss_bytes": rss(),
        "latency_p50_ms": sorted(latencies)[10],
        "latency_p100_ms": max(latencies),
    }

    print(json.dumps(steps, indent=2, ensure_ascii=False))
