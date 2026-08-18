"""Deep memory audit of LiteE5Embedder only, at finer granularity than ADR-006's single
"embedder only" data point. Diagnostic-only -- does not modify src/vrag/index/embedder.py.
Production's `_ensure_loaded()` loads tokenizer-then-session in one call; this script replicates
the same underlying operations with an RSS checkpoint inserted between each sub-step, including
splitting "read the ONNX file into memory" (`onnx.load()`) from "build the inference session from
it" (`InferenceSession(model_proto.SerializeToString())`) -- two things `InferenceSession(path)`
normally does in one call.

Usage: python scripts/audit_embedder_memory.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

MODEL_DIR = REPO_ROOT / "data" / "onnx" / "multilingual-e5-small"
MODEL_FILE = MODEL_DIR / "onnx" / "model_quint8_avx2.onnx"
TOKENIZER_FILE = MODEL_DIR / "tokenizer.json"


class RSSSampler:
    def __init__(self, interval_s: float = 0.01) -> None:
        self._proc = psutil.Process()
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak_rss = self._proc.memory_info().rss

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss
            if rss > self.peak_rss:
                self.peak_rss = rss
            time.sleep(self._interval_s)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join()
        return max(self.peak_rss, self._proc.memory_info().rss)


def rss() -> int:
    return psutil.Process().memory_info().rss


def check_dtypes(model_path: Path) -> dict:
    """Inspects every initializer (weight tensor) in the ONNX graph and buckets total bytes by
    dtype -- answers "is it actually int8, or does it contain float32 weights" precisely, not
    from the filename alone."""
    import onnx
    from onnx import TensorProto

    model = onnx.load(str(model_path))
    dtype_names = {v: k for k, v in TensorProto.DataType.items()}
    by_dtype: dict[str, dict] = {}
    for init in model.graph.initializer:
        dtype = dtype_names.get(init.data_type, f"UNKNOWN({init.data_type})")
        n_elements = 1
        for d in init.dims:
            n_elements *= d
        entry = by_dtype.setdefault(dtype, {"n_tensors": 0, "n_elements": 0})
        entry["n_tensors"] += 1
        entry["n_elements"] += n_elements
    return by_dtype


if __name__ == "__main__":
    steps: dict[str, dict] = {}

    sampler = RSSSampler()
    sampler.start()

    steps["1_before_import"] = {"rss_bytes": rss()}

    from vrag import index as _vrag_index_pkg  # noqa: F401
    from vrag.index import embedder as embedder_module  # noqa: F401

    steps["2_after_importing_embedder_module"] = {"rss_bytes": rss()}

    import onnxruntime as ort

    steps["3a_after_importing_onnxruntime"] = {
        "rss_bytes": rss(),
        "onnxruntime_version": ort.__version__,
        "available_providers": ort.get_available_providers(),
    }

    import onnx

    model_proto = onnx.load(str(MODEL_FILE))
    steps["3b_after_loading_onnx_model_into_memory"] = {
        "rss_bytes": rss(),
        "onnx_file_size_bytes": MODEL_FILE.stat().st_size,
    }

    from tokenizers import Tokenizer

    steps["4a_after_importing_tokenizers_lib"] = {"rss_bytes": rss()}

    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
    tokenizer.enable_truncation(max_length=512)
    steps["4b_after_loading_tokenizer"] = {
        "rss_bytes": rss(),
        "tokenizer_file_size_bytes": TOKENIZER_FILE.stat().st_size,
    }

    sess_options = ort.SessionOptions()
    session = ort.InferenceSession(
        model_proto.SerializeToString(), sess_options=sess_options
    )
    steps["5_after_creating_inference_session"] = {
        "rss_bytes": rss(),
        "providers_selected": session.get_providers(),
        "intra_op_num_threads": sess_options.intra_op_num_threads,
        "inter_op_num_threads": sess_options.inter_op_num_threads,
        "execution_mode": str(sess_options.execution_mode),
        "graph_optimization_level": str(sess_options.graph_optimization_level),
        "enable_cpu_mem_arena": sess_options.enable_cpu_mem_arena,
        "enable_mem_pattern": sess_options.enable_mem_pattern,
        "enable_mem_reuse": getattr(sess_options, "enable_mem_reuse", "N/A (not exposed)"),
    }

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
    run_one("query: वार्म-अप क्वेरी जो पहला डमी इनफरेंस चलाती है")
    first_inference_ms = (time.perf_counter() - t0) * 1000
    steps["6_after_first_dummy_inference"] = {
        "rss_bytes": rss(),
        "first_inference_latency_ms": first_inference_ms,
    }

    latencies = []
    for i in range(20):
        t0 = time.perf_counter()
        run_one(f"query: क्वेरी संख्या {i}")
        latencies.append((time.perf_counter() - t0) * 1000)
    steps["7_after_20_subsequent_inferences"] = {
        "rss_bytes": rss(),
        "latency_p50_ms": sorted(latencies)[10],
        "latency_p100_ms": max(latencies),
        "latency_mean_ms": sum(latencies) / len(latencies),
    }

    peak = sampler.stop()

    dtype_breakdown = check_dtypes(MODEL_FILE)

    # Duplicate-session check: does calling embed_queries()/embed_passages() via the real
    # LiteE5Embedder class ever create a second session for the same instance?
    from vrag.index.embedder import LiteE5Embedder

    real_embedder = LiteE5Embedder()
    real_embedder.embed_queries(["first call"])
    session_id_after_first_call = id(real_embedder._session)
    real_embedder.embed_queries(["second call"])
    session_id_after_second_call = id(real_embedder._session)
    real_embedder.embed_passages(["third call, different method"])
    session_id_after_third_call = id(real_embedder._session)

    report = {
        "steps": steps,
        "peak_rss_bytes_across_all_steps": peak,
        "batch_size_in_production": (
            "1 -- interface.py/hybrid.py always call embed_queries([single_query]) per request"
        ),
        "dtype_breakdown_of_onnx_weights": dtype_breakdown,
        "duplicate_session_check": {
            "same_session_object_across_calls": (
                session_id_after_first_call
                == session_id_after_second_call
                == session_id_after_third_call
            ),
            "session_id_after_1st_call": session_id_after_first_call,
            "session_id_after_2nd_call": session_id_after_second_call,
            "session_id_after_3rd_call": session_id_after_third_call,
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
