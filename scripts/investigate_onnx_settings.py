"""Investigation only -- per the user's explicit instruction, does not change production code.
Sweeps ONNX Runtime SessionOptions that could plausibly reduce LiteE5Embedder's session-creation
RSS, measuring both the memory effect and the real single-query inference latency tradeoff for
each, one setting at a time (never combining two changes in one measurement, so the effect of each
is attributable). Each variant runs in an isolated subprocess so session state from one variant
can't pollute another's memory reading.

Usage: python scripts/investigate_onnx_settings.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_MODEL_DIR = REPO_ROOT / "data" / "onnx" / "multilingual-e5-small"
MODEL_FILE = _MODEL_DIR / "onnx" / "model_quint8_avx2.onnx"
TOKENIZER_FILE = _MODEL_DIR / "tokenizer.json"

VARIANTS = {
    "baseline_(current_production_defaults)": {},
    "disable_cpu_mem_arena": {"enable_cpu_mem_arena": False},
    "disable_mem_pattern": {"enable_mem_pattern": False},
    "intra_op_threads_1": {"intra_op_num_threads": 1},
    "disable_graph_optimization": {"graph_optimization_level": "ORT_DISABLE_ALL"},
    "all_memory_settings_combined": {
        "enable_cpu_mem_arena": False,
        "enable_mem_pattern": False,
        "intra_op_num_threads": 1,
    },
}

_WORKER = f'''
import sys, json, time
sys.path.insert(0, r"{REPO_ROOT / "src"}")
import psutil
import onnxruntime as ort
from tokenizers import Tokenizer
import numpy as np

config = json.loads(sys.argv[1])

def rss():
    return psutil.Process().memory_info().rss

baseline = rss()
tokenizer = Tokenizer.from_file(r"{TOKENIZER_FILE}")
tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
tokenizer.enable_truncation(max_length=512)
after_tokenizer = rss()

opts = ort.SessionOptions()
if "enable_cpu_mem_arena" in config:
    opts.enable_cpu_mem_arena = config["enable_cpu_mem_arena"]
if "enable_mem_pattern" in config:
    opts.enable_mem_pattern = config["enable_mem_pattern"]
if "intra_op_num_threads" in config:
    opts.intra_op_num_threads = config["intra_op_num_threads"]
if "graph_optimization_level" in config:
    level_name = config["graph_optimization_level"]
    opts.graph_optimization_level = getattr(ort.GraphOptimizationLevel, level_name)

session = ort.InferenceSession(r"{MODEL_FILE}", sess_options=opts)
after_session = rss()

def run_one(text):
    enc = tokenizer.encode_batch([text])
    input_ids = np.array([e.ids for e in enc], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)
    feed = {{
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }}
    session.run(None, feed)

run_one("query: warm up")  # discarded, matches every other bench script's warm-up convention
after_warmup = rss()

latencies = []
for i in range(30):
    t0 = time.perf_counter()
    run_one(f"query: latency probe number {{i}}")
    latencies.append((time.perf_counter() - t0) * 1000)

latencies.sort()
print(json.dumps({{
    "baseline_rss": baseline,
    "after_tokenizer_rss": after_tokenizer,
    "after_session_rss": after_session,
    "after_warmup_rss": after_warmup,
    "latency_p50_ms": latencies[15],
    "latency_p100_ms": latencies[-1],
    "latency_mean_ms": sum(latencies) / len(latencies),
}}))
'''

if __name__ == "__main__":
    results = {}
    for name, config in VARIANTS.items():
        print(f"\n=== {name} ===", file=sys.stderr)
        proc = subprocess.run(
            [sys.executable, "-c", _WORKER, json.dumps(config)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"FAILED: {proc.stderr[-2000:]}", file=sys.stderr)
            continue
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        results[name] = result
        print(json.dumps(result, indent=2), file=sys.stderr)

    baseline = results.get("baseline_(current_production_defaults)")
    print("\n=== Summary (session RSS delta vs. baseline, latency P50) ===")
    header = (
        f"{'variant':40s} {'session_rss_mb':>15s} {'delta_mb':>10s} "
        f"{'p50_ms':>8s} {'p100_ms':>8s}"
    )
    print(header)
    for name, r in results.items():
        rss_mb = r["after_session_rss"] / 1e6
        base_rss = baseline["after_session_rss"] if baseline else r["after_session_rss"]
        delta = (r["after_session_rss"] - base_rss) / 1e6
        print(
            f"{name:40s} {rss_mb:15.1f} {delta:10.1f} "
            f"{r['latency_p50_ms']:8.3f} {r['latency_p100_ms']:8.3f}"
        )

    with open(REPO_ROOT / "eval" / "onnx_settings_investigation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {REPO_ROOT / 'eval' / 'onnx_settings_investigation.json'}")
