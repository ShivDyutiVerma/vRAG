"""R-029 core test: does sentencepiece (Google's original library, loading the raw
sentencepiece.bpe.model) reproduce LiteE5Embedder's exact token IDs, at what memory cost, and what
latency? Investigation only -- does not modify src/vrag/index/embedder.py. Per the user's explicit
requirement, no replacement is recommended unless token-ID equivalence is 100%.

Pipeline replicated manually for the sentencepiece candidate, based on tokenizer.json's actual
config (verified, not assumed): sentencepiece's own .model already embeds the Precompiled
normalizer (charsmap) and the Metaspace ("_") pre-tokenization scheme natively -- those don't need
separate reimplementation, sentencepiece.encode() does them internally. What sentencepiece does
NOT do automatically is the HF post_processor's <s> ... </s> wrapping (TemplateProcessing in
tokenizer.json), so that's added manually: [BOS=0] + sp.encode(text) + [EOS=2].

Usage: python scripts/test_tokenizer_equivalence.py
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
TOKENIZER_JSON = MODEL_DIR / "tokenizer.json"
SPM_MODEL = MODEL_DIR / "sentencepiece.bpe.model"
CORPUS_PATH = REPO_ROOT / "eval" / "tokenizer_test_corpus.json"
RESULTS_PATH = REPO_ROOT / "eval" / "tokenizer_equivalence_results.json"

BOS_ID = 0
EOS_ID = 2


def rss() -> int:
    return psutil.Process().memory_info().rss


def percentile(values: list[float], pct: float) -> float:
    s = sorted(values)
    return s[min(int(len(s) * pct / 100), len(s) - 1)]


def measure_current_tokenizer(rows: list[dict]) -> dict:
    baseline = rss()
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(TOKENIZER_JSON))
    after_load = rss()

    all_ids = []
    latencies = []
    for row in rows:
        t0 = time.perf_counter()
        ids = tokenizer.encode(row["prefixed_text"]).ids
        latencies.append((time.perf_counter() - t0) * 1000)
        all_ids.append(ids)

    return {
        "backend": "tokenizers (HF, Rust) -- CURRENT PRODUCTION",
        "rss_baseline_bytes": baseline,
        "rss_after_load_bytes": after_load,
        "rss_net_bytes": after_load - baseline,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "all_ids": all_ids,
    }


def measure_sentencepiece_candidate(rows: list[dict]) -> dict:
    baseline = rss()
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(SPM_MODEL))
    after_load = rss()

    # ID remap, verified directly against sp.id_to_piece()/tokenizer.json's added_tokens, not
    # guessed: raw sentencepiece's own layout is 0=<unk> 1=<s> 2=</s> 3=<first real piece>...,
    # while HF's tokenizer.json conversion renumbered specials to 0=<s> 1=<pad> 2=</s> 3=<unk>
    # and shifted every real piece up by 1 (HF id = raw id + 1) to make room for the inserted
    # <pad> token, which sentencepiece's own vocab has no concept of (sp.pad_id() == -1).
    UNK_HF_ID = 3

    def remap(raw_id: int) -> int:
        if raw_id == 0:  # sentencepiece's own <unk>
            return UNK_HF_ID
        return raw_id + 1  # every real piece, verified via the first-mismatch diagnostic

    all_ids = []
    latencies = []
    for row in rows:
        t0 = time.perf_counter()
        core_ids = sp.encode(row["prefixed_text"], out_type=int)
        ids = [BOS_ID, *(remap(i) for i in core_ids), EOS_ID]
        latencies.append((time.perf_counter() - t0) * 1000)
        all_ids.append(ids)

    return {
        "backend": "sentencepiece (Google, C++) -- CANDIDATE",
        "rss_baseline_bytes": baseline,
        "rss_after_load_bytes": after_load,
        "rss_net_bytes": after_load - baseline,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "all_ids": all_ids,
        "spm_vocab_size": sp.vocab_size(),
    }


if __name__ == "__main__":
    rows = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(rows)} test strings")

    # Each backend measured in isolation would be more rigorous (avoids one polluting the other's
    # baseline), but the RSS numbers here are cross-checked against R-028's already-isolated
    # figure (262.3MB net for `tokenizers`) before being trusted -- see the report.
    print("\nMeasuring current tokenizer (tokenizers/Rust)...")
    current = measure_current_tokenizer(rows)
    print(f"  RSS net: {current['rss_net_bytes'] / 1e6:.1f}MB "
          f"(cross-check vs R-028's isolated 262.3MB)")
    print(f"  Latency P50/P95: {current['latency_p50_ms']:.4f}/{current['latency_p95_ms']:.4f}ms")

    print("\nMeasuring sentencepiece candidate...")
    candidate = measure_sentencepiece_candidate(rows)
    print(f"  RSS net: {candidate['rss_net_bytes'] / 1e6:.1f}MB")
    print(f"  Latency P50/P95: {candidate['latency_p50_ms']:.4f}/"
          f"{candidate['latency_p95_ms']:.4f}ms")
    print(f"  spm vocab size: {candidate['spm_vocab_size']}")

    print("\nComparing token IDs...")
    n_exact = 0
    n_total = len(rows)
    mismatches = []
    for i, row in enumerate(rows):
        cur_ids = current["all_ids"][i]
        cand_ids = candidate["all_ids"][i]
        if cur_ids == cand_ids:
            n_exact += 1
        else:
            mismatches.append({
                "index": i,
                "category": row["category"],
                "text": row["prefixed_text"],
                "current_ids": cur_ids,
                "candidate_ids": cand_ids,
            })

    equivalence_rate = n_exact / n_total
    print(f"\n=== EXACT TOKEN-ID EQUIVALENCE: {n_exact}/{n_total} = {equivalence_rate:.4%} ===")

    by_category: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_category.setdefault(row["category"], []).append(i)
    for cat, indices in by_category.items():
        cat_exact = sum(
            1 for i in indices if current["all_ids"][i] == candidate["all_ids"][i]
        )
        print(f"  {cat}: {cat_exact}/{len(indices)} = {cat_exact / len(indices):.4%}")

    if mismatches:
        print(f"\nFirst {min(5, len(mismatches))} mismatches:")
        for m in mismatches[:5]:
            print(f"  [{m['category']}] {m['text'][:60]!r}")
            print(f"    current:   {m['current_ids']}")
            print(f"    candidate: {m['candidate_ids']}")

    report = {
        "n_total": n_total,
        "n_exact_match": n_exact,
        "equivalence_rate": equivalence_rate,
        "current_tokenizer": {k: v for k, v in current.items() if k != "all_ids"},
        "candidate_tokenizer": {k: v for k, v in candidate.items() if k != "all_ids"},
        "mismatches_sample": mismatches[:50],
        "n_mismatches_total": len(mismatches),
    }
    RESULTS_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")
