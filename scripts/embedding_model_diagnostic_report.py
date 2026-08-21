"""Phase 8 (docs/DECISIONS.md ADR-017): aggregates the three per-model worker outputs
(eval/embedding_diagnostic_{e5small,bgem3,e5base}.json) into the final comparison table and
applies the decision rule -- adopt a candidate only if it materially improves Recall@10/MRR/
difficult-language performance without unacceptable memory/CPU cost, never on raw cosine score
alone (a rescaled/differently-distributed embedding space can look "more confident" while
retrieving the same or worse passages -- the decision rule explicitly guards against exactly this).

Usage: python scripts/embedding_model_diagnostic_report.py
Output: eval/embedding_diagnostic_comparison.json
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval"

MODEL_KEYS = ["e5small", "bgem3", "e5base"]
FOCUS_LANGS = ["san_Deva", "tam_Taml", "urd_Arab", "hin_Deva", "eng_Latn"]

# Real, measured disk sizes (du -sh on the HF cache), checked before this report was written.
DISK_SIZES_MB = {
    "e5small": 471,
    "bgem3": 2271,  # true single-snapshot model size; local cache holds 2 revisions (4.3GB total)
    "e5base": 1100,
}


def main() -> None:
    results = {}
    for key in MODEL_KEYS:
        path = EVAL_DIR / f"embedding_diagnostic_{key}.json"
        if not path.exists():
            print(f"MISSING: {path} -- run the worker with --model-key {key} first")
            continue
        results[key] = json.loads(path.read_text(encoding="utf-8"))

    if len(results) < len(MODEL_KEYS):
        print(f"\nOnly {len(results)}/{len(MODEL_KEYS)} models available -- partial report.")

    print("\n=== Model comparison ===")
    print(
        f"{'Model':10s} {'Dim':>6s} {'Disk(MB)':>10s} {'RAM(MB)':>10s} "
        f"{'Embed(ms)':>11s} {'Search(ms)':>12s}"
    )
    for key in MODEL_KEYS:
        if key not in results:
            continue
        r = results[key]
        print(
            f"{key:10s} {r['dim']:6d} {DISK_SIZES_MB[key]:10d} {r['final_rss_mb']:10.1f} "
            f"{r['embed_queries_ms_per_item']:11.2f} {r['avg_search_ms']:12.3f}"
        )

    print("\n=== Aggregate retrieval metrics (shared 28,565-chunk subset, 532 real queries) ===")
    print(f"{'Model':10s} {'Recall@1':>10s} {'Recall@5':>10s} {'Recall@10':>10s} {'MRR@10':>10s}")
    for key in MODEL_KEYS:
        if key not in results:
            continue
        g = results[key]["global_metrics"]
        print(
            f"{key:10s} {g['recall@1']:10.4f} {g['recall@5']:10.4f} "
            f"{g['recall@10']:10.4f} {g['mrr@10']:10.4f}"
        )

    print("\n=== Focus languages (Sanskrit/Tamil/Urdu/Hindi/English) ===")
    for lang in FOCUS_LANGS:
        print(f"\n  {lang}:")
        for key in MODEL_KEYS:
            if key not in results:
                continue
            m = results[key]["per_language_metrics"].get(lang)
            if m:
                print(
                    f"    {key:10s} R@1={m['recall@1']:.4f} R@5={m['recall@5']:.4f} "
                    f"R@10={m['recall@10']:.4f} MRR@10={m['mrr@10']:.4f}"
                )

    print("\n=== Critical cases: capital-of-India ===")
    known_distractors = {"hin_Deva::1001095_3", "hin_Deva::1149223_6", "eng_Latn::1012189_7"}
    for key in MODEL_KEYS:
        if key not in results:
            continue
        for c in results[key]["critical_cases"]:
            flag = (
                "STILL FOOLED"
                if c["top1_doc_id"] in known_distractors
                else "not the known distractor"
            )
            print(
                f"  {key:10s} {c['label']:28s} top1_score={c['top1_score']:.4f} "
                f"doc={c['top1_doc_id']} [{flag}]"
            )

    # Decision rule: material improvement on Recall@10 AND MRR AND difficult-language performance,
    # without unacceptable RAM/CPU cost. Never decided on raw score alone.
    print("\n=== Decision rule applied ===")
    if "e5small" in results:
        baseline = results["e5small"]["global_metrics"]
        baseline_focus = {
            lang: results["e5small"]["per_language_metrics"].get(lang) for lang in FOCUS_LANGS
        }
        for key in ["bgem3", "e5base"]:
            if key not in results:
                continue
            g = results[key]["global_metrics"]
            d_recall10 = g["recall@10"] - baseline["recall@10"]
            d_mrr = g["mrr@10"] - baseline["mrr@10"]
            focus_deltas = {
                lang: (
                    results[key]["per_language_metrics"][lang]["recall@10"]
                    - baseline_focus[lang]["recall@10"]
                )
                for lang in FOCUS_LANGS
                if lang in results[key]["per_language_metrics"] and baseline_focus[lang]
            }
            ram_delta_mb = results[key]["final_rss_mb"] - results["e5small"]["final_rss_mb"]
            latency_ratio = (
                results[key]["embed_queries_ms_per_item"]
                / results["e5small"]["embed_queries_ms_per_item"]
            )
            print(f"\n  {key}: dRecall@10={d_recall10:+.4f} dMRR@10={d_mrr:+.4f}")
            print(f"    focus-language dRecall@10: {focus_deltas}")
            print(f"    dRAM={ram_delta_mb:+.1f}MB  latency_ratio={latency_ratio:.2f}x")

    out = {
        "model_keys_available": list(results.keys()),
        "disk_sizes_mb": DISK_SIZES_MB,
        "results_by_model": results,
    }
    out_path = EVAL_DIR / "embedding_diagnostic_comparison.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote -> {out_path}")


if __name__ == "__main__":
    main()
