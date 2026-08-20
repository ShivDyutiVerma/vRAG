"""Phase 4 (docs/DECISIONS.md ADR-013): threshold sweep + rule comparison over the raw
calibration data collected by scripts/calibrate_g3_collect.py.

Pure analysis over eval/g3_calibration_multilingual_100k_raw.json -- no model/index loading, no
network, runs in under a second. Produces the sweep table, global-vs-per-language-vs-normalized
comparison, and a stability check, then writes everything to
eval/g3_threshold_sweep_multilingual_100k.json.

Definitions (all query-level, judged against the real gold passage set):
- "correct" / relevant_in_top1 == True: the retrieved top-1 passage IS one of the query's gold
  passages -- i.e. answering with it would be a grounded, correct answer.
- false-refusal: G3 abstains on a query where top-1 IS correct (a good answer needlessly refused).
- false-accept: G3 passes a query where top-1 is NOT correct (a wrong answer confidently given --
  the dangerous case; this is what "capital of India -> Bangkok" would look like if not caught).
- precision_of_accepted: of the queries G3 lets through, what fraction are actually correct.

Does NOT modify src/vrag/guardrails/g3_confidence.py or any other production file.

Usage: python scripts/calibrate_g3_sweep.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = REPO_ROOT / "eval" / "g3_calibration_multilingual_100k_raw.json"
OUT_PATH = REPO_ROOT / "eval" / "g3_threshold_sweep_multilingual_100k.json"

CURRENT_TAU = 0.8835
CURRENT_MARGIN = 0.0

# Sweep range: just below the observed global top1 minimum to just above the observed maximum
# (computed from the loaded data itself below, not hardcoded) -- covers the entire range where
# the threshold can possibly change any query's decision.
SWEEP_STEP = 0.0025


def g3_pass(top1: float, weakest: float, n: int, tau: float, margin: float) -> bool:
    if n == 0:
        return False
    if top1 < tau:
        return False
    return not (n >= 2 and (top1 - weakest) < margin)


def confusion(rows: list[dict], tau: float, margin: float) -> dict:
    tp = fp = fn = tn = 0
    for r in rows:
        n = min(5, r["n_hits_wide"])
        passed = g3_pass(r["top1_score"], r["weakest5_score"], n, tau, margin)
        correct = r["relevant_in_top1"]
        if passed and correct:
            tp += 1
        elif passed and not correct:
            fp += 1
        elif not passed and correct:
            fn += 1
        else:
            tn += 1
    n_total = len(rows)
    answered = tp + fp
    abstained = fn + tn
    return {
        "tau": round(tau, 4),
        "margin": margin,
        "n": n_total,
        "answered": answered,
        "abstained": abstained,
        "abstain_rate": round(abstained / n_total, 4) if n_total else None,
        "true_accept": tp,
        "false_accept": fp,
        "false_refusal": fn,
        "true_refusal": tn,
        "false_accept_rate": round(fp / n_total, 4) if n_total else None,
        "false_refusal_rate": round(fn / n_total, 4) if n_total else None,
        "precision_of_accepted": round(tp / answered, 4) if answered else None,
    }


def retrieval_metrics(rows: list[dict]) -> dict:
    """Threshold-independent -- these are pure retrieval numbers, constant across every sweep
    row, reported once rather than repeated per threshold."""
    return {
        "recall@1": round(statistics.mean(r["recall@1"] for r in rows), 4),
        "recall@5": round(statistics.mean(r["recall@5"] for r in rows), 4),
        "recall@10": round(statistics.mean(r["recall@10"] for r in rows), 4),
        "mrr@10": round(statistics.mean(r["mrr@10"] for r in rows), 4),
    }


def sweep_range(rows: list[dict]) -> list[float]:
    scores = [r["top1_score"] for r in rows]
    lo = round(min(scores) - 0.01, 4)
    hi = round(max(scores) + 0.005, 4)
    out = []
    tau = lo
    while tau <= hi:
        out.append(round(tau, 4))
        tau += SWEEP_STEP
    return out


def best_by_rule(sweep_table: list[dict], precision_floor: float) -> dict | None:
    """Among thresholds meeting a minimum precision_of_accepted, pick the one with the most
    answered queries (ties broken toward the higher/stricter tau). Returns None if no threshold
    in the swept range meets the floor."""
    candidates = [
        row
        for row in sweep_table
        if row["answered"] > 0
        and row["precision_of_accepted"] is not None
        and row["precision_of_accepted"] >= precision_floor
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (r["answered"], r["tau"]))


def per_language_analysis(rows: list[dict], taus: list[float], precision_floor: float) -> dict:
    langs = sorted({r["language"] for r in rows})
    out = {}
    for lang in langs:
        lang_rows = [r for r in rows if r["language"] == lang]
        table = [confusion(lang_rows, tau, CURRENT_MARGIN) for tau in taus]
        baseline = confusion(lang_rows, CURRENT_TAU, CURRENT_MARGIN)
        chosen = best_by_rule(table, precision_floor)
        out[lang] = {
            "n": len(lang_rows),
            "baseline_at_current_tau": baseline,
            "chosen_tau": chosen["tau"] if chosen else None,
            "chosen_row": chosen,
            "median_top1": round(statistics.median(r["top1_score"] for r in lang_rows), 4),
        }
    return out


def stability_check(rows: list[dict], taus: list[float], precision_floor: float) -> dict:
    """Split each language's queries by query_id parity (even/odd) into two halves; independently
    pick the best-by-rule tau on each half; report whether the two halves agree. n=38/language is
    small for tuning a free threshold -- this is a direct, cheap check for overfitting to this
    particular 532-query draw, not a substitute for a larger held-out set."""
    langs = sorted({r["language"] for r in rows})
    out = {}
    for lang in langs:
        lang_rows = [r for r in rows if r["language"] == lang]
        half_a = [r for r in lang_rows if r["query_id"] % 2 == 0]
        half_b = [r for r in lang_rows if r["query_id"] % 2 == 1]
        table_a = [confusion(half_a, tau, CURRENT_MARGIN) for tau in taus]
        table_b = [confusion(half_b, tau, CURRENT_MARGIN) for tau in taus]
        chosen_a = best_by_rule(table_a, precision_floor)
        chosen_b = best_by_rule(table_b, precision_floor)
        out[lang] = {
            "n_half_a": len(half_a),
            "n_half_b": len(half_b),
            "tau_half_a": chosen_a["tau"] if chosen_a else None,
            "tau_half_b": chosen_b["tau"] if chosen_b else None,
            "agree_within_0.02": (
                chosen_a is not None
                and chosen_b is not None
                and abs(chosen_a["tau"] - chosen_b["tau"]) <= 0.02
            ),
        }
    return out


def normalized_offset_rule(rows: list[dict], base_tau: float, cap: float = 0.03) -> dict:
    """Rule C: TAU_lang = base_tau - (median_top1_global - median_top1_lang), clipped to
    +-cap. A single global anchor (base_tau) plus one number per language (its own median top1
    score, computable offline at calibration time, no free per-language grid search needed) --
    the "explainable formula" alternative to B's independently-optimized per-language thresholds.
    """
    global_median = statistics.median(r["top1_score"] for r in rows)
    langs = sorted({r["language"] for r in rows})
    offsets = {}
    for lang in langs:
        lang_rows = [r for r in rows if r["language"] == lang]
        lang_median = statistics.median(r["top1_score"] for r in lang_rows)
        raw_offset = global_median - lang_median
        clipped = max(-cap, min(cap, raw_offset))
        tau_lang = round(base_tau - clipped, 4)
        offsets[lang] = {
            "median_top1_lang": round(lang_median, 4),
            "raw_offset": round(raw_offset, 4),
            "clipped_offset": round(clipped, 4),
            "tau_lang": tau_lang,
            "confusion": confusion(lang_rows, tau_lang, CURRENT_MARGIN),
        }
    return {
        "global_median_top1": round(global_median, 4),
        "base_tau": base_tau,
        "cap": cap,
        "per_language": offsets,
    }


def aggregate_rows(per_lang_confusions: list[dict]) -> dict:
    tp = sum(c["true_accept"] for c in per_lang_confusions)
    fp = sum(c["false_accept"] for c in per_lang_confusions)
    fn = sum(c["false_refusal"] for c in per_lang_confusions)
    tn = sum(c["true_refusal"] for c in per_lang_confusions)
    n = tp + fp + fn + tn
    answered = tp + fp
    return {
        "n": n,
        "answered": answered,
        "abstained": fn + tn,
        "abstain_rate": round((fn + tn) / n, 4) if n else None,
        "true_accept": tp,
        "false_accept": fp,
        "false_refusal": fn,
        "true_refusal": tn,
        "false_accept_rate": round(fp / n, 4) if n else None,
        "false_refusal_rate": round(fn / n, 4) if n else None,
        "precision_of_accepted": round(tp / answered, 4) if answered else None,
    }


def margin_grid(rows: list[dict], taus: list[float]) -> list[dict]:
    """Small (TAU, MARGIN) grid -- checks whether a non-zero margin (top1 vs 5th-place score gap)
    adds real discriminating power at THIS candidate's distribution, per g3_confidence.py's own
    documented rule ("if TAU is ever recalibrated, MARGIN must be re-swept at the new value too,
    the two are not independent"). Narrow margin range: gaps observed in the raw data are all
    under 0.09, so anything larger would abstain on everything.
    """
    margins = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03]
    out = []
    for tau in taus:
        for margin in margins:
            out.append(confusion(rows, tau, margin))
    return out


def stress_cases(rows: list[dict]) -> list[dict]:
    by_type = {
        "correct_top1": [r for r in rows if r["relevant_in_top1"]],
        "wrong_top1_high_score": sorted(
            (r for r in rows if not r["relevant_in_top1"]), key=lambda r: -r["top1_score"]
        )[:8],
        "correct_in_top5_not_top1": [
            r for r in rows if r["relevant_in_top5"] and not r["relevant_in_top1"]
        ],
        "correct_in_top10_not_top5": [
            r for r in rows if r["relevant_in_top10"] and not r["relevant_in_top5"]
        ],
        "low_evidence_quality": sorted(rows, key=lambda r: r["top1_score"])[:8],
    }
    samples = []
    for label, group in by_type.items():
        for r in group[:5]:
            samples.append(
                {
                    "category": label,
                    "query_id": r["query_id"],
                    "language": r["language"],
                    "query": r["query"],
                    "top1_score": r["top1_score"],
                    "relevant_in_top1": r["relevant_in_top1"],
                    "current_g3_passed": r["current_g3_passed"],
                    "top1_hit_preview": r["top_hits"][0] if r["top_hits"] else None,
                }
            )
    return samples


def main() -> None:
    data = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    rows = data["rows"]
    assert data["current_tau"] == CURRENT_TAU and data["current_margin"] == CURRENT_MARGIN

    taus = sweep_range(rows)
    print(
        f"n_queries={len(rows)}  sweep: {taus[0]}..{taus[-1]} "
        f"step={SWEEP_STEP} ({len(taus)} points)"
    )

    global_baseline = confusion(rows, CURRENT_TAU, CURRENT_MARGIN)
    print("\n=== Global baseline (current TAU=0.8835) ===")
    print(json.dumps(global_baseline, indent=2))

    global_sweep = [confusion(rows, tau, CURRENT_MARGIN) for tau in taus]
    ret_metrics = retrieval_metrics(rows)
    print(f"\nretrieval metrics (threshold-independent): {ret_metrics}")

    precision_floor = global_baseline[
        "precision_of_accepted"
    ]  # 0.348 -- don't get worse than today
    best_global = best_by_rule(global_sweep, precision_floor)
    print(f"\nBest global TAU meeting precision >= baseline ({precision_floor}): {best_global}")

    # separation diagnostics
    correct_scores = [r["top1_score"] for r in rows if r["relevant_in_top1"]]
    wrong_scores = [r["top1_score"] for r in rows if not r["relevant_in_top1"]]
    separation = {
        "correct_top1_median": round(statistics.median(correct_scores), 4),
        "correct_top1_mean": round(statistics.mean(correct_scores), 4),
        "wrong_top1_median": round(statistics.median(wrong_scores), 4),
        "wrong_top1_mean": round(statistics.mean(wrong_scores), 4),
        "note": (
            "Heavy overlap: wrong-top1 max exceeds correct-top1 median, and correct-top1 min is "
            "below wrong-top1 median -- top1 score alone is a WEAK discriminator between correct "
            "and incorrect top-1 hits on this multilingual candidate (unlike R-015's Hindi-only "
            "in-domain-vs-OOD calibration, where the signal was cleaner)."
        ),
    }
    print(f"\nseparation diagnostics: {json.dumps(separation, indent=2)}")

    per_lang = per_language_analysis(rows, taus, precision_floor)
    print("\n=== Per-language (precision-floor-constrained) ===")
    for lang, info in per_lang.items():
        print(
            f"  {lang:10s} n={info['n']:3d} median_top1={info['median_top1']:.4f} "
            f"baseline_answered={info['baseline_at_current_tau']['answered']:2d} "
            f"chosen_tau={info['chosen_tau']} "
            f"chosen_answered={(info['chosen_row'] or {}).get('answered')}"
        )

    per_lang_confusions_at_chosen = [
        info["chosen_row"] if info["chosen_row"] else info["baseline_at_current_tau"]
        for info in per_lang.values()
    ]
    aggregate_B = aggregate_rows(per_lang_confusions_at_chosen)
    print(f"\nAggregate under per-language rule (B): {aggregate_B}")

    stability = stability_check(rows, taus, precision_floor)
    n_stable = sum(1 for v in stability.values() if v["agree_within_0.02"])
    print(f"\nStability check (even/odd query_id split): {n_stable}/{len(stability)} agree <=0.02")
    for lang, info in stability.items():
        print(
            f"  {lang:10s} half_a={info['tau_half_a']} half_b={info['tau_half_b']} "
            f"agree={info['agree_within_0.02']}"
        )

    rule_c = normalized_offset_rule(rows, base_tau=CURRENT_TAU, cap=0.03)
    per_lang_confusions_c = [v["confusion"] for v in rule_c["per_language"].values()]
    aggregate_C = aggregate_rows(per_lang_confusions_c)
    print(f"\nAggregate under normalized-offset rule (C, cap=0.03): {aggregate_C}")
    for lang, info in rule_c["per_language"].items():
        print(
            f"  {lang:10s} tau_lang={info['tau_lang']} offset={info['clipped_offset']} "
            f"answered={info['confusion']['answered']}"
        )

    margin_results = margin_grid(
        rows, [CURRENT_TAU] + ([best_global["tau"]] if best_global else [])
    )
    print("\n=== Margin grid at current + best-global TAU ===")
    for row in margin_results:
        print(
            f"  tau={row['tau']} margin={row['margin']:.3f} answered={row['answered']} "
            f"precision={row['precision_of_accepted']}"
        )

    stress = stress_cases(rows)

    out = {
        "current_tau": CURRENT_TAU,
        "current_margin": CURRENT_MARGIN,
        "n_queries": len(rows),
        "retrieval_metrics": ret_metrics,
        "global_baseline": global_baseline,
        "separation_diagnostics": separation,
        "global_sweep": global_sweep,
        "best_global_meeting_baseline_precision": best_global,
        "per_language": per_lang,
        "aggregate_rule_B_per_language_optimum": aggregate_B,
        "stability_check_even_odd_split": stability,
        "rule_C_normalized_offset": rule_c,
        "aggregate_rule_C_normalized_offset": aggregate_C,
        "margin_grid_at_key_taus": margin_results,
        "stress_case_samples": stress,
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote -> {OUT_PATH}")


if __name__ == "__main__":
    main()
