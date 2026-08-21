"""Phase 9 (docs/DECISIONS.md ADR-018): a shrinkage-based per-language TAU -- the one genuinely
new idea this phase adds to ADR-013's already-exhausted global/per-language/normalized-offset
sweep. ADR-013 found: a fully free per-language TAU (its rule B) fit the aggregate numbers well
but failed its own even/odd stability check (2/14 languages agreed within 0.02) -- classic
overfitting to n=38/language. A formula-based per-language offset (its rule C) was stable but
*underperformed* the global baseline. This tests the middle ground neither ADR-013 nor Phase 4
tried: shrink each language's free-optimized TAU toward the global TAU by a tunable pseudo-count
k (standard empirical-Bayes/James-Stein-style partial pooling) --

    TAU_lang = (n_lang * TAU_lang_free + k * TAU_global) / (n_lang + k)

Large k = mostly the safe global default; small k = mostly the (overfit-prone) free per-language
value. Swept over a few k values and re-run through the SAME even/odd stability check ADR-013
used, so a shrunk rule is only ever reported as viable if it actually passes the same bar the free
version failed.

Reuses eval/g3_calibration_multilingual_100k_raw.json (Phase 4's real collection) -- no new
retrieval calls.

Usage: python scripts/calibrate_g3_shrinkage.py
Output: eval/g3_shrinkage_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = REPO_ROOT / "eval" / "g3_calibration_multilingual_100k_raw.json"
OUT_PATH = REPO_ROOT / "eval" / "g3_shrinkage_results.json"

CURRENT_TAU = 0.8835
CURRENT_MARGIN = 0.0
PRECISION_FLOOR = 0.3483  # same anchor as ADR-013 -- don't ship worse precision than today
SWEEP_STEP = 0.0025
K_VALUES = [20, 50, 100, 200, 400]


def g3_pass(
    top1: float, weakest: float, n: int, tau: float, margin: float = CURRENT_MARGIN
) -> bool:
    if n == 0:
        return False
    if top1 < tau:
        return False
    return not (n >= 2 and (top1 - weakest) < margin)


def confusion(rows: list[dict], tau: float) -> dict:
    tp = fp = fn = tn = 0
    for r in rows:
        n = min(5, r["n_hits_wide"])
        passed = g3_pass(r["top1_score"], r["weakest5_score"], n, tau)
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
    return {
        "tau": round(tau, 4),
        "n": n_total,
        "answered": answered,
        "abstained": n_total - answered,
        "abstain_rate": round((n_total - answered) / n_total, 4) if n_total else None,
        "true_accept": tp,
        "false_accept": fp,
        "false_refusal": fn,
        "true_refusal": tn,
        "false_accept_rate": round(fp / n_total, 4) if n_total else None,
        "false_refusal_rate": round(fn / n_total, 4) if n_total else None,
        "precision_of_accepted": round(tp / answered, 4) if answered else None,
    }


def sweep_range(rows: list[dict]) -> list[float]:
    scores = [r["top1_score"] for r in rows]
    lo = round(min(scores) - 0.01, 4)
    hi = round(max(scores) + 0.005, 4)
    out, tau = [], lo
    while tau <= hi:
        out.append(round(tau, 4))
        tau += SWEEP_STEP
    return out


def best_free_tau(rows: list[dict], taus: list[float]) -> float | None:
    """The free (unconstrained) per-language optimum, same rule ADR-013's rule B used: max
    answered subject to precision >= floor."""
    best = None
    for tau in taus:
        row = confusion(rows, tau)
        if row["answered"] == 0 or row["precision_of_accepted"] is None:
            continue
        if row["precision_of_accepted"] < PRECISION_FLOOR:
            continue
        if best is None or row["answered"] > best[1]:
            best = (tau, row["answered"])
    return best[0] if best else None


def shrink(n_lang: int, tau_free: float | None, tau_global: float, k: int) -> float:
    if tau_free is None:
        return tau_global
    return (n_lang * tau_free + k * tau_global) / (n_lang + k)


def aggregate(per_lang_confusions: list[dict]) -> dict:
    tp = sum(c["true_accept"] for c in per_lang_confusions)
    fp = sum(c["false_accept"] for c in per_lang_confusions)
    fn = sum(c["false_refusal"] for c in per_lang_confusions)
    tn = sum(c["true_refusal"] for c in per_lang_confusions)
    n = tp + fp + fn + tn
    answered = tp + fp
    return {
        "n": n,
        "answered": answered,
        "abstained": n - answered,
        "abstain_rate": round((n - answered) / n, 4) if n else None,
        "true_accept": tp,
        "false_accept": fp,
        "false_refusal": fn,
        "true_refusal": tn,
        "false_accept_rate": round(fp / n, 4) if n else None,
        "false_refusal_rate": round(fn / n, 4) if n else None,
        "precision_of_accepted": round(tp / answered, 4) if answered else None,
    }


def main() -> None:
    data = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    rows = data["rows"]
    assert data["current_tau"] == CURRENT_TAU

    langs = sorted({r["language"] for r in rows})
    taus = sweep_range(rows)

    # free per-language optimum (ADR-013's rule B, recomputed here for the shrinkage formula)
    tau_free_by_lang = {}
    for lang in langs:
        lang_rows = [r for r in rows if r["language"] == lang]
        tau_free_by_lang[lang] = best_free_tau(lang_rows, taus)
    print("Free per-language optima:", tau_free_by_lang)

    results_by_k = {}
    for k in K_VALUES:
        tau_shrunk = {
            lang: round(shrink(38, tau_free_by_lang[lang], CURRENT_TAU, k), 4) for lang in langs
        }
        per_lang_confusions = []
        for lang in langs:
            lang_rows = [r for r in rows if r["language"] == lang]
            per_lang_confusions.append(confusion(lang_rows, tau_shrunk[lang]))
        agg = aggregate(per_lang_confusions)

        # stability check: even/odd split, does the SHRUNK tau (using the free optimum computed
        # on each half) stay close between halves? A stable shrinkage should show much smaller
        # half-to-half swings than ADR-013's free version did.
        stability = {}
        for lang in langs:
            lang_rows = [r for r in rows if r["language"] == lang]
            half_a = [r for r in lang_rows if r["query_id"] % 2 == 0]
            half_b = [r for r in lang_rows if r["query_id"] % 2 == 1]
            free_a = best_free_tau(half_a, taus)
            free_b = best_free_tau(half_b, taus)
            shrunk_a = shrink(len(half_a), free_a, CURRENT_TAU, k)
            shrunk_b = shrink(len(half_b), free_b, CURRENT_TAU, k)
            stability[lang] = {
                "shrunk_half_a": round(shrunk_a, 4),
                "shrunk_half_b": round(shrunk_b, 4),
                "abs_diff": round(abs(shrunk_a - shrunk_b), 4),
            }
        max_diff = max(s["abs_diff"] for s in stability.values())
        mean_diff = sum(s["abs_diff"] for s in stability.values()) / len(stability)

        results_by_k[k] = {
            "tau_shrunk_by_lang": tau_shrunk,
            "aggregate": agg,
            "stability": stability,
            "max_half_diff": round(max_diff, 4),
            "mean_half_diff": round(mean_diff, 4),
        }
        print(
            f"k={k:4d}  answered={agg['answered']:3d} precision={agg['precision_of_accepted']} "
            f"false_accept={agg['false_accept']:3d}  max_half_diff={max_diff:.4f} "
            f"mean_half_diff={mean_diff:.4f}"
        )

    baseline = confusion(rows, CURRENT_TAU)
    print(f"\nBaseline (global TAU={CURRENT_TAU}): {baseline}")

    out = {
        "current_tau": CURRENT_TAU,
        "precision_floor": PRECISION_FLOOR,
        "baseline": baseline,
        "tau_free_by_lang": tau_free_by_lang,
        "k_values_tested": K_VALUES,
        "results_by_k": results_by_k,
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote -> {OUT_PATH}")


if __name__ == "__main__":
    main()
