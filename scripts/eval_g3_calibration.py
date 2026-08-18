"""G3 confidence-gate calibration (docs/EVAL_PROTOCOL.md "Guardrail calibration (G3)",
docs/BUILD_PLAN.md P5 task 3). Sweeps `TAU` (top1 retrieval score threshold) and `MARGIN`
(top1 - weakest-of-top5) against `eval/calibration.json`'s 150 in-domain + 150 out-of-domain
queries, replicating `g3_confidence.py`'s exact refusal logic so the sweep measures what the real
guardrail actually does, not an approximation of it.

Retrieval config matches production exactly: metadata_aware/multilingual-e5-small/dense-only,
efSearch=64 (A1-A3 + efSearch winners, docs/DECISIONS_R.md R-004/R-009/R-010/R-014).

Two-stage sweep, since TAU and MARGIN interact (refuse = top1 < TAU OR margin < MARGIN, an OR
gate — sweeping both as one 2D grid muddies which threshold is doing the work):
  1. Sweep TAU alone (MARGIN fixed at 0.0, which neutralises that gate) -> a clean ROC-style curve,
     chart saved to docs/assets/g3_calibration.png.
  2. At the chosen TAU, sweep MARGIN alone to see its marginal contribution before finalising both.

Usage: python scripts/eval_g3_calibration.py --index-dir data/index/metadata_aware
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_index
import matplotlib

from vrag.index.embedder import E5Embedder

matplotlib.use("Agg")  # headless backend -- must be set before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_PATH = REPO_ROOT / "eval" / "calibration.json"
ASSETS_DIR = REPO_ROOT / "docs" / "assets"

TARGET_FALSE_REFUSAL = 0.10  # in-domain, want below
TARGET_CORRECT_REFUSAL = 0.80  # out-of-domain, want above


def _g3_would_refuse(top1: float, weakest: float, n_chunks: int, tau: float, margin: float) -> bool:
    """Exact replica of g3_confidence.py's check() decision logic (not re-imported, since that
    module operates on RetrievedChunk objects from a live request -- this operates on precomputed
    scores for a bulk sweep). Keep in sync with src/vrag/guardrails/g3_confidence.py by hand."""
    if n_chunks == 0:
        return True
    if top1 < tau:
        return True
    return bool(n_chunks >= 2 and (top1 - weakest) < margin)


def get_scores(index_dir: str, k: int = 10) -> list[dict]:
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    dense, _sparse, _chunk_lookup = build_index.load(index_dir)
    embedder = E5Embedder()

    results = []
    for row in calibration:
        query_vec = embedder.embed_queries([row["query"]])[0]
        hits = dense.search(query_vec, k=k)
        scores = [max(0.0, min(1.0, s)) for _cid, s in hits]  # same clamp as hybrid.py
        top1 = scores[0] if scores else 0.0
        weakest = scores[min(4, len(scores) - 1)] if scores else 0.0
        results.append(
            {
                "query_id": row["query_id"],
                "label": row["label"],
                "top1": top1,
                "weakest": weakest,
                "n_chunks": len(scores),
            }
        )
    return results


def _refusal_count(rows: list[dict], tau: float, margin: float) -> int:
    return sum(
        1 for r in rows if _g3_would_refuse(r["top1"], r["weakest"], r["n_chunks"], tau, margin)
    )


def sweep_tau(scored: list[dict], margin: float, n_points: int = 101) -> list[dict]:
    top1_values = [r["top1"] for r in scored]
    lo, hi = min(top1_values), max(top1_values)
    span = hi - lo if hi > lo else 1.0
    taus = [lo + span * i / (n_points - 1) for i in range(n_points)]

    in_domain = [r for r in scored if r["label"] == "in_domain"]
    out_domain = [r for r in scored if r["label"] == "out_domain"]

    curve = []
    for tau in taus:
        false_refusals = _refusal_count(in_domain, tau, margin)
        correct_refusals = _refusal_count(out_domain, tau, margin)
        curve.append(
            {
                "tau": tau,
                "false_refusal_rate": false_refusals / len(in_domain),
                "correct_refusal_rate": correct_refusals / len(out_domain),
            }
        )
    return curve


def sweep_margin(scored: list[dict], tau: float, n_points: int = 51) -> list[dict]:
    margins_observed = [r["top1"] - r["weakest"] for r in scored if r["n_chunks"] >= 2]
    lo, hi = 0.0, max(margins_observed) if margins_observed else 0.1
    span = hi - lo if hi > lo else 1.0

    in_domain = [r for r in scored if r["label"] == "in_domain"]
    out_domain = [r for r in scored if r["label"] == "out_domain"]

    curve = []
    for i in range(n_points):
        margin = lo + span * i / (n_points - 1)
        false_refusals = _refusal_count(in_domain, tau, margin)
        correct_refusals = _refusal_count(out_domain, tau, margin)
        curve.append(
            {
                "margin": margin,
                "false_refusal_rate": false_refusals / len(in_domain),
                "correct_refusal_rate": correct_refusals / len(out_domain),
            }
        )
    return curve


def pick_operating_point(curve: list[dict], key: str) -> dict:
    """Prefer points meeting both targets; among those, the one with the best correct-refusal rate
    (most conservative on the out-of-domain side without breaking the false-refusal budget). If no
    point meets both targets, fall back to the one minimising the sum of target shortfalls."""
    meeting_both = [
        p
        for p in curve
        if p["false_refusal_rate"] <= TARGET_FALSE_REFUSAL
        and p["correct_refusal_rate"] >= TARGET_CORRECT_REFUSAL
    ]
    if meeting_both:
        return max(meeting_both, key=lambda p: p["correct_refusal_rate"])

    def shortfall(p: dict) -> float:
        false_over = max(0.0, p["false_refusal_rate"] - TARGET_FALSE_REFUSAL)
        correct_under = max(0.0, TARGET_CORRECT_REFUSAL - p["correct_refusal_rate"])
        return false_over + correct_under

    return min(curve, key=shortfall)


def plot_tau_curve(curve: list[dict], chosen: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    false_rates = [p["false_refusal_rate"] for p in curve]
    correct_rates = [p["correct_refusal_rate"] for p in curve]
    ax.plot(false_rates, correct_rates, marker=".", linestyle="-", color="tab:blue", alpha=0.7)
    ax.scatter(
        [chosen["false_refusal_rate"]],
        [chosen["correct_refusal_rate"]],
        color="tab:red",
        s=120,
        zorder=5,
        label=f"chosen tau={chosen['tau']:.3f}",
    )
    ax.axvline(
        TARGET_FALSE_REFUSAL, color="gray", linestyle="--", linewidth=1, label="target: false < 10%"
    )
    ax.axhline(
        TARGET_CORRECT_REFUSAL, color="gray", linestyle=":", linewidth=1, label="target: correct>80"
    )
    ax.set_xlabel("False-refusal rate (in-domain, want low)")
    ax.set_ylabel("Correct-refusal rate (out-of-domain, want high)")
    ax.set_title("G3 calibration: TAU sweep (MARGIN=0)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default="data/index/metadata_aware")
    args = parser.parse_args()

    print("Scoring 300 calibration queries against the production index...")
    scored = get_scores(args.index_dir)

    top1_in = [r["top1"] for r in scored if r["label"] == "in_domain"]
    top1_out = [r["top1"] for r in scored if r["label"] == "out_domain"]
    print(
        f"in_domain top1:  min={min(top1_in):.3f} max={max(top1_in):.3f} "
        f"mean={sum(top1_in) / len(top1_in):.3f}"
    )
    print(
        f"out_domain top1: min={min(top1_out):.3f} max={max(top1_out):.3f} "
        f"mean={sum(top1_out) / len(top1_out):.3f}"
    )

    print("\nSweeping TAU (MARGIN=0)...")
    tau_curve = sweep_tau(scored, margin=0.0)
    chosen_tau_point = pick_operating_point(tau_curve, key="tau")

    print("\n--- reference points along the curve ---")
    for target_false in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        candidates = [p for p in tau_curve if p["false_refusal_rate"] <= target_false]
        if candidates:
            best = max(candidates, key=lambda p: p["correct_refusal_rate"])
            print(
                f"false<={target_false:.2f}: tau={best['tau']:.4f} "
                f"actual_false={best['false_refusal_rate']:.3f} "
                f"correct={best['correct_refusal_rate']:.3f}"
            )
        else:
            print(f"false<={target_false:.2f}: unreachable at any tau")

    print("\n--- chosen (sum-shortfall-minimizing) point ---")
    print(json.dumps(chosen_tau_point, indent=2))

    plot_tau_curve(tau_curve, chosen_tau_point, ASSETS_DIR / "g3_calibration.png")
    print(f"Saved chart to {ASSETS_DIR / 'g3_calibration.png'}")

    print(f"\nSweeping MARGIN at TAU={chosen_tau_point['tau']:.4f}...")
    margin_curve = sweep_margin(scored, tau=chosen_tau_point["tau"])
    chosen_margin_point = pick_operating_point(margin_curve, key="margin")
    print(json.dumps(chosen_margin_point, indent=2))

    print("\n=== Recommended operating point ===")
    print(f"TAU    = {chosen_tau_point['tau']:.4f}")
    print(f"MARGIN = {chosen_margin_point['margin']:.4f}")
    print(
        f"At (TAU={chosen_tau_point['tau']:.4f}, MARGIN={chosen_margin_point['margin']:.4f}): "
        f"false_refusal_rate(in-domain)={chosen_margin_point['false_refusal_rate']:.3f}, "
        f"correct_refusal_rate(out-domain)={chosen_margin_point['correct_refusal_rate']:.3f}"
    )
