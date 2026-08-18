"""A1 chunking-strategy comparison bar chart (docs/BUILD_PLAN.md P2 task 6: "markdown table + a bar
chart into docs/assets/"). The table has existed in docs/EVAL_RESULTS.md §1 since A1 completed; this
was the one missing deliverable. Reads real rows straight from eval/ablation_ledger.csv rather than
hardcoding numbers, so the chart can't drift from the ledger that's the actual source of truth.

Usage: python scripts/plot_chunking_chart.py
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend -- must be set before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = REPO_ROOT / "eval" / "ablation_ledger.csv"
OUT_PATH = REPO_ROOT / "docs" / "assets" / "chunking_comparison.png"

# The one canonical A1 run_id per strategy (default config, dense-only, e5-small, no rerank) --
# metadata_aware uses all 3 noise-floor run_ids so its bar gets a real error bar, not a point
# estimate (docs/DECISIONS_R.md R-004). sentence_window uses the R-006-corrected re-run, not the
# original buggy one (docs/DECISIONS_R.md R-006).
STRATEGY_RUN_IDS = {
    "passage_native": ["passage_native_1786981991"],
    "fixed_overlap": ["fixed_overlap_1786985157"],
    "metadata_aware": [
        "metadata_aware_1787001805",
        "metadata_aware_1787024699",
        "metadata_aware_1787024922",
    ],
    "hierarchical": ["hierarchical_1787004288"],
    "semantic": ["semantic_1786999464"],
    "sentence_window": ["sentence_window_1787024474"],
}
STRATEGY_ORDER = [
    "metadata_aware",
    "passage_native",
    "fixed_overlap",
    "semantic",
    "hierarchical",
    "sentence_window",
]


def load_ledger_rows() -> dict[str, dict]:
    with LEDGER_PATH.open(encoding="utf-8") as f:
        return {row["run_id"]: row for row in csv.DictReader(f)}


def main() -> None:
    rows = load_ledger_rows()

    recall5_mean, recall5_err, recall10_mean = [], [], []
    for strategy in STRATEGY_ORDER:
        run_ids = STRATEGY_RUN_IDS[strategy]
        recall5_vals = [float(rows[rid]["recall@5"]) for rid in run_ids]
        recall10_vals = [float(rows[rid]["recall@10"]) for rid in run_ids]
        recall5_mean.append(statistics.mean(recall5_vals))
        recall5_err.append(
            (max(recall5_vals) - min(recall5_vals)) / 2 if len(recall5_vals) > 1 else 0.0
        )
        recall10_mean.append(statistics.mean(recall10_vals))

    x = range(len(STRATEGY_ORDER))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars5 = ax.bar(
        [i - width / 2 for i in x],
        recall5_mean,
        width,
        yerr=recall5_err,
        capsize=4,
        label="Recall@5",
        color="tab:blue",
    )
    bars10 = ax.bar(
        [i + width / 2 for i in x], recall10_mean, width, label="Recall@10", color="tab:orange"
    )

    ax.set_ylabel("Recall")
    ax.set_title("A1 chunking strategy comparison (dense-only, multilingual-e5-small)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(STRATEGY_ORDER, rotation=20, ha="right")
    ax.set_ylim(0, 0.85)
    ax.legend()
    ax.bar_label(bars5, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(bars10, fmt="%.3f", padding=3, fontsize=8)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved chart to {OUT_PATH}")


if __name__ == "__main__":
    main()
