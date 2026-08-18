"""P6 latency campaign, task 7 (docs/BUILD_PLAN.md) -- charts from scripts/bench_latency.py's real
output (eval/latency_results.json, eval/latency_track_b_results.json). Never fabricates data; reads
only what the benchmark actually measured.

Usage: python scripts/make_latency_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "eval" / "latency_results.json"
ASSETS_DIR = REPO_ROOT / "docs" / "assets"


def make_breakdown_chart(samples: list[dict]) -> None:
    """Stacked bar: true per-stage cost (server timings_ms) vs. the doomed Track B wait, for
    "answered" queries specifically -- the split this whole benchmark exists to make visible."""
    answered = [s for s in samples if s["status"] == "answered"]
    stage_names = [
        "input_guard", "scope_guard", "retrieve", "ground_gate", "extract_answer", "output_guard",
    ]
    stage_medians = []
    for stage in stage_names:
        vals = [s["timings_ms"].get(stage, 0.0) for s in answered]
        stage_medians.append(float(np.median(vals)))

    wall_median = float(np.median([s["wall_ms"] for s in answered]))
    stage_sum = sum(stage_medians)
    track_b_wait = wall_median - stage_sum

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(stage_names) + 1))
    bottom = 0.0
    stage_colors = colors[: len(stage_names)]
    for name, val, color in zip(stage_names, stage_medians, stage_colors, strict=True):
        ax.bar("Answered query\n(P50)", val, bottom=bottom, label=name, color=color)
        bottom += val
    ax.bar(
        "Answered query\n(P50)", track_b_wait, bottom=bottom,
        label="doomed Track B wait\n(shed under budget)", color="lightgray", hatch="//",
    )
    ax.set_ylabel("Latency (ms)")
    ax.set_title(
        f"Where the P50={wall_median:.0f}ms actually goes\n"
        f"(true Track A stage cost: {stage_sum:.1f}ms — the rest is a doomed Track B attempt)"
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    out = ASSETS_DIR / "latency_breakdown.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


def make_cdf_chart(samples: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for status, color in [("answered", "C0"), ("abstained", "C1"), ("refused", "C2")]:
        vals = sorted(s["wall_ms"] for s in samples if s["status"] == status)
        if not vals:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, label=f"{status} (n={len(vals)})", color=color)
    ax.set_xlabel("Wall-clock latency (ms), log scale")
    ax.set_ylabel("Cumulative fraction of requests")
    ax.set_xscale("log")
    ax.set_title("t_pipeline CDF by response status (500 samples, real local run)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = ASSETS_DIR / "latency_cdf.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    samples = data["samples"]
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    make_breakdown_chart(samples)
    make_cdf_chart(samples)
