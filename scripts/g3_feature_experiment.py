"""Phase 5 (docs/DECISIONS.md ADR-014): cheap deterministic confidence-signal experiment.

Phase 4 (ADR-013) found top1 cosine score is a weak discriminator between correct and incorrect
top-1 retrieval on the multilingual candidate (correct/wrong distributions heavily overlap). This
script investigates whether OTHER cheap, deterministic, CPU-only signals -- computable from query
text, retrieved passage text/scores/metadata alone, with NO neural model, NO LLM, NO network call
-- do better, either alone or in simple two-feature combinations.

Gold labels (`relevant_in_top1`, from eval/heldout_queries_multilingual.json's real MSMARCO-XI
is_selected passages) are used ONLY to evaluate each feature (AUC / precision / coverage), never
to construct a feature value. Every feature function takes only (query text, retrieved hits: text
+ score + language + rank) as input -- exactly what's available at real inference time.

**Real tokenization bug found and fixed while building this:** Python's `re` module's `\\w` does
NOT include Unicode combining marks (categories Mn/Mc) -- naive `\\w+` tokenization shatters
Devanagari/Bengali/Gujarati/etc. text at every vowel sign (matra), e.g. "भारत" (4 real characters)
splits into 4 single-character garbage tokens instead of staying whole. `tokenize()` below fixes
this by treating L*/N*/Mn/Mc as word-continuing, verified against real Hindi and English query
text before being used in any feature.

Input: eval/g3_feature_experiment_raw.json (scripts/collect_g3_feature_data.py).
Output: eval/g3_feature_experiment_results.json.

Usage: python scripts/g3_feature_experiment.py
"""

from __future__ import annotations

import json
import time
import unicodedata
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = REPO_ROOT / "eval" / "g3_feature_experiment_raw.json"
OUT_PATH = REPO_ROOT / "eval" / "g3_feature_experiment_results.json"

CURRENT_TAU = 0.8835
PRECISION_FLOOR = 0.3483  # Phase 4 (ADR-013) baseline precision_of_accepted -- same anchor,
# so Phase 5 candidates are directly comparable to Phase 4's, not judged against a new bar.


# --------------------------------------------------------------------------------------------
# Tokenization (see module docstring for the Mn/Mc bug this fixes)
# --------------------------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    cur: list[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "N") or cat in ("Mn", "Mc"):
            cur.append(ch)
        else:
            if cur:
                tokens.append("".join(cur))
                cur = []
    if cur:
        tokens.append("".join(cur))
    return tokens


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# --------------------------------------------------------------------------------------------
# Feature extraction -- every function receives ONLY query text + retrieved hits (text, score,
# language, rank) + the query's own target language. No gold label ever enters here.
# --------------------------------------------------------------------------------------------


def extract_features(query: str, hits: list[dict], target_lang: str) -> dict:
    scores = [h["score"] for h in hits]
    n = len(scores)
    top1 = scores[0] if n >= 1 else 0.0
    top2 = scores[1] if n >= 2 else top1
    top3 = scores[2] if n >= 3 else top2
    top5_window = scores[: min(5, n)]
    top10_window = scores[: min(10, n)]

    gap12 = top1 - top2
    gap15mean = top1 - (float(np.mean(top5_window)) if top5_window else 0.0)

    if len(top10_window) >= 2 and float(np.std(top10_window)) > 1e-9:
        zscore_top1 = (top1 - float(np.mean(top10_window))) / float(np.std(top10_window))
    else:
        zscore_top1 = 0.0

    same_lang_consistency = (
        sum(1 for h in hits[:10] if h["language"] == target_lang) / len(hits[:10])
        if hits[:10]
        else 0.0
    )

    query_tokens = [t.lower() for t in tokenize(query)]
    query_set = set(query_tokens)
    query_content_set = {t for t in query_set if len(t) >= 4}

    top1_text = hits[0]["text"] if hits else ""
    top1_tokens = [t.lower() for t in tokenize(top1_text)]
    top1_set = set(top1_tokens)
    top1_content_set = {t for t in top1_set if len(t) >= 4}

    lexical_overlap_top1 = jaccard(query_set, top1_set)
    content_overlap_top1 = jaccard(query_content_set, top1_content_set)

    # Mutual redundancy: do the top-3 passages textually agree with EACH OTHER (a proxy for
    # "multiple independent hits converge on the same content", without any entity/NER model)?
    top3_token_sets = [set(t.lower() for t in tokenize(h["text"])) for h in hits[:3]]
    pair_jaccards = []
    for i in range(len(top3_token_sets)):
        for j in range(i + 1, len(top3_token_sets)):
            pair_jaccards.append(jaccard(top3_token_sets[i], top3_token_sets[j]))
    mutual_redundancy_top3 = float(np.mean(pair_jaccards)) if pair_jaccards else 0.0

    concentration_ratio = top1 / sum(top5_window) if sum(top5_window) > 1e-9 else 0.0
    score_std_top5 = float(np.std(top5_window)) if top5_window else 0.0

    # Entropy of a LINEAR (not softmax -- no arbitrary temperature hyperparameter) normalization
    # of the top-10 score window: p_i = (s_i - min) / sum(s_i - min). 0 = fully peaked (all mass
    # on rank 1), log(k)-normalized 1.0 = perfectly uniform (no signal which rank is "the" hit).
    if len(top10_window) >= 2:
        arr = np.array(top10_window, dtype=float)
        shifted = arr - arr.min()
        total = shifted.sum()
        if total > 1e-9:
            p = shifted / total
            p_nonzero = p[p > 1e-12]
            ent = -float(np.sum(p_nonzero * np.log(p_nonzero)))
            entropy_norm = ent / np.log(len(top10_window))
        else:
            entropy_norm = 1.0  # all scores identical -> maximally uniform/unconfident
    else:
        entropy_norm = 0.0

    return {
        "top1": top1,
        "top2": top2,
        "top3": top3,
        "gap12": gap12,
        "gap15mean": gap15mean,
        "zscore_top1": zscore_top1,
        "same_lang_consistency": same_lang_consistency,
        "lexical_overlap_top1": lexical_overlap_top1,
        "content_overlap_top1": content_overlap_top1,
        "mutual_redundancy_top3": mutual_redundancy_top3,
        "concentration_ratio": concentration_ratio,
        "score_std_top5": score_std_top5,
        "entropy_norm": entropy_norm,
        "n_hits": float(n),
        "query_len_tokens": float(len(query_tokens)),
    }


FEATURE_NAMES = [
    "top1",
    "top2",
    "top3",
    "gap12",
    "gap15mean",
    "zscore_top1",
    "same_lang_consistency",
    "lexical_overlap_top1",
    "content_overlap_top1",
    "mutual_redundancy_top3",
    "concentration_ratio",
    "score_std_top5",
    "entropy_norm",
    "n_hits",
    "query_len_tokens",
]


# --------------------------------------------------------------------------------------------
# Evaluation: AUC (manual rank-sum / Mann-Whitney formula, no scipy dependency), threshold
# sweep, per-language stability.
# --------------------------------------------------------------------------------------------


def auc_score(values: np.ndarray, labels: np.ndarray) -> float | None:
    """Rank-based AUC (Mann-Whitney U / rank-sum formula), average ranks for ties."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    sum_rank_pos = ranks[labels == 1].sum()
    return float((sum_rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def confusion_at_threshold(
    values: np.ndarray, labels: np.ndarray, threshold: float, higher_is_better: bool
) -> dict:
    accept = values >= threshold if higher_is_better else values <= threshold
    tp = int(np.sum(accept & (labels == 1)))
    fp = int(np.sum(accept & (labels == 0)))
    fn = int(np.sum(~accept & (labels == 1)))
    tn = int(np.sum(~accept & (labels == 0)))
    n = len(labels)
    answered = tp + fp
    return {
        "threshold": float(threshold),
        "higher_is_better": higher_is_better,
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


def best_single_threshold(
    values: np.ndarray, labels: np.ndarray, higher_is_better: bool, precision_floor: float
) -> dict | None:
    candidates = sorted(set(values.tolist()))
    best = None
    for t in candidates:
        row = confusion_at_threshold(values, labels, t, higher_is_better)
        if row["answered"] == 0 or row["precision_of_accepted"] is None:
            continue
        if row["precision_of_accepted"] < precision_floor:
            continue
        if best is None or row["answered"] > best["answered"]:
            best = row
    return best


def evaluate_feature(
    name: str, values: np.ndarray, labels: np.ndarray, languages: np.ndarray
) -> dict:
    auc = auc_score(values, labels)
    higher_is_better = auc is None or auc >= 0.5
    best = best_single_threshold(values, labels, higher_is_better, PRECISION_FLOOR)

    baseline_at_median = confusion_at_threshold(
        values, labels, float(np.median(values)), higher_is_better
    )

    per_lang_auc = {}
    for lang in sorted(set(languages.tolist())):
        mask = languages == lang
        lv, ll = values[mask], labels[mask]
        per_lang_auc[lang] = auc_score(lv, ll)

    valid_aucs = [a for a in per_lang_auc.values() if a is not None]
    return {
        "name": name,
        "auc": auc,
        "higher_is_better": higher_is_better,
        "value_range": [float(values.min()), float(values.max())],
        "best_threshold_at_precision_floor": best,
        "confusion_at_median_threshold": baseline_at_median,
        "per_language_auc": per_lang_auc,
        "per_language_auc_std": float(np.std(valid_aucs)) if len(valid_aucs) > 1 else None,
        "per_language_auc_mean": float(np.mean(valid_aucs)) if valid_aucs else None,
    }


def stability_split(
    values: np.ndarray, labels: np.ndarray, query_ids: np.ndarray, higher_is_better: bool
) -> dict:
    """Global even/odd query_id split (not per-language -- ~266 queries/half, learned from
    Phase 4's rule B, where per-language n=38 splits were too small to trust)."""
    even = query_ids % 2 == 0
    odd = ~even
    best_a = best_single_threshold(values[even], labels[even], higher_is_better, PRECISION_FLOOR)
    best_b = best_single_threshold(values[odd], labels[odd], higher_is_better, PRECISION_FLOOR)
    ta = best_a["threshold"] if best_a else None
    tb = best_b["threshold"] if best_b else None
    agree = ta is not None and tb is not None
    return {
        "threshold_half_a": ta,
        "threshold_half_b": tb,
        "n_half_a": int(even.sum()),
        "n_half_b": int(odd.sum()),
        "both_found": agree,
    }


# --------------------------------------------------------------------------------------------
# Combinations: simple 2D grid, global (not per-language) operating point.
# --------------------------------------------------------------------------------------------


def sweep_combination(
    feat_a: np.ndarray,
    hib_a: bool,
    feat_b: np.ndarray,
    hib_b: bool,
    labels: np.ndarray,
    precision_floor: float,
    n_grid: int = 25,
) -> dict | None:
    cand_a = np.quantile(feat_a, np.linspace(0.0, 1.0, n_grid))
    cand_b = np.quantile(feat_b, np.linspace(0.0, 1.0, n_grid))
    best = None
    for ta in cand_a:
        accept_a = feat_a >= ta if hib_a else feat_a <= ta
        for tb in cand_b:
            accept_b = feat_b >= tb if hib_b else feat_b <= tb
            accept = accept_a & accept_b
            tp = int(np.sum(accept & (labels == 1)))
            fp = int(np.sum(accept & (labels == 0)))
            answered = tp + fp
            if answered == 0:
                continue
            precision = tp / answered
            if precision < precision_floor:
                continue
            if best is None or answered > best["answered"]:
                fn = int(np.sum(~accept & (labels == 1)))
                tn = int(np.sum(~accept & (labels == 0)))
                n = len(labels)
                best = {
                    "threshold_a": float(ta),
                    "threshold_b": float(tb),
                    "n": n,
                    "answered": answered,
                    "abstained": fn + tn,
                    "abstain_rate": round((fn + tn) / n, 4),
                    "true_accept": tp,
                    "false_accept": fp,
                    "false_refusal": fn,
                    "true_refusal": tn,
                    "false_accept_rate": round(fp / n, 4),
                    "false_refusal_rate": round(fn / n, 4),
                    "precision_of_accepted": round(precision, 4),
                }
    return best


# --------------------------------------------------------------------------------------------
# Offline-only diagnostic: manual logistic regression (numpy, already a declared project
# dependency -- no new dependency added), strict train/val split, reported but NOT shippable.
# --------------------------------------------------------------------------------------------


def fit_logistic_diagnostic(
    X: np.ndarray, y: np.ndarray, query_ids: np.ndarray, feature_names: list[str]
) -> dict:
    train_mask = query_ids % 2 == 0
    val_mask = ~train_mask
    Xtr, ytr = X[train_mask], y[train_mask]
    Xva, yva = X[val_mask], y[val_mask]

    mu, sigma = Xtr.mean(axis=0), Xtr.std(axis=0)
    sigma[sigma < 1e-9] = 1.0
    Xtr_n = (Xtr - mu) / sigma
    Xva_n = (Xva - mu) / sigma

    n_feat = Xtr_n.shape[1]
    w = np.zeros(n_feat)
    b = 0.0
    lr = 0.1
    l2 = 0.01
    for _ in range(2000):
        z = Xtr_n @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad_w = Xtr_n.T @ (p - ytr) / len(ytr) + l2 * w
        grad_b = float(np.mean(p - ytr))
        w -= lr * grad_w
        b -= lr * grad_b

    train_scores = Xtr_n @ w + b
    val_scores = Xva_n @ w + b
    train_auc = auc_score(train_scores, ytr)
    val_auc = auc_score(val_scores, yva)

    return {
        "features_used": feature_names,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "weights": dict(zip(feature_names, w.tolist(), strict=True)),
        "bias": float(b),
        "train_auc": train_auc,
        "val_auc": val_auc,
        "generalizes": (
            val_auc is not None and train_auc is not None and (train_auc - val_auc) < 0.08
        ),
        "note": (
            "OFFLINE DIAGNOSTIC ONLY -- numpy gradient descent, train/val split by query_id "
            "parity, standardized features, L2=0.01, 2000 steps, lr=0.1. Not wired into "
            "production; g3_confidence.py is untouched regardless of this result."
        ),
    }


# --------------------------------------------------------------------------------------------
# Regression / stress cases
# --------------------------------------------------------------------------------------------


async def run_regression_cases(feature_evals: dict) -> list[dict]:
    import os
    import sys

    os.environ.setdefault("VRAG_INDEX_DIR", str(REPO_ROOT / "data" / "index" / "multilingual_100k"))
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from vrag.retrieval.interface import retrieve

    cases = [
        ("hindi_capital_of_india", "भारत की राजधानी क्या है?", "hin_Deva", "hi-IN"),
        ("english_capital_of_india", "What is the capital of India?", "hin_Deva", "hi-IN"),
    ]
    results = []
    for label, query, target_lang, sarvam in cases:
        chunks = await retrieve(query, k=20, language=sarvam)
        hits = [
            {
                "rank": i + 1,
                "chunk_id": c.chunk_id,
                "passage_id": c.passage_id,
                "score": c.score,
                "language": c.language,
                "text": c.text,
            }
            for i, c in enumerate(chunks)
        ]
        feats = extract_features(query, hits, target_lang)
        results.append(
            {
                "label": label,
                "query": query,
                "top1_text_preview": hits[0]["text"][:200] if hits else None,
                "top1_passage_id": hits[0]["passage_id"] if hits else None,
                "features": feats,
                "current_g3_would_accept": feats["top1"] >= CURRENT_TAU,
            }
        )
    return results


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------


def main() -> None:
    data = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    rows = data["rows"]
    print(f"n_queries={len(rows)}")

    t0 = time.perf_counter()
    feat_rows = []
    for r in rows:
        feats = extract_features(r["query"], r["hits"], r["language"])
        feat_rows.append(feats)
    t1 = time.perf_counter()
    per_query_us = (t1 - t0) / len(rows) * 1e6
    print(
        f"Feature extraction: {per_query_us:.1f} us/query "
        f"(all {len(FEATURE_NAMES)} features combined)"
    )

    labels = np.array([1 if r["relevant_in_top1"] else 0 for r in rows], dtype=int)
    languages = np.array([r["language"] for r in rows])
    query_ids = np.array([r["query_id"] for r in rows], dtype=np.int64)

    feature_arrays = {
        name: np.array([f[name] for f in feat_rows], dtype=float) for name in FEATURE_NAMES
    }

    n_wrong = len(labels) - int(labels.sum())
    print(f"\nBaseline label distribution: correct={int(labels.sum())} wrong={n_wrong}")

    feature_results = {}
    for name in FEATURE_NAMES:
        result = evaluate_feature(name, feature_arrays[name], labels, languages)
        feature_results[name] = result
        best = result["best_threshold_at_precision_floor"]
        print(
            f"  {name:24s} AUC={result['auc']}"
            f"  best_answered={(best or {}).get('answered')}"
            f"  best_precision={(best or {}).get('precision_of_accepted')}"
        )

    print("\n=== Stability check (global even/odd split) for top-5 AUC features ===")
    ranked = sorted(
        (n for n in FEATURE_NAMES if feature_results[n]["auc"] is not None),
        key=lambda n: abs(feature_results[n]["auc"] - 0.5),
        reverse=True,
    )
    stability_results = {}
    for name in ranked[:6]:
        hib = feature_results[name]["higher_is_better"]
        stab = stability_split(feature_arrays[name], labels, query_ids, hib)
        stability_results[name] = stab
        print(f"  {name:24s} half_a={stab['threshold_half_a']} half_b={stab['threshold_half_b']}")

    print("\n=== Combinations ===")
    combos = {
        "A_top1_gap12": ("top1", "gap12"),
        "B_top1_concentration": ("top1", "concentration_ratio"),
        "C_top1_lexical": ("top1", "content_overlap_top1"),
        "D_top1_samelang": ("top1", "same_lang_consistency"),
    }
    combo_results = {}
    for combo_name, (fa, fb) in combos.items():
        hib_a = feature_results[fa]["higher_is_better"]
        hib_b = feature_results[fb]["higher_is_better"]
        best = sweep_combination(
            feature_arrays[fa], hib_a, feature_arrays[fb], hib_b, labels, PRECISION_FLOOR
        )
        even = query_ids % 2 == 0
        best_half_a = sweep_combination(
            feature_arrays[fa][even],
            hib_a,
            feature_arrays[fb][even],
            hib_b,
            labels[even],
            PRECISION_FLOOR,
        )
        best_half_b = sweep_combination(
            feature_arrays[fa][~even],
            hib_a,
            feature_arrays[fb][~even],
            hib_b,
            labels[~even],
            PRECISION_FLOOR,
        )
        combo_results[combo_name] = {
            "features": [fa, fb],
            "global_best": best,
            "stability_half_a": best_half_a,
            "stability_half_b": best_half_b,
        }
        print(
            f"  {combo_name}: answered={(best or {}).get('answered')} "
            f"precision={(best or {}).get('precision_of_accepted')}"
        )

    print("\n=== Offline logistic regression diagnostic (not for production) ===")
    diag_features = ["top1", "gap12", "concentration_ratio", "content_overlap_top1"]
    X = np.stack([feature_arrays[f] for f in diag_features], axis=1)
    y = labels.astype(float)
    logreg = fit_logistic_diagnostic(X, y, query_ids, diag_features)
    print(
        f"  train_auc={logreg['train_auc']} val_auc={logreg['val_auc']} "
        f"generalizes={logreg['generalizes']}"
    )

    print("\n=== Regression cases ===")
    import asyncio

    regression = asyncio.run(run_regression_cases(feature_results))
    for r in regression:
        print(
            f"  [{r['label']}] top1={r['features']['top1']:.4f} "
            f"current_g3_accept={r['current_g3_would_accept']} "
            f"top1_passage={r['top1_passage_id']}"
        )

    out = {
        "n_queries": len(rows),
        "precision_floor_anchor": PRECISION_FLOOR,
        "feature_extraction_us_per_query": per_query_us,
        "label_distribution": {
            "correct": int(labels.sum()),
            "wrong": len(labels) - int(labels.sum()),
        },
        "features": feature_results,
        "stability_top_features": stability_results,
        "combinations": combo_results,
        "logistic_diagnostic": logreg,
        "regression_cases": regression,
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote -> {OUT_PATH}")


if __name__ == "__main__":
    main()
