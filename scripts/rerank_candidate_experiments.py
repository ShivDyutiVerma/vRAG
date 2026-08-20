"""Phase 7 (docs/DECISIONS.md ADR-016), Part 3: cheap, deterministic reranking-candidate
experiments over the already-retrieved top-20 window (no new retrieval calls -- reuses the real
data in eval/g3_feature_experiment_raw.json, Phase 5's collection). No neural model, no LLM, no
network call -- exactly what was asked for; the old cross-encoder (R-038, net-negative on the
Hindi-only corpus) is NOT re-tested here per the explicit instruction not to repeat that
experiment unless diagnostics prove it's specifically required.

Candidates tested (letters match the Phase 7 instructions, distinct from ADR-013/016's
abstention-taxonomy A-F -- careful not to conflate the two elsewhere):
  B. Rerank by query/document lexical overlap alone (Jaccard over all tokens).
  C. Rerank by a cheap "entity" proxy: numeric-token overlap (dates/codes/quantities) weighted
     above generic content-word overlap (tokens >= 4 chars) -- no NER model exists, this is the
     deterministic proxy available without one, documented as such.
  E. Reciprocal-rank fusion (RRF, the same formula already implemented in
     src/vrag/index/fusion.py) between the dense rank and the lexical-overlap rank.

For every candidate: Recall@1 (global + per-language), how many of the 354 known abstentions
would flip to a correct rank-1 AND separately whether that flip would actually clear G3's real
TAU=0.8835 using the promoted passage's own real dense cosine score (reranking changes ORDER, not
the score G3 reads -- explicitly checked, not assumed), and -- the critical safety check, matching
R-038's rigor -- whether any candidate DEMOTES a currently-correct rank-1 out of first place
(a regression / new-false-refusal risk in the other direction).

Usage: python scripts/rerank_candidate_experiments.py
Output: eval/g3_rerank_candidate_results.json
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = REPO_ROOT / "eval" / "g3_feature_experiment_raw.json"
HELDOUT_PATH = REPO_ROOT / "eval" / "heldout_queries_multilingual.json"
OUT_PATH = REPO_ROOT / "eval" / "g3_rerank_candidate_results.json"

CURRENT_TAU = 0.8835
_DIGIT_RUN = re.compile(r"\d+")


def tokenize(text: str) -> list[str]:
    """Same Mn/Mc-combining-mark-aware tokenizer as Phase 5 (ADR-014) -- naive \\w+ shatters
    Devanagari/Bengali/etc. at every vowel sign."""
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
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def entity_proxy_score(query_tokens: list[str], hit_tokens: list[str]) -> float:
    """No NER model available (none permitted) -- deterministic proxy: numeric-token overlap
    (dates/codes/quantities -- the single strongest signal observed in the case inspection, e.g.
    zip codes, routing numbers, years) weighted 2x above generic content-word overlap (tokens
    >=4 chars, same threshold Phase 5 used)."""
    q_nums = {t for t in query_tokens if _DIGIT_RUN.fullmatch(t)}
    h_nums = {t for t in hit_tokens if _DIGIT_RUN.fullmatch(t)}
    q_content = {t.lower() for t in query_tokens if len(t) >= 4 and not _DIGIT_RUN.fullmatch(t)}
    h_content = {t.lower() for t in hit_tokens if len(t) >= 4 and not _DIGIT_RUN.fullmatch(t)}
    num_overlap = jaccard(q_nums, h_nums)
    content_overlap = jaccard(q_content, h_content)
    return 2.0 * num_overlap + content_overlap


def rrf_fuse(dense_rank: int, lexical_rank: int, k: int = 60) -> float:
    return 1.0 / (k + dense_rank) + 1.0 / (k + lexical_rank)


def dedupe_by_passage(hits: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for h in hits:
        if h["passage_id"] not in seen:
            seen.add(h["passage_id"])
            out.append(h)
    return out


def rerank_row(row: dict) -> dict:
    hits = dedupe_by_passage(row["hits"])
    if not hits:
        return {"dense": [], "lexical": [], "entity": [], "rrf": []}

    query_tokens = tokenize(row["query"])
    query_set = {t.lower() for t in query_tokens}

    scored = []
    for h in hits:
        h_tokens = tokenize(h["text"])
        h_set = {t.lower() for t in h_tokens}
        lex = jaccard(query_set, h_set)
        ent = entity_proxy_score(query_tokens, h_tokens)
        scored.append({**h, "lexical_score": lex, "entity_score": ent})

    dense_order = scored  # already dense-ranked, as retrieved
    lexical_order = sorted(scored, key=lambda x: -x["lexical_score"])
    entity_order = sorted(scored, key=lambda x: -x["entity_score"])

    dense_rank_of = {h["passage_id"]: i + 1 for i, h in enumerate(dense_order)}
    lexical_rank_of = {h["passage_id"]: i + 1 for i, h in enumerate(lexical_order)}
    rrf_order = sorted(
        scored,
        key=lambda x: -rrf_fuse(dense_rank_of[x["passage_id"]], lexical_rank_of[x["passage_id"]]),
    )

    return {
        "dense": dense_order,
        "lexical": lexical_order,
        "entity": entity_order,
        "rrf": rrf_order,
    }


def main() -> None:
    data = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    held = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    held_by_id = {q["query_id"]: q for q in held}
    rows = data["rows"]

    strategies = ["dense", "lexical", "entity", "rrf"]
    correct_at_1: dict[str, int] = dict.fromkeys(strategies, 0)
    per_lang_correct: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(strategies, 0))
    per_lang_n: dict[str, int] = defaultdict(int)

    abstained_flip_count: dict[str, int] = dict.fromkeys(strategies, 0)
    abstained_flip_and_clears_tau: dict[str, int] = dict.fromkeys(strategies, 0)
    regressions: dict[str, int] = dict.fromkeys(strategies, 0)  # was correct@1 dense, now isn't

    per_query_out = []
    for row in rows:
        q = held_by_id[row["query_id"]]
        gold_doc_ids = {p["passage_id"] for p in q["relevant_passages"]}
        rankings = rerank_row(row)
        was_correct_dense = (
            bool(rankings["dense"]) and rankings["dense"][0]["passage_id"] in gold_doc_ids
        )
        per_lang_n[row["language"]] += 1

        row_out = {"query_id": row["query_id"], "language": row["language"]}
        for strat in strategies:
            ranked = rankings[strat]
            is_correct = bool(ranked) and ranked[0]["passage_id"] in gold_doc_ids
            if is_correct:
                correct_at_1[strat] += 1
                per_lang_correct[row["language"]][strat] += 1
            row_out[f"{strat}_top1_correct"] = is_correct
            row_out[f"{strat}_top1_score"] = ranked[0]["score"] if ranked else None

            if not was_correct_dense and is_correct:
                abstained_flip_count[strat] += 1
                if ranked[0]["score"] >= CURRENT_TAU:
                    abstained_flip_and_clears_tau[strat] += 1
            if was_correct_dense and not is_correct:
                regressions[strat] += 1
        per_query_out.append(row_out)

    print(f"n_queries={len(rows)}")
    print("\n=== Recall@1 by strategy ===")
    for strat in strategies:
        rate = correct_at_1[strat] / len(rows)
        print(f"  {strat:10s} {correct_at_1[strat]}/{len(rows)} = {rate:.4f}")

    print("\n=== Currently-abstained queries (dense top1 wrong) that flip to correct top1 ===")
    for strat in strategies:
        if strat == "dense":
            continue
        print(
            f"  {strat:10s} flips_to_correct={abstained_flip_count[strat]}"
            f"  AND_clears_TAU={abstained_flip_and_clears_tau[strat]}"
        )

    print("\n=== Regressions (currently-correct top1 demoted to wrong) ===")
    for strat in strategies:
        if strat == "dense":
            continue
        print(f"  {strat:10s} regressions={regressions[strat]}")

    print(f"\n{'Language':10s}" + "".join(f"{s:>10s}" for s in strategies) + f"{'n':>6s}")
    for lang in sorted(per_lang_n):
        row_str = "".join(f"{per_lang_correct[lang][s]:10d}" for s in strategies)
        print(f"{lang:10s}{row_str}{per_lang_n[lang]:6d}")

    out = {
        "n_queries": len(rows),
        "current_tau": CURRENT_TAU,
        "recall_at_1_by_strategy": {s: correct_at_1[s] / len(rows) for s in strategies},
        "recall_at_1_counts": correct_at_1,
        "abstained_flip_to_correct": abstained_flip_count,
        "abstained_flip_and_clears_tau": abstained_flip_and_clears_tau,
        "regressions_currently_correct_demoted": regressions,
        "per_language_correct_at_1": {
            lang: {**per_lang_correct[lang], "n": per_lang_n[lang]} for lang in per_lang_n
        },
        "per_query": per_query_out,
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote -> {OUT_PATH}")


if __name__ == "__main__":
    main()
