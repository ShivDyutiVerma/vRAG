"""Phase 0 dataset reconnaissance (AGENT_BUILD_SPEC.md §6.1, docs/BUILD_PLAN.md P0 task 4).

Reads ai4bharat/MSMARCO-XI's Hindi train file directly (`train/hintrain.parquet`, downloaded once
and cached by huggingface_hub) rather than going through `datasets`' generic loader. Two dead ends
that led here, recorded so nobody re-walks them:

  1. `load_dataset("ai4bharat/MSMARCO-XI", "hi", ...)` fails — the repo's custom loading script
     defines per-language configs named "hi", "bn", etc., but `datasets` 5.x doesn't execute
     community loading scripts by default (security default), so only an auto-detected "default"
     config exists, silently concatenating every language's parquet file into one stream.
  2. Streaming the "default" config and filtering client-side on `target_lang == "hi"` works but is
     slow (scans through all 13 languages' data). Streaming the Hindi parquet file directly via
     `hf://` also stalled — the `fsspec`/`aiohttp` path underneath doesn't appear to pick up the
     process-wide IPv4 DNS fix the same way `huggingface_hub`'s own httpx-based downloader does
     (see scripts/_netcompat.py and docs/RISKS.md for the underlying IPv6 issue).

`hf_hub_download` reliably fetches the single Hindi file (3.7GB, one-time cost, cached after), then
everything below reads it locally and lazily via pyarrow row-group batches — no need to hold the
whole file in memory to inspect a sample.

Confirmed schema (from the parquet file itself, 2026-08-17): query, Answer, query_id, query_type,
passages (dict: is_selected, English_passages, Translated_passages), source_lang, target_lang,
meta, Eng_Query, Eng_Answer.
"""

from __future__ import annotations

import argparse
import itertools
import statistics

import _dataset


def inspect(n_samples: int, n_eyeball: int) -> None:
    path = _dataset.local_parquet_path()
    print(f"Reading first {n_samples} rows from local file: {path}\n")
    rows = list(itertools.islice(_dataset.iter_rows(), n_samples))
    if not rows:
        print("No rows returned — check the downloaded file.")
        return

    print("## Schema (keys of first row)\n")
    print(sorted(rows[0].keys()))

    print(f"\n## {n_eyeball} samples for eyeball quality-check")
    print("(impressions go in DECISIONS_R.md)\n")
    for row in rows[:n_eyeball]:
        translated = row["passages"].get("Translated_passages", [])
        first_passage = translated[0] if translated else "(no translated passage)"
        print(f"--- query_id={row['query_id']} query_type={row['query_type']} ---")
        print(f"query (hi): {row['query']}")
        print(f"query (en): {row.get('Eng_Query', '(n/a)')}")
        print(f"first passage (hi): {first_passage[:300]}")
        print()

    passage_lengths: list[int] = []
    selected_counts: list[int] = []
    for row in rows:
        translated = row["passages"].get("Translated_passages", [])
        is_selected = row["passages"].get("is_selected", [])
        selected_counts.append(sum(1 for s in is_selected if s))
        for p in translated:
            passage_lengths.append(len(p.split()))

    print("## Passage-length distribution (whitespace token count, over sampled rows)\n")
    if passage_lengths:
        pl = sorted(passage_lengths)
        print(
            f"n={len(pl)}  min={pl[0]}  p50={statistics.median(pl):.0f}  "
            f"p95={pl[int(len(pl) * 0.95)]}  max={pl[-1]}"
        )
    else:
        print("No translated passages found in sample.")

    print(f"\nMean selected (relevant) passages per query: {statistics.mean(selected_counts):.2f}")
    total_translated_per_query = statistics.mean(
        len(row["passages"].get("Translated_passages", [])) for row in rows
    )
    print(f"Mean total translated passages per query: {total_translated_per_query:.2f}")

    print("\n## Chunk-count estimate (passage-native strategy: 1 chunk per translated passage)\n")
    for n_queries in (10_000, 20_000, 30_000, 50_000):
        est = int(n_queries * total_translated_per_query)
        in_range = "OK — within 50k-200k target" if 50_000 <= est <= 200_000 else ""
        print(f"{n_queries:>7} queries -> ~{est:>7} chunks   {in_range}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500, help="rows to read for stats (default 500)")
    parser.add_argument(
        "--eyeball", type=int, default=20, help="rows to print for spot-check (default 20)"
    )
    args = parser.parse_args()
    inspect(args.n, args.eyeball)
