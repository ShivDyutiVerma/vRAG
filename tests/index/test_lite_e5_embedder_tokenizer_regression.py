"""Regression test for R-030's tokenizer swap (docs/DECISIONS_R.md): LiteE5Embedder's new
sentencepiece-based tokenization must keep producing the exact token IDs the old `tokenizers`
(Rust) library did, on every test run -- not just verified once at implementation time (R-029).
If a future model re-export or vocab change ever breaks the special-token remap, this fails loudly
here instead of silently producing wrong embeddings in production.

Covers the same 1,020-string corpus R-029 verified (eval/tokenizer_test_corpus.json — 500 real
Hindi held-out queries, 500 real English queries, 20 mixed/romanized) plus additional edge cases:
truncation at the 512-token boundary, exotic/mixed-script text, empty string, whitespace-only,
single word, pure digits, and a realistic long multi-sentence passage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "eval" / "tokenizer_test_corpus.json"

sys.path.insert(0, str(REPO_ROOT / "src"))

from vrag.index.embedder import DEFAULT_ONNX_MODEL_DIR, LiteE5Embedder  # noqa: E402


@pytest.fixture(scope="module")
def old_tokenizer():
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(f"{DEFAULT_ONNX_MODEL_DIR}/tokenizer.json")
    tok.enable_truncation(max_length=512)
    return tok


@pytest.fixture(scope="module")
def new_embedder() -> LiteE5Embedder:
    return LiteE5Embedder()


def test_1020_real_strings_produce_identical_token_ids(old_tokenizer, new_embedder) -> None:
    """The core R-029/R-030 regression guard -- same corpus, same prefix, must match exactly."""
    rows = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    assert len(rows) >= 1000, "test corpus should have at least 1,000 strings"

    mismatches = []
    for row in rows:
        text = row["prefixed_text"]
        old_ids = old_tokenizer.encode(text).ids
        new_ids = new_embedder._tokenize_batch([text])[0][0].tolist()
        if old_ids != new_ids:
            mismatches.append((row["category"], text, old_ids, new_ids))

    if mismatches:
        sample = "\n".join(
            f"  [{cat}] {text[:60]!r}\n    old: {old}\n    new: {new}"
            for cat, text, old, new in mismatches[:5]
        )
        pytest.fail(
            f"{len(mismatches)}/{len(rows)} strings produced different token IDs "
            f"(expected 0 — R-029 verified 100% equivalence):\n{sample}"
        )


@pytest.mark.parametrize(
    "text",
    [
        "",  # empty string
        "   ",  # whitespace only
        "hello",  # single English word
        "नमस्ते",  # single Hindi word
        "123456789",  # pure digits
        "🎉🔥emoji test ζωή русский العربية",  # exotic/mixed scripts, tests unk/byte handling
        "भारत की राजधानी नई दिल्ली है। " * 100,  # >512 tokens, exercises truncation
        (
            "This is a realistic long English passage with multiple sentences, meant to "
            "exercise normal prose tokenization rather than a single short query. It "
            "includes punctuation, numbers like 42 and 3.14, and a mix of common words "
            "to approximate what a real retrieved passage might look like in production."
        ),
    ],
    ids=[
        "empty",
        "whitespace_only",
        "single_english_word",
        "single_hindi_word",
        "pure_digits",
        "exotic_mixed_scripts",
        "over_512_tokens_truncation",
        "realistic_long_passage",
    ],
)
def test_edge_cases_produce_identical_token_ids(old_tokenizer, new_embedder, text: str) -> None:
    prefixed = f"query: {text}"
    old_ids = old_tokenizer.encode(prefixed).ids
    new_ids = new_embedder._tokenize_batch([prefixed])[0][0].tolist()
    assert old_ids == new_ids, f"mismatch for {text[:60]!r}:\n  old: {old_ids}\n  new: {new_ids}"


def test_batch_padding_matches_old_tokenizer_batch_behavior(old_tokenizer, new_embedder) -> None:
    """Old tokenizer pads a batch to the batch's own longest sequence, right-padded, pad_id=1,
    attention_mask=0 on pad positions (verified empirically before implementing, R-030) -- must
    hold for the new implementation too, not just single-string encoding."""
    old_tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
    texts = ["query: छोटा", "query: यह एक बहुत लंबा वाक्य है जो अधिक टोकन उत्पन्न करेगा"]

    old_encodings = old_tokenizer.encode_batch(texts)
    old_ids = [e.ids for e in old_encodings]
    old_mask = [e.attention_mask for e in old_encodings]

    new_ids, new_mask = new_embedder._tokenize_batch(texts)

    assert new_ids.tolist() == old_ids
    assert new_mask.tolist() == old_mask
    old_tokenizer.no_padding()


def test_embed_queries_returns_correct_dimension_and_is_l2_normalized(
    new_embedder: LiteE5Embedder,
) -> None:
    """End-to-end sanity check through the real public interface, not just tokenization."""
    import math

    vectors = new_embedder.embed_queries(["भारत की राजधानी क्या है?"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 384
    norm = math.sqrt(sum(v * v for v in vectors[0]))
    assert abs(norm - 1.0) < 1e-5
