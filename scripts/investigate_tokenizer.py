"""Tokenizer-only memory investigation (docs/DECISIONS_R.md R-029), following R-028's finding that
the tokenizer (262MB) costs more than the ONNX session itself (137MB). Investigation only -- does
not modify production code (src/vrag/index/embedder.py untouched).

Usage: python scripts/investigate_tokenizer.py --stage where_262mb_goes
       python scripts/investigate_tokenizer.py --stage backend_identity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import psutil

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

MODEL_DIR = REPO_ROOT / "data" / "onnx" / "multilingual-e5-small"
TOKENIZER_JSON = MODEL_DIR / "tokenizer.json"
SPM_MODEL = MODEL_DIR / "sentencepiece.bpe.model"


def rss() -> int:
    return psutil.Process().memory_info().rss


def stage_where_262mb_goes() -> None:
    baseline = rss()
    print(f"1. baseline: {baseline / 1e6:.1f}MB")

    raw_text = TOKENIZER_JSON.read_text(encoding="utf-8")
    after_read = rss()
    print(f"2. after reading file as text ({len(raw_text.encode('utf-8')) / 1e6:.1f}MB on disk): "
          f"{after_read / 1e6:.1f}MB (+{(after_read - baseline) / 1e6:.1f}MB)")

    data = json.loads(raw_text)
    after_json_parse = rss()
    print(f"3. after json.loads() (Python dict/list objects): {after_json_parse / 1e6:.1f}MB "
          f"(+{(after_json_parse - after_read) / 1e6:.1f}MB)")

    vocab = data["model"]["vocab"]
    print(f"   vocab entries: {len(vocab)}")

    del data, raw_text
    import gc

    gc.collect()
    after_gc = rss()
    print(f"4. after del + gc.collect() (should drop back toward baseline if nothing else "
          f"holds a reference): {after_gc / 1e6:.1f}MB")

    from tokenizers import Tokenizer

    after_import = rss()
    print(f"5. after importing tokenizers lib: {after_import / 1e6:.1f}MB "
          f"(+{(after_import - after_gc) / 1e6:.1f}MB)")

    _tokenizer = Tokenizer.from_file(str(TOKENIZER_JSON))
    after_load = rss()
    print(f"6. after Tokenizer.from_file() (Rust-side trie/automaton + its own copy of the "
          f"vocab): {after_load / 1e6:.1f}MB (+{(after_load - after_import) / 1e6:.1f}MB)")
    print(f"\nConclusion: pure-Python JSON holding costs "
          f"~{(after_json_parse - after_read) / 1e6:.0f}MB; the Rust tokenizer's own internal "
          f"structures cost an additional ~{(after_load - after_import) / 1e6:.0f}MB on top of "
          f"that (they do NOT share memory with the transient Python JSON object, which was "
          f"freed before this step) -- both costs are real and additive in the staged view here, "
          f"but production only ever pays the Rust-side cost (~262MB), never the transient "
          f"Python JSON one, since _ensure_loaded() calls Tokenizer.from_file() directly on the "
          f"file path without ever materialising a full Python dict of the vocab.")


def stage_backend_identity() -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import Unigram

    tokenizer = Tokenizer.from_file(str(TOKENIZER_JSON))
    print(f"Python type: {type(tokenizer)}")
    print(f"Python module: {type(tokenizer).__module__}")
    model = tokenizer.model
    print(f"Tokenizer's model type: {type(model)}")
    print(f"Is a tokenizers.models.Unigram: {isinstance(model, Unigram)}")

    import tokenizers as tok_pkg

    print(f"\ntokenizers package version: {tok_pkg.__version__}")
    print(f"tokenizers package file: {tok_pkg.__file__}")

    try:
        import sentencepiece as spm

        spm_version = getattr(spm, "__version__", "unknown")
        print(f"\nsentencepiece package version: {spm_version}")
        print(f"sentencepiece package file: {spm.__file__}")
        print("\nThese are two DIFFERENT codebases (HF's Rust `tokenizers` crate vs. Google's "
              "C++ `sentencepiece` library) that both implement the Unigram algorithm from the "
              "same published SentencePiece paper/format -- not the same binary, not the same "
              "code, just the same *algorithm* and (potentially) the same trained model file.")
    except ImportError:
        print("\nsentencepiece not installed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True, choices=["where_262mb_goes", "backend_identity"]
    )
    args = parser.parse_args()
    if args.stage == "where_262mb_goes":
        stage_where_262mb_goes()
    else:
        stage_backend_identity()
