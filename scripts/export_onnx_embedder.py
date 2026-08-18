"""ONNX + int8 quantisation for the production embedder (docs/BUILD_PLAN.md P6 task 5 — "ONNX +
int8 the embedder (CPU only)"). CLAUDE.md's hot-path invariant is explicit: "ONNX int8 is for CPU
only — on GPU it is slower than FP32; use FP16 there" — this targets the production deploy shape
(CPU-only host, AGENT_BUILD_SPEC.md §5.3), not this dev machine's GPU.

Two-step export via `sentence-transformers`' own ONNX backend (docs/DECISIONS_R.md R-019):
  1. Export `intfloat/multilingual-e5-small` to plain ONNX (FP32).
  2. Dynamic int8 quantisation (`export_dynamic_quantized_onnx_model`, "avx2" config — the broadest
     generically-supported x86_64 instruction set, since the exact cloud CPU Render provisions isn't
     known in advance; no calibration dataset needed for dynamic quantisation).

Output directory (`data/onnx/multilingual-e5-small/`) is gitignored, like `data/` generally —
regenerate with this script, don't commit ONNX binaries to git.

Usage: python scripts/export_onnx_embedder.py
"""

from __future__ import annotations

from pathlib import Path

from sentence_transformers import SentenceTransformer
from sentence_transformers.backend import export_dynamic_quantized_onnx_model

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "intfloat/multilingual-e5-small"
OUT_DIR = REPO_ROOT / "data" / "onnx" / "multilingual-e5-small"


def main() -> None:
    print(f"Exporting {MODEL_NAME} to ONNX (FP32) -> {OUT_DIR}")
    model = SentenceTransformer(MODEL_NAME, backend="onnx", model_kwargs={"export": True})
    model.save_pretrained(str(OUT_DIR))

    print("Quantising to int8 (dynamic, avx2 config, CPU-only per CLAUDE.md hot-path invariant)...")
    export_dynamic_quantized_onnx_model(model, "avx2", str(OUT_DIR))

    print(f"\nDone. ONNX files in {OUT_DIR}:")
    for f in sorted(OUT_DIR.glob("**/*.onnx")):
        print(f"  {f.relative_to(OUT_DIR)}  ({f.stat().st_size / 1e6:.1f}MB)")


if __name__ == "__main__":
    main()
