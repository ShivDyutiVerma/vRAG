# 3.12, not 3.11 (pyproject.toml's stated floor): the `retrieval-lean` extra's numpy>=2.5 pin
# (docs/DECISIONS_R.md R-001) requires Python >=3.12 -- only discovered here, since retrieval-lean
# had never actually been installed inside a 3.11 environment before this Docker validation (the
# dev machine that pinned it runs 3.13). pyproject.toml's ">=3.11" is still satisfied; this is a
# base-image compatibility fix, not a project requirement change.
FROM python:3.12-slim

WORKDIR /app

# System deps: build-essential for future R-side libraries (faiss, onnxruntime); curl to fetch the
# pre-built index artifact below (never build the index inside the container — AGENT_BUILD_SPEC.md
# §5.3).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/
# `retrieval-lean` (docs/DECISIONS_R.md R-023/R-030): onnxruntime + faiss-cpu + bm25s + sentencepiece
# only — NOT torch/transformers/sentence-transformers, which is what made the pre-R-023 memory
# budget impossible on Render's free tier. This is the runtime-only subset; `retrieve()`'s "dense"
# mode (the A3 winner) never touches bm25s at request time either (ADR-007), but it's still
# installed since HybridRetriever's constructor is shared code.
RUN pip install --no-cache-dir -e ".[retrieval-lean]"

COPY frontend/ ./frontend/

# Pre-built retrieval index (docs/DECISIONS_R.md R-018/R-021), downloaded at build time rather than
# built in the container (AGENT_BUILD_SPEC.md §5.3). v2 adds chunk_lookup.sqlite3, which is what
# load_built_index_lean() (src/vrag/index/persistence.py) actually reads — v1's eager
# chunk_lookup.json-only layout no longer matches the code path this image runs.
RUN mkdir -p data/index \
    && curl -fsSL --retry 5 --retry-delay 2 --http1.1 \
       "https://github.com/ShivDyutiVerma/vRAG/releases/download/index-metadata_aware-v2/metadata_aware_index_v2.tar.gz" \
       -o /tmp/index.tar.gz \
    && tar --no-same-owner -xzf /tmp/index.tar.gz -C data/index \
    && rm /tmp/index.tar.gz

# Lean embedder bundle for LiteE5Embedder (docs/DECISIONS_R.md R-029/R-030): sentencepiece.bpe.model
# + the int8 ONNX model only, no tokenizer.json — v2 of the release asset, since v1 predates the
# sentencepiece tokenizer swap and only shipped the old HF tokenizers.json file. Renamed on extract
# to match LiteE5Embedder's DEFAULT_ONNX_MODEL_DIR (src/vrag/index/embedder.py).
RUN mkdir -p /tmp/embedder_extract data/onnx \
    && curl -fsSL --retry 5 --retry-delay 2 --http1.1 \
       "https://github.com/ShivDyutiVerma/vRAG/releases/download/embedder-lite-onnx-v2/multilingual_e5_small_lite_v2.tar.gz" \
       -o /tmp/embedder.tar.gz \
    && tar --no-same-owner -xzf /tmp/embedder.tar.gz -C /tmp/embedder_extract \
    && mv /tmp/embedder_extract/multilingual-e5-small-lite data/onnx/multilingual-e5-small \
    && rm -rf /tmp/embedder.tar.gz /tmp/embedder_extract

# Hugging Face Spaces (Docker SDK) expects the app to listen on $PORT, default 7860.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn vrag.api.main:app --app-dir src --host 0.0.0.0 --port ${PORT}"]
