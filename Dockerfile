# 3.12, not 3.11 (pyproject.toml's stated minimum) -- numpy>=2.5 (retrieval-lean) requires 3.12+.
# Never surfaced before: this is the first time retrieval extras are actually installed in the
# image, since real retrieval was stubbed out at the Docker level until now (docs/DECISIONS_P.md
# P-018/R-023).
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
# retrieval-lean (docs/DECISIONS_R.md R-023): numpy/faiss-cpu/bm25s/onnxruntime/tokenizers only —
# deliberately excludes torch/transformers/sentence-transformers, which is what made real
# retrieval unaffordable on Render free tier's 512MB (docs/RISKS.md R4). Real (non-stub) retrieval
# was verified end-to-end by R in a fresh venv with exactly this extra installed: 727MB RSS,
# still over budget but the closest either track has gotten — trying it live is the fastest way
# to get a real answer (docs/DECISIONS_P.md P-018 update), with the previous stub-serving deploy
# as the rollback if it OOMs.
RUN pip install --no-cache-dir -e ".[retrieval-lean]"

COPY frontend/ ./frontend/

# Pre-built retrieval index v2 (docs/DECISIONS_R.md R-023 — adds chunk_lookup.sqlite3 alongside
# the JSON, R-021's lean format) and the lean ONNX embedder (R-022's LiteE5Embedder, torch-free),
# both downloaded at build time, never built in the container (AGENT_BUILD_SPEC.md §5.3).
# src/vrag/retrieval/interface.py's `_get_real_retriever()` still falls back to the Day-0 stub if
# either artifact is missing or fails to load, so a partial/failed download degrades safely rather
# than crashing.
RUN mkdir -p data/index \
    && curl -fsSL "https://github.com/ShivDyutiVerma/vRAG/releases/download/index-metadata_aware-v2/metadata_aware_index_v2.tar.gz" \
       -o /tmp/index.tar.gz \
    && tar --no-same-owner -xzf /tmp/index.tar.gz -C data/index \
    && rm /tmp/index.tar.gz
# --strip-components=1: the release tarball's own top-level directory is named
# multilingual-e5-small-lite/, not multilingual-e5-small/ (LiteE5Embedder's DEFAULT_ONNX_MODEL_DIR,
# src/vrag/index/embedder.py) — verified by inspecting the tarball before writing this, not assumed
# from the naming pattern of the index release.
RUN mkdir -p data/onnx/multilingual-e5-small \
    && curl -fsSL "https://github.com/ShivDyutiVerma/vRAG/releases/download/embedder-lite-onnx-v1/multilingual-e5-small-lite-onnx.tar.gz" \
       -o /tmp/embedder.tar.gz \
    && tar --no-same-owner --strip-components=1 -xzf /tmp/embedder.tar.gz -C data/onnx/multilingual-e5-small \
    && rm /tmp/embedder.tar.gz

# Hugging Face Spaces (Docker SDK) expects the app to listen on $PORT, default 7860.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn vrag.api.main:app --app-dir src --host 0.0.0.0 --port ${PORT}"]
