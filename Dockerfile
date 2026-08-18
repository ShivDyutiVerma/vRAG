FROM python:3.11-slim

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
RUN pip install --no-cache-dir -e .

COPY frontend/ ./frontend/

# Pre-built retrieval index (docs/DECISIONS_R.md R-018), downloaded at build time rather than
# built in the container. Staged ahead of the `retrieval` extras actually being installed here —
# real retrieval is still blocked on a leaner, torch/transformers-free embedder inference path
# (docs/DECISIONS_P.md P-018, memory budget doesn't fit today's sentence-transformers stack on
# Render's free tier). Safe to add now regardless: src/vrag/retrieval/interface.py's
# `_get_real_retriever()` falls back to the Day-0 stub if loading fails for any reason, including
# missing dependencies — this line alone does not change runtime behavior on its own.
RUN mkdir -p data/index \
    && curl -fsSL "https://github.com/ShivDyutiVerma/vRAG/releases/download/index-metadata_aware-v1/metadata_aware_index.tar.gz" \
       -o /tmp/index.tar.gz \
    && tar --no-same-owner -xzf /tmp/index.tar.gz -C data/index \
    && rm /tmp/index.tar.gz

# Hugging Face Spaces (Docker SDK) expects the app to listen on $PORT, default 7860.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn vrag.api.main:app --app-dir src --host 0.0.0.0 --port ${PORT}"]
