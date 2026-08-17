FROM python:3.11-slim

WORKDIR /app

# System deps for future R-side libraries (faiss, onnxruntime) land here once needed —
# kept minimal for the Day 1 walking skeleton.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

COPY frontend/ ./frontend/

# Hugging Face Spaces (Docker SDK) expects the app to listen on $PORT, default 7860.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn vrag.api.main:app --app-dir src --host 0.0.0.0 --port ${PORT}"]
