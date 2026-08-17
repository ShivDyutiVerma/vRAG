"""multilingual-e5-small embedder (AGENT_BUILD_SPEC.md §5.2, §7.1 gotchas). PyTorch first here per
docs/BUILD_PLAN.md P1 — ONNX int8 quantisation is a Phase 6 optimisation, not done yet.

E5 was TRAINED with literal "query: " / "passage: " string prefixes on its inputs and produces
worse (not erroring, just silently worse) embeddings without them — this is the "no error, just
wrong" bug class CLAUDE.md calls out explicitly, which is why `format_query`/`format_passage` are
separated out as pure, fast-testable functions rather than buried inline in the model call.

Embeddings are L2-normalised at encode time so FAISS inner-product search equals cosine similarity
(AGENT_BUILD_SPEC.md §5.2 gotcha).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"


def format_query(text: str) -> str:
    return f"{QUERY_PREFIX}{text}"


def format_passage(text: str) -> str:
    return f"{PASSAGE_PREFIX}{text}"


class E5Embedder:
    """Lazy-loads the model on first real use — importing this module (e.g. transitively, from
    something that only needs format_query/format_passage) must not trigger a model download."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _model_instance(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        prefixed = [format_query(t) for t in texts]
        vectors = self._model_instance().encode(prefixed, normalize_embeddings=True)
        return vectors.tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed = [format_passage(t) for t in texts]
        vectors = self._model_instance().encode(prefixed, normalize_embeddings=True)
        return vectors.tolist()
