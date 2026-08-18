"""Embedder backends for the A2 ablation (docs/TECH_MENU.md §S4, docs/BUILD_PLAN.md P3). Four
candidates, one shared shape (`embed_queries`/`embed_passages` -> L2-normalised vectors so FAISS
inner-product search equals cosine similarity, AGENT_BUILD_SPEC.md §5.2 gotcha):

  - E5Embedder (multilingual-e5-small) — the A1 default. PyTorch first per docs/BUILD_PLAN.md P1;
    ONNX int8 quantisation is a Phase 6 optimisation, not done yet. Requires literal "query: " /
    "passage: " prefixes at inference time — trained with them, silently worse (not an error)
    without them, which is why format_query/format_passage are pure, separately-tested functions
    rather than buried inline in the model call.
  - Model2VecEmbedder (potion-multilingual-128M) — static (non-contextual) lookup+mean-pool
    embeddings, no transformer forward pass, no prefix convention.
  - BGEM3Embedder (BAAI/bge-m3) — dense mode only here (it also does sparse + multi-vector, out of
    scope for a dense-only A2 comparison). No prefix required — verified against the model card,
    not assumed: BGE-M3 explicitly dropped the older BGE models' query-instruction requirement.
  - VyakyarthEmbedder (krutrim-ai-labs/Vyakyarth) — the Indic-specialist wildcard. No prefix
    required — also verified against the model card.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

    name = "multilingual-e5-small"

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


DEFAULT_ONNX_MODEL_DIR = "data/onnx/multilingual-e5-small"
DEFAULT_ONNX_FILE_NAME = "onnx/model_quint8_avx2.onnx"


class ONNXE5Embedder:
    """`multilingual-e5-small`, ONNX-exported and dynamically int8-quantised for CPU inference
    (docs/BUILD_PLAN.md P6 task 5, docs/DECISIONS_R.md R-019). CPU only, per CLAUDE.md's hot-path
    invariant: "ONNX int8 is for CPU only — on GPU it is slower than FP32; use FP16 there" — this
    is the production embedder shape, not for use on this dev machine's GPU (`E5Embedder` stays the
    right choice for offline GPU index-building, docs/DECISIONS_R.md R-005).

    Requires `scripts/export_onnx_embedder.py` to have been run first — this class loads a local,
    pre-exported model directory, it does not export on the fly (exporting needs `torch` +
    `optimum`'s export path, which are offline/dev-time tools; the deployed production path should
    only need `onnxruntime` + `sentence-transformers[onnx]`'s inference-only backend).

    Same interface, same query:/passage: prefix requirement as `E5Embedder` — this is a drop-in
    replacement at the embedding layer, not a different model.
    """

    name = "multilingual-e5-small-onnx-int8"

    def __init__(
        self,
        model_dir: str = DEFAULT_ONNX_MODEL_DIR,
        file_name: str = DEFAULT_ONNX_FILE_NAME,
    ) -> None:
        self._model_dir = model_dir
        self._file_name = file_name
        self._model: SentenceTransformer | None = None

    def _model_instance(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._model_dir,
                backend="onnx",
                model_kwargs={"file_name": self._file_name},
            )
        return self._model

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        prefixed = [format_query(t) for t in texts]
        vectors = self._model_instance().encode(prefixed, normalize_embeddings=True)
        return vectors.tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed = [format_passage(t) for t in texts]
        vectors = self._model_instance().encode(prefixed, normalize_embeddings=True)
        return vectors.tolist()


class LiteE5Embedder:
    """Same ONNX int8 model as `ONNXE5Embedder`, but bypasses `sentence-transformers` entirely
    (docs/DECISIONS_R.md R-022) — raw `onnxruntime.InferenceSession` + the `tokenizers` library's
    Rust tokenizer, with mean-pooling and L2-normalisation done by hand in numpy. Verified
    byte-identical output to `ONNXE5Embedder` (cosine similarity 1.0, max abs diff ~1.5e-8, pure
    float32 rounding noise) — this is the exact same model and pipeline, not an approximation.

    The reason this class exists: `sentence-transformers` imports `torch` as a hard dependency
    regardless of which backend actually does inference (R-020's finding) — `import torch` alone
    measured at ~383MB RSS, the dominant cost in `ONNXE5Embedder`'s ~980MB embedder footprint. This
    class needs only `onnxruntime` (already a `retrieval` extra dependency) and `tokenizers`
    (already installed transitively via `transformers`/`sentence-transformers` — no new
    dependency), neither of which touches `torch` at import time.

    Same interface, same query:/passage: prefix requirement as `E5Embedder`/`ONNXE5Embedder` — a
    drop-in replacement at the embedding layer.
    """

    name = "multilingual-e5-small-lite-onnx-int8"

    def __init__(
        self,
        model_dir: str = DEFAULT_ONNX_MODEL_DIR,
        file_name: str = DEFAULT_ONNX_FILE_NAME,
        max_length: int = 512,
    ) -> None:
        self._model_dir = model_dir
        self._file_name = file_name
        self._max_length = max_length
        self._session: Any = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._session is None:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            self._tokenizer = Tokenizer.from_file(f"{self._model_dir}/tokenizer.json")
            self._tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
            self._tokenizer.enable_truncation(max_length=self._max_length)
            self._session = ort.InferenceSession(f"{self._model_dir}/{self._file_name}")
        return self._session, self._tokenizer

    def _embed(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        session, tokenizer = self._ensure_loaded()
        if not texts:
            return []

        encodings = tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)

        (last_hidden_state,) = session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        mask = attention_mask[..., None].astype(np.float32)
        mean_pooled = (last_hidden_state * mask).sum(axis=1) / mask.sum(axis=1)
        normalized = mean_pooled / np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        return normalized.tolist()

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed([format_query(t) for t in texts])

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._embed([format_passage(t) for t in texts])


class Model2VecEmbedder:
    """potion-multilingual-128M — static (non-contextual) embeddings via lookup + mean-pool, no
    transformer forward pass at all (docs/TECH_MENU.md §S4: "up to 500x faster than the source
    transformer"). No query/passage prefix convention — Model2Vec's training doesn't use
    instruction prefixes the way E5 does.

    Loaded via `sentence_transformers.SentenceTransformer` (its `StaticEmbedding` backend natively
    supports Model2Vec models), not the dedicated `model2vec` package the model card recommends as
    "the fastest and most lightweight way to run Model2Vec models" —
    `model2vec.StaticModel.from_pretrained` fetches the *entire* HF repo including ~25 benchmark
    eval-result YAMLs unrelated to inference, which hung repeatedly on this network.
    `sentence-transformers` only fetches the files it actually needs. The model's core speed
    advantage (no transformer forward pass, a static lookup+mean-pool) comes from its architecture,
    not the loading library, so this doesn't misrepresent the property being tested — see
    docs/DECISIONS_R.md R-007 for the full account,
    including the `model2vec` dependency that's now unused (left installed rather than reverted
    mid-ablation; harmless either way).
    """

    name = "potion-multilingual-128M"
    DEFAULT_MODEL_NAME = "minishlab/potion-multilingual-128M"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _model_instance(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model_instance().encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model_instance().encode(texts, normalize_embeddings=True)
        return vectors.tolist()


class BGEM3Embedder:
    """BAAI/bge-m3, dense mode only (it also does sparse + multi-vector retrieval from the same
    model — out of scope for a dense-only A2 comparison, see docs/TECH_MENU.md §S4's "special
    case" note). No prefix required — verified against the model card, not assumed: BGE-M3
    explicitly dropped the older BGE models' query-instruction requirement.
    """

    name = "bge-m3"
    DEFAULT_MODEL_NAME = "BAAI/bge-m3"

    # BGE-M3 is the largest of the 4 A2 candidates (568M params, 1024-dim hidden states). The
    # default sentence-transformers batch_size (32) repeatedly hit CUDA OOM on this machine's 6GB
    # laptop GPU (docs/DECISIONS_R.md R-008) — smaller batches trade a bit of throughput for
    # actually fitting in VRAM.
    BATCH_SIZE = 8

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _model_instance(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model_instance().encode(
            texts, normalize_embeddings=True, batch_size=self.BATCH_SIZE
        )
        return vectors.tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model_instance().encode(
            texts, normalize_embeddings=True, batch_size=self.BATCH_SIZE
        )
        return vectors.tolist()


class VyakyarthEmbedder:
    """krutrim-ai-labs/Vyakyarth — the Indic-specialist wildcard (docs/TECH_MENU.md §S4), XLM-R
    based, contrastively trained on Indic retrieval benchmarks. No prefix required — verified
    against the model card, not assumed.
    """

    name = "vyakyarth"
    DEFAULT_MODEL_NAME = "krutrim-ai-labs/Vyakyarth"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _model_instance(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model_instance().encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model_instance().encode(texts, normalize_embeddings=True)
        return vectors.tolist()


EMBEDDER_REGISTRY: dict[str, type] = {
    E5Embedder.name: E5Embedder,
    ONNXE5Embedder.name: ONNXE5Embedder,
    LiteE5Embedder.name: LiteE5Embedder,
    Model2VecEmbedder.name: Model2VecEmbedder,
    BGEM3Embedder.name: BGEM3Embedder,
    VyakyarthEmbedder.name: VyakyarthEmbedder,
}
