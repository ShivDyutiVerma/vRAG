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

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"


class EmbedderProtocol(Protocol):
    """The shape every class in EMBEDDER_REGISTRY already shares structurally (duck-typed there
    too — the registry is `dict[str, type]`, not `dict[str, type[EmbedderProtocol]]`). Exists so
    consumers outside this module (src/vrag/retrieval/hybrid.py) can type-hint "any embedder"
    without hard-importing a specific class — added when R-023 needed HybridRetriever to accept
    LiteE5Embedder as well as E5Embedder."""

    name: str

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...
    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...


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


DEFAULT_SPM_FILE_NAME = "sentencepiece.bpe.model"

# Special-token ID remap, verified byte-exact against 1,020 real strings + truncation/exotic-
# script edge cases before being trusted (docs/DECISIONS_R.md R-029, R-030). Raw sentencepiece's
# own vocab layout for this model is 0=<unk> 1=<s> 2=</s> 3=<first real piece>... (sentencepiece
# has no native pad concept: SentencePieceProcessor.pad_id() == -1). The ONNX model was exported
# via sentence-transformers' HF tokenizer.json conversion, which renumbers specials to
# 0=<s> 1=<pad> 2=</s> 3=<unk> and shifts every real piece up by 1 to make room for <pad> — the
# ONNX model's embedding table is indexed by *those* IDs, not sentencepiece's native ones, so this
# remap is required for correctness, not an optional detail.
_SPM_BOS_ID = 0  # HF <s> — prepended to every sequence, not present in sp.encode()'s own output
_SPM_EOS_ID = 2  # HF </s> — appended to every sequence
_SPM_PAD_ID = 1  # HF <pad> — matches tokenizers.Tokenizer.enable_padding(pad_id=1) exactly
_SPM_UNK_HF_ID = 3  # HF <unk> — sentencepiece's own <unk> is raw id 0, remapped to this
_SPM_NATIVE_UNK_ID = 0  # sentencepiece's own <unk>, per SentencePieceProcessor.unk_id()


def _remap_spm_id(raw_id: int) -> int:
    if raw_id == _SPM_NATIVE_UNK_ID:
        return _SPM_UNK_HF_ID
    return raw_id + 1


class LiteE5Embedder:
    """Same ONNX int8 model as `ONNXE5Embedder`, but bypasses `sentence-transformers` entirely
    (docs/DECISIONS_R.md R-022) — raw `onnxruntime.InferenceSession` + Google's `sentencepiece`
    library (R-029/R-030 — replaced the HF `tokenizers` Rust library, which cost ~262MB RSS for
    this model's 250,002-token vocabulary, ~2x the ONNX session's own ~137MB), with padding,
    truncation, attention-mask generation, and mean-pooling/L2-normalisation all done by hand in
    numpy. Verified byte-identical output to `ONNXE5Embedder` (cosine similarity 1.0, max abs diff
    ~1.5e-8, pure float32 rounding noise) — this is the exact same model and pipeline, not an
    approximation.

    `sentencepiece` produces the exact same token IDs as `tokenizers` did, verified against 1,020
    real strings (500 Hindi, 500 English, 20 mixed) plus truncation-boundary and exotic-script
    edge cases — 100.0000% exact match, once the special-token remap above is applied (see the
    module-level comment). `tests/index/test_lite_e5_embedder_tokenizer_regression.py` re-verifies
    this live against the real `tokenizers` library on every test run, not just once at
    implementation time, so any future drift (a re-exported model, a changed vocab) fails loudly
    instead of silently producing wrong embeddings.

    The reason this class avoids `sentence-transformers`: it imports `torch` as a hard dependency
    regardless of which backend actually does inference (R-020's finding) — `import torch` alone
    measured at ~383MB RSS, the dominant cost in `ONNXE5Embedder`'s ~980MB embedder footprint. This
    class needs only `onnxruntime` (already a `retrieval` extra dependency) and `sentencepiece`
    (added as a new `retrieval-lean` dependency, R-029/R-030 — small, ~a few MB, no `torch`), never
    touching `torch` at import time.

    Same interface, same query:/passage: prefix requirement as `E5Embedder`/`ONNXE5Embedder` — a
    drop-in replacement at the embedding layer.
    """

    name = "multilingual-e5-small-lite-onnx-int8"

    def __init__(
        self,
        model_dir: str = DEFAULT_ONNX_MODEL_DIR,
        file_name: str = DEFAULT_ONNX_FILE_NAME,
        spm_file_name: str = DEFAULT_SPM_FILE_NAME,
        max_length: int = 512,
    ) -> None:
        self._model_dir = model_dir
        self._file_name = file_name
        self._spm_file_name = spm_file_name
        self._max_length = max_length
        self._session: Any = None
        self._sp: Any = None

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._session is None:
            import onnxruntime as ort
            import sentencepiece as spm

            self._sp = spm.SentencePieceProcessor(
                model_file=f"{self._model_dir}/{self._spm_file_name}"
            )
            self._session = ort.InferenceSession(f"{self._model_dir}/{self._file_name}")
        return self._session, self._sp

    def _tokenize_batch(self, texts: list[str]) -> tuple[Any, Any]:
        """Reproduces exactly what `tokenizers.Tokenizer.enable_padding(pad_id=1)` +
        `enable_truncation(max_length=...)` + `encode_batch()` did: right-pad every sequence in
        the batch to the batch's own longest sequence (not always `max_length` — verified against
        the old tokenizer's real behavior, docs/DECISIONS_R.md R-029), truncate to `max_length`
        keeping BOS+EOS, attention_mask=0 on pad positions."""
        import numpy as np

        _session, sp = self._ensure_loaded()
        # sp.encode() handles this model's own normalizer (Precompiled charsmap) and pre-tokenizer
        # (Metaspace "▁") internally, exactly as tokenizer.json's config specifies — verified
        # directly against tokenizer.json's config before assuming, not guessed (R-029).
        batch_core_ids = sp.encode(texts, out_type=int)

        # Genuine divergence found by regression testing (R-030), not assumed away: the old
        # tokenizer's Metaspace pre-tokenizer emits one trailing "▁" piece when the input ends in
        # whitespace (any amount — one space or many, always exactly one token); raw
        # sentencepiece's own encode() silently strips trailing whitespace instead. Verified this
        # is the *entire* discrepancy via a systematic sweep (leading/internal whitespace never
        # differs; only a non-empty string ending in whitespace does) before patching narrowly
        # rather than broadly. `piece_to_id` looked up dynamically, not hardcoded, so this stays
        # correct even if the metaspace piece's ID ever shifts in a future model export.
        trailing_space_id = sp.piece_to_id("▁")

        wrapped = []
        for text, core_ids in zip(texts, batch_core_ids, strict=True):
            if text and text != text.rstrip():
                core_ids = [*core_ids, trailing_space_id]
            # Truncate core tokens to leave room for BOS+EOS, keeping EOS as the true last token
            # — verified against the real tokenizer's truncation behavior on a >512-token string
            # (R-029's edge-case test), not assumed from a general HF convention.
            truncated_core = core_ids[: self._max_length - 2]
            ids = [_SPM_BOS_ID, *(_remap_spm_id(i) for i in truncated_core), _SPM_EOS_ID]
            wrapped.append(ids)

        max_len_in_batch = max((len(ids) for ids in wrapped), default=0)
        input_ids = np.full((len(wrapped), max_len_in_batch), _SPM_PAD_ID, dtype=np.int64)
        attention_mask = np.zeros((len(wrapped), max_len_in_batch), dtype=np.int64)
        for row, ids in enumerate(wrapped):
            input_ids[row, : len(ids)] = ids
            attention_mask[row, : len(ids)] = 1
        return input_ids, attention_mask

    def _embed(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        session, _sp = self._ensure_loaded()
        if not texts:
            return []

        input_ids, attention_mask = self._tokenize_batch(texts)
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
