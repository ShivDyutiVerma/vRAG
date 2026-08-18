"""The R/P seam. Agreed jointly on Day 0 (docs/TEAM_SPLIT.md §1, docs/FRIEND_BRIEF.md §3,
docs/API_CONTRACTS.md). Changing this signature after Day 0 is a joint decision — record a new ADR
in docs/DECISIONS.md immediately if it ever needs to change.

Workstream P's harness calls retrieve() and only retrieve(). Workstream R owns the real
implementation.

Real path: loads the persisted `metadata_aware` index (docs/DECISIONS_R.md R-004 — the A1 ablation
winner, chosen as "cheapest among five statistically-tied strategies") from `data/index/`, built
offline by `scripts/build_index.py --save-dir` (AGENT_BUILD_SPEC.md §5.3 — never build the index at
container start). Delegates to `HybridRetriever` (src/vrag/retrieval/hybrid.py) in `retrieval_mode=
"dense"` — the A3 ablation winner (docs/DECISIONS_R.md R-010) — not the originally-assumed hybrid
RRF path; see that ADR and the module docstring in hybrid.py for the measured reason and its
consequences for RetrievedChunk.score's scale.

Fallback: if the persisted index isn't present on disk (e.g. a fresh clone, or CI, where the
multi-GB corpus + model haven't been downloaded), falls back to the Day-1 stub instead of raising —
this file's tests and Workstream P's harness both keep working either way; only the *content* of
what comes back differs. The stub's shape is identical to the real path's output, so nothing
downstream needs to know which one answered.

Also falls back to the stub if the index *files* are present but loading them fails for any reason
(missing retrieval dependencies, corrupt/partial artifact, etc.) — found while staging the
deployment-memory fix in docs/DECISIONS_P.md P-018: a container could plausibly have the index
downloaded before the leaner embedder dependencies are installed, and `retrieve()`'s own docstring
already promises "Never raises." A missing import is exactly the kind of internal failure that
promise is supposed to cover, not just a missing file.

Embedder/chunk-lookup choice (docs/DECISIONS_R.md R-023): uses `LiteE5Embedder` (raw onnxruntime +
tokenizers, no torch/transformers import) and `load_built_index_lean()` (SQLite-backed chunk lookup
when the artifact has one, else the plain dict) — the R-021/R-022 memory fixes, verified
byte-identical to the original `E5Embedder`+dict path, not an approximation. Needs the
`retrieval-lean` extra installed (onnxruntime + tokenizers + faiss-cpu + bm25s — NOT torch/
transformers/sentence-transformers) and the lean ONNX model directory on disk; see docs/RISKS.md R4
for what's still needed on the deploy side to actually reach these code paths in production.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vrag.retrieval.hybrid import HybridRetriever

_INDEX_DIR = Path(__file__).resolve().parents[3] / "data" / "index" / "metadata_aware"

# A3 winner, docs/DECISIONS_R.md R-010 — single source of truth, not a literal repeated at each
# call site below. Both load_built_index_lean() (which decides whether to load the BM25/sparse
# index at all, docs/DECISIONS.md ADR-006) and HybridRetriever's own retrieval_mode must agree on
# this value or construction raises immediately (see hybrid.py's __init__ guard) rather than
# silently loading a sparse index that's never used, or skipping one that's needed.
_RETRIEVAL_MODE = "dense"


class RetrievedChunk(BaseModel):
    chunk_id: str
    passage_id: str
    text: str
    score: float = Field(ge=0.0, le=1.0, description="Fused/reranked relevance, 0-1")
    language: str  # ISO code detected/tagged at index time


_STUB_CHUNKS: list[RetrievedChunk] = [
    RetrievedChunk(
        chunk_id="stub-chunk-001",
        passage_id="stub-passage-4192",
        text=(
            "कंचनजंगा भारत में स्थित सबसे ऊँचा पर्वत है, जिसकी ऊँचाई लगभग 8,586 मीटर है। "
            "यह हिमालय पर्वत श्रृंखला का हिस्सा है और भारत-नेपाल सीमा पर स्थित है।"
        ),
        score=0.91,
        language="hi",
    ),
    RetrievedChunk(
        chunk_id="stub-chunk-002",
        passage_id="stub-passage-0891",
        text=(
            "हिमालय पर्वत श्रृंखला दुनिया की सबसे ऊँची पर्वत श्रृंखला है, जिसमें एवरेस्ट सहित "
            "कई प्रमुख चोटियाँ शामिल हैं।"
        ),
        score=0.74,
        language="hi",
    ),
    RetrievedChunk(
        chunk_id="stub-chunk-003",
        passage_id="stub-passage-1027",
        text="भारत का भूगोल पर्वत, पठार, मैदान और तटीय क्षेत्रों में विभाजित है।",
        score=0.52,
        language="hi",
    ),
]

_retriever: HybridRetriever | None = None
_retriever_load_attempted = False


def _get_real_retriever() -> HybridRetriever | None:
    """Lazy singleton — loading the index (FAISS mmap + BM25 + ~100k-entry chunk lookup) costs
    real time and memory, so it happens once, on first actual use, not at import time (every test
    file that imports this module would otherwise pay that cost even when never calling
    retrieve())."""
    global _retriever, _retriever_load_attempted
    if _retriever_load_attempted:
        return _retriever
    _retriever_load_attempted = True

    if not (_INDEX_DIR / "chunk_lookup.json").exists() and not (
        _INDEX_DIR / "chunk_lookup.sqlite3"
    ).exists():
        return None

    try:
        from vrag.index.embedder import LiteE5Embedder
        from vrag.index.persistence import load_built_index_lean
        from vrag.retrieval.hybrid import HybridRetriever

        dense, sparse, chunk_lookup = load_built_index_lean(
            _INDEX_DIR, retrieval_mode=_RETRIEVAL_MODE
        )
        _retriever = HybridRetriever(
            dense=dense,
            sparse=sparse,
            embedder=LiteE5Embedder(),
            chunk_lookup=chunk_lookup,
            retrieval_mode=_RETRIEVAL_MODE,
        )
    except Exception:
        logger.exception(
            "Real retriever failed to load despite index files being present "
            "(missing dependency? corrupt artifact?) — falling back to the stub"
        )
        return None
    return _retriever


async def retrieve(query: str, k: int = 5) -> list[RetrievedChunk]:
    """The one function Workstream P's harness calls to get retrieved context.

    Never raises. Returns [] on internal failure — the guardrail layer treats an empty list as
    "nothing relevant found" and routes to abstention (G3).
    """
    if not query or not query.strip():
        return []

    retriever = _get_real_retriever()
    if retriever is not None:
        return await retriever.retrieve(query, k)

    return _STUB_CHUNKS[:k]
