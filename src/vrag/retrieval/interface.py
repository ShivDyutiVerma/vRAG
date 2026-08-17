"""The R/P seam. Agreed jointly on Day 0 (docs/TEAM_SPLIT.md §1, docs/FRIEND_BRIEF.md §3,
docs/API_CONTRACTS.md). Changing this signature after Day 0 is a joint decision — record a new ADR
in docs/DECISIONS.md immediately if it ever needs to change.

Workstream P's harness calls retrieve() and only retrieve(). Workstream R owns the real
implementation. Currently stubbed — Workstream R replaces the body with the real hybrid
dense+sparse retrieval pipeline (see src/vrag/retrieval/hybrid.py's HybridRetriever, built and
tested, not yet wired in here pending the chunking ablation's winner); the signature does not
change when that happens.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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


async def retrieve(query: str, k: int = 5) -> list[RetrievedChunk]:
    """The one function Workstream P's harness calls to get retrieved context.

    Never raises. Returns [] on internal failure — the guardrail layer treats an empty list as
    "nothing relevant found" and routes to abstention (G3).

    STUB (Day 1): ignores `query`, returns up to `k` hardcoded fake chunks so the rest of the
    pipeline is testable end to end immediately. Swapped for Workstream R's real
    HybridRetriever.retrieve at the Day 2 integration sync (docs/TEAM_SPLIT.md §5) — should be a
    one-line import change.
    """
    if not query or not query.strip():
        return []
    return _STUB_CHUNKS[:k]
