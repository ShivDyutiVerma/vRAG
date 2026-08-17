"""The seam between Workstream R (retrieval) and Workstream P (everything else).

Workstream R owns the real implementation of `retrieve()`. This file currently holds a stub that
returns fake data so the harness, guardrails, and generation stages can be built and tested end to
end without waiting on the real index. Do not add retrieval logic here — see
`docs/TEAM_SPLIT.md` §1 and §2 for the ownership boundary. Changing this signature after Day 0
requires a joint ADR in `docs/DECISIONS.md`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

_STUB_CHUNKS: list[dict[str, str | float]] = [
    {
        "chunk_id": "stub-chunk-001",
        "passage_id": "stub-passage-4192",
        "text": (
            "कंचनजंगा भारत में स्थित सबसे ऊँचा पर्वत है, जिसकी ऊँचाई लगभग 8,586 मीटर है। "
            "यह हिमालय पर्वत श्रृंखला का हिस्सा है और भारत-नेपाल सीमा पर स्थित है।"
        ),
        "score": 0.91,
        "language": "hi",
    },
    {
        "chunk_id": "stub-chunk-002",
        "passage_id": "stub-passage-0891",
        "text": (
            "हिमालय पर्वत श्रृंखला दुनिया की सबसे ऊँची पर्वत श्रृंखला है, जिसमें एवरेस्ट सहित "
            "कई प्रमुख चोटियाँ शामिल हैं।"
        ),
        "score": 0.74,
        "language": "hi",
    },
    {
        "chunk_id": "stub-chunk-003",
        "passage_id": "stub-passage-1027",
        "text": "भारत का भूगोल पर्वत, पठार, मैदान और तटीय क्षेत्रों में विभाजित है।",
        "score": 0.52,
        "language": "hi",
    },
]


class RetrievedChunk(BaseModel):
    chunk_id: str
    passage_id: str
    text: str
    score: float = Field(ge=0.0, le=1.0, description="Fused/reranked relevance, 0-1")
    language: str


async def retrieve(query: str, k: int = 5) -> list[RetrievedChunk]:
    """The one function Workstream P's harness calls to get retrieved context.

    Never raises. Returns [] on internal failure — the guardrail layer treats an empty list as
    "nothing relevant found" and routes to abstention (G3).

    STUB (Day 1): ignores `query`, returns up to `k` hardcoded fake chunks so the rest of the
    pipeline is testable end to end immediately. Swapped for Workstream R's real implementation
    at the Day 2 integration sync (docs/TEAM_SPLIT.md §5) — should be a one-line import change.
    """
    if not query or not query.strip():
        return []
    return [RetrievedChunk(**chunk) for chunk in _STUB_CHUNKS[:k]]
