"""The R/P seam. Agreed jointly on Day 0 (docs/TEAM_SPLIT.md §1, docs/FRIEND_BRIEF.md §3,
docs/API_CONTRACTS.md). Changing this signature after Day 0 is a joint decision — record a new ADR
in docs/DECISIONS.md immediately if it ever needs to change.

Workstream P's harness calls retrieve() and only retrieve(). Workstream R owns the real
implementation. Currently stubbed — Workstream R replaces the body with the real hybrid
dense+sparse retrieval pipeline; the signature does not change when that happens.
"""

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    chunk_id: str
    passage_id: str
    text: str
    score: float  # fused/reranked relevance score, 0-1
    language: str  # ISO code detected/tagged at index time


async def retrieve(query: str, k: int = 5) -> list[RetrievedChunk]:
    """Never raises. Returns [] on internal failure — the harness's grounding gate treats
    an empty list as "nothing relevant found" and routes to abstention.

    STUB (Day 0): returns canned data so Workstream P can build the whole pipeline against
    a stable shape immediately. Replaced with the real hybrid retrieve() during Phase 2/3.
    """
    return [
        RetrievedChunk(
            chunk_id="stub-chunk-0001",
            passage_id="stub-passage-0001",
            text=(
                "यह एक स्टब पैसेज है जो असली रिट्रीवल लागू होने तक "
                "पाइपलाइन का परीक्षण करने के लिए उपयोग किया जाता है।"
            ),
            score=0.91,
            language="hi",
        ),
        RetrievedChunk(
            chunk_id="stub-chunk-0002",
            passage_id="stub-passage-0002",
            text=(
                "दूसरा स्टब पैसेज, जिसमें अलग स्कोर है ताकि रैंकिंग व्यवहार का "
                "परीक्षण किया जा सके।"
            ),
            score=0.77,
            language="hi",
        ),
    ][:k]
