"""Structured output schema for Track B generation. See AGENT_BUILD_SPEC.md §7.2 item 6 and §11
(S11): the reasoning field comes before the answer field so the model has somewhere to "think"
before committing — this is a hard-path invariant per CLAUDE.md, do not reorder these fields.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GeneratedAnswer(BaseModel):
    reasoning: str = Field(
        description=(
            "Brief internal reasoning connecting the retrieved context to the answer. "
            "Not shown to the user."
        )
    )
    answer: str = Field(
        description="The final, fluent answer to the user's question, grounded only in the "
        "provided context. If the context doesn't support an answer, say so plainly."
    )
    cited_chunk_ids: list[str] = Field(
        description="chunk_id values from the provided context that support the answer. "
        "Must be a subset of the chunk_ids actually given — never invent one."
    )


GENERATED_ANSWER_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "answer": {"type": "string"},
        "cited_chunk_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reasoning", "answer", "cited_chunk_ids"],
}
