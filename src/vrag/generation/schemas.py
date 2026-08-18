"""Structured output schema for Track B generation. See AGENT_BUILD_SPEC.md §7.2 item 6 and §11
(S11): the reasoning field comes before the answer field so the model has somewhere to "think"
before committing — this is a hard-path invariant per CLAUDE.md, do not reorder these fields.

`cited_chunk_ids` is a comma-separated string in the wire schema, not a JSON array, despite that
being the more natural type. Found (and reported in docs/DECISIONS_P.md) a real bug in Sarvam's
`response_format: json_schema` mode: any `array`-typed field causes the model to correctly fill
the preceding fields, then pad the rest of `max_tokens` with whitespace instead of emitting the
array and closing the object — confirmed reproducible and isolated by removing the array field
alone and observing `finish_reason` flip from "length" to "stop". A comma-separated string
sidesteps the provider bug entirely while keeping the response genuinely structured.
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
    cited_chunk_ids_csv: str = Field(
        description="Comma-separated chunk_id values from the provided context that support "
        "the answer (e.g. 'chunk_001,chunk_004'). Must be a subset of the chunk_ids actually "
        "given — never invent one. Empty string if none apply."
    )

    @property
    def cited_chunk_ids(self) -> list[str]:
        return [c.strip() for c in self.cited_chunk_ids_csv.split(",") if c.strip()]


GENERATED_ANSWER_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "answer": {"type": "string"},
        "cited_chunk_ids_csv": {"type": "string"},
    },
    "required": ["reasoning", "answer", "cited_chunk_ids_csv"],
}
