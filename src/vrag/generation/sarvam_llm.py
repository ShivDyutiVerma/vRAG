"""Sarvam LLM client for Track B generation. Real network call — the one thing on the request
path other than Sarvam STT allowed to touch the network, per CLAUDE.md hot-path rules.

Uses Sarvam's OpenAI-compatible chat completions endpoint with provider-native JSON schema mode
(docs.sarvam.ai/api-reference/chat/chat-completions, fetched 2026-08-18) rather than prompt-
engineered JSON + manual parsing — docs/TECH_MENU.md §S11 ranks this the top choice when the
provider supports it, and Sarvam does.

Streaming (`stream: true`), not a single blocking POST — built after live probing
(docs/DECISIONS_P.md P-017) found a real Sarvam bug distinct from P-R15's array-field issue: with
a multi-chunk (k=5-ish) context, the model sometimes completes `reasoning` and `answer` correctly
but then pads pure whitespace toward `max_tokens` instead of continuing to `cited_chunk_ids_csv`
and closing the object — a genuine failure, not budget starvation, but one that (non-streaming)
took 4-7s+ to even detect, since the only signal was the eventual `max_tokens` cutoff. Streaming
lets `_call_once_streaming` watch for that specific pattern (many consecutive whitespace/empty
content deltas with no forward progress) and abort in a few hundred ms instead of waiting out the
full token budget — see STALL_THRESHOLD_CHUNKS below for the evidence behind the threshold.

`search_corpus` tool (docs/DECISIONS_P.md P-019, AGENT_BUILD_SPEC.md §7.2 item 5, and the brief's
explicit "tool calls" requirement, C5) — real OpenAI-style function-calling, not simulated. Not
combined into the same request as `response_format: json_schema`: live-tested, and Sarvam doesn't
handle that combination reliably either (the model ignores the tool and tries to force an answer
into the schema, hitting the same whitespace-padding bug). Instead: the normal structured call
(unchanged, same cost as before) now also emits `needs_more_context`; only when that's true does
`generate()` escalate to a real tool-calling round (`tool_choice: "required"`, since we already
know a call is needed) followed by one final structured re-answer with the expanded context — tool
depth capped at 1 by construction (the follow-up call never offers the tool again).
"""

from __future__ import annotations

import json
import logging
from typing import NamedTuple

import httpx
from pydantic import ValidationError

from vrag.config import settings
from vrag.generation.schemas import GENERATED_ANSWER_JSON_SCHEMA, GeneratedAnswer
from vrag.retrieval.interface import RetrievedChunk, retrieve

logger = logging.getLogger(__name__)

_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"
# sarvam-105b-conversations is marketed as tuned for real-time conversational/voice-agent
# workloads, but measured (docs/DECISIONS_P.md) to be broken for structured JSON output — it pads
# pure whitespace instead of emitting content at all under response_format:json_schema, even with
# reasoning disabled. Plain sarvam-105b produces correct structured output; use that instead.
_MODEL = "sarvam-105b"

_SYSTEM_PROMPT = (
    "You are a grounded question-answering assistant for a Hindi voice product. Answer ONLY "
    "using the numbered context passages given. If the passages don't contain the answer, say "
    "so plainly in the `answer` field rather than guessing. "
    "CRITICAL: the `answer` field MUST be written in the same script and language as the "
    "context passages (Hindi/Devanagari) — never answer in English even if you reason in "
    "English internally. An answer in the wrong language is treated as a failure. "
    "Cite only chunk_ids that were actually given to you — never invent one. "
    "Keep the `reasoning` field to at most one short sentence (under 20 words) — it exists to "
    "nudge you to check the context before answering, not to record a full chain of thought. "
    "Spend your token budget on the answer, not on reasoning."
)

# Evidence: docs/DECISIONS_P.md P-017. Real streamed runs against the live API never showed more
# than 2 consecutive whitespace-only/empty content deltas during genuine JSON formatting (e.g. the
# newline+indent between fields) in any successful completion. The real bug this guards against
# (see module docstring) produces runs of hundreds of such chunks. 20 gives a large (10x) safety
# margin above legitimate formatting gaps while aborting within roughly a few hundred ms of the
# stall starting, not after the full max_tokens budget.
STALL_THRESHOLD_CHUNKS = 20

# OpenAI-style function definition, live-verified against Sarvam's chat completions endpoint
# (docs/DECISIONS_P.md P-019) — sarvam-105b correctly emits a `tool_calls` response when offered
# this, with well-formed JSON arguments.
SEARCH_CORPUS_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "search_corpus",
        "description": (
            "Search the knowledge corpus for additional passages when the given context is "
            "insufficient to answer the question. Returns up to k more passages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Follow-up search query"},
                "k": {"type": "integer", "description": "Number of passages to retrieve"},
            },
            "required": ["query", "k"],
        },
    },
}


class GenerationResult(NamedTuple):
    """generate()'s success return: the final answer plus every chunk that was actually
    available when it was produced. When the search_corpus follow-up fires, `chunks` is the
    original chunks plus whatever the tool call fetched, deduped — GenerateStage must validate
    citations (G4) against this full set, not just RetrieveStage's original list, or a real
    tool-fetched citation would be wrongly flagged as invented."""

    answer: GeneratedAnswer
    chunks: list[RetrievedChunk]


class _GenerationStalled(Exception):
    """Internal signal that a streamed response stopped making forward progress (see
    STALL_THRESHOLD_CHUNKS). Caught by generate()'s repair loop — never escapes this module."""

    def __init__(self, partial_content: str) -> None:
        super().__init__("generation stalled: too many consecutive empty/whitespace-only chunks")
        self.partial_content = partial_content


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    return "\n".join(f"[chunk_id={c.chunk_id}] {c.text}" for c in chunks)


async def search_corpus(query: str, k: int) -> list[RetrievedChunk]:
    """The tool Track B's LLM can call (AGENT_BUILD_SPEC.md §7.2 item 5) — a thin wrapper over
    the R/P seam. Never raises: retrieve() already guarantees that, and this adds nothing that
    could fail on its own."""
    k = max(1, min(k, 10))  # a runaway/adversarial k from the model shouldn't blow the budget
    return await retrieve(query, k=k)


async def _call_once_streaming(client: httpx.AsyncClient, messages: list[dict]) -> str:
    """Streams the completion and reconstructs the full content string, same contract as a
    non-streaming call's `choices[0].message.content` — except it can fail fast via
    `_GenerationStalled` when the response stops making forward progress (see module docstring
    and STALL_THRESHOLD_CHUNKS)."""
    payload = {
        "model": _MODEL,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "GeneratedAnswer",
                "schema": GENERATED_ANSWER_JSON_SCHEMA,
                "strict": True,
            },
        },
        "temperature": 0.2,
        "max_tokens": 512,
        # sarvam-105b is a reasoning model: by default it emits a reasoning_content chain-of-
        # thought that's billed and counted against max_tokens *before* the actual structured
        # content, and a short budget gets entirely consumed by reasoning with the real answer
        # never produced. Disabling it is Sarvam's own documented recommendation for
        # latency-sensitive/live-call use (docs/DECISIONS_P.md) and is essential here regardless
        # of latency, since 512 tokens of reasoning would otherwise starve the answer itself.
        "reasoning_effort": None,
        "stream": True,
    }
    headers = {
        "api-subscription-key": settings.sarvam_api_key,
        "Content-Type": "application/json",
    }

    accumulated = ""
    consecutive_stall_chunks = 0
    async with client.stream("POST", _CHAT_URL, headers=headers, json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:") :].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                choices = chunk.get("choices") or []
                # .get("content", "") only supplies the default when the key is absent -- Sarvam's
                # SSE stream sometimes sends an explicit "content": null (e.g. a trailing/role-only
                # delta), which .get() then returns as None, not "". `or ""` normalises both cases.
                delta = (choices[0]["delta"].get("content") or "") if choices else ""
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if delta and delta.strip():
                consecutive_stall_chunks = 0
            else:
                consecutive_stall_chunks += 1
                if consecutive_stall_chunks >= STALL_THRESHOLD_CHUNKS:
                    raise _GenerationStalled(accumulated)
            accumulated += delta
    return accumulated


async def _generate_structured(
    client: httpx.AsyncClient, messages: list[dict]
) -> GeneratedAnswer | None:
    """One structured-answer attempt with one repair-on-failure retry (parse failure or stall) —
    AGENT_BUILD_SPEC.md §7.2 item 6. `messages` is used as-is and mutated with repair turns if
    needed; the caller's own copy of the conversation (if it needs one) should be built before
    calling this. Returns None if both attempts fail. Never raises."""
    for attempt in range(2):
        try:
            raw = await _call_once_streaming(client, messages)
        except _GenerationStalled as exc:
            if attempt == 0:
                logger.info("Track B: generation stalled, retrying once")
                messages.append({"role": "assistant", "content": exc.partial_content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That response stalled before completing valid JSON. Reply "
                            "again with ONLY valid, complete JSON matching the schema."
                        ),
                    }
                )
                continue
            logger.error("Track B: repair attempt also stalled")
            return None
        try:
            parsed = json.loads(raw)
            return GeneratedAnswer.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as exc:
            if attempt == 0:
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"That wasn't valid JSON matching the schema ({exc}). "
                            "Reply again with ONLY valid JSON matching the schema."
                        ),
                    }
                )
                continue
            logger.error("Track B: repair attempt also failed to parse: %s", exc)
            return None
    return None


async def _call_tool_decision(
    client: httpx.AsyncClient, messages: list[dict]
) -> dict[str, str | int] | None:
    """Only reached when a structured call's `needs_more_context` was true — asks the model to
    actually call `search_corpus`. Non-streaming: live-observed responses here are short (~30
    completion tokens, just a tool call, no prose), so the stall-detection machinery built for
    long structured answers isn't needed. `tool_choice: "required"` since we already know a call
    is needed, not asking the model to reconsider. Returns None (not raises) if the model doesn't
    return a usable tool call — the caller falls back to the original answer rather than erroring,
    same "always have a fallback" pattern as everything else in this module."""
    payload = {
        "model": _MODEL,
        "messages": messages,
        "tools": [SEARCH_CORPUS_TOOL],
        "tool_choice": "required",
        "temperature": 0.2,
        "max_tokens": 128,
        "reasoning_effort": None,
    }
    headers = {
        "api-subscription-key": settings.sarvam_api_key,
        "Content-Type": "application/json",
    }
    resp = await client.post(_CHAT_URL, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    tool_calls = data["choices"][0]["message"].get("tool_calls") or []
    if not tool_calls:
        return None
    try:
        args = json.loads(tool_calls[0]["function"]["arguments"])
        query = str(args["query"])
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return None
    try:
        k = int(args.get("k", 5))
    except (TypeError, ValueError):
        k = 5
    return {"query": query, "k": k}


def _messages_for(query: str, chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{_build_context_block(chunks)}\n\nQuestion: {query}",
        },
    ]


async def generate(
    query: str, chunks: list[RetrievedChunk], timeout_s: float = 10.0
) -> GenerationResult | None:
    """Real network call. Returns None on any failure (missing key, timeout, HTTP error, or a
    parse failure that survives one repair attempt per AGENT_BUILD_SPEC.md §7.2 item 6) — the
    caller (GenerateStage) falls back to Track A. Never raises.

    Up to three round trips when the model signals `needs_more_context`: the initial structured
    answer, a tool-calling round to fetch more context (depth capped at 1 — this follow-up never
    offers the tool again), and a final structured re-answer over the expanded context. The common
    case (context already sufficient) costs exactly one round trip, same as before this feature.
    """
    if not settings.sarvam_api_key:
        return None

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            first = await _generate_structured(client, _messages_for(query, chunks))
            if first is None:
                return None
            if not first.needs_more_context:
                return GenerationResult(answer=first, chunks=chunks)

            tool_args = await _call_tool_decision(client, _messages_for(query, chunks))
            if tool_args is None:
                logger.info(
                    "Track B: needs_more_context=True but no tool call followed, "
                    "keeping the original answer"
                )
                return GenerationResult(answer=first, chunks=chunks)

            fetched = await search_corpus(str(tool_args["query"]), int(tool_args["k"]))
            if not fetched:
                return GenerationResult(answer=first, chunks=chunks)

            seen_ids = {c.chunk_id for c in chunks}
            expanded_chunks = chunks + [c for c in fetched if c.chunk_id not in seen_ids]

            second = await _generate_structured(client, _messages_for(query, expanded_chunks))
            if second is None:
                return GenerationResult(answer=first, chunks=chunks)
            return GenerationResult(answer=second, chunks=expanded_chunks)
    except Exception:
        logger.exception("Track B generation call failed")
        return None
