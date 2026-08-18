"""Sarvam LLM client for Track B generation. Real network call — the one thing on the request
path other than Sarvam STT allowed to touch the network, per CLAUDE.md hot-path rules.

Uses Sarvam's OpenAI-compatible chat completions endpoint with provider-native JSON schema mode
(docs.sarvam.ai/api-reference/chat/chat-completions, fetched 2026-08-18) rather than prompt-
engineered JSON + manual parsing — docs/TECH_MENU.md §S11 ranks this the top choice when the
provider supports it, and Sarvam does.

Non-streaming for now: a real, structured, grounded answer, correctly parsed and repaired on one
failure, is worth more today than a token-streaming version that isn't built yet. Streaming is a
follow-up (docs/DECISIONS_P.md).
"""

from __future__ import annotations

import json
import logging

import httpx
from pydantic import ValidationError

from vrag.config import settings
from vrag.generation.schemas import GENERATED_ANSWER_JSON_SCHEMA, GeneratedAnswer
from vrag.retrieval.interface import RetrievedChunk

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
    "Cite only chunk_ids that were actually given to you — never invent one."
)


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    return "\n".join(f"[chunk_id={c.chunk_id}] {c.text}" for c in chunks)


async def _call_once(client: httpx.AsyncClient, messages: list[dict]) -> str:
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
    }
    headers = {
        "api-subscription-key": settings.sarvam_api_key,
        "Content-Type": "application/json",
    }
    resp = await client.post(_CHAT_URL, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


async def generate(
    query: str, chunks: list[RetrievedChunk], timeout_s: float = 10.0
) -> GeneratedAnswer | None:
    """Real network call. Returns None on any failure (missing key, timeout, HTTP error, or a
    parse failure that survives one repair attempt per AGENT_BUILD_SPEC.md §7.2 item 6) — the
    caller (GenerateStage) falls back to Track A. Never raises."""
    if not settings.sarvam_api_key:
        return None

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{_build_context_block(chunks)}\n\nQuestion: {query}",
        },
    ]

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            for attempt in range(2):  # one repair attempt on parse failure
                raw = await _call_once(client, messages)
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
    except Exception:
        logger.exception("Track B generation call failed")
        return None
    return None
