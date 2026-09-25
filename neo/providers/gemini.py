"""Gemini brain (google-genai SDK) — the default agentic + vision provider.

Free tier of Gemini 3.x Flash is the only free option with enough TPM for an agent loop.
It is also the one that gets overloaded, so every request walks a model list
(3.8 → 3.7 → 3.5 Flash) with a short backoff on 503 before giving up to the next brain.

Gemini 3 requires the ``thought_signature`` of every function call to be echoed back when the
history is replayed; calls that came from another brain (Groq, Claude) carry none, so we send
Google's documented bypass value for those.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Literal

from google import genai
from google.genai import errors as gerrors
from google.genai import types as gt

from neo.config import settings
from neo.providers.base import (
    Message,
    ProviderError,
    RateLimited,
    ToolCall,
    ToolSpec,
    Turn,
    Usage,
)

# The SDK nags about automatic function calling on the async path even though we never use it.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

_LEVEL = {"low": gt.ThinkingLevel.LOW, "medium": gt.ThinkingLevel.MEDIUM, "high": gt.ThinkingLevel.HIGH}
_SKIP_SIGNATURE = b"skip_thought_signature_validator"
_RETRY_CODES = {500, 502, 503, 504}
_QUOTA_COOLDOWN_S = 600.0  # a (model, key) that 429'd is skipped for this long
_OVERLOAD_COOLDOWN_S = 90.0  # a model that 5xx'd after a retry is skipped for this long
_BACKOFF_S = (1.2,)  # one short retry on 5xx; a bad night should fall through to Groq fast


def _to_contents(messages: list[Message]) -> list[gt.Content]:
    out: list[gt.Content] = []
    for m in messages:
        if m.role == "user":
            parts: list[gt.Part] = []
            if m.text:
                parts.append(gt.Part.from_text(text=m.text))
            for im in m.images:
                parts.append(gt.Part.from_bytes(data=im.data, mime_type=im.mime))
            out.append(gt.Content(role="user", parts=parts or [gt.Part.from_text(text="")]))
        elif m.role == "assistant":
            parts = []
            if m.text:
                parts.append(gt.Part.from_text(text=m.text))
            for c in m.tool_calls:
                parts.append(
                    gt.Part(
                        function_call=gt.FunctionCall(id=c.id or None, name=c.name, args=c.args),
                        thought_signature=c.signature or _SKIP_SIGNATURE,
                    )
                )
            if parts:
                out.append(gt.Content(role="model", parts=parts))
        else:  # tool results → a user turn of function responses (+ any images)
            parts = []
            for r in m.tool_results:
                parts.append(
                    gt.Part(
                        function_response=gt.FunctionResponse(
                            id=r.call_id or None, name=r.name, response={"result": r.content, "ok": r.ok}
                        )
                    )
                )
                for im in r.images:
                    parts.append(gt.Part.from_bytes(data=im.data, mime_type=im.mime))
            out.append(gt.Content(role="user", parts=parts))
    return out


def _tools(specs: list[ToolSpec] | None) -> list[gt.Tool] | None:
    if not specs:
        return None
    return [
        gt.Tool(
            function_declarations=[
                gt.FunctionDeclaration(
                    name=s.name, description=s.description, parameters_json_schema=s.parameters
                )
                for s in specs
            ]
        )
    ]


def _pretty(model: str) -> str:
    return model.replace("gemini-", "Gemini ").replace("-flash", " Flash").replace("-lite", " Lite")


async def _note(text: str) -> None:
    from neo.events import bus

    await bus().note(text)


def _code(e: Exception) -> int | None:
    return getattr(e, "code", None) if isinstance(e, gerrors.APIError) else None


def _map_error(e: Exception) -> ProviderError:
    if _code(e) == 429:
        return RateLimited(str(e))
    return ProviderError(str(e)[:300])


class GeminiProvider:
    name = "gemini"
    supports_vision = True
    supports_tools = True

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        s = settings()
        key = api_key or s.gemini_api_key
        if not key:
            raise ProviderError("GEMINI_API_KEY is not set")
        self.model = model or s.gemini_model
        lite = "lite" in self.model
        self.models = [self.model] + (
            [] if lite else [m for m in s.gemini_fallback_models if m != self.model]
        )
        keys = [key] if api_key else s.gemini_keys
        self._clients = [genai.Client(api_key=k) for k in keys]
        self._client = self._clients[0]
        self._exhausted: dict[tuple[str, int], float] = {}  # (model, key index) → retry-after timestamp

    async def _call(self, fn, **kw):
        """Walk model × key. 5xx → short backoff then next model; 429 → next key for this model
        (free-tier quota is per key *and* per model); other errors raise immediately."""
        last: Exception | None = None
        import time as _time

        for model in self.models:
            overloaded = False
            if self._exhausted.get((model, -1), 0.0) > _time.time():
                continue  # recently overloaded — skip the whole model for a bit
            for ki, client in enumerate(self._clients):
                if self._exhausted.get((model, ki), 0.0) > _time.time():
                    continue  # quota known to be spent — don't pay a round trip to learn it again
                for i, delay in enumerate((0.0, *_BACKOFF_S)):
                    if delay:
                        await asyncio.sleep(delay)
                    try:
                        r = await fn(client, model, **kw)
                        self.model, self._client = model, client
                        return r
                    except Exception as e:  # noqa: BLE001
                        last = e
                        code = _code(e)
                        if code in _RETRY_CODES and i < len(_BACKOFF_S):
                            continue
                        if code == 429:
                            self._exhausted[(model, ki)] = _time.time() + _QUOTA_COOLDOWN_S
                            print(f"[gemini] {model} key#{ki + 1} quota exhausted; trying next key")
                            await _note(f"{_pretty(model)} key {ki + 1}: no quota, trying next")
                            break
                        if code in _RETRY_CODES:
                            print(f"[gemini] {model} overloaded; trying next model")
                            self._exhausted[(model, -1)] = _time.time() + _OVERLOAD_COOLDOWN_S
                            await _note(f"{_pretty(model)} overloaded, trying next")
                            overloaded = True
                            break
                        raise _map_error(e) from e
                if overloaded:
                    break
        raise _map_error(last or ProviderError("gemini: no models"))

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        effort: Literal["low", "medium", "high"] = "medium",
        max_tokens: int = 4096,
    ) -> Turn:
        cfg = gt.GenerateContentConfig(
            system_instruction=system or None,
            tools=_tools(tools),
            thinking_config=gt.ThinkingConfig(thinking_level=_LEVEL[effort]),
            max_output_tokens=max_tokens,
        )
        resp = await self._call(
            lambda client, model, **kw: client.aio.models.generate_content(model=model, **kw),
            contents=_to_contents(messages),
            config=cfg,
        )

        turn = Turn()
        cand = (resp.candidates or [None])[0]
        for part in (cand.content.parts if cand and cand.content else []) or []:
            if part.function_call:
                fc = part.function_call
                turn.tool_calls.append(
                    ToolCall(
                        id=fc.id or uuid.uuid4().hex[:12],
                        name=fc.name or "",
                        args=dict(fc.args or {}),
                        signature=part.thought_signature or None,
                    )
                )
            elif part.text:
                if getattr(part, "thought", False):
                    turn.thinking += part.text
                else:
                    turn.text += part.text
        um = resp.usage_metadata
        if um:
            turn.usage = Usage(um.prompt_token_count or 0, um.candidates_token_count or 0)
        return turn

    async def stream(
        self, messages: list[Message], *, system: str = "", max_tokens: int = 2048
    ) -> AsyncIterator[str]:
        cfg = gt.GenerateContentConfig(
            system_instruction=system or None,
            thinking_config=gt.ThinkingConfig(thinking_level=gt.ThinkingLevel.LOW),
            max_output_tokens=max_tokens,
        )
        contents = _to_contents(messages)

        async def first_chunk(client, model: str, **kw):
            # The SDK issues the request lazily on first iteration, so the retry/model-walk has to
            # cover pulling the first chunk, not just creating the iterator.
            it = await client.aio.models.generate_content_stream(model=model, **kw)
            ait = it.__aiter__()
            try:
                head = await ait.__anext__()
            except StopAsyncIteration:
                head = None
            return head, ait

        head, ait = await self._call(first_chunk, contents=contents, config=cfg)
        try:
            if head is not None and head.text:
                yield head.text
            async for chunk in ait:
                if chunk.text:
                    yield chunk.text
        except Exception as e:  # noqa: BLE001
            raise _map_error(e) from e
