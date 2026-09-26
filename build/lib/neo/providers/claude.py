"""Optional Claude brain (paid). Off unless ANTHROPIC_API_KEY is set and `brain=claude`.

Uses adaptive thinking, `output_config.effort`, and server-side refusal fallbacks
(`fallbacks="default"`) so a policy decline re-runs on a fallback model in the same call.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from typing import Any, Literal

import anthropic

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

_EFFORT = {"low": "low", "medium": "medium", "high": "xhigh"}


def _img(im) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": im.mime, "data": base64.b64encode(im.data).decode()},
    }


def _to_messages(messages: list[Message]) -> list[dict[str, Any]]:
    # Anthropic requires the transcript to open with a user turn: drop any orphaned prefix.
    start = next((i for i, m in enumerate(messages) if m.role == "user"), len(messages))
    out: list[dict[str, Any]] = []
    for m in messages[start:]:
        if m.role == "user":
            content: list[dict[str, Any]] = [_img(i) for i in m.images]
            if m.text or not content:
                content.append({"type": "text", "text": m.text or "(empty)"})
            out.append({"role": "user", "content": content})
        elif m.role == "assistant":
            if m.raw is not None:  # replay Claude's own blocks (incl. thinking + signatures) verbatim
                out.append({"role": "assistant", "content": m.raw})
                continue
            content = []
            if m.text:
                content.append({"type": "text", "text": m.text})
            for c in m.tool_calls:
                content.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.args})
            if content:
                out.append({"role": "assistant", "content": content})
        else:
            content = []
            for r in m.tool_results:
                rc: list[dict[str, Any]] = [_img(i) for i in r.images]
                if r.content or not rc:
                    rc.insert(0, {"type": "text", "text": r.content or "(no output)"})
                content.append(
                    {"type": "tool_result", "tool_use_id": r.call_id, "content": rc, "is_error": not r.ok}
                )
            out.append({"role": "user", "content": content})
    return out


def _tools(specs: list[ToolSpec] | None) -> list[dict[str, Any]]:
    return [{"name": s.name, "description": s.description, "input_schema": s.parameters} for s in specs or []]


class ClaudeProvider:
    name = "claude"
    supports_vision = True
    supports_tools = True

    def __init__(self, model: str | None = None) -> None:
        s = settings()
        if not s.anthropic_api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        self.model = model or s.claude_model
        self._client = anthropic.AsyncAnthropic(api_key=s.anthropic_api_key)

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        effort: Literal["low", "medium", "high"] = "medium",
        max_tokens: int = 4096,
    ) -> Turn:
        try:
            resp = await self._client.beta.messages.create(
                model=self.model,
                max_tokens=max(max_tokens, 16000),
                system=system or anthropic.NOT_GIVEN,
                messages=_to_messages(messages),
                tools=_tools(tools) or anthropic.NOT_GIVEN,
                thinking={"type": "adaptive"},
                output_config={"effort": _EFFORT[effort]},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.RateLimitError as e:
            raise RateLimited(str(e)) from e
        except anthropic.APIStatusError as e:
            raise ProviderError(f"claude: {e}") from e
        except anthropic.APIConnectionError as e:
            raise ProviderError(f"claude: connection error: {e}") from e

        if resp.stop_reason == "refusal":
            return Turn(text="I can't help with that one.")
        turn = Turn(raw=[b.model_dump(exclude_none=True) for b in resp.content])
        for b in resp.content:
            if b.type == "text":
                turn.text += b.text
            elif b.type == "tool_use":
                turn.tool_calls.append(ToolCall(id=b.id, name=b.name, args=dict(b.input or {})))
            elif b.type == "thinking" and getattr(b, "thinking", ""):
                turn.thinking += b.thinking
        turn.usage = Usage(resp.usage.input_tokens, resp.usage.output_tokens)
        return turn

    async def stream(
        self, messages: list[Message], *, system: str = "", max_tokens: int = 2048
    ) -> AsyncIterator[str]:
        try:
            async with self._client.beta.messages.stream(
                model=self.model,
                max_tokens=max(max_tokens, 16000),
                system=system or anthropic.NOT_GIVEN,
                messages=_to_messages(messages),
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as st:
                async for text in st.text_stream:
                    yield text
        except anthropic.RateLimitError as e:
            raise RateLimited(str(e)) from e
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            raise ProviderError(f"claude: {e}") from e
