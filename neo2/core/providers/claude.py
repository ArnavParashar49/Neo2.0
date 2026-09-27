"""Optional Claude brain (paid). Off unless ANTHROPIC_API_KEY is set and `brain=claude`.

Uses adaptive thinking, `output_config.effort`, and server-side refusal fallbacks
(`fallbacks="default"`) so a policy decline re-runs on a fallback model in the same call.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from typing import Any, Literal

import anthropic

from neo.config import Settings, settings
from neo.providers.base import (
    BadRequest,
    Message,
    Overloaded,
    ProviderError,
    RateLimited,
    RequestTooLarge,
    ToolCall,
    ToolSpec,
    Turn,
    Unavailable,
    Usage,
)
from neo.providers.fit import fit
from neo.providers.history import normalize

_EFFORT = {"low": "low", "medium": "medium", "high": "xhigh"}
_TOKEN_BUDGET = 200_000  # context window


def _img(im) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": im.mime, "data": base64.b64encode(im.data).decode()},
    }


def _to_messages(messages: list[Message]) -> list[dict[str, Any]]:
    # Opens with a user turn, every tool_use answered at once (neo.providers.history).
    out: list[dict[str, Any]] = []
    for m in normalize(messages):
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


def _error(e: Exception) -> ProviderError:
    if isinstance(e, anthropic.RateLimitError):
        return RateLimited(str(e))
    if isinstance(e, anthropic.APIStatusError):
        low = str(e).lower()
        if e.status_code == 413 or "prompt is too long" in low or "too many tokens" in low:
            return RequestTooLarge(f"claude: {e}")
        if e.status_code == 529 or e.status_code >= 500:
            return Overloaded(f"claude: {e}")
        if 400 <= e.status_code < 500:
            return BadRequest(f"claude: {e}")
        return ProviderError(f"claude: {e}")
    if isinstance(e, anthropic.APIConnectionError):
        return Unavailable(f"claude: connection error: {e}")
    return ProviderError(f"claude: {e}")


class ClaudeProvider:
    name = "claude"
    supports_vision = True
    supports_tools = True
    token_budget = _TOKEN_BUDGET

    def __init__(
        self, model: str | None = None, *, api_key: str | None = None, client: anthropic.AsyncAnthropic | None = None
    ) -> None:
        """Given a key or a client, it's the host's: nothing is read from settings, and the model
        defaults to the release's (a host may build it before any agent exists — NEO.app's
        settings would read ~/.neo). Given neither, the key and model come from settings."""
        key = api_key
        own = client is not None or key is not None
        if not own:
            key = settings().anthropic_api_key
        if client is None and not key:
            raise Unavailable("ANTHROPIC_API_KEY is not set")
        self.model = model or (Settings.model_fields["claude_model"].default if own else settings().claude_model)
        self._client = client or anthropic.AsyncAnthropic(api_key=key)

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        effort: Literal["low", "medium", "high"] = "medium",
        max_tokens: int = 4096,
    ) -> Turn:
        out = max(max_tokens, 16000)
        f = fit(
            messages,
            system=system,
            tools=tools,
            budget=self.token_budget,
            max_tokens=out,
            vision=True,
            name=self.name,
        )
        try:
            resp = await self._client.beta.messages.create(
                model=self.model,
                max_tokens=f.max_tokens,
                system=system or anthropic.NOT_GIVEN,
                messages=_to_messages(f.messages),
                tools=_tools(f.tools) or anthropic.NOT_GIVEN,
                thinking={"type": "adaptive"},
                output_config={"effort": _EFFORT[effort]},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            raise _error(e) from e

        if resp.stop_reason == "refusal":
            return Turn(text="I can't help with that one.")
        turn = Turn(raw=[b.model_dump(exclude_none=True) for b in resp.content])
        for b in resp.content:
            if b.type == "text":
                turn.text += b.text
            elif b.type == "tool_use":
                turn.tool_calls.append(
                    ToolCall(id=b.id, name=b.name, args=dict(b.input or {}), source=self.model)
                )
            elif b.type == "thinking" and getattr(b, "thinking", ""):
                turn.thinking += b.thinking
        turn.usage = Usage(resp.usage.input_tokens, resp.usage.output_tokens)
        turn.offered = f.offered()
        return turn

    async def stream(
        self, messages: list[Message], *, system: str = "", max_tokens: int = 2048
    ) -> AsyncIterator[str]:
        f = fit(
            messages,
            system=system,
            budget=self.token_budget,
            max_tokens=max(max_tokens, 16000),
            vision=True,
            name=self.name,
        )
        try:
            async with self._client.beta.messages.stream(
                model=self.model,
                max_tokens=f.max_tokens,
                system=system or anthropic.NOT_GIVEN,
                messages=_to_messages(f.messages),
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as st:
                async for text in st.text_stream:
                    yield text
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            raise _error(e) from e
