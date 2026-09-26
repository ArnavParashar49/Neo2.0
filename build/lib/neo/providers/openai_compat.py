"""OpenAI-compatible brains: Groq (fast free tier) and the local mlx-lm server.

One class, parameterised by ``base_url`` / ``model``. Both speak the chat-completions
protocol with tool calling; Groq also honours ``reasoning_effort`` for gpt-oss models.
"""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

from openai import APIStatusError, AsyncOpenAI, RateLimitError

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


def _to_messages(messages: list[Message], system: str, vision: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        if m.role == "user":
            if m.images and vision:
                content: list[dict[str, Any]] = [{"type": "text", "text": m.text}]
                for im in m.images:
                    b64 = base64.b64encode(im.data).decode()
                    content.append(
                        {"type": "image_url", "image_url": {"url": f"data:{im.mime};base64,{b64}"}}
                    )
                out.append({"role": "user", "content": content})
            else:
                text = m.text + ("\n[image omitted: this brain has no vision]" if m.images else "")
                out.append({"role": "user", "content": text})
        elif m.role == "assistant":
            if not m.text and not m.tool_calls:
                continue  # content:null with no tool_calls is rejected by OpenAI-compatible APIs
            msg: dict[str, Any] = {"role": "assistant", "content": m.text or None}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.args)},
                    }
                    for c in m.tool_calls
                ]
            out.append(msg)
        else:
            for r in m.tool_results:
                text = r.content + ("\n[screenshot attached]" if r.images and not vision else "")
                out.append({"role": "tool", "tool_call_id": r.call_id, "content": text})
                if r.images and vision:
                    content = [{"type": "text", "text": f"Image from {r.name}:"}]
                    for im in r.images:
                        b64 = base64.b64encode(im.data).decode()
                        content.append(
                            {"type": "image_url", "image_url": {"url": f"data:{im.mime};base64,{b64}"}}
                        )
                    out.append({"role": "user", "content": content})
    return out


def _tools(specs: list[ToolSpec] | None) -> list[dict[str, Any]] | None:
    if not specs:
        return None
    return [
        {
            "type": "function",
            "function": {"name": s.name, "description": s.description, "parameters": s.parameters},
        }
        for s in specs
    ]


class OpenAICompatProvider:
    supports_tools = True

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        supports_vision: bool = False,
        reasoning_effort: bool = False,
    ) -> None:
        self.name = name
        self.model = model
        self.supports_vision = supports_vision
        self._reasoning = reasoning_effort
        self._base_url = base_url
        self._is_local = "127.0.0.1" in base_url or "localhost" in base_url
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "none",
            timeout=180.0 if self._is_local else 60.0,
            max_retries=0 if self._is_local else 1,
        )

    async def _preflight(self) -> None:
        """Fail fast when the local server isn't up instead of hanging on connect/retry."""
        if self._is_local:
            from neo.providers.local_server import is_up

            port = int(self._base_url.rsplit(":", 1)[1].split("/")[0])
            if not await is_up(port):
                raise ProviderError("local model server not running (start it with `python -m neo --local`)")

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        effort: Literal["low", "medium", "high"] = "medium",
        max_tokens: int = 4096,
    ) -> Turn:
        kwargs: dict[str, Any] = {}
        if self._reasoning:
            kwargs["reasoning_effort"] = effort
        t = _tools(tools)
        if t:
            kwargs["tools"] = t
        await self._preflight()
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=_to_messages(messages, system, self.supports_vision),
                max_tokens=max_tokens,
                **kwargs,
            )
        except RateLimitError as e:
            raise RateLimited(str(e)) from e
        except APIStatusError as e:
            raise ProviderError(f"{self.name}: {e}") from e
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"{self.name}: {e}") from e

        choice = resp.choices[0].message
        turn = Turn(text=choice.content or "")
        for tc in choice.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            turn.tool_calls.append(
                ToolCall(id=tc.id or uuid.uuid4().hex[:12], name=tc.function.name, args=args)
            )
        if resp.usage:
            turn.usage = Usage(resp.usage.prompt_tokens or 0, resp.usage.completion_tokens or 0)
        return turn

    async def stream(
        self, messages: list[Message], *, system: str = "", max_tokens: int = 2048
    ) -> AsyncIterator[str]:
        await self._preflight()
        try:
            st = await self._client.chat.completions.create(
                model=self.model,
                messages=_to_messages(messages, system, self.supports_vision),
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in st:
                d = chunk.choices[0].delta if chunk.choices else None
                if d and d.content:
                    yield d.content
        except RateLimitError as e:
            raise RateLimited(str(e)) from e
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"{self.name}: {e}") from e


def groq(model: str | None = None) -> OpenAICompatProvider:
    s = settings()
    if not s.groq_api_key:
        raise ProviderError("GROQ_API_KEY is not set")
    return OpenAICompatProvider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key=s.groq_api_key,
        model=model or s.groq_model,
        reasoning_effort=True,
    )


def local(model: str | None = None, port: int = 8080) -> OpenAICompatProvider:
    """Local mlx-lm server (see neo.providers.local_server to launch it)."""
    return OpenAICompatProvider(
        name="local",
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key="local",
        model=model or settings().local_model,
    )
