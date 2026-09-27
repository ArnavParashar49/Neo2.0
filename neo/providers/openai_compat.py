"""OpenAI-compatible brains: Groq (fast free tier) and the local mlx-lm server.

One class, parameterised by ``base_url`` / ``model``. Both speak the chat-completions
protocol with tool calling; Groq also honours ``reasoning_effort`` for gpt-oss models.

Both have little room per request (Groq's free tier: ~8K tokens a minute, prompt and answer
together; the local server: its context), so every request is fitted to ``token_budget`` first
(neo.providers.fit) — Groq's as the per-minute rate it is, which every step pays back before the
next one goes. When the server still says it's too large, it also says how large it
really was: the estimate is corrected from that and the request fitted again. The correction is
capped, and every request that goes through reports its real size, so one odd refusal (a prompt of
dense digits, a 413 with no numbers) doesn't leave every later request over-trimmed.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError

from neo.config import settings
from neo.providers.base import (
    BLIND_PICTURE,
    BadRequest,
    Message,
    ProviderError,
    RateLimited,
    RequestTooLarge,
    ToolCall,
    ToolSpec,
    Turn,
    Unavailable,
    Usage,
)
from neo.providers.fit import Fitted, fit
from neo.providers.history import normalize

# Tokens one request may carry (prompt + max answer) on Groq's free tier, per model: its
# tokens-per-minute limit. Groq's 413 names the real number, which then replaces this.
_GROQ_REQUEST_TOKENS = {
    "openai/gpt-oss-120b": 8000,
    "openai/gpt-oss-20b": 8000,
    "llama-3.3-70b-versatile": 12000,
}
_GROQ_DEFAULT_TOKENS = 6000
_LOCAL_CONTEXT_TOKENS = 32768
_MAX_SCALE = 2.5  # the estimate is never believed to be further off than this
_FOLLOW = 0.5  # a request that went through moves the correction this far toward its real size
_DRIFT = 0.1  # …or, when the server didn't say, this far back toward none
_LIMIT_REQUESTED = re.compile(r"Limit\s+(\d+),\s*Requested\s+(\d+)", re.I)
_TRY_AGAIN = re.compile(r"try again in\s+(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?", re.I)

log = logging.getLogger(__name__)


def _to_messages(messages: list[Message], system: str, vision: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in normalize(messages):
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
                text = r.content + ("\n" + BLIND_PICTURE if r.images and not vision else "")
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


def _retry_after(e: Exception) -> float | None:
    """Seconds a 429 asks us to wait: the Retry-After header, or Groq's "try again in 1m2.5s"."""
    resp = getattr(e, "response", None)
    try:
        h = resp.headers.get("retry-after") if resp is not None else None
        if h:
            return float(h)
    except (TypeError, ValueError, AttributeError):
        pass
    m = _TRY_AGAIN.search(str(e))
    if not m or not any(m.groups()):
        return None
    h, mi, sec = m.groups()
    return int(h or 0) * 3600 + int(mi or 0) * 60 + float(sec or 0)


def _too_large(e: APIStatusError) -> bool:
    text = str(e).lower()
    return e.status_code == 413 or "context_length_exceeded" in text or "reduce the length" in text


class Truncated(BadRequest):
    """The answer ran out of room before it began: a reasoning model spent all of `max_tokens`
    thinking (finish_reason "length", no text and no tool call). Request-shaped, like any 400: the
    brain is fine, and asking again with more room or less thinking usually works."""


def _ran_out(choice: Any) -> bool:
    """A reply cut off by max_tokens with nothing to show for it (reasoning spent it all)."""
    msg = getattr(choice, "message", None)
    return getattr(choice, "finish_reason", None) == "length" and not (
        getattr(msg, "content", None) or getattr(msg, "tool_calls", None)
    )


class OpenAICompatProvider:
    supports_tools = True
    # A reply whose reasoning used up all of max_tokens (no text, no call) raises Truncated instead
    # of coming back empty: a subclass that can retry with more room (neo.providers.cerebras) says so.
    raise_truncated = False
    # reasoning_effort for streamed chat answers (None: the model's default), for brains with one
    stream_effort: str | None = None

    def __init__(
        self,
        *,
        name: str,
        model: str,
        base_url: str = "",
        api_key: str = "",
        supports_vision: bool = False,
        reasoning_effort: bool = False,
        token_budget: int = _LOCAL_CONTEXT_TOKENS,
        per_minute: bool = False,
        client: AsyncOpenAI | None = None,
    ) -> None:
        """`client`: the host's own configured AsyncOpenAI (its base URL, key, retries…)."""
        self.name = name
        self.model = model
        self.supports_vision = supports_vision
        self.token_budget = token_budget
        self.per_minute = per_minute  # the budget is tokens per minute, not a context window
        self._scale = 1.0  # real tokens / estimated, learned from the server's own 413
        self._cool_until = 0.0  # a 429 that said how long to wait
        self._reasoning = reasoning_effort
        self._base_url = base_url or str(getattr(client, "base_url", "") or "")
        # NEO's own local server (mlx-lm): checked before each call. A host's client is theirs.
        self._is_local = client is None and ("127.0.0.1" in base_url or "localhost" in base_url)
        # nothing sent to it leaves the Mac (a host's own client may point at its local server too)
        self.on_device = "127.0.0.1" in self._base_url or "localhost" in self._base_url
        self._client = client or AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "none",
            timeout=180.0 if self._is_local else 60.0,
            max_retries=0 if self._is_local else 1,
        )

    def cooling_until(self) -> float:
        """When this brain can be asked again (0: now)."""
        return self._cool_until if self._cool_until > time.time() else 0.0

    def client(self) -> AsyncOpenAI:
        """The client requests go through (a subclass may hand out one per event loop)."""
        return self._client

    def _spent(self, usage: Any, f: Fitted, text: str = "") -> None:
        """Every answer's token usage as the server reported it (None: it didn't — `text` is what
        streamed). Nothing here: a paid brain counts it (neo.providers.cerebras)."""

    async def _preflight(self) -> None:
        """Fail fast when the local server isn't up instead of hanging on connect/retry."""
        if self._is_local:
            from neo.providers.local_server import is_up

            port = int(self._base_url.rsplit(":", 1)[1].split("/")[0])
            if not await is_up(port):
                raise Unavailable("local model server not running (start it with `python -m neo --local`)")

    def _fit(self, messages, system, tools, max_tokens) -> Fitted:
        return fit(
            messages,
            system=system,
            tools=tools,
            budget=self.token_budget,
            max_tokens=max_tokens,
            vision=self.supports_vision,
            name=self.name,
            scale=self._scale,
            per_minute=self.per_minute,
        )

    def _learn_size(self, e: APIStatusError, f: Fitted) -> None:
        """The server refused a request we thought fit: believe its numbers from now on."""
        m = _LIMIT_REQUESTED.search(str(e))
        if m:
            limit, requested = int(m.group(1)), int(m.group(2))
            self.token_budget = min(self.token_budget, limit)
            real_prompt = max(1, requested - f.max_tokens)
            self._scale = max(self._scale, self._scale * real_prompt / max(1, f.tokens) * 1.05)
        else:
            self._scale *= 1.4
        self._scale = min(self._scale, _MAX_SCALE)
        log.warning(
            f"[brain] {self.name}: request too large; now budget {self.token_budget}, estimate ×{self._scale:.2f}"
        )

    def _learn_real(self, f: Fitted, usage: Any) -> None:
        """A request that went through: its real prompt size (when the server says) is what the
        estimate is corrected toward, down as well as up; otherwise the correction fades."""
        real = getattr(usage, "prompt_tokens", None) if usage is not None else None
        raw = f.tokens / self._scale  # the uncorrected estimate of what was sent
        if isinstance(real, int) and real > 0 and raw >= 1:
            target, rate = min(_MAX_SCALE, max(1.0, real / raw * 1.05)), _FOLLOW
        else:
            target, rate = 1.0, _DRIFT
        self._scale += (target - self._scale) * rate

    def _error(self, e: Exception) -> ProviderError:
        if isinstance(e, RateLimitError):
            wait = _retry_after(e)
            if wait:
                self._cool_until = time.time() + wait
            return RateLimited(f"{self.name}: {e}", retry_after=wait)
        if isinstance(e, APIStatusError):
            if _too_large(e):
                return RequestTooLarge(f"{self.name}: {e}")
            if 400 <= e.status_code < 500:
                return BadRequest(f"{self.name}: {e}")
            return ProviderError(f"{self.name}: {e}")
        if isinstance(e, APIConnectionError):
            return Unavailable(f"{self.name}: {e}")
        return ProviderError(f"{self.name}: {e}")

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        effort: Literal["low", "medium", "high"] = "medium",
        max_tokens: int = 4096,
    ) -> Turn:
        await self._preflight()
        f = self._fit(messages, system, tools, max_tokens)
        tries = 3  # a 413 teaches the real size; a second one (the estimate still low) teaches more
        for attempt in range(tries):
            kwargs: dict[str, Any] = {}
            if self._reasoning:
                # Reasoning spends the answer's tokens: with the room cut to fit, think briefly.
                kwargs["reasoning_effort"] = effort if f.max_tokens >= max_tokens else "low"
            t = _tools(f.tools)
            if t:
                kwargs["tools"] = t
            try:
                resp = await self.client().chat.completions.create(
                    model=self.model,
                    messages=_to_messages(f.messages, system, self.supports_vision),
                    max_tokens=f.max_tokens,
                    **kwargs,
                )
                break
            except APIStatusError as e:
                if attempt < tries - 1 and not isinstance(e, RateLimitError) and _too_large(e):
                    self._learn_size(e, f)
                    f = self._fit(messages, system, tools, max_tokens)  # RequestTooLarge if it can't
                    continue
                raise self._error(e) from e
            except Exception as e:  # noqa: BLE001
                raise self._error(e) from e

        self._learn_real(f, resp.usage)
        self._spent(resp.usage, f)
        if self.raise_truncated and _ran_out(resp.choices[0]):
            raise Truncated(f"{self.name}: no room for the answer (reasoning used all {f.max_tokens} tokens)")
        choice = resp.choices[0].message
        turn = Turn(text=choice.content or "")
        for tc in choice.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            turn.tool_calls.append(
                ToolCall(
                    id=tc.id or uuid.uuid4().hex[:12], name=tc.function.name, args=args, source=self.model
                )
            )
        if resp.usage:
            turn.usage = Usage(resp.usage.prompt_tokens or 0, resp.usage.completion_tokens or 0)
        turn.offered = f.offered()
        turn.sees = bool(self.supports_vision)
        return turn

    async def stream(
        self, messages: list[Message], *, system: str = "", max_tokens: int = 2048
    ) -> AsyncIterator[str]:
        await self._preflight()
        f = self._fit(messages, system, None, max_tokens)
        said: list[str] = []
        usage: Any = None
        try:
            extra: dict[str, Any] = {}
            if self._reasoning and self.stream_effort:
                extra["reasoning_effort"] = self.stream_effort
            st = await self.client().chat.completions.create(
                model=self.model,
                messages=_to_messages(f.messages, system, self.supports_vision),
                max_tokens=f.max_tokens,
                stream=True,
                **extra,
            )
            async for chunk in st:
                usage = getattr(chunk, "usage", None) or usage  # (a server that reports it: the last chunk)
                d = chunk.choices[0].delta if chunk.choices else None
                if d and d.content:
                    said.append(d.content)
                    yield d.content
        except Exception as e:  # noqa: BLE001
            raise self._error(e) from e
        finally:
            if said or usage is not None:
                self._spent(usage, f, "".join(said))


def groq(model: str | None = None) -> OpenAICompatProvider:
    s = settings()
    if not s.groq_api_key:
        raise Unavailable("GROQ_API_KEY is not set")
    model = model or s.groq_model
    return OpenAICompatProvider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key=s.groq_api_key,
        model=model,
        reasoning_effort=True,
        token_budget=_GROQ_REQUEST_TOKENS.get(model, _GROQ_DEFAULT_TOKENS),
        per_minute=True,
    )


def local(model: str | None = None, port: int = 8080) -> OpenAICompatProvider:
    """Local mlx-lm server (see neo.providers.local_server to launch it)."""
    return OpenAICompatProvider(
        name="local",
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key="local",
        model=model or settings().local_model,
        token_budget=_LOCAL_CONTEXT_TOKENS,
    )
