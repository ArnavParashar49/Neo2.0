"""Optional Claude brain (paid). Off unless ANTHROPIC_API_KEY is set and a role names Claude
(`brain=claude`, or a reference such as ``claude@claude-haiku-5-5``).

Uses adaptive thinking and `output_config.effort`, through the official anthropic SDK. What each
request carries depends on the model (:data:`PROFILES`): server-side refusal fallbacks
(`fallbacks="default"`), how NEO's three levels map to the API's effort, and the thinking-binding
controls. claude-opus-5 sends exactly what it always did. On the 5.5 models and Fable 5.1 a thinking
block replayed after NEO trimmed or rewrote an earlier turn (fit, the context diet) fails the API's
conversation check — on every account made since 2026-08-31 a 400 on every multi-step run — so they
ask the API to drop such a block instead (`prefix_mismatch_behavior: "drop_block"`), and the log
says how many it dropped (never what was in them). Haiku 5.5 has no server-side fallback: it's
never sent one. A model not in the table gets adaptive thinking and plain effort, nothing more
(the Brains & Keys picker says it may not work with NEO).

A refused key (401, 403) is KeyRefused — the chain benches the brain for an hour and the health
list says so; a 429 says how long to wait (its retry-after); a 400 or 404 about the model puts
"Claude rejected model X" in the health list.
"""

from __future__ import annotations

import base64
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import anthropic

from neo2.core import logger, runtime
from neo2.core.config import Settings, settings
from neo2.core.providers.base import (
    BadRequest,
    KeyRefused,
    Message,
    Overloaded,
    ProviderError,
    QuotaExhausted,
    RateLimited,
    RequestTooLarge,
    ToolCall,
    ToolSpec,
    Turn,
    Unavailable,
    Usage,
)
from neo2.core.providers.fit import fit
from neo2.core.providers.history import normalize

log = logger(__name__)

_TOKEN_BUDGET = 200_000  # context window
FALLBACK_BETA = "server-side-fallback-2026-07-01"  # (the "default" form of `fallbacks`)
BINDING_BETA = "thinking-binding-controls-2026-08-01"
_DROP_BLOCK = {"prefix_mismatch_behavior": "drop_block"}
_NO_CREDIT_S = 3600.0  # a 402 (no credit): asked again after this long
_LMH: Mapping[str, str] = MappingProxyType({"low": "low", "medium": "medium", "high": "high"})


@dataclass(frozen=True)
class Profile:
    """What a request to one model carries. `fallbacks`: fallbacks="default" (server-side refusal
    fallback). `effort`: NEO's level → the API's effort. `binding`: the thinking-binding controls
    (drop a replayed block that fails the conversation check, rather than a 400). `known`: the model
    is one NEO was built against."""

    fallbacks: bool
    effort: Mapping[str, str]
    binding: bool
    known: bool = True


PROFILES: Mapping[str, Profile] = MappingProxyType(
    {
        # exactly the request NEO always sent: its effort "high" is the API's xhigh
        "claude-opus-5": Profile(True, MappingProxyType({"low": "low", "medium": "medium", "high": "xhigh"}), False),
        "claude-opus-5-5": Profile(True, _LMH, True),
        "claude-sonnet-5-5": Profile(True, _LMH, True),  # ("default" fallbacks: the Claude API's form)
        "claude-fable-5-1": Profile(True, _LMH, True),
        "claude-haiku-5-5": Profile(False, _LMH, True),  # no server-side fallback for Haiku 5.5
    }
)
UNKNOWN = Profile(False, _LMH, False, known=False)


def profile(model: str) -> Profile:
    """The request profile for `model` (an exact id; any other gets :data:`UNKNOWN`)."""
    return PROFILES.get(model or "", UNKNOWN)


def _img(im) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": im.mime, "data": base64.b64encode(im.data).decode()},
    }


def _to_messages(messages: list[Message]) -> list[dict[str, Any]]:
    # Opens with a user turn, every tool_use answered at once (neo2.core.providers.history).
    out: list[dict[str, Any]] = []
    for m in normalize(messages):
        if m.role == "user":
            content: list[dict[str, Any]] = [_img(i) for i in m.images]
            if m.text or not content:
                content.append({"type": "text", "text": m.text or "(empty)"})
            out.append({"role": "user", "content": content})
        elif m.role == "assistant":
            # Claude's own blocks (incl. thinking + signatures), verbatim. Another brain's replay
            # payload (OpenAI's output items) is never Claude's: its text and calls go as below.
            if isinstance(m.raw, list):
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


def _retry_after(e: Exception) -> float | None:
    """Seconds a 429 (or 529) asks us to wait: its retry-after header."""
    resp = getattr(e, "response", None)
    try:
        h = resp.headers.get("retry-after") if resp is not None else None
        return max(0.0, float(h)) if h else None
    except (TypeError, ValueError, AttributeError):
        return None


def _message(e: Exception) -> str:
    """The API's own error message (its body's error.message), else the exception's."""
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return err["message"]
    return str(getattr(e, "message", "") or e)


def model_rejected(e: Exception, model: str) -> bool:
    """A 400 or 404 about the model itself: a model id the API doesn't know, or this key may not use."""
    code = getattr(e, "status_code", 0)
    low = _message(e).lower()
    if code == 404:
        return "model" in low
    if code == 400:
        return low.startswith("model:") or (bool(model) and f"model: {model.lower()}" in low)
    return False


def _error(e: Exception, name: str = "claude") -> ProviderError:
    if isinstance(e, anthropic.RateLimitError):
        return RateLimited(f"{name}: {e}", retry_after=_retry_after(e))
    if isinstance(e, anthropic.APIStatusError):
        code = e.status_code
        if code in (401, 403):  # fixed words: never the API's text, which may quote the key
            return KeyRefused(f"{name}: Anthropic refused the API key (HTTP {code})")
        if code == 402:
            return QuotaExhausted(f"{name}: the Anthropic account has no credit (HTTP 402)", until=time.time() + _NO_CREDIT_S)
        low = str(e).lower()
        if code == 413 or "prompt is too long" in low or "too many tokens" in low:
            return RequestTooLarge(f"{name}: {e}")
        if code == 529 or code >= 500:
            return Overloaded(f"{name}: {e}", retry_after=_retry_after(e))
        if 400 <= code < 500:
            return BadRequest(f"{name}: {e}")
        return ProviderError(f"{name}: {e}")
    if isinstance(e, anthropic.APIConnectionError):
        return Unavailable(f"{name}: connection error: {e}")
    return ProviderError(f"{name}: {e}")


def _kept(blocks: list[Any]) -> list[int]:
    """The reply's blocks to keep (indexes). After a server-side fallback (a `fallback` block marks
    each switch) the thinking and tool calls of the model that declined, before the last switch,
    are not the reply's; the marker itself is only an audit note."""
    types = [getattr(b, "type", "") for b in blocks]
    last = max((i for i, t in enumerate(types) if t == "fallback"), default=-1)
    if last < 0:
        return list(range(len(blocks)))
    drop = {"thinking", "redacted_thinking", "tool_use", "server_tool_use"}
    return [i for i, t in enumerate(types) if t != "fallback" and not (i < last and t in drop)]


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
        self._own = own  # a host's: NEO.app's health list and spend caps are never touched by it
        if own:
            self._cap = None
        else:
            from neo2.core.providers import spend

            self._cap = spend.cap_for("anthropic", settings())

    @property
    def profile(self) -> Profile:
        return profile(self.model)

    # ---- what each request carries -----------------------------------------------------------
    def _extras(self, *, thinking: bool) -> dict[str, Any]:
        """The model's own part of a request: its thinking config, betas and fallbacks."""
        pr = self.profile
        out: dict[str, Any] = {}
        if pr.binding:
            out["thinking"] = {"type": "adaptive", "block_binding": dict(_DROP_BLOCK)}
        elif thinking:
            out["thinking"] = {"type": "adaptive"}
        betas = ([FALLBACK_BETA] if pr.fallbacks else []) + ([BINDING_BETA] if pr.binding else [])
        if betas:
            out["betas"] = betas
        if pr.fallbacks:
            out["fallbacks"] = "default"
        return out

    def _dropped(self, msg: Any) -> None:
        """How many replayed thinking blocks the API dropped (the binding controls): a count only."""
        def field(x: Any, k: str) -> Any:
            return x.get(k) if isinstance(x, dict) else getattr(x, k, None)

        try:
            dropped = [x for x in getattr(msg, "input_transformations", None) or [] if field(x, "type") == "thinking_dropped"]
        except TypeError:
            return
        if dropped:
            trimmed = sum(1 for x in dropped if field(x, "reason") == "prefix_binding_mismatch")
            log.info(
                f"[brain] {self.name}: the API dropped {len(dropped)} earlier thinking block(s) "
                f"({trimmed} after NEO trimmed the history, {len(dropped) - trimmed} for another reason)"
            )

    def _rejected(self, e: Exception) -> None:
        """A 400 or 404 about the model: one health problem (NEO.app's own brains only)."""
        if self._own or not runtime.process_wide_ok() or not model_rejected(e, self.model):
            return
        from neo2.core import brainref, health

        shown = brainref.shown(self.model) or "it was set to"
        health.report(
            f"model:anthropic:{self.model}",
            f"Claude rejected model {shown} — pick a current one in Brains & Keys.",
        )

    def _works(self) -> None:
        if not self._own and runtime.process_wide_ok():
            from neo2.core import health

            health.report(f"model:anthropic:{self.model}", None)

    async def _admit(self) -> None:
        if self._cap is not None and (refused := self._cap.refusal(self.name)) is not None:
            await self._cap.tell()
            raise refused

    async def _count(self, usage: Any) -> None:
        if self._cap is None or usage is None:
            return
        self._cap.add(self.model, int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0))
        await self._cap.tell()

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        effort: Literal["low", "medium", "high"] = "medium",
        max_tokens: int = 4096,
    ) -> Turn:
        await self._admit()
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
        pr = self.profile
        try:
            resp = await self._client.beta.messages.create(
                model=self.model,
                max_tokens=f.max_tokens,
                system=system or anthropic.NOT_GIVEN,
                messages=_to_messages(f.messages),
                tools=_tools(f.tools) or anthropic.NOT_GIVEN,
                output_config={"effort": pr.effort.get(effort, "medium")},
                **self._extras(thinking=True),
            )
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            self._rejected(e)
            raise _error(e, self.name) from e

        if pr.binding:
            self._dropped(resp)
        self._works()
        await self._count(getattr(resp, "usage", None))
        if resp.stop_reason == "refusal":
            return Turn(text="I can't help with that one.")
        keep = _kept(resp.content)
        turn = Turn(raw=[resp.content[i].model_dump(exclude_none=True) for i in keep])
        for b in (resp.content[i] for i in keep):
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
        await self._admit()
        f = fit(
            messages,
            system=system,
            budget=self.token_budget,
            max_tokens=max(max_tokens, 16000),
            vision=True,
            name=self.name,
        )
        pr = self.profile
        try:
            async with self._client.beta.messages.stream(
                model=self.model,
                max_tokens=f.max_tokens,
                system=system or anthropic.NOT_GIVEN,
                messages=_to_messages(f.messages),
                output_config={"effort": pr.effort.get("low", "low")},
                **self._extras(thinking=False),
            ) as st:
                async for text in st.text_stream:
                    yield text
                final = None
                if pr.binding or self._cap is not None:
                    try:
                        final = st.current_message_snapshot
                    except Exception:  # noqa: BLE001 — no snapshot to read: nothing to log or count
                        final = None
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            self._rejected(e)
            raise _error(e, self.name) from e
        if final is not None:
            if pr.binding:
                self._dropped(final)
            await self._count(getattr(final, "usage", None))
        self._works()
