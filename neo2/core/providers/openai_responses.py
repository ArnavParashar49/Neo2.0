"""OpenAI's own models (paid), on the Responses API: brain ``openai`` (``openai@gpt-6-luna``…).

Not the chat-completions class (neo2.core.providers.openai_compat): OpenAI's docs say GPT-6 Astra
and GPT-6.1 Sol call tools only on the Responses API, and Chat Completions allows tools on Luna only
with reasoning off. Only where the user gave OpenAI a role: a key alone sends nothing (R5).

* The request: ``instructions`` (the system prompt), ``input`` (the conversation as items),
  function tools, ``reasoning.effort``, ``max_output_tokens``, ``store=False`` with the reasoning
  kept between steps as encrypted items (``include=["reasoning.encrypted_content"]``), and
  ``parallel_tool_calls``.
* The conversation: a user turn is input_text and input_image (data: URLs); a reply of this model is
  replayed verbatim (its own output items, the reasoning items too — Turn.raw), another model's as
  its text and function calls; a tool's result is a function_call_output whose output holds its
  pictures (input_image) next to its text.
* A model that refuses the reasoning parameter (a 400 that names it) is asked again once without it,
  and never sent it again.
* Errors: a refused key (401, a 403 about the key) is KeyRefused; out of quota (insufficient_quota)
  is QuotaExhausted for an hour; another 429 is RateLimited with its retry-after; a request too long
  is RequestTooLarge; a 5xx is Overloaded; no connection is Unavailable. The words are NEO's own,
  never an error's text that may quote the key.
"""

from __future__ import annotations

import base64
import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    RateLimitError,
)

from neo2.core import logger, runtime
from neo2.core.providers import ImageLimit
from neo2.core.providers.base import (
    BLIND_PICTURE,
    BadRequest,
    ImagePart,
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
from neo2.core.providers.openai_compat import Truncated

log = logger(__name__)

PROVIDER = "openai"
_TOKEN_BUDGET = 200_000
_MIN_OUTPUT = 16_000  # reasoning is spent from max_output_tokens before the answer
_QUOTA_S = 3600.0  # insufficient_quota / no credit: asked again after this long
_TIMEOUT_S = 180.0
_MB = 1024 * 1024
# A conservative picture limit: OpenAI fits a picture into 2048 px before it looks (verify against
# the live API; a larger one is only shrunk, never refused).
IMAGE_LIMIT = ImageLimit(2048, None, 20 * _MB)
_EFFORT = {"minimal": "low", "low": "low", "medium": "medium", "high": "high"}
_DETAIL = {"low": "low", "high": "high", "ultra": "high"}  # (auto, medium: the API's own choice)
_ABOUT_KEY = re.compile(r"api[ _-]?key|invalid[_ ]key|incorrect[_ ]key|unauthori[sz]ed|revoked", re.I)
_RAW = "openai_responses"  # Turn.raw's tag: this provider's own output items


def _sees(model: str) -> bool:
    """A model NEO lists says; another GPT model sees (the catalogue's rule)."""
    from neo2.core.providers import catalog

    m = catalog.known_model(PROVIDER, model)
    return m.sees if m is not None else model.startswith("gpt-")


def _image(im: ImagePart) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": "input_image",
        "image_url": f"data:{im.mime};base64,{base64.b64encode(im.data).decode()}",
    }
    out["detail"] = _DETAIL.get(im.detail, "auto")
    return out


def _call_item(c: ToolCall) -> dict[str, Any]:
    return {"type": "function_call", "call_id": c.id, "name": c.name, "arguments": json.dumps(c.args)}


def _own(raw: Any, model: str) -> list[dict[str, Any]] | None:
    """`raw` as this model's own output items, to replay verbatim; None when it's another's (another
    model's reasoning items are its own: that reply goes as its text and calls instead)."""
    if isinstance(raw, dict) and raw.get("provider") == _RAW and raw.get("model") == model:
        items = raw.get("items")
        if isinstance(items, list) and all(isinstance(i, dict) for i in items):
            return items
    return None


def _to_input(messages: list[Message], vision: bool, model: str) -> list[dict[str, Any]]:
    """The conversation as Responses input items (it opens with a user turn and every call is
    answered: neo2.core.providers.history)."""
    out: list[dict[str, Any]] = []
    for m in normalize(messages):
        if m.role == "user":
            content: list[dict[str, Any]] = [{"type": "input_text", "text": m.text or "(empty)"}]
            if m.images and vision:
                content += [_image(im) for im in m.images]
            elif m.images:
                content[0]["text"] += "\n[image omitted: this brain has no vision]"
            out.append({"role": "user", "content": content})
        elif m.role == "assistant":
            own = _own(m.raw, model)
            if own is not None:
                out.extend(own)
                continue
            if m.text:
                out.append({"role": "assistant", "content": m.text})
            out.extend(_call_item(c) for c in m.tool_calls)
        else:
            for r in m.tool_results:
                text = r.content or "(no output)"
                if r.images and vision:
                    output: Any = [{"type": "input_text", "text": text}, *(_image(im) for im in r.images)]
                else:
                    output = text + ("\n" + BLIND_PICTURE if r.images else "")
                out.append({"type": "function_call_output", "call_id": r.call_id, "output": output})
    return out


def _tools(specs: list[ToolSpec] | None) -> list[dict[str, Any]]:
    # strict: False — NEO's schemas have optional fields, which strict mode would make required
    return [
        {"type": "function", "name": s.name, "description": s.description, "parameters": s.parameters, "strict": False}
        for s in specs or []
    ]


def _dump(item: Any) -> dict[str, Any]:
    """An output item as the input item it replays as (by its API names: `async`, not `async_`)."""
    if isinstance(item, dict):
        return item
    return item.model_dump(exclude_none=True, by_alias=True, mode="json")


def _message(e: Exception) -> str:
    body = getattr(e, "body", None)
    if isinstance(body, dict) and isinstance(body.get("message"), str):
        return body["message"]
    return str(getattr(e, "message", "") or "")


def _code(e: Exception) -> str:
    body = getattr(e, "body", None)
    fields = [getattr(e, "code", None), getattr(e, "type", None)]
    if isinstance(body, dict):
        fields += [body.get("code"), body.get("type")]
    return " ".join(str(f) for f in fields if f)


def _retry_after(e: Exception) -> float | None:
    resp = getattr(e, "response", None)
    try:
        h = resp.headers.get("retry-after") if resp is not None else None
        return max(0.0, float(h)) if h else None
    except (TypeError, ValueError, AttributeError):
        return None


def _about_reasoning(e: Exception) -> bool:
    """A 400 that refuses the reasoning parameter (or the encrypted reasoning it asks to include)."""
    param = str(getattr(e, "param", "") or "")
    return param.startswith(("reasoning", "include")) or "reasoning" in _message(e).lower()


def _too_large(e: APIStatusError) -> bool:
    low = (_message(e) + " " + _code(e)).lower()
    return e.status_code == 413 or "context_length_exceeded" in low or "context window" in low


class OpenAIResponsesProvider:
    supports_tools = True
    token_budget = _TOKEN_BUDGET
    image_limit = IMAGE_LIMIT

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
        name: str = PROVIDER,
        vision: bool | None = None,
        cap: Any = None,
        base_url: str | None = None,
    ) -> None:
        """`client`: the host's own configured AsyncOpenAI. Nothing here reads NEO's settings (the
        factory :func:`openai` does, for NEO.app's own brain); `cap`: its daily cap (spend.Cap)."""
        if client is None and not api_key:
            raise Unavailable("OPENAI_API_KEY is not set")
        self.name = name
        self.model = model
        self.supports_vision = _sees(model) if vision is None else bool(vision)
        # `base_url`: where requests go (NEO.app's own brain: OpenAI's, always — an OPENAI_BASE_URL
        # set for other tools never sends its key and screenshots elsewhere). None: the SDK's own.
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=_TIMEOUT_S, max_retries=1)
        self._reasoning = True  # False once this model refused the reasoning parameter
        self._cool_until = 0.0
        self._cap = cap
        self._app = False  # NEO.app's own (built from its settings): may say so in its health list

    def cooling_until(self) -> float:
        return self._cool_until if self._cool_until > time.time() else 0.0

    # ---- the request -------------------------------------------------------------------------------
    def _request(self, messages: list[Message], system: str, tools: list[ToolSpec] | None, effort: str, out: int) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "model": self.model,
            "input": _to_input(messages, self.supports_vision, self.model),
            "max_output_tokens": out,
            "store": False,
        }
        if system:
            kw["instructions"] = system
        t = _tools(tools)
        if t:
            kw["tools"] = t
            kw["parallel_tool_calls"] = True
        if self._reasoning:
            kw["reasoning"] = {"effort": _EFFORT.get(effort, "medium")}
            kw["include"] = ["reasoning.encrypted_content"]
        return kw

    def _error(self, e: Exception) -> ProviderError:
        name = self.name
        if isinstance(e, RateLimitError):
            if "insufficient_quota" in _code(e):
                return QuotaExhausted(f"{name}: the OpenAI account is out of quota", until=time.time() + _QUOTA_S)
            wait = _retry_after(e)
            if wait:
                self._cool_until = time.time() + wait
            return RateLimited(f"{name}: rate-limited (HTTP 429)", retry_after=wait)
        if isinstance(e, AuthenticationError):
            return KeyRefused(f"{name}: OpenAI refused the API key (HTTP 401)")
        if isinstance(e, PermissionDeniedError) and _ABOUT_KEY.search(_message(e) + " " + _code(e)):
            return KeyRefused(f"{name}: OpenAI refused the API key (HTTP 403)")
        if isinstance(e, APIStatusError):
            code = e.status_code
            if code == 402:
                return QuotaExhausted(f"{name}: the OpenAI account has no credit (HTTP 402)", until=time.time() + _QUOTA_S)
            if _too_large(e):
                return RequestTooLarge(f"{name}: the request is too long for {self.model}")
            if code >= 500:
                return Overloaded(f"{name}: OpenAI is overloaded (HTTP {code})", retry_after=_retry_after(e))
            if 400 <= code < 500:
                return BadRequest(f"{name}: HTTP {code}: {_message(e)[:200]}")
            return ProviderError(f"{name}: HTTP {code}")
        if isinstance(e, APIConnectionError):
            return Unavailable(f"{name}: can't reach OpenAI ({type(e).__name__})")
        return ProviderError(f"{name}: {type(e).__name__}")

    def _rejected(self, e: Exception) -> None:
        """A 404 (or a 403) about the model: one health problem (NEO.app's own brain only)."""
        code = getattr(e, "status_code", 0)
        low = (_message(e) + " " + _code(e)).lower()
        if not self._app or not runtime.process_wide_ok() or code not in (403, 404) or "model" not in low:
            return
        if code == 403 and _ABOUT_KEY.search(low):  # (the key's: KeyRefused says that one)
            return
        from neo2.core import brainref, health

        shown = brainref.shown(self.model) or "it was set to"
        health.report(f"model:{PROVIDER}:{self.model}", f"OpenAI rejected model {shown} — pick a current one in Brains & Keys.")

    def _works(self) -> None:
        if self._app and runtime.process_wide_ok():
            from neo2.core import health

            health.report(f"model:{PROVIDER}:{self.model}", None)

    async def _admit(self) -> None:
        if self._cap is not None and (refused := self._cap.refusal(self.name)) is not None:
            await self._cap.tell()
            raise refused

    async def _count(self, usage: Any) -> None:
        if self._cap is None or usage is None:
            return
        self._cap.add(self.model, int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0))
        await self._cap.tell()

    async def _create(self, build, **extra: Any) -> Any:
        """One request; a model that refuses the reasoning parameter is asked once more without it."""
        for attempt in range(2):
            try:
                return await self._client.responses.create(**build(), **extra)
            except BadRequestError as e:
                if attempt == 0 and self._reasoning and _about_reasoning(e):
                    self._reasoning = False
                    log.info(f"[brain] {self.name}: {self.model} doesn't take a reasoning effort; asking without one")
                    continue
                self._rejected(e)
                raise self._error(e) from e
            except (APIStatusError, APIConnectionError) as e:
                self._rejected(e)
                raise self._error(e) from e
        raise ProviderError(f"{self.name}: no answer")  # (unreachable: the loop returns or raises)

    # ---- the answer --------------------------------------------------------------------------------
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
        f = fit(
            messages,
            system=system,
            tools=tools,
            budget=self.token_budget,
            max_tokens=max(max_tokens, _MIN_OUTPUT),
            vision=self.supports_vision,
            name=self.name,
        )
        resp = await self._create(lambda: self._request(f.messages, system, f.tools, effort, f.max_tokens))
        self._works()
        await self._count(getattr(resp, "usage", None))
        items = list(getattr(resp, "output", None) or [])
        turn = Turn(raw={"provider": _RAW, "model": self.model, "items": [_dump(i) for i in items]})
        refused = ""
        for it in items:
            kind = getattr(it, "type", "")
            if kind == "message":
                for c in getattr(it, "content", None) or []:
                    if getattr(c, "type", "") == "output_text":
                        turn.text += c.text
                    elif getattr(c, "type", "") == "refusal":
                        refused = getattr(c, "refusal", "") or "I can't help with that one."
            elif kind == "function_call":
                try:
                    args = json.loads(it.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": it.arguments}
                if not isinstance(args, dict):
                    args = {"_raw": it.arguments}
                turn.tool_calls.append(
                    ToolCall(id=it.call_id or uuid.uuid4().hex[:12], name=it.name, args=args, source=self.model)
                )
            elif kind == "reasoning":
                for s in getattr(it, "summary", None) or []:
                    turn.thinking += getattr(s, "text", "") or ""
        if refused and not turn.text and not turn.tool_calls:
            turn.text = refused
        if not turn.text and not turn.tool_calls and getattr(resp, "status", "") == "incomplete":
            raise Truncated(f"{self.name}: no room for the answer (reasoning used all {f.max_tokens} tokens)")
        u = getattr(resp, "usage", None)
        if u is not None:
            turn.usage = Usage(int(u.input_tokens or 0), int(u.output_tokens or 0))
        turn.offered = f.offered()
        turn.sees = bool(self.supports_vision)
        return turn

    async def stream(self, messages: list[Message], *, system: str = "", max_tokens: int = 2048) -> AsyncIterator[str]:
        """A chat answer's text as it's written (no tools; quick thinking)."""
        await self._admit()
        f = fit(
            messages,
            system=system,
            budget=self.token_budget,
            max_tokens=max(max_tokens, _MIN_OUTPUT),
            vision=self.supports_vision,
            name=self.name,
        )
        st = await self._create(lambda: self._request(f.messages, system, None, "low", f.max_tokens), stream=True)
        usage: Any = None
        try:
            async for ev in st:
                kind = getattr(ev, "type", "")
                if kind == "response.output_text.delta":
                    if ev.delta:
                        yield ev.delta
                elif kind in ("response.completed", "response.incomplete"):
                    usage = getattr(getattr(ev, "response", None), "usage", None)
                elif kind in ("response.failed", "error"):
                    raise Overloaded(f"{self.name}: the answer stopped part way ({kind})")
        except (APIStatusError, APIConnectionError) as e:
            raise self._error(e) from e
        self._works()
        await self._count(usage)


def openai(model: str | None = None) -> OpenAIResponsesProvider:
    """NEO.app's OpenAI brain (or an embedded agent's): its key, model and daily cap from settings."""
    from neo2.core.config import settings
    from neo2.core.providers import catalog, spend

    s = settings()
    if not s.openai_api_key:
        raise Unavailable("OPENAI_API_KEY is not set")
    model = model or s.openai_model
    p = OpenAIResponsesProvider(
        model,
        api_key=s.openai_api_key,
        vision=catalog.describe(f"openai@{model}", s).sees,
        cap=spend.cap_for(PROVIDER, s),
        base_url=catalog.PROVIDERS["openai"].base_url,  # (where Brains & Keys tested the key, too)
    )
    p._app = True
    return p
