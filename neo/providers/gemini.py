"""Gemini brain (google-genai SDK) — the default agentic + vision provider.

Free tier of Gemini 3.x Flash is the only free option with enough TPM for an agent loop.
It is also the one that gets overloaded, so every request walks a model list
(3.8 → 3.7 → 3.5 Flash) with a short backoff on 503 before giving up to the next brain.

Free-tier quota is per key *and* per model. A (model, key) that says it's out of quota is left
alone for the rest of its quota window — until the daily reset for a per-day quota, the delay the
server names for a per-minute one — and once every pair is out, a request fails at once with
QuotaExhausted instead of paying a round trip per pair on every agent step.

Gemini 3 requires the ``thought_signature`` of every function call to be echoed back when the
history is replayed. A signature is only good for the model that wrote it: calls that came from
another brain (Groq, Claude) — or from another Gemini model before a quota/overload switch — get
Google's documented bypass value instead.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx
from google import genai
from google.genai import errors as gerrors
from google.genai import types as gt

from neo.config import settings
from neo.providers.base import (
    BadRequest,
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
from neo.providers.fit import fit
from neo.providers.history import flatten, normalize

# The SDK nags about automatic function calling on the async path even though we never use it.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

_LEVEL = {"low": gt.ThinkingLevel.LOW, "medium": gt.ThinkingLevel.MEDIUM, "high": gt.ThinkingLevel.HIGH}
_SKIP_SIGNATURE = b"skip_thought_signature_validator"
_RETRY_CODES = {500, 502, 503, 504}
_QUOTA_COOLDOWN_S = 600.0  # a (model, key) that 429'd without saying which quota: skipped this long
_OVERLOAD_COOLDOWN_S = 90.0  # a model that 5xx'd after a retry is skipped for this long
_BACKOFF_S = (1.2,)  # one short retry on 5xx; a bad night should fall through to Groq fast
_SLOW_FAIL_S = 3.0  # a 5xx that took this long to arrive is real overload, not a blip: no retry
_REQUEST_TIMEOUT_MS = 60_000  # never let one brain call eat the whole agent budget
_COOLDOWN_FILE = "brain_cooldowns.json"  # (model, key) quota state survives a restart
_TOKEN_BUDGET = 250_000  # free-tier tokens per minute (the context itself is 1M)
_RESET_TZ = "America/Los_Angeles"  # free-tier daily quotas reset at midnight Pacific
# A 400 about the shape of the tool turns: the history is re-sent once with them written as text.
_TURN_SHAPE = ("function response", "function call", "thought_signature", "thought signature")
_TOO_LONG = ("exceeds the maximum number of tokens", "input token count", "too many tokens")


def _signature(c: ToolCall, model: str) -> bytes:
    if c.signature and (not c.source or not model or c.source == model):
        return c.signature
    return _SKIP_SIGNATURE


def _to_contents(messages: list[Message], model: str = "") -> list[gt.Content]:
    """The history as Gemini contents for `model` (whose own signatures are echoed; any other
    brain's calls get the bypass). Normalized first: see neo.providers.history."""
    out: list[gt.Content] = []
    for m in normalize(messages):
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
                        thought_signature=_signature(c, model),
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
    code, msg = _code(e), str(e)[:300]
    if code == 429:
        return RateLimited(msg)
    if code in (400, 413):
        low = msg.lower()
        return RequestTooLarge(msg) if code == 413 or any(k in low for k in _TOO_LONG) else BadRequest(msg)
    if code in _RETRY_CODES:
        return Overloaded(msg)
    if isinstance(e, httpx.TransportError):
        return Unavailable(f"gemini: {type(e).__name__}")
    return ProviderError(msg)


def _seconds(v: Any) -> float | None:
    """ "37s" / "1.5s" (a protobuf Duration as JSON) → seconds."""
    try:
        return float(str(v).rstrip("s"))
    except (TypeError, ValueError):
        return None


def _to_daily_reset(now: float | None = None) -> float:
    """Seconds until the free tier's daily quota resets (midnight Pacific)."""
    import datetime as dt

    now = time.time() if now is None else now
    try:
        from zoneinfo import ZoneInfo

        tz: dt.tzinfo = ZoneInfo(_RESET_TZ)
    except Exception:  # noqa: BLE001 — no tz database: Pacific standard time is close enough
        tz = dt.timezone(dt.timedelta(hours=-8))
    here = dt.datetime.fromtimestamp(now, tz)
    midnight = (here + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(60.0, midnight.timestamp() - now)


def _quota_window(e: Exception) -> tuple[float, str]:
    """(seconds this (model, key) stays out of quota, "day" | "minute" | "?") from a 429's
    details: a per-day quota lasts to the daily reset, a per-minute one as long as the server's
    retry delay; anything else a fixed cooldown."""
    info = getattr(e, "details", None)
    err = info.get("error", info) if isinstance(info, dict) else {}
    daily = minute = False
    delay: float | None = None
    for d in (err.get("details") if isinstance(err, dict) else None) or []:
        if not isinstance(d, dict):
            continue
        kind = str(d.get("@type", ""))
        if kind.endswith("QuotaFailure"):
            for v in d.get("violations") or []:
                q = f"{v.get('quotaId', '')} {v.get('quotaMetric', '')}".lower()
                daily = daily or "perday" in q or "per_day" in q
                minute = minute or "perminute" in q or "per_minute" in q
        elif kind.endswith("RetryInfo"):
            delay = _seconds(d.get("retryDelay"))
    if daily:
        return _to_daily_reset(), "day"
    if delay is not None:
        return max(delay, 1.0) + 1.0, "minute"
    if minute:
        return 61.0, "minute"
    return _QUOTA_COOLDOWN_S, "?"


class GeminiProvider:
    name = "gemini"
    supports_vision = True
    supports_tools = True
    token_budget = _TOKEN_BUDGET

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        s = settings()
        key = api_key or s.gemini_api_key
        if not key:
            raise Unavailable("GEMINI_API_KEY is not set")
        self.model = model or s.gemini_model
        lite = "lite" in self.model
        self.models = [self.model] + (
            [] if lite else [m for m in s.gemini_fallback_models if m != self.model]
        )
        keys = [key] if api_key else s.gemini_keys
        http = gt.HttpOptions(timeout=_REQUEST_TIMEOUT_MS)
        self._clients = [genai.Client(api_key=k, http_options=http) for k in keys]
        self._client = self._clients[0]
        self._exhausted: dict[tuple[str, int], float] = {}  # (model, key index | -1) → retry-after timestamp
        self._load_cooldowns()

    # ---- cooldown persistence -------------------------------------------------------------
    # Learning that nine (model, key) pairs are out of quota costs nine round trips; a restart
    # must not pay that again, so the map is kept on disk with its expiry times.
    def _cooldown_path(self):
        return settings().data_dir / _COOLDOWN_FILE

    def _load_cooldowns(self) -> None:
        import json

        try:
            raw = json.loads(self._cooldown_path().read_text())
        except Exception:  # noqa: BLE001 — missing/corrupt file: start clean
            return
        now = time.time()
        for k, until in raw.items():
            model, _, ki = k.rpartition("|")
            if model and float(until) > now:
                self._exhausted[(model, int(ki))] = float(until)

    def _save_cooldowns(self) -> None:
        import json

        now = time.time()
        # Merge with what's on disk: the lite brain is another provider sharing the file.
        try:
            disk = {k: float(t) for k, t in json.loads(self._cooldown_path().read_text()).items()}
        except Exception:  # noqa: BLE001
            disk = {}
        disk.update({f"{m}|{ki}": t for (m, ki), t in self._exhausted.items()})
        live = {k: t for k, t in disk.items() if t > now}
        try:
            self._cooldown_path().write_text(json.dumps(live))
        except Exception:  # noqa: BLE001 — best effort
            pass

    # ---- quota state ----------------------------------------------------------------------
    def _blocked(self, model: str, ki: int, now: float) -> bool:
        return self._exhausted.get((model, -1), 0.0) > now or self._exhausted.get((model, ki), 0.0) > now

    def cooling_until(self) -> float:
        """When some (model, key) can be asked again, if none can now (0: one can now)."""
        now = time.time()
        ends = []
        for model in self.models:
            for ki in range(len(self._clients)):
                if not self._blocked(model, ki, now):
                    return 0.0
                ends.append(max(self._exhausted.get((model, -1), 0.0), self._exhausted.get((model, ki), 0.0)))
        return min(ends) if ends else 0.0

    def _all_blocked_error(self) -> ProviderError:
        """Nothing left to try: say whether that's quota (for how long) or overload."""
        now = time.time()
        quota, busy = [], []
        for model in self.models:
            over = self._exhausted.get((model, -1), 0.0)
            for ki in range(len(self._clients)):
                q = self._exhausted.get((model, ki), 0.0)
                if q > now:
                    quota.append(q)
                elif over > now:
                    busy.append(over)
        if busy:
            return Overloaded("gemini: every model is overloaded", retry_after=min(busy) - now)
        if quota:
            return QuotaExhausted("429 gemini: every key is out of quota", until=min(quota))
        if not self.models or not self._clients:
            return ProviderError("gemini: no models")
        return RateLimited(
            "429 gemini: rate-limited, try again shortly", retry_after=1.0
        )  # a window just ended

    async def _call(self, fn, **kw):
        """Walk model × key. 5xx → short backoff then next model; 429 → next key for this model
        (free-tier quota is per key *and* per model); other errors raise immediately. Pairs known
        to be out of quota or overloaded are skipped without a round trip; when none is left, the
        error says which (QuotaExhausted with its reset time, or Overloaded)."""
        for model in self.models:
            overloaded = False
            for ki, client in enumerate(self._clients):
                if self._blocked(model, ki, time.time()):
                    continue  # known spent / overloaded — don't pay a round trip to learn it again
                for i, delay in enumerate((0.0, *_BACKOFF_S)):
                    if delay:
                        await asyncio.sleep(delay)
                    t0 = time.time()
                    try:
                        r = await fn(client, model, **kw)
                        self.model, self._client = model, client
                        return r
                    except Exception as e:  # noqa: BLE001
                        code = _code(e)
                        took = time.time() - t0
                        if code in _RETRY_CODES and i < len(_BACKOFF_S) and took < _SLOW_FAIL_S:
                            continue  # a quick blip is worth one retry; a slow 5xx is not
                        if code == 429:
                            window, per = _quota_window(e)
                            self._exhausted[(model, ki)] = time.time() + window
                            self._save_cooldowns()
                            span = "for today" if per == "day" else f"for {int(window)}s"
                            print(f"[gemini] {model} key#{ki + 1} quota exhausted ({span}); trying next key")
                            await _note(f"{_pretty(model)} key {ki + 1}: no quota, trying next")
                            break
                        if code in _RETRY_CODES:
                            print(f"[gemini] {model} overloaded ({took:.0f}s); trying next model")
                            self._exhausted[(model, -1)] = time.time() + _OVERLOAD_COOLDOWN_S
                            self._save_cooldowns()
                            await _note(f"{_pretty(model)} overloaded, trying next")
                            overloaded = True
                            break
                        raise _map_error(e) from e
                if overloaded:
                    break
        raise self._all_blocked_error()

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        effort: Literal["low", "medium", "high"] = "medium",
        max_tokens: int = 4096,
    ) -> Turn:
        f = fit(
            messages, system=system, tools=tools, budget=self.token_budget, max_tokens=max_tokens,
            vision=True, name=self.name,
        )  # fmt: skip
        cfg = gt.GenerateContentConfig(
            system_instruction=system or None,
            tools=_tools(f.tools),
            thinking_config=gt.ThinkingConfig(thinking_level=_LEVEL[effort]),
            max_output_tokens=f.max_tokens,
        )
        answered = {"model": self.model}

        def request(msgs: list[Message]):
            async def fn(client, model: str):
                answered["model"] = model
                return await client.aio.models.generate_content(
                    model=model, contents=_to_contents(msgs, model), config=cfg
                )

            return fn

        try:
            resp = await self._call(request(f.messages))
        except BadRequest as e:
            if isinstance(e, RequestTooLarge) or not any(k in str(e).lower() for k in _TURN_SHAPE):
                raise
            # A shape rule nothing above foresaw: the task goes on with the tool turns as text.
            print(f"[gemini] history rejected ({str(e)[:100]}); retrying with tool turns as text")
            resp = await self._call(request(flatten(f.messages)))

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
                        source=answered["model"],
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
        f = fit(
            messages,
            system=system,
            budget=self.token_budget,
            max_tokens=max_tokens,
            vision=True,
            name=self.name,
        )
        cfg = gt.GenerateContentConfig(
            system_instruction=system or None,
            thinking_config=gt.ThinkingConfig(thinking_level=gt.ThinkingLevel.LOW),
            max_output_tokens=f.max_tokens,
        )

        async def first_chunk(client, model: str, **kw):
            # The SDK issues the request lazily on first iteration, so the retry/model-walk has to
            # cover pulling the first chunk, not just creating the iterator.
            it = await client.aio.models.generate_content_stream(
                model=model, contents=_to_contents(f.messages, model), **kw
            )
            ait = it.__aiter__()
            try:
                head = await ait.__anext__()
            except StopAsyncIteration:
                head = None
            return head, ait

        head, ait = await self._call(first_chunk, config=cfg)
        try:
            if head is not None and head.text:
                yield head.text
            async for chunk in ait:
                if chunk.text:
                    yield chunk.text
        except Exception as e:  # noqa: BLE001
            raise _map_error(e) from e
