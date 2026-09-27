"""Gemini brain (google-genai SDK) — the default agentic + vision provider.

Free tier of Gemini 3.x Flash is the only free option with enough TPM for an agent loop.
It is also the one that gets overloaded, so every request walks a model list
(3.8 → 3.7 → 3.5 Flash) with a short backoff on 503 before giving up to the next brain.

Free-tier quota is per key *and* per model. A (model, key) that says it's out of quota is left
alone for the rest of its quota window — until the daily reset for a per-day quota, the delay the
server names for a per-minute one — and once every pair is out, a request fails at once with
QuotaExhausted instead of paying a round trip per pair on every agent step. That state follows the
key itself (a short hash of it), not its place in the list: a key the user adds after hearing
"out of quota" is asked at once. A key Gemini refuses outright (not valid, suspended, the API off
for its project) is skipped for the next one; a refusal of one model only (a 403 that names
neither the key nor its project) leaves that key's other models to answer.

The brains NEO builds from settings keep that state in their data dir (NEO.app's, or an embedded
agent's own). One a host builds from its own keys (neo2.providers.Gemini) keeps it in memory, in
the object: nothing is written for it.

How hard it thinks is the chain's choice per step (neo.providers.effort). "minimal" is asked only of
a model that takes it: one that says it doesn't (a 400 naming the thinking level) is asked at
"low" from then on. A picture's ImagePart.detail sets its media resolution (Gemini bills 280 /
560 / 1,120 / 2,240 tokens a picture); "auto" leaves the model's own default.

Gemini 3 requires the ``thought_signature`` of every function call to be echoed back when the
history is replayed. A signature is only good for the model that wrote it: calls that came from
another brain (Groq, Claude) — or from another Gemini model before a quota/overload switch — get
Google's documented bypass value instead.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, Literal

import httpx
from google import genai
from google.genai import errors as gerrors
from google.genai import types as gt

from neo.config import Settings, settings
from neo.providers.base import (
    BadRequest,
    Level,
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

log = logging.getLogger(__name__)

_LEVEL = {
    "minimal": gt.ThinkingLevel.MINIMAL,
    "low": gt.ThinkingLevel.LOW,
    "medium": gt.ThinkingLevel.MEDIUM,
    "high": gt.ThinkingLevel.HIGH,
}
# A 400 that says the model won't take the thinking level it was sent ("minimal" on a model that
# dropped it): that model is asked at "low" from then on.
_NO_MINIMAL = re.compile(r"thinking[ _]?level|thinking_config|\bminimal\b", re.I)
# ImagePart.detail → Gemini's per-picture media resolution ("auto": not sent, the model's default).
_RESOLUTION = {
    "low": gt.PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW,
    "medium": gt.PartMediaResolutionLevel.MEDIA_RESOLUTION_MEDIUM,
    "high": gt.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH,
    "ultra": gt.PartMediaResolutionLevel.MEDIA_RESOLUTION_ULTRA_HIGH,
}
# A 400 about the per-picture resolution (a model that doesn't take it): sent again without it.
_NO_RESOLUTION = re.compile(r"media[ _]?resolution", re.I)
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
# A key the server won't take at all (revoked, expired, suspended, the API off for its project):
# no model will answer on it, so it is skipped for the next key rather than failing the request.
_BAD_KEY = re.compile(r"api[ _]key[ _](?:not[ _]valid|invalid|expired)", re.I)
# …which a 403 says in its ErrorInfo reason or its message (not the metadata: every Google 403
# names the "consumer" project there). A 403 that says neither is about one model ("Permission
# denied … for model X"): only that (model, key) pair is left alone.
_KEY_REASONS = re.compile(r"^(?:API_KEY_\w+|SERVICE_DISABLED|CONSUMER_\w+|BILLING_DISABLED)$")
_KEY_WORDS = re.compile(r"api[ _]?key|suspended|has not been used in project|billing", re.I)
_BAD_KEY_COOLDOWN_S = 3600.0
_ANY_KEY = "*"  # a model's own trouble (overload), whichever key asked
_SAVED = re.compile(r"\|(?:[0-9a-f]{12}|\*)$")  # "model|<key id>": older files named the key's place


def _key_id(key: str) -> str:
    """Which key a saved cooldown belongs to, without writing the key to disk."""
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _reasons(e: Exception) -> list[str]:
    info = getattr(e, "details", None)
    err = info.get("error", info) if isinstance(info, dict) else {}
    details = (err.get("details") if isinstance(err, dict) else None) or []
    return [str(d.get("reason", "")) for d in details if isinstance(d, dict) and d.get("reason")]


def _refused(e: Exception) -> Literal["key", "model"] | None:
    """Whether the server refused the key itself (no model will answer on it), only this model
    on it, or neither."""
    code = _code(e)
    if code == 401 or (code == 400 and _BAD_KEY.search(str(e))):
        return "key"
    if code != 403:
        return None
    message = getattr(e, "message", None) or str(e)
    if any(_KEY_REASONS.match(r) for r in _reasons(e)) or _KEY_WORDS.search(message):
        return "key"
    return "model"


def _signature(c: ToolCall, model: str) -> bytes:
    if c.signature and (not c.source or not model or c.source == model):
        return c.signature
    return _SKIP_SIGNATURE


def _picture(im, detail: bool = True) -> gt.Part:
    """A picture as a Gemini part, at the media resolution its detail asks for."""
    res = _RESOLUTION.get(getattr(im, "detail", "auto")) if detail else None
    if res is None:
        return gt.Part.from_bytes(data=im.data, mime_type=im.mime)
    return gt.Part.from_bytes(data=im.data, mime_type=im.mime, media_resolution=res)


def _to_contents(messages: list[Message], model: str = "", *, detail: bool = True) -> list[gt.Content]:
    """The history as Gemini contents for `model` (whose own signatures are echoed; any other
    brain's calls get the bypass). Normalized first: see neo.providers.history. `detail`: send each
    picture's media resolution (off for a model that refused it)."""
    out: list[gt.Content] = []
    for m in normalize(messages):
        if m.role == "user":
            parts: list[gt.Part] = []
            if m.text:
                parts.append(gt.Part.from_text(text=m.text))
            for im in m.images:
                parts.append(_picture(im, detail))
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
                    parts.append(_picture(im, detail))
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


def _pictures_with_detail(messages: list[Message]) -> bool:
    return any(
        getattr(im, "detail", "auto") in _RESOLUTION
        for m in messages
        for im in (*m.images, *(i for r in m.tool_results for i in r.images))
    )


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
    supports_vision = True
    supports_tools = True
    token_budget = _TOKEN_BUDGET
    efforts = frozenset({"minimal", "low", "medium", "high"})  # (minimal: where the model takes it)

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        *,
        keys: Sequence[str] | None = None,
        fallback_models: Sequence[str] | None = None,
        name: str = "gemini",
    ) -> None:
        """`keys`: every key, in priority order (quota is per key). Given a key (or keys), it's
        the host's: nothing is read from settings, the model and fallbacks not passed are the
        release's (a host may build it before any agent exists — NEO.app's settings would read
        ~/.neo), and what it learns about its keys stays in this object. Given no key, the keys
        and what isn't passed come from settings, and the quota state is kept in their data dir."""
        # The SDK nags about automatic function calling on the async path even though we never use it.
        logging.getLogger("google_genai.models").setLevel(logging.ERROR)
        own = keys is not None or bool(api_key)
        d = Settings.model_fields
        if keys is not None:
            key_list = list(dict.fromkeys(k.strip() for k in keys if k and k.strip()))
        elif api_key:
            key_list = [api_key]
        else:
            key_list = settings().gemini_keys if settings().gemini_api_key else []
        if not key_list:
            raise Unavailable("GEMINI_API_KEY is not set")
        # The brain this is ("gemini", "gemini_lite"): what the chain demotes and the user hears.
        self.name = name
        self.model = model or (d["gemini_model"].default if own else settings().gemini_model)
        lite = "lite" in self.model
        if fallback_models is not None:
            fallbacks = list(fallback_models)
        else:
            fallbacks = list(d["gemini_fallback_models"].default if own else settings().gemini_fallback_models)
        self.models = [self.model] + ([] if lite else [m for m in fallbacks if m != self.model])
        http = gt.HttpOptions(timeout=_REQUEST_TIMEOUT_MS)
        self._clients = [genai.Client(api_key=k, http_options=http) for k in key_list]
        self._client = self._clients[0]
        self._key_ids = [_key_id(k) for k in key_list]
        # (model, key id | _ANY_KEY) → when it may be asked again
        self._exhausted: dict[tuple[str, str], float] = {}
        self._bad_keys: dict[str, float] = {}  # key id → when to try a key the server refused again
        self._denied: dict[tuple[str, str], float] = {}  # (model, key id) refused: when to try again
        self._no_minimal: set[str] = set()  # models that said they don't take the "minimal" level
        self._no_detail: set[str] = set()  # models that refused a per-picture media resolution
        # Where the quota state is kept: the data dir of the settings it was built from (NEO.app's,
        # or the embedded agent's whose pool built it — fixed now, so no other agent's call can
        # move it). A host's own provider: nowhere, it stays in this object.
        self._cooldown_file: Path | None = None if own else settings().data_dir / _COOLDOWN_FILE
        self._load_cooldowns()

    # ---- cooldown persistence -------------------------------------------------------------
    # Learning that nine (model, key) pairs are out of quota costs nine round trips; a restart
    # must not pay that again, so the map is kept on disk with its expiry times.
    def _cooldown_path(self) -> Path | None:
        return self._cooldown_file

    def _load_cooldowns(self) -> None:
        import json

        path = self._cooldown_path()
        if path is None:
            return
        try:
            raw = json.loads(path.read_text())
        except Exception:  # noqa: BLE001 — missing/corrupt file: start clean
            return
        now = time.time()
        mine = {*self._key_ids, _ANY_KEY}
        for k, until in raw.items():
            model, _, kid = str(k).rpartition("|")
            try:
                t = float(until)
            except (TypeError, ValueError):
                continue
            if model and kid in mine and t > now:  # (a key no longer configured: forgotten)
                self._exhausted[(model, kid)] = t

    def _save_cooldowns(self) -> None:
        import json

        path = self._cooldown_path()
        if path is None:
            return
        now = time.time()
        # Merge with what's on disk: the lite brain is another provider sharing the file.
        try:
            disk = {k: float(t) for k, t in json.loads(path.read_text()).items()}
        except Exception:  # noqa: BLE001
            disk = {}
        disk.update({f"{m}|{kid}": t for (m, kid), t in self._exhausted.items()})
        live = {k: t for k, t in disk.items() if t > now and _SAVED.search(k)}
        try:
            path.write_text(json.dumps(live))
        except Exception:  # noqa: BLE001 — best effort
            pass

    # ---- quota state ----------------------------------------------------------------------
    def _kid(self, ki: int) -> str:
        return self._key_ids[ki] if ki < len(self._key_ids) else f"#{ki}"

    def _until(self, model: str, ki: int) -> float:
        """When (model, key) may be asked again: the latest of its quota, the model's overload and
        the key's own refusal."""
        kid = self._kid(ki)
        return max(
            self._exhausted.get((model, _ANY_KEY), 0.0),
            self._exhausted.get((model, kid), 0.0),
            self._bad_keys.get(kid, 0.0),
            self._denied.get((model, kid), 0.0),
        )

    def _blocked(self, model: str, ki: int, now: float) -> bool:
        return self._until(model, ki) > now

    def cooling_until(self) -> float:
        """When some (model, key) can be asked again, if none can now (0: one can now)."""
        now = time.time()
        ends = []
        for model in self.models:
            for ki in range(len(self._clients)):
                if not self._blocked(model, ki, now):
                    return 0.0
                ends.append(self._until(model, ki))
        return min(ends) if ends else 0.0

    def cooling_error(self) -> ProviderError | None:
        """While no (model, key) can be asked: why — the error asking now would end in (a spent
        quota and when it resets, an overload, keys refused). None when one can be asked. The chain
        records it for a brain it skips inside a run (neo.providers._too_late)."""
        return self._all_blocked_error() if self.cooling_until() else None

    def _all_blocked_error(self) -> ProviderError:
        """Nothing left to try: say whether that's quota (for how long), overload or the keys."""
        now = time.time()
        quota, busy, refused, denied = [], [], 0, 0
        for model in self.models:
            over = self._exhausted.get((model, _ANY_KEY), 0.0)
            for ki in range(len(self._clients)):
                q = self._exhausted.get((model, self._kid(ki)), 0.0)
                if self._bad_keys.get(self._kid(ki), 0.0) > now:
                    refused += 1
                elif self._denied.get((model, self._kid(ki)), 0.0) > now:
                    denied += 1
                elif q > now:
                    quota.append(q)
                elif over > now:
                    busy.append(over)
        if busy:
            return Overloaded("gemini: every model is overloaded", retry_after=min(busy) - now)
        if quota:
            return QuotaExhausted("429 gemini: every key is out of quota", until=min(quota))
        if refused:
            return Unavailable("API key not valid")
        if denied:
            return Unavailable("gemini: permission denied for these models")
        if not self.models or not self._clients:
            return ProviderError("gemini: no models")
        return RateLimited(
            "429 gemini: rate-limited, try again shortly", retry_after=1.0
        )  # a window just ended

    async def _call(self, fn, **kw):
        """Walk model × key. 5xx → short backoff then next model; 429 → next key for this model
        (free-tier quota is per key *and* per model); a key the server refuses (not valid, 401, a
        403 about the key) → the next key, for every model; a 403 about this model only → the next
        key for it; other errors raise immediately. Pairs known to be
        out of quota or overloaded, and refused keys, are skipped without a round trip; when none
        is left, the error says which (QuotaExhausted with its reset time, Overloaded, or
        Unavailable for keys that don't work)."""
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
                        refused = _refused(e)
                        if refused == "key":
                            self._bad_keys[self._kid(ki)] = time.time() + _BAD_KEY_COOLDOWN_S
                            log.warning(f"[gemini] key#{ki + 1} refused ({code}: {str(e)[:80]}); trying next key")
                            await _note(f"Gemini key {ki + 1} isn't accepted, trying next")
                            break
                        if refused == "model":
                            self._denied[(model, self._kid(ki))] = time.time() + _BAD_KEY_COOLDOWN_S
                            log.warning(f"[gemini] {model} refused on key#{ki + 1} ({str(e)[:80]}); trying next")
                            await _note(f"{_pretty(model)} not allowed on key {ki + 1}, trying next")
                            break
                        if code == 429:
                            window, per = _quota_window(e)
                            self._exhausted[(model, self._kid(ki))] = time.time() + window
                            self._save_cooldowns()
                            span = "for today" if per == "day" else f"for {int(window)}s"
                            log.warning(f"[gemini] {model} key#{ki + 1} quota exhausted ({span}); trying next key")
                            await _note(f"{_pretty(model)} key {ki + 1}: no quota, trying next")
                            break
                        if code in _RETRY_CODES:
                            log.warning(f"[gemini] {model} overloaded ({took:.0f}s); trying next model")
                            self._exhausted[(model, _ANY_KEY)] = time.time() + _OVERLOAD_COOLDOWN_S
                            self._save_cooldowns()
                            await _note(f"{_pretty(model)} overloaded, trying next")
                            overloaded = True
                            break
                        raise _map_error(e) from e
                if overloaded:
                    break
        raise self._all_blocked_error()

    def _level(self, model: str, effort: str) -> Level:
        """The thinking level to ask `model` at: `effort`, but "low" for "minimal" on a model that
        doesn't take it (and "medium" for anything unknown)."""
        if effort == "minimal" and model in self._no_minimal:
            return "low"
        return effort if effort in _LEVEL else "medium"  # type: ignore[return-value]

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        effort: Level = "medium",
        max_tokens: int = 4096,
    ) -> Turn:
        f = fit(
            messages, system=system, tools=tools, budget=self.token_budget, max_tokens=max_tokens,
            vision=True, name=self.name,
        )  # fmt: skip
        answered = {"model": self.model, "level": ""}

        def config(model: str) -> gt.GenerateContentConfig:
            level = self._level(model, effort)
            answered["level"] = level
            return gt.GenerateContentConfig(
                system_instruction=system or None,
                tools=_tools(f.tools),
                thinking_config=gt.ThinkingConfig(thinking_level=_LEVEL[level]),
                max_output_tokens=f.max_tokens,
            )

        def request(msgs: list[Message]):
            async def fn(client, model: str):
                answered["model"] = model
                return await client.aio.models.generate_content(
                    model=model,
                    contents=_to_contents(msgs, model, detail=model not in self._no_detail),
                    config=config(model),
                )

            return fn

        msgs = f.messages
        for _ in range(3):  # a model's refusal of "minimal" or of a picture's resolution is learned once
            try:
                resp = await self._call(request(msgs))
                break
            except BadRequest as e:
                if isinstance(e, RequestTooLarge):
                    raise
                said, model = str(e), answered["model"]
                if answered["level"] == "minimal" and model not in self._no_minimal and _NO_MINIMAL.search(said):
                    log.warning(f"[gemini] {model} doesn't take thinking level minimal; asking at low from now on")
                    self._no_minimal.add(model)
                    continue
                if model not in self._no_detail and _pictures_with_detail(msgs) and _NO_RESOLUTION.search(said):
                    log.warning(f"[gemini] {model} refused a picture's media resolution; sending without it")
                    self._no_detail.add(model)
                    continue
                if not any(k in said.lower() for k in _TURN_SHAPE) or msgs is not f.messages:
                    raise
                # A shape rule nothing above foresaw: the task goes on with the tool turns as text.
                log.warning(f"[gemini] history rejected ({said[:100]}); retrying with tool turns as text")
                msgs = flatten(f.messages)
        else:
            resp = await self._call(request(msgs))

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
        turn.offered = f.offered()
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
