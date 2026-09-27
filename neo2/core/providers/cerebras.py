"""Cerebras: the fast paid brain (decision 49).

Two brains on one account (CEREBRAS_API_KEY, which the user adds to ~/.neo/.env):

- ``cerebras`` — gpt-oss-120b, text only: the fast text brain for steps that need no picture,
  tried ahead of Groq;
- ``cerebras_vision`` — qwen-3.8-27b, which sees: a seeing fallback when Gemini is overloaded,
  tried ahead of Groq's vision brain. At most 2 pictures and 10 MiB per request (the API's
  limits): the newest and the user's own go first, others are left out with a note, and when even
  the newest can't go the request is refused (RequestTooLarge) — a brain that answers as if it saw
  must have seen. An agent's step goes on the context diet (neo2.core.providers.diet) before its
  pictures are counted, and before its size is checked against the rate limits.

Both speak the OpenAI chat-completions protocol at https://api.cerebras.ai/v1
(neo2.core.providers.openai_compat), with these differences:

Speed. The server answers in ~15 ms, but a cold round trip (TCP + TLS) is ~0.9 s. So an agent's
two Cerebras brains share one connection pool (one per agent, per event loop), kept open for two
minutes between steps, and :func:`prewarm_soon` opens it at the wake word (the start of the user's
speech): ``GET /v1/models``, which spends no tokens and also refreshes which models this key may
use. It pays off only when it has that head start — the utterance and its transcript take longer
than the handshake; one started a few ms before the first request saves nothing — and only where
Cerebras is asked: first for chat ("fast"), after Gemini for an agent's steps. A step that starts
while the prewarm is still connecting waits for its connection rather than open a second one.

Thinking. Both models reason inside ``max_tokens`` (``max_tokens=16`` came back with no text and
finish_reason "length"). A request passes ``reasoning_effort`` (the effort the caller asked for: high
for an agent task, low for UI steps) and always leaves room for that thinking (:data:`ROOM`, by
effort) — unused room costs nothing. (The chess log: "high" in 4096 tokens ran qwen out of room twice
in one task, and each time the whole prompt was sent and paid for again.) A reply that runs out of
room anyway is asked once more with low effort and twice the room before it counts as a failure
(Truncated: request-shaped, so the chain tries the next brain without benching this one).

Limits. Every response's ``x-ratelimit-*`` headers (limit / remaining / reset, per window) are kept
per model; a request that would not fit a window is not sent: a wait of up to
:data:`SLEEP_MAX_S` is slept, a longer one is a RateLimited with that wait (the chain fails over,
or sits it out when every brain is down). Nothing is hardcoded: what the server says is the limit.

Money. Each answer's usage × the price table (neo2.core.providers.spend) is added to today's total,
against a daily budget (``cerebras_daily_budget``, $0.50 by default; 0 turns Cerebras off: no
chain names it). Crossing 80% is said once a day; once it's spent, Cerebras is skipped until
midnight, and NEO says so. An invalid key (401) or an account out of credits (402, or a billing
code; billing words only on a 403 or 429) is refused for the day with the reason said; a model
this key may not use (403, 404) takes only that brain out. Said: shown as the turn's progress
line (:meth:`Account.announce`) and heard with the reply (:func:`add_spoken_notes`); in a stop
message, a refusal's own words (``Refused.words``).

Settings (read with defaults, so this module works before and after they are declared in
neo2/core/config.py): ``cerebras_api_key`` (CEREBRAS_API_KEY), ``cerebras_model``,
``cerebras_vision_model``, ``cerebras_daily_budget``, ``cerebras_prices``. An embedded agent's
account, ledger and connections are its own (its BrainPool, its data dir).
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import threading
import time
import weakref
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from neo2.core import logger, runtime
from neo2.core.providers import cerebras_hooks as hooks
from neo2.core.providers import diet, spend
from neo2.core.providers.base import (
    BadRequest,
    Message,
    Overloaded,
    ProviderError,
    QuotaExhausted,
    RateLimited,
    RequestTooLarge,
    ToolSpec,
    Turn,
    Unavailable,
)

# (what the chain, the session and the voice call: kept importable without the openai SDK)
from neo2.core.providers.cerebras_hooks import (  # noqa: F401 — re-exported
    DEFAULT_BUDGET,
    TEXT_BRAIN,
    VISION_BRAIN,
    add_spoken_notes,
    enabled,
    prewarm_soon,
    spoken_notes,
    with_cerebras,
    with_cerebras_eyes,
    with_spoken_notes,
)
from neo2.core.providers.fit import Fitted, estimate
from neo2.core.providers.history import normalize
from neo2.core.providers.openai_compat import OpenAICompatProvider, Truncated, _retry_after

log = logger(__name__)

BASE_URL = "https://api.cerebras.ai/v1"
TEXT_MODEL = "gpt-oss-120b"
VISION_MODEL = "qwen-3.8-27b"
SPOKEN = {TEXT_BRAIN: "Cerebras", VISION_BRAIN: "Cerebras vision"}

# Context windows on a paid key (inference-docs.cerebras.ai); a model not listed gets the smaller
# free-tier window, and a 413 teaches the rest (openai_compat).
CONTEXT = {"gpt-oss-120b": 131_072, "qwen-3.8-27b": 131_072}
DEFAULT_CONTEXT = 65_536
MAX_PICTURES = 2
MAX_REQUEST_BYTES = 10 * 1024 * 1024
_REQUEST_SLACK = 64 * 1024  # JSON framing, tool schemas' escaping: the estimate is an estimate
# Room a reply always gets, by effort: the reasoning is spent from max_tokens before the answer.
ROOM: dict[str, int] = {"low": 8192, "medium": 8192, "high": 16_384}
MAX_ROOM = 32_768  # a retry's room never grows past this (paid-tier output limit: 40K)
_STREAM_REASONING = 256  # a streamed answer's unseen reasoning, when the server reports no usage

KEEPALIVE_S = 120.0  # a warm connection outlives the few seconds between an agent's steps
TIMEOUT = httpx.Timeout(30.0, connect=5.0)  # it answers in well under a second: a hang fails over
PREWARM_TIMEOUT_S = 5.0
PREWARM_IDLE_S = 20.0  # a connection used this recently is still warm: no prewarm
# A step that starts while the prewarm is still connecting waits this long for it (a cold round
# trip is ~0.9 s), then goes ahead on a connection of its own.
PREWARM_JOIN_S = 1.5
_HEARD_MAX = 4  # lines kept for the next reply (each is said once a day at most)
MODELS_TTL_S = 6 * 3600.0
SLEEP_MAX_S = 2.0  # a rate window that reopens this soon is waited for, not failed over
DAILY_S = 3600.0  # a limit that reopens later than this is a daily one: out until then

# The account can't pay: HTTP 402, or an error whose code or type field says so. A billing code
# counts on a 403 or a 429 too, and so do the same words in its message, whole words only: those
# are the statuses a refused account comes with. A 400/422 quotes the request ("desktop update",
# "no credit card") and a 5xx is the server's own trouble: neither ever refuses the account.
_BILLING_CODE = re.compile(
    r"(?:^|_)(?:payment_required|insufficient_(?:funds|credits?|balance|quota)|billing|out_of_credits?"
    r"|credits?_(?:exhausted|depleted))(?:_|$)",
    re.I,
)
_BILLING_WORDS = re.compile(
    r"\b(?:insufficient[_ ](?:funds|credits?|balance|quota)|out of credits?|no (?:remaining )?credits? (?:left|remaining)"
    r"|payment[_ ]required|top[- ]up your (?:account|balance|credits?)|add (?:a )?payment method)\b",
    re.I,
)
# The key itself is refused: a 401, or a 403 whose code or message says it's the key.
_BAD_KEY_CODE = re.compile(r"(?:invalid|wrong|incorrect)_api_key|api_key_(?:invalid|revoked)", re.I)
_BAD_KEY_WORDS = re.compile(
    r"\b(?:invalid[_ ]api[_ ]key|incorrect api key|wrong api key|api key (?:is )?(?:invalid|revoked))\b", re.I
)
KEY_REFUSED = "Cerebras refused the API key (CEREBRAS_API_KEY)"
NO_CREDITS = "Cerebras says the account is out of credits"

# Tests hand in an httpx.MockTransport here (nothing leaves the process), and a fake sleep.
_transport: httpx.AsyncBaseTransport | None = None
_sleep = asyncio.sleep


class Refused(QuotaExhausted):
    """Cerebras is out until `until` for a reason no retry fixes today: the key was refused, the
    account is out of credits, today's budget is spent, or a daily limit is used up. `words`: the
    reason as the user hears it."""

    def __init__(self, msg: str = "", *, until: float, words: str) -> None:
        super().__init__(msg, until=until)
        self.words = words


# ---- settings --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Conf:
    api_key: str = field(default="", repr=False)
    model: str = TEXT_MODEL
    vision_model: str = VISION_MODEL
    daily_budget: float = DEFAULT_BUDGET
    prices: Mapping[str, tuple[float, float]] = field(default_factory=lambda: dict(spend.PRICES))
    data_dir: Path | None = None  # the ledger's folder (None: in memory only)


def conf(s: Any = None) -> Conf:
    """Cerebras' settings from `s` (default: the bound agent's settings, else NEO.app's)."""
    if s is None:
        from neo2.core.config import settings

        s = settings()
    dd = getattr(s, "data_dir", None)
    return Conf(
        api_key=str(getattr(s, "cerebras_api_key", "") or "").strip(),
        model=str(getattr(s, "cerebras_model", "") or TEXT_MODEL).strip(),
        vision_model=str(getattr(s, "cerebras_vision_model", "") or VISION_MODEL).strip(),
        daily_budget=spend.budget(getattr(s, "cerebras_daily_budget", DEFAULT_BUDGET), DEFAULT_BUDGET),
        prices=spend.prices(getattr(s, "cerebras_prices", None)),
        data_dir=Path(dd) if dd else None,
    )


# ---- rate limits -----------------------------------------------------------------------------

_HEADER = re.compile(r"x-ratelimit-(limit|remaining|reset)-(requests|tokens)(?:-([a-z]+))?")
_DURATION = re.compile(r"(?:([\d.]+)h)?(?:([\d.]+)m(?!s))?(?:([\d.]+)s)?(?:([\d.]+)ms)?")


def _seconds(v: str) -> float | None:
    """A reset header's value: seconds ("11.38"), or a duration ("1m2.5s", "250ms")."""
    v = v.strip()
    try:
        f = float(v)
        return f if math.isfinite(f) and f >= 0 else None
    except ValueError:
        pass
    m = _DURATION.fullmatch(v)
    if not v or not m or not any(m.groups()):
        return None
    try:
        h, mi, s, ms = (float(x) if x else 0.0 for x in m.groups())
    except ValueError:
        return None
    return h * 3600 + mi * 60 + s + ms / 1000


@dataclass
class Window:
    limit: float | None = None
    remaining: float | None = None
    reset_at: float | None = None  # epoch seconds


class Limits:
    """One model's rate windows, as the server's x-ratelimit-* headers last described them."""

    def __init__(self) -> None:
        self.windows: dict[tuple[str, str], Window] = {}

    def update(self, headers: Mapping[str, str], now: float) -> None:
        for k, v in headers.items():
            m = _HEADER.fullmatch(k.lower())
            if not m:
                continue
            what, kind, span = m.groups()
            w = self.windows.setdefault((kind, span or ""), Window())
            if what == "reset":
                if (secs := _seconds(str(v))) is not None:
                    w.reset_at = now + secs
                continue
            try:
                n = float(v)
            except ValueError:
                continue
            if math.isfinite(n):
                setattr(w, what, n)

    def wait(self, tokens: int, now: float) -> tuple[float, str] | None:
        """(seconds, which window) until a request of ~`tokens` prompt tokens fits every window
        the server reported; None when it fits now (or nothing was reported)."""
        worst: tuple[float, str] | None = None
        for (kind, span), w in self.windows.items():
            if w.remaining is None or w.reset_at is None or w.reset_at <= now:
                continue
            need = 1 if kind == "requests" else max(1, tokens)
            if w.remaining < need:
                secs = w.reset_at - now
                if worst is None or secs > worst[0]:
                    worst = (secs, f"{kind} per {span or 'minute'}")
        return worst

    def refused_for(self, now: float, tokens: int = 1) -> float | None:
        """After a 429: seconds until every window too small for the refused request (~`tokens`
        prompt tokens) reopens, else the soonest per-minute one; None when the server described
        none. A window that still has room isn't waited for, however low it runs: a day window
        at 8% of 720M tokens takes the next step, and waiting for it would bench a brain for a
        day over a minute's limit."""
        out, minute = [], []
        for (kind, span), w in self.windows.items():
            if w.reset_at is None or w.reset_at <= now:
                continue
            need = 1 if kind == "requests" else max(1, tokens)
            if w.remaining is not None and w.remaining < need:
                out.append(w.reset_at - now)
            elif span in ("", "minute"):
                minute.append(w.reset_at - now)
        if out:
            return max(out)
        return min(minute) if minute else None


# ---- the account -----------------------------------------------------------------------------


class Account:
    """One Cerebras key as one agent uses it: its connections, rate windows, the models it may
    use, today's spend and refusals. Shared by that agent's two Cerebras brains."""

    def __init__(self, c: Conf, *, ledger: spend.Ledger | None = None, base_url: str = BASE_URL) -> None:
        self.conf = c
        self.base_url = base_url
        path = Path(c.data_dir) / spend.FILE if c.data_dir else None
        self.ledger = ledger or spend.Ledger(
            path, daily_budget=c.daily_budget, price_table=c.prices, key=c.api_key
        )
        self.limits: dict[str, Limits] = {}
        self.offered: set[str] | None = None  # model ids /v1/models listed for this key
        self.offered_at = 0.0
        self.missing: dict[str, float] = {}  # model → until: "no such model" (404)
        self.last_io = 0.0  # time.monotonic() of the last response: the connection is warm
        self._asked: dict[str, int] = {}  # model → the last request's size (tokens): what a 429 refused
        self._pending: list[str] = []  # lines for the turn's progress line (bus note)
        self._heard: list[str] = []  # the same lines, for the reply the user hears (spoken_notes)
        self._warming: asyncio.Future | None = None  # the prewarm's GET /v1/models, while in flight
        self._lock = threading.Lock()
        self._loop: Any = None  # the event loop the connections belong to (None: not used yet)
        self._transport: httpx.AsyncBaseTransport | None = None
        self._clients: dict[str, AsyncOpenAI] = {}

    # connections ------------------------------------------------------------------------------
    def client(self, model: str) -> AsyncOpenAI:
        """The client for `model`'s requests. Every model's client shares one connection pool
        (a warm connection serves both brains); a new event loop gets a new pool, since
        connections belong to the loop that opened them."""
        try:
            loop: Any = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        with self._lock:
            if loop is not None and self._loop is not None and self._loop() is not loop:
                self._transport, self._clients, self._loop = None, {}, None
            if loop is not None and self._loop is None:
                self._loop = weakref.ref(loop)
            if self._transport is None:
                self._transport = _transport or httpx.AsyncHTTPTransport(
                    limits=httpx.Limits(
                        max_connections=32, max_keepalive_connections=8, keepalive_expiry=KEEPALIVE_S
                    )
                )
            c = self._clients.get(model)
            if c is None:
                http = httpx.AsyncClient(
                    transport=self._transport,
                    timeout=TIMEOUT,
                    event_hooks={"response": [self._hook(model)]},
                )
                c = self._clients[model] = AsyncOpenAI(
                    base_url=self.base_url,
                    api_key=self.conf.api_key or "none",
                    http_client=http,
                    timeout=TIMEOUT,
                    max_retries=0,  # a 429 fails over at once; a stale connection is retried here
                )
            return c

    def _hook(self, model: str):
        async def on_response(resp: httpx.Response) -> None:
            self.last_io = time.monotonic()
            if model:
                self.limits.setdefault(model, Limits()).update(resp.headers, time.time())

        return on_response

    # what may go ------------------------------------------------------------------------------
    def blocked_until(self, model: str, now: float | None = None) -> float:
        """When this model can be asked again without a refusal or a closed rate window (0: now)."""
        now = time.time() if now is None else now
        t = 0.0
        for scope in ("account", model):
            if (r := self.ledger.refusal(scope)) is not None:
                t = max(t, r[1])
        if self.ledger.over():
            t = max(t, self.ledger.midnight())
        if (lim := self.limits.get(model)) is not None and (w := lim.wait(1, now)) is not None:
            t = max(t, now + w[0])
        t = max(t, self.missing.get(model, 0.0))
        return t if t > now else 0.0

    def refusal(self, model: str, name: str, now: float | None = None) -> Refused | None:
        """Why `model` is out for the day, if it is (no network): a refused key or account, a
        model this key may not use, today's budget spent (said once a day), or a daily rate limit
        used up. None: it may be asked (a short wait or a missing model aside)."""
        now = time.time() if now is None else now
        for scope in ("account", model):
            if (r := self.ledger.refusal(scope)) is not None:
                return Refused(f"{name}: {r[0]}", until=r[1], words=r[0])
        if self.ledger.over():
            b = self.ledger.daily_budget
            words = (
                f"Cerebras' daily budget of {_usd(b)} is used up" if b > 0 else "Cerebras' daily budget is $0"
            )
            if b > 0:  # (a budget of 0 is the user's own off switch: nothing to tell them)
                self.say_once("spent", f"{words}; skipping it until tomorrow")
            return Refused(f"{name}: {words}", until=self.ledger.midnight(), words=words)
        if (lim := self.limits.get(model)) is not None and (w := lim.wait(1, now)) is not None:
            secs, which = w
            if secs > DAILY_S:
                words = f"Cerebras' {which} limit is used up"
                return Refused(f"{name}: {words}", until=now + secs, words=words)
        return None

    async def admit(self, model: str, tokens: int, name: str) -> None:
        """Raise unless a request of ~`tokens` prompt tokens may go to `model` now. Nothing here
        touches the network: every refusal was learnt earlier (or is today's budget)."""
        now = time.time()
        if (r := self.refusal(model, name, now)) is not None:
            raise r
        if self.missing.get(model, 0.0) > now or (self.offered and model not in self.offered):
            raise Unavailable(f"{name}: Cerebras doesn't offer {model} to this key")
        self._asked[model] = tokens
        if (lim := self.limits.get(model)) is not None and (w := lim.wait(tokens, now)) is not None:
            secs, which = w
            if secs <= SLEEP_MAX_S:
                log.info(f"[brain] {name}: waiting {secs:.1f}s for its {which} limit")
                await _sleep(secs)
                return
            if secs > DAILY_S:
                words = f"Cerebras' {which} limit is used up"
                raise Refused(f"{name}: {words}", until=now + secs, words=words)
            raise RateLimited(f"{name}: {which} limit reached", retry_after=secs)

    # errors -----------------------------------------------------------------------------------
    def account_refusal(self, e: APIStatusError, name: str) -> Refused | None:
        """An error that refuses the whole account, refused for the day (and said): the key (a
        401, or a 403 that says it's the key), or credits (a 402, or a billing code or words on a
        403/429). None for anything else: a 400/422 quotes the request, a 5xx is the server's."""
        code, text = e.status_code, _error_text(e)
        fields = [str(x) for x in (getattr(e, "code", None), getattr(e, "type", None)) if x]
        if code == 401 or (
            code == 403 and (any(_BAD_KEY_CODE.search(f) for f in fields) or _BAD_KEY_WORDS.search(text))
        ):
            return self._refuse("account", KEY_REFUSED, name, e)
        if code == 402 or (
            code in (403, 429)
            and (any(_BILLING_CODE.search(f) for f in fields) or _BILLING_WORDS.search(text))
        ):
            return self._refuse("account", NO_CREDITS, name, e)
        return None

    def classify(self, e: APIStatusError, model: str, name: str) -> ProviderError | None:
        """Cerebras' own meaning of an HTTP error (None: the generic mapping applies)."""
        code = e.status_code
        now = time.time()
        if (refused := self.account_refusal(e, name)) is not None:
            return refused
        if code == 403:
            return self._refuse(model, f"Cerebras doesn't let this key use {model}", name, e)
        if code == 404:
            self.missing[model] = now + MODELS_TTL_S
            return Unavailable(f"{name}: Cerebras doesn't offer {model} to this key")
        if code == 429:
            waits = []
            lim = self.limits.get(model)
            if lim is not None and (w := lim.refused_for(now, self._asked.get(model, 1))) is not None:
                waits.append(w)
            if (ra := _retry_after(e)) is not None:
                waits.append(ra)
            wait = max(waits) if waits else None
            if wait is not None and wait > DAILY_S:
                words = "Cerebras' daily limit is used up"
                return Refused(f"{name}: {e}", until=now + wait, words=words)
            return RateLimited(f"{name}: {e}", retry_after=wait)
        if code >= 500:
            return Overloaded(f"{name}: {e}", retry_after=_retry_after(e))
        return None

    def _refuse(self, scope: str, words: str, name: str, e: Exception) -> Refused:
        until = self.ledger.refuse(scope, words)
        self.say_once(f"refused:{scope}", f"{words}; skipping it for today")
        if scope == "account" and self.ledger.path is not None and not runtime.library_mode():
            from neo2.core import health  # NEO.app's problem list: something only the user can fix

            health.report("cerebras", f"{words}. NEO skips Cerebras until midnight (Settings & API Keys…).")
        log.warning(
            f"[brain] {name}: {words} (HTTP {getattr(e, 'status_code', '?')}); refused until midnight"
        )
        return Refused(f"{name}: {words}", until=until, words=words)

    # money ------------------------------------------------------------------------------------
    def spent(self, model: str, tokens_in: int, tokens_out: int) -> None:
        crossed = self.ledger.add(model, tokens_in, tokens_out)
        b = self.ledger.daily_budget
        if "warn" in crossed and "spent" not in crossed:
            self._tell(f"Cerebras has used {_usd(self.ledger.today())} of today's {_usd(b)} budget")
        if "spent" in crossed:
            self._tell(f"Cerebras' daily budget of {_usd(b)} is used up; I'll skip it until tomorrow")

    # what the user hears ----------------------------------------------------------------------
    def say_once(self, tag: str, text: str) -> None:
        """Tell the user `text`, once a day per `tag`."""
        if self.ledger.mark_said(tag):
            self._tell(text)

    def _tell(self, text: str) -> None:
        """Queue `text` twice: for the turn's progress line (:meth:`announce`) and for the reply
        the user hears (:func:`spoken_notes`, which the session adds to it). A host's own provider
        called with no agent bound has no one to hear it: only logged."""
        self._pending.append(text)
        if runtime.current_scope() is None and runtime.library_mode():
            return
        with self._lock:
            if text not in self._heard:
                self._heard = [*self._heard, text][-_HEARD_MAX:]

    def take_heard(self) -> list[str]:
        """What's to be said out loud since the last call (drained)."""
        with self._lock:
            out, self._heard = self._heard, []
        return out

    async def announce(self) -> None:
        """Show what's queued (the budget warning, a refusal) on the agent's own bus: the turn's
        progress line, as every brain switch is. Never to another agent's or NEO.app's. (What the
        user hears is the reply's: :func:`spoken_notes`.)"""
        while self._pending:
            text = self._pending.pop(0)
            log.warning(f"[brain] {text}")
            if runtime.current_scope() is None and runtime.library_mode():
                continue  # a host's own provider, called with no agent bound: nobody to tell
            try:
                from neo2.core.events import bus

                await bus().note(text)
            except Exception as e:  # noqa: BLE001 — a note must never fail the answer
                log.info(f"[brain] couldn't show a Cerebras note: {type(e).__name__}")

    # warming up -------------------------------------------------------------------------------
    async def prewarm(self, *, idle_s: float = PREWARM_IDLE_S) -> bool:
        """Open (or keep open) the connection before it's needed, and refresh which models this
        key may use. Free: /v1/models spends no tokens. False when nothing was done. A step that
        starts while it's still connecting waits for it (:meth:`join_prewarm`) rather than open
        a second, cold connection next to it."""
        if self.ledger.refusal("account") is not None or self.ledger.over():
            return False
        fresh = self.offered is not None and time.time() - self.offered_at < MODELS_TTL_S
        if time.monotonic() - self.last_io < idle_s and fresh:
            return False
        if self._warming is not None and not self._warming.done():
            return False  # one is already on its way
        call = asyncio.ensure_future(self.client("").models.list())
        self._warming = call
        try:
            page = await asyncio.wait_for(call, PREWARM_TIMEOUT_S)
        except APIStatusError as e:
            # Only what's about the account is learnt here (a refused key, no credits): the models
            # list isn't a model's request, and its 403/404/429/5xx say nothing about one.
            if self.account_refusal(e, TEXT_BRAIN) is None:
                log.info(f"[brain] cerebras prewarm: /v1/models answered {e.status_code}; nothing learnt")
            await self.announce()
            return False
        except Exception as e:  # noqa: BLE001 — a failed warm-up costs nothing; the request will tell
            log.info(f"[brain] cerebras prewarm failed: {type(e).__name__}")
            return False
        finally:
            if self._warming is call:
                self._warming = None
        ids = {str(getattr(m, "id", "")) for m in getattr(page, "data", None) or []} - {""}
        if ids:
            self.offered, self.offered_at = ids, time.time()
        return True

    async def join_prewarm(self) -> None:
        """A prewarm still connecting on this event loop: wait for it (at most
        :data:`PREWARM_JOIN_S`), then this request takes its connection — the rest of one
        handshake instead of a whole second one."""
        call = self._warming
        if call is None or call.done():
            return
        try:
            if call.get_loop() is not asyncio.get_running_loop():
                return
        except RuntimeError:
            return
        await asyncio.wait({call}, timeout=PREWARM_JOIN_S)  # (never raises; never cancels it)


def _usd(x: float) -> str:
    """Dollars as said: "$0.42" (a budget under a cent keeps its digits: "$0.0010")."""
    return f"${x:.2f}" if x == 0 or x >= 0.01 else f"${x:.4f}"


def _error_text(e: APIStatusError) -> str:
    body = getattr(e, "body", None)
    try:
        extra = json.dumps(body) if body is not None else ""
    except (TypeError, ValueError):
        extra = str(body)
    return f"{e} {extra}"


# ---- one agent's account ---------------------------------------------------------------------


def account(c: Conf | None = None) -> Account:
    """The bound agent's Cerebras account (its BrainPool's), else NEO.app's; made on first use,
    and again when the settings changed (a new key)."""
    from neo2.core.providers import pool

    c = c or conf()
    if not c.api_key:
        raise Unavailable("CEREBRAS_API_KEY is not set")
    owner = pool()
    with hooks._accounts_lock:
        a = hooks._accounts.get(owner)
        if a is None or a.conf != c:
            a = hooks._accounts[owner] = Account(c)
        return a


# ---- the brains ------------------------------------------------------------------------------


class CerebrasProvider(OpenAICompatProvider):
    raise_truncated = True
    stream_effort = "low"  # a chat answer: quick thinking
    stream_usage = True  # (off for this brain once the server refuses the option: see stream)

    def __init__(self, acct: Account, *, model: str, name: str, vision: bool) -> None:
        super().__init__(
            name=name,
            model=model,
            base_url=acct.base_url,
            supports_vision=vision,
            reasoning_effort=True,
            token_budget=CONTEXT.get(model, DEFAULT_CONTEXT),
            client=acct.client(model),
        )
        self.account = acct
        # what one request may carry (providers.image_limit reads these): only the seeing brain
        self.max_images = MAX_PICTURES if vision else 0
        self.max_request_bytes = MAX_REQUEST_BYTES

    def client(self) -> AsyncOpenAI:
        return self.account.client(self.model)

    def cooling_until(self) -> float:
        now = time.time()
        t = max(super().cooling_until(), self.account.blocked_until(self.model, now))
        return t if t > now else 0.0

    def cooling_reason(self) -> Refused | None:
        """Why :meth:`cooling_until` is hours away, when it's because Cerebras is out for the day
        (a refused key or account, today's budget spent, a daily limit used up): the error to
        report for it, in its own words — not "no time left" (a run's budget, which the chain
        checks against cooling_until). None when it's a short wait, or none."""
        return self.account.refusal(self.model, self.name)

    def _error(self, e: Exception) -> ProviderError:
        if (
            isinstance(e, APIStatusError)
            and (mine := self.account.classify(e, self.model, self.name)) is not None
        ):
            if isinstance(mine, RateLimited) and mine.retry_after:
                self._cool_until = time.time() + mine.retry_after
            return _without_key(mine, self.account.conf.api_key)
        return _without_key(super()._error(e), self.account.conf.api_key)

    def _spent(self, usage: Any, f: Fitted, text: str = "") -> None:
        tin = getattr(usage, "prompt_tokens", None)
        tout = getattr(usage, "completion_tokens", None)
        if not isinstance(tin, int):
            tin = f.tokens  # the estimate of what was sent
        if not isinstance(tout, int):
            # a stream that said nothing spent its room thinking (Truncated); else its text plus some
            tout = (
                math.ceil(len(text.encode("utf-8", "ignore")) / 3) + _STREAM_REASONING
                if text
                else f.max_tokens
            )
        try:
            self.account.spent(self.model, tin, tout)
        except Exception as e:  # noqa: BLE001 — counting must never lose the answer it counts
            log.warning(f"[brain] {self.name}: couldn't count its spend ({type(e).__name__})")

    def _prepare(self, messages: list[Message], system: str, tools: list[ToolSpec] | None) -> list[Message]:
        if not self.supports_vision:
            return messages
        room = self.max_request_bytes - _text_bytes(messages, system, tools) - _REQUEST_SLACK
        return cap_pictures(messages, most=self.max_images, max_bytes=room, name=self.name)

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        effort: Literal["low", "medium", "high"] = "medium",
        max_tokens: int = 4096,
        **extra: Any,  # whatever else the chain passes (brain-effort's seams): handed on as is
    ) -> Turn:
        effort = effort if effort in ROOM else "low"  # (the values both models take)
        try:
            if tools is not None:  # an agent's step: what it sends is what's counted (fit diets it again: a no-op)
                messages = diet.diet(normalize(messages))
            messages = self._prepare(messages, system, tools)
            await self.account.join_prewarm()
            await self.account.admit(
                self.model, estimate(messages, system, tools, vision=self.supports_vision), self.name
            )
            room = max(max_tokens, ROOM[effort])
            retried: set[str] = set()
            while True:
                try:
                    return await super().complete(
                        messages, system=system, tools=tools, effort=effort, max_tokens=room, **extra
                    )
                except Truncated:
                    if "room" in retried:
                        raise
                    retried.add("room")
                    effort, room = "low", min(room * 2, MAX_ROOM)
                    log.info(
                        f"[brain] {self.name}: no room left for the answer; again with {room} tokens, low effort"
                    )
                except Unavailable as e:
                    if "connection" in retried or not _stale(e):
                        raise
                    retried.add("connection")  # a kept-alive connection the server had closed
                except RateLimited as e:
                    wait = e.retry_after
                    if (
                        "rate" in retried
                        or isinstance(e, QuotaExhausted)
                        or wait is None
                        or wait > SLEEP_MAX_S
                    ):
                        raise
                    retried.add("rate")
                    await _sleep(wait)
        finally:
            await self.account.announce()

    async def stream(
        self, messages: list[Message], *, system: str = "", max_tokens: int = 2048
    ) -> AsyncIterator[str]:
        try:
            messages = self._prepare(messages, system, None)
            await self.account.join_prewarm()
            await self.account.admit(
                self.model, estimate(messages, system, None, vision=self.supports_vision), self.name
            )
            room = max(max_tokens, ROOM["low"])
            produced = False
            try:
                async for chunk in super().stream(messages, system=system, max_tokens=room):
                    produced = True
                    yield chunk
            except BadRequest as e:
                # A server that won't take stream_options (its usage then comes unasked, or is
                # estimated): asked once more without them, and they stay off when that goes
                # through. A 400 that comes again is the request's own: raised, the option kept.
                if produced or type(e) is not BadRequest or not self.stream_usage:
                    raise
                self.stream_usage = False
                try:
                    async for chunk in super().stream(messages, system=system, max_tokens=room):
                        produced = True
                        yield chunk
                except BadRequest:
                    self.stream_usage = not produced
                    raise
                log.info(f"[brain] {self.name}: the server refused stream_options; its usage is estimated")
            if not produced:  # its thinking used all the room: nothing was said, another brain may say it
                raise Truncated(f"{self.name}: no room left for the answer")
        finally:
            await self.account.announce()


def _without_key(e: ProviderError, key: str) -> ProviderError:
    """`e` with the key cut out of its message, should a server have echoed it back: the chain logs
    a failure's message and a stop message may say it."""
    if key and any(key in str(a) for a in e.args):
        e.args = tuple(str(a).replace(key, "[CEREBRAS_API_KEY]") for a in e.args)
    return e


def _stale(e: ProviderError) -> bool:
    """A connection error that isn't a timeout: most likely a kept-alive connection the server
    had closed just as it was reused. Asked again at once, on a new one."""
    cause = e.__cause__
    return isinstance(cause, APIConnectionError) and not isinstance(cause, APITimeoutError)


def _text_bytes(messages: list[Message], system: str, tools: list[ToolSpec] | None) -> int:
    n = len(system.encode("utf-8", "ignore"))
    for m in messages:
        n += len(m.text.encode("utf-8", "ignore"))
        n += sum(len(r.content.encode("utf-8", "ignore")) for r in m.tool_results)
        n += sum(len(json.dumps(c.args, ensure_ascii=False).encode()) for c in m.tool_calls)
    for t in tools or ():
        n += len(t.description.encode("utf-8", "ignore")) + len(json.dumps(t.parameters).encode())
    return n


def _b64(n: int) -> int:
    return 4 * math.ceil(n / 3)


def cap_pictures(
    messages: list[Message], *, most: int, max_bytes: int, name: str = VISION_BRAIN
) -> list[Message]:
    """`messages` with at most `most` pictures whose base64 fits in `max_bytes`: the newest, then the
    user's own (the photo a request is about), then the newest of the rest; the others are left out
    with a note. RequestTooLarge when even the newest can't go: a brain that sees must not answer
    about a picture it never got. Never mutates `messages`."""
    out = list(messages)
    note = f" [older picture left out: this brain takes at most {most} per request]"
    slots: list[tuple[int, int, int]] = []  # (turn, result or -1 for the turn's own, picture): newest first
    for i in range(len(out) - 1, -1, -1):
        m = out[i]
        for j in range(len(m.tool_results) - 1, -1, -1):
            slots += [(i, j, k) for k in range(len(m.tool_results[j].images) - 1, -1, -1)]
        slots += [(i, -1, k) for k in range(len(m.images) - 1, -1, -1)]

    def image(s: tuple[int, int, int]):
        i, j, k = s
        return (out[i].images if j < 0 else out[i].tool_results[j].images)[k]

    users = [s for s in slots[1:] if s[1] < 0 and out[s[0]].role == "user"]
    kept: set[tuple[int, int, int]] = set()
    size = 0
    for s in slots[:1] + users + [s for s in slots[1:] if s not in users]:
        b = _b64(len(image(s).data))
        if len(kept) < most and size + b <= max_bytes:
            kept.add(s)
            size += b
        elif s == slots[0]:
            raise RequestTooLarge(f"{name}: the picture is too large to send ({b / 1048576:.1f} MiB as base64)")
    for i, m in enumerate(out):
        results = [
            replace(r, content=r.content + note, images=ims)
            if len(ims := [im for k, im in enumerate(r.images) if (i, j, k) in kept]) < len(r.images)
            else r
            for j, r in enumerate(m.tool_results)
        ]
        if any(a is not b for a, b in zip(results, m.tool_results, strict=True)):
            m = replace(m, tool_results=results)
        if len(ims := [im for k, im in enumerate(m.images) if (i, -1, k) in kept]) < len(m.images):
            m = replace(m, text=m.text + note, images=ims)
        out[i] = m
    return out


def cerebras(model: str | None = None) -> CerebrasProvider:
    """The text brain (gpt-oss-120b by default) on the bound agent's account."""
    c = conf()
    if not c.api_key:
        raise Unavailable("CEREBRAS_API_KEY is not set")
    return CerebrasProvider(account(c), model=model or c.model, name=TEXT_BRAIN, vision=False)


def cerebras_vision(model: str | None = None) -> CerebrasProvider:
    """The seeing brain (qwen-3.8-27b by default) on the bound agent's account."""
    c = conf()
    if not c.api_key:
        raise Unavailable("CEREBRAS_API_KEY is not set")
    return CerebrasProvider(account(c), model=model or c.vision_model, name=VISION_BRAIN, vision=True)


def hosted(
    api_key: str,
    *,
    vision: bool = False,
    model: str | None = None,
    daily_budget: float = DEFAULT_BUDGET,
    prices: Any = None,
) -> CerebrasProvider:
    """A host's own Cerebras brain (neo2.providers): its account, spend and limits live in the
    object, in memory only (never a file, a global or ~/.neo)."""
    if not api_key:
        raise ProviderError("Cerebras needs an API key")
    c = Conf(
        api_key=api_key, daily_budget=spend.budget(daily_budget, DEFAULT_BUDGET), prices=spend.prices(prices)
    )
    return CerebrasProvider(
        Account(c),
        model=model or (VISION_MODEL if vision else TEXT_MODEL),
        name=VISION_BRAIN if vision else TEXT_BRAIN,
        vision=vision,
    )


async def prewarm() -> bool:
    """Open the connection to Cerebras before a request needs it (a cold round trip is ~0.9 s of
    it). Nothing when Cerebras isn't :func:`enabled`, after a refusal, or when the connection is
    still warm; never raises."""
    try:
        c = conf()
        if not c.api_key or c.daily_budget <= 0:
            return False
        return await account(c).prewarm()
    except Exception as e:  # noqa: BLE001 — a warm-up must never fail the task it's for
        log.info(f"[brain] cerebras prewarm skipped: {type(e).__name__}")
        return False


def status() -> dict[str, Any]:
    """Today's Cerebras spend, for `python -m neo2 --doctor` (no network)."""
    c = conf()
    if not c.api_key:
        return {"key": False}
    a = account(c)
    r = a.ledger.refusal("account")
    return {
        "key": True,
        "today_usd": round(a.ledger.today(), 4),
        "budget_usd": a.ledger.daily_budget,
        "refused": r[0] if r else "",
    }
