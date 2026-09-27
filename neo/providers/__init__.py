"""Brain factory + failover chain.

``brain("agent")`` returns the provider chain for agentic turns, ``brain("fast")`` for
chat-only turns. A chain tries providers in order and moves on when one is rate-limited or
unreachable, so a free-tier 429 degrades gracefully instead of failing the turn.

The brains made so far, and which of them failed lately, live in a :class:`BrainPool`: NEO.app's
(the module's `_cache` and `_demoted`), or an embedded agent's own (neo.runtime) — built from
that agent's keys, and with the host's own providers for any purpose it supplied them for.

A step that works from a picture (a screenshot the run took, a photo the user sent) goes only to
brains that can see it: the chain's own, then the seeing fallbacks (:data:`EYES`) — never to a
text-only cloud brain, which would act blind. When none can answer, the run stops and says so
(NoEyes). How hard each step thinks follows the run (neo.providers.effort), and inside an agent run
the chain keeps to the time the run has left (neo.providers.budget).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time as _time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from neo import runtime
from neo.config import BrainName, settings
from neo.providers import budget
from neo.providers.base import (
    BLIND_REFUSAL,
    AllBrainsFailed,
    Blind,
    Level,
    Message,
    NoEyes,
    OutOfTime,
    Provider,
    ProviderError,
    RateLimited,
    ToolSpec,
    TooSlow,
    Turn,
    spoken_name,
)
from neo.providers.effort import Effort, for_brain

__all__ = [
    "EYES",
    "AllBrainsFailed",
    "BrainPool",
    "Chain",
    "Effort",
    "ImageLimit",
    "NoEyes",
    "OutOfTime",
    "ProviderError",
    "RateLimited",
    "brain",
    "image_limit",
    "make",
    "pool",
]

log = logging.getLogger(__name__)

Purpose = Literal["agent", "fast", "vision", "offline", "light"]

_cache: dict[str, Provider] = {}


class BrainPool:
    """One agent's brains: those made from its settings (by name), how recently each failed, and
    the host's own providers per purpose (`overrides`), used instead of the named brains.

    What a brain learns on the way (a Gemini key's quota window or refusal, a model's overload, a
    server's real request size) is kept by the provider object itself, so the brains made here —
    and their file in the agent's data dir (brain_cooldowns.json) — are this agent's alone. A
    host's own provider object keeps what it learns in itself: one object handed to two agents is
    shared by both."""

    def __init__(self, overrides: Mapping[Purpose, Sequence[Provider]] | None = None) -> None:
        self._made: dict[str, Provider] = {}
        self._failed: dict[str, float] = {}
        self.overrides: dict[str, list[Provider]] = {k: list(v) for k, v in (overrides or {}).items() if v}

    def cache(self) -> dict[str, Provider]:
        return self._made

    def demoted(self) -> dict[str, float]:
        return self._failed

    def make(self, name: BrainName) -> Provider:
        c = self.cache()
        if name not in c:
            c[name] = _build(name)
        return c[name]


class _AppPool(BrainPool):
    """NEO.app's pool: the module's `_cache` and `_demoted`, looked up at call time."""

    def cache(self) -> dict[str, Provider]:
        return _cache

    def demoted(self) -> dict[str, float]:
        return _demoted


_app_pool = _AppPool()


def pool() -> BrainPool:
    """The bound agent's brains (neo.runtime), else NEO.app's."""
    rt = runtime.current_runtime()
    if rt is not None:
        return rt.get_or_create("brains", BrainPool)
    runtime.check_unbound("the brains")
    return _app_pool


def make(name: BrainName) -> Provider:
    return pool().make(name)


def _build(name: BrainName) -> Provider:
    if name == "gemini":
        from neo.providers.gemini import GeminiProvider

        p: Provider = GeminiProvider()
    elif name == "groq":
        from neo.providers.openai_compat import groq

        p = groq()
    elif name == "gemini_lite":
        from neo.providers.gemini import GeminiProvider

        p = GeminiProvider(model=settings().gemini_lite_model, name="gemini_lite")  # its own brain
    elif name == "local":
        from neo.providers.openai_compat import local

        p = local()
    elif name == "claude":
        from neo.providers.claude import ClaudeProvider

        p = ClaudeProvider()
    elif name == "groq_vision":
        p = _groq_vision()
    else:
        raise ProviderError(f"unknown brain {name}")
    return p


# Groq's free tier, per model: tokens a minute (prompt and answer together) — one request's most.
_GROQ_VISION_TOKENS = 8000


def _groq_vision() -> Provider:
    """Groq's vision model (settings.groq_vision_model): the last seeing fallback. Its own brain —
    its own rate limit and demotion — on the same key as the text one. It takes pictures (a few per
    request, ~2K tokens each), so its budget is a per-minute rate every step pays back."""
    from neo.providers.base import Unavailable
    from neo.providers.openai_compat import OpenAICompatProvider

    s = settings()
    if not s.groq_api_key:
        raise Unavailable("GROQ_API_KEY is not set")
    if not s.groq_vision_model:
        raise Unavailable("no Groq vision model is set (NEO_GROQ_VISION_MODEL)")
    return OpenAICompatProvider(
        name="groq_vision",
        base_url="https://api.groq.com/openai/v1",
        api_key=s.groq_api_key,
        model=s.groq_vision_model,
        supports_vision=True,
        # Not its reasoning switch: a vision model's values differ from gpt-oss's low/medium/high.
        reasoning_effort=False,
        token_budget=_GROQ_VISION_TOKENS,
        per_minute=True,
    )


_DEMOTE_S = 300.0  # a brain that just failed goes to the back of the line for this long
# Every brain failed but one only asked for a short wait: wait, then ask it again. A per-minute
# limit is always clear within the minute, and on Groq's 8K a step of 4–6K tokens is ~30–45s of it:
# the one brain left carries a long task through that wait rather than ending it.
_WAIT_MAX_S = 65.0

# Shared across chains: once Gemini has fallen over in one request, the next request (a fresh
# Chain each time) should not pay the same failed round trips before reaching Groq. NEO.app's;
# an embedded agent's pool keeps its own.
_demoted: dict[str, float] = {}

# A request-shaped failure says nothing about the brain's health: the next request may be fine.
_NO_DEMOTE = ("bad_request", "too_large")


def _cooling(p: Provider) -> float:
    """When a brain that knows it's spent (every Gemini key out of quota) can be asked again."""
    try:
        return float(getattr(p, "cooling_until", lambda: 0.0)() or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


# A brain slower than this to answer one step: the user hears that NEO is waiting on it.
_PATIENCE_S = 20.0
# What a brain needs to answer once a rate-limit wait is over: a wait that leaves less of the run's
# time than this isn't worth starting.
_ANSWER_S = 3.0


def _too_late(p: Provider) -> TooSlow | None:
    """Within a run: the reason not to ask this brain now — the run's time is up, or the wait it
    asked for (a spent per-minute budget) ends after it. None: ask it."""
    left = budget.left()
    if left is None:
        return None
    if left <= 0:
        return TooSlow("the run's time is up")
    wait = _cooling(p) - _time.time()
    if wait > left:
        return TooSlow(f"{p.name} asked to wait {wait:.0f}s, past the {left:.0f}s left")
    return None


async def _patience(name: str) -> None:
    await asyncio.sleep(_PATIENCE_S)
    from neo.events import bus

    await bus().note(f"Still waiting on {spoken_name(name)}…")


class Chain:
    """Ordered providers with automatic failover on RateLimited / ProviderError.

    A provider that fails is *demoted* for a few minutes (a spent quota: until it resets): still
    available, but tried after the ones that work. `sticky=False` keeps the given order regardless
    (the vision chain must not let a text-only brain answer first and silently drop the screenshot).
    When every brain fails, the error is an AllBrainsFailed that says, per brain, why — quota,
    too large, not running — so the task can end with that reason spoken. Demotions are the
    agent's own (its BrainPool), NEO.app's module `_demoted` when nothing is bound — except in a
    chain of a host's own providers that the host calls itself (neo2.providers.Failover, no agent
    bound): that one keeps them itself and touches nothing of NEO.app's.

    `eyes`: the seeing brains a step working from a picture may also ask, after the chain's own
    ones that see (an agent chain's screenshot while Gemini is overloaded goes to Flash-Lite or
    Groq's vision model, not to Groq's text model). `effort`: the run's thinking policy — each
    step's level is read from the transcript (neo.providers.effort); without one, the level the
    caller passes is used as it is."""

    def __init__(
        self,
        names: Sequence[BrainName | Provider],
        *,
        sticky: bool = True,
        eyes: Sequence[BrainName] = (),
        effort: Effort | None = None,
    ) -> None:
        self.names = list(names)  # brain names, or provider instances (a host's own), in order
        self.sticky = sticky
        self.eyes = [n for n in eyes if n not in self.names]
        self.effort = effort
        self.last_used: str = ""
        self.last_model: str = ""
        self.last_effort: str = ""
        self.levels: list[str] = []  # the level of each step asked through this chain (the A/B's record)
        self._missing: list[tuple[str, ProviderError]] = []
        self._own_demoted: dict[str, float] = {}

    def _hosts_only(self) -> bool:
        """Only a host's own provider objects: no brain here is made from anyone's settings."""
        return all(not isinstance(n, str) for n in self.names)

    def _alone(self) -> bool:
        """A host's own chain, called by the host itself with no agent bound (not NEO.app's)."""
        return self._hosts_only() and runtime.current_scope() is None

    def _demotions(self) -> dict[str, float]:
        return self._own_demoted if self._alone() else pool().demoted()

    def _providers(self) -> list[Provider]:
        out: list[Provider] = []
        self._missing = []
        for n in self._order():
            if not isinstance(n, str):
                out.append(n)
                continue
            try:
                out.append(make(n))
            except ProviderError as e:  # missing key → skip silently
                log.info(f"[brain] {n} unavailable: {e}")
                self._missing.append((n, e))
        if self.sticky:
            # A brain that knows it's spent (all its quota) goes last: the next one is asked at once.
            now = _time.time()
            out.sort(key=lambda p: _cooling(p) > now)
        return out

    def _order(self) -> list[BrainName | Provider]:
        if not self.sticky:
            return list(self.names)
        now = _time.time()
        demoted = self._demotions()
        healthy = [n for n in self.names if demoted.get(_key(n), 0.0) <= now]
        return healthy + [n for n in self.names if n not in healthy]

    def _demote(self, name: str, e: ProviderError | None = None) -> None:
        if e is not None and getattr(e, "kind", "") in _NO_DEMOTE:
            return
        until = _time.time() + _DEMOTE_S
        if e is not None and getattr(e, "kind", "") == "quota":
            until = max(until, float(getattr(e, "until", 0.0) or 0.0))
        self._demotions()[name] = until

    @property
    def name(self) -> str:
        return "+".join(_key(n) for n in self.names)

    @property
    def supports_vision(self) -> bool:
        return any(p.supports_vision for p in self._providers()) or any(
            getattr(p, "supports_vision", False) for p in self._eye_brains()
        )

    supports_tools = True

    def _eye_brains(self) -> list[Provider]:
        """The seeing fallbacks that are set up (a missing key: skipped, as in _providers)."""
        out: list[Provider] = []
        for n in self.eyes:
            try:
                p = make(n)
            except Exception as e:  # noqa: BLE001 — not set up (no key, no package): it can't look
                log.info(f"[brain] {n} unavailable: {e}")
                continue
            if getattr(p, "supports_vision", False):
                out.append(p)
        return out

    def _for_pixels(self, providers: list[Provider], failures: list[tuple[str, ProviderError]]) -> list[Provider]:
        """Who may answer a step working from a picture: the brains that can see it — the chain's
        own first, then its seeing fallbacks — and a text-only brain on this Mac only where the user
        put it ahead of a cloud brain (local-first: the screen stays on the Mac while a brain here
        can answer; the fallbacks, cloud brains all, come after it). Every other text-only brain is
        left out, as Blind: it would act on a picture it never saw. A host's own chain keeps its
        brains as the host gave them."""
        if self._hosts_only():
            return _eyes_first(providers)
        have = {p.name for p in providers}
        extra = [p for p in self._eye_brains() if p.name not in have]

        def cloud(q: Provider) -> bool:
            return not getattr(q, "on_device", False)

        # A brain here guards the screen where the user put it before a cloud brain, and a cloud
        # brain that sees would come after it (one of the chain's own, or a fallback).
        guards = {
            p.name
            for i, p in enumerate(providers)
            if not cloud(p)
            and any(cloud(q) for q in providers[i + 1 :])
            and any(cloud(q) and q.supports_vision for q in providers[i + 1 :] + extra)
        }
        out: list[Provider] = []
        for p in _eyes_first(providers):
            if getattr(p, "supports_vision", False) or p.name in guards:
                out.append(p)
            else:
                failures.append((p.name, Blind(f"{p.name}: this model has no vision")))
        now = _time.time()
        return out + sorted(extra, key=lambda p: _cooling(p) > now)  # (stable: EYES order otherwise)

    def _no_brain(self, failures: list[tuple[str, ProviderError]], *, pixels: bool = False) -> ProviderError:
        if any(getattr(e, "kind", "") == "budget" for _, e in failures):
            return OutOfTime(failures, limit_s=budget.limit())
        if self._hosts_only():  # no key of NEO's could have been missing
            return AllBrainsFailed(failures + self._missing)
        if pixels and any(getattr(e, "kind", "") != "blind" for _, e in failures):
            return NoEyes(failures)  # a brain that sees was asked and couldn't answer
        s = settings()
        if not pool().overrides and not (s.gemini_api_key or s.groq_api_key or s.anthropic_api_key):
            return ProviderError(
                "no API key set — add GEMINI_API_KEY to .env (free at aistudio.google.com/apikey)"
            )
        if pixels:
            return NoEyes(failures or self._missing)
        return AllBrainsFailed(failures + self._missing)

    async def _failed(self, p: Provider, e: ProviderError, failures: list[tuple[str, ProviderError]]) -> None:
        log.warning(f"[brain] {p.name} failed ({type(e).__name__}: {str(e)[:80]}); trying next")
        failures.append((p.name, e))
        if self.sticky:
            self._demote(p.name, e)
        if self._alone():
            return  # no conversation to tell (and never NEO.app's)
        from neo.events import bus

        who = spoken_name(p.name)  # "Gemini Lite", "the local model"
        await bus().note(f"{who[:1].upper()}{who[1:]} unavailable, switching brain")

    @staticmethod
    def _short_wait(failures: list[tuple[str, ProviderError]]) -> tuple[float, str] | None:
        """(seconds, brain) when some brain failed only for a moment (a per-minute limit, a busy
        spell) and asked us to wait no longer than _WAIT_MAX_S."""
        waits = [
            (float(w), n)
            for n, e in failures
            if (w := getattr(e, "retry_after", None)) is not None and 0 <= float(w) <= _WAIT_MAX_S
        ]
        return min(waits) if waits else None

    def level(self, messages: list[Message], asked: Any = "medium") -> Level:
        """This step's thinking level: the run's policy's, else what the caller asked for."""
        lv = self.effort.at(messages) if self.effort is not None else asked
        lv = lv if lv in ("minimal", "low", "medium", "high") else "medium"
        self.last_effort = lv
        self.levels.append(lv)
        return lv

    async def _ask(
        self, p: Provider, messages: list[Message], system: str, tools: list[ToolSpec] | None, level: str,
        max_tokens: int,
    ) -> Turn:  # fmt: skip
        """One brain's answer — within the run's time left, when a run is bound (neo.providers.budget):
        a call still out when it's up is given up on (TooSlow). A call slower than _PATIENCE_S says
        so to the user instead of leaving them waiting in silence."""
        call = p.complete(messages, system=system, tools=tools, effort=for_brain(p, level), max_tokens=max_tokens)
        left = budget.left()
        note = None if self._alone() else asyncio.ensure_future(_patience(p.name))
        try:
            if left is None:
                return await call
            try:
                return await asyncio.wait_for(call, timeout=max(0.0, left))
            except TimeoutError:
                raise TooSlow(f"{p.name}: no answer before the run's time was up") from None
        finally:
            if note is not None:
                note.cancel()

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        effort: Any = "medium",
        max_tokens: int = 4096,
    ) -> Turn:
        level = self.level(messages, effort)
        failures: list[tuple[str, ProviderError]] = []
        providers = self._providers()
        pixels = _needs_eyes(messages)
        if pixels:
            providers = self._for_pixels(providers, failures)
        for p in providers:
            if (late := _too_late(p)) is not None:
                failures.append((p.name, late))
                continue
            try:
                turn = await self._ask(p, messages, system, tools, level, max_tokens)
                self.last_used, self.last_model = p.name, getattr(p, "model", "")
                return turn
            except TooSlow as e:  # the run's time is up: no other brain could answer in none
                failures.append((p.name, e))
                raise self._no_brain(failures, pixels=pixels) from None
            except ProviderError as e:
                await self._failed(p, e, failures)
        if (wait := self._short_wait(failures)) is not None:
            # Every brain is out, but one only for seconds (Groq's per-minute budget): a long
            # task is worth that pause rather than ending here — unless it would end past the
            # run's time (then the run stops now, and says why).
            secs, name = wait
            left = budget.left()
            if left is not None and secs + _ANSWER_S > left:
                log.warning(f"[brain] every brain failed; {name}'s {secs:.0f}s wait is past the {left:.0f}s left")
                raise OutOfTime(failures, limit_s=budget.limit())
            p = next(p for p in providers if p.name == name)
            log.warning(f"[brain] every brain failed; waiting {secs:.0f}s for {name}")
            from neo.events import bus

            await bus().note(f"Waiting {secs:.0f}s for {spoken_name(name)}'s rate limit")
            await asyncio.sleep(secs + 0.5)
            try:
                turn = await self._ask(p, messages, system, tools, level, max_tokens)
                self.last_used, self.last_model = p.name, getattr(p, "model", "")
                return turn
            except TooSlow as e:
                failures.append((p.name, e))
            except ProviderError as e:
                failures = [(n, x) for n, x in failures if n != name]
                await self._failed(p, e, failures)
        raise self._no_brain(failures, pixels=pixels)

    async def stream(
        self, messages: list[Message], *, system: str = "", max_tokens: int = 2048
    ) -> AsyncIterator[str]:
        failures: list[tuple[str, ProviderError]] = []
        for p in self._providers():
            produced = False
            try:
                async for chunk in p.stream(messages, system=system, max_tokens=max_tokens):
                    produced = True
                    yield chunk
                self.last_used, self.last_model = p.name, getattr(p, "model", "")
                return
            except ProviderError as e:
                if produced:  # can't restart a half-streamed answer
                    raise
                failures.append((p.name, e))
                if self.sticky:
                    self._demote(p.name, e)
        raise self._no_brain(failures)


def _eyes_first(providers: list[Provider]) -> list[Provider]:
    """For a request working from a picture (a screenshot of a board): a brain that can see it goes
    first, however recently it failed — a text-only one would act blind. But never ahead of a brain
    on this Mac that came before it: the user who put a local brain first keeps their screen on the
    Mac, and a cloud brain gets the picture only when the local one can't answer."""
    now = _time.time()
    local_before = 0
    order: list[tuple[int, bool, bool, int]] = []
    for i, p in enumerate(providers):
        order.append((local_before, _cooling(p) > now, not getattr(p, "supports_vision", False), i))
        if getattr(p, "on_device", False):
            local_before += 1
    return [providers[k[-1]] for k in sorted(order)]


def _has_picture(messages: list[Message]) -> bool:
    """Does the request being worked on carry a picture (a screenshot it took, a photo sent)?"""
    from neo.providers.history import request_start

    start = request_start(messages) if messages else 0
    return any(m.images or any(r.images for r in m.tool_results) for m in messages[start:])


def _needs_eyes(messages: list[Message]) -> bool:
    """Is the request being worked on a pixel task: it carries a picture, or a brain that can't see
    already tried to act at screen coordinates in it and was refused (BLIND_REFUSAL) — the task
    needs a brain that sees, and another turn of a blind one would only repeat that."""
    from neo.providers.history import request_start

    if _has_picture(messages):
        return True
    start = request_start(messages) if messages else 0
    return any(not r.ok and BLIND_REFUSAL in r.content for m in messages[start:] for r in m.tool_results)


# The brains that can see, in the order a step working from a picture asks them:
# 1. Gemini Flash (3.8 → 3.7 → 3.5, its own model walk);
# 2. Flash-Lite: a quota of its own, and it sees (the Chess run's Flash models were all overloaded
#    and it was never asked);
# 3. Claude, when set up;
# 4. last, Groq's vision model: 8K tokens a minute, ~30–40 s a picture step — the run's budget
#    (neo.providers.budget) stops it rather than let it run past the task's limit.
# A local model joins only once it can see (supports_vision), wherever the user put it. A seeing
# fallback faster than Groq's goes in before "groq_vision".
EYES: tuple[BrainName, ...] = ("gemini", "gemini_lite", "claude", "groq_vision")
# Brains that never see (a picture step never goes to them; they take the steps without one).
_TEXT_ONLY: frozenset[str] = frozenset({"groq", "local"})


def brain(purpose: Purpose = "agent", *, effort: Effort | None = None) -> Chain:
    """The chain for `purpose`. `effort`: the run's thinking policy (neo.providers.effort)."""
    if own := _hosts(purpose):
        return Chain(own, sticky=purpose != "vision", effort=effort)
    s = settings()
    if purpose == "fast":
        order: list[BrainName | None] = [s.fast_brain, s.brain, s.offline_brain]
    elif purpose == "vision":
        # The brains that can see the screenshot come first; text-only ones take only the steps that
        # carry no picture (an AX-only step): given one, they would act blind.
        vision: list[BrainName | None] = [n for n in (s.brain, *EYES) if n not in _TEXT_ONLY]
        order = vision + [n for n in (s.brain, s.fast_brain, s.offline_brain) if n not in vision]
        return Chain(_dedupe(order), sticky=False, effort=effort)
    elif purpose == "offline":
        return Chain(_dedupe([s.offline_brain]), effort=effort)  # (no cloud eyes: offline means here)
    elif purpose == "light":
        # One-tool actions that missed the regex fast path: cheapest capable brain first.
        order = ["gemini_lite", s.fast_brain, s.brain, s.offline_brain]
    else:
        order = [s.brain, s.fast_brain, s.offline_brain]
    return Chain(_dedupe(order), eyes=EYES, effort=effort)


# ---- how big a picture may be --------------------------------------------------------------------


@dataclass(frozen=True)
class ImageLimit:
    """The most a brain takes of one picture without shrinking it itself: its longer side (px),
    its pixel count, its encoded size (bytes). None: no limit of that kind."""

    long_edge: int | None = None
    pixels: int | None = None
    bytes: int | None = None

    def scale(self, width: int, height: int) -> float:
        """The factor (≤ 1) a width × height picture is scaled by to fit."""
        s = 1.0
        if width <= 0 or height <= 0:
            return s
        if self.long_edge:
            s = min(s, self.long_edge / max(width, height))
        if self.pixels:
            s = min(s, (self.pixels / (width * height)) ** 0.5)
        return s

    def tighter(self, other: ImageLimit | None) -> ImageLimit:
        """The stricter of the two, limit by limit."""
        if other is None:
            return self

        def least(a: int | None, b: int | None) -> int | None:
            return b if a is None else a if b is None else min(a, b)

        return ImageLimit(
            least(self.long_edge, other.long_edge), least(self.pixels, other.pixels), least(self.bytes, other.bytes)
        )


_MB = 1024 * 1024
# Anthropic: a picture over 1568 px on its longer side or ~1.15 MP is scaled down by the API;
# Opus 4.7 and later take 2576 px / 3.75 MP. 5 MB per picture.
_CLAUDE_OLD = ImageLimit(1568, 1_150_000, 5 * _MB)
_CLAUDE_NEW = ImageLimit(2576, 3_750_000, 5 * _MB)
# Groq: 33 megapixels per picture; a request of base64 pictures at most 4 MB (≈3 MB of picture).
_GROQ = ImageLimit(None, 33_177_600, 3 * _MB)
# By brain name (a new seeing brain adds its own row). Gemini: none (it bills a fixed number of
# tokens per resolution level instead: ImagePart.detail).
IMAGE_LIMITS: dict[str, ImageLimit | None] = {"gemini": None, "groq": _GROQ}
_CLAUDE_MODEL = re.compile(r"claude-(opus|sonnet|haiku)-(\d+)(?:[-.](\d{1,2}))?(?!\d)", re.I)


def _claude_limit(model: str) -> ImageLimit:
    m = _CLAUDE_MODEL.search(model or "")
    if m and m.group(1).lower() == "opus" and (int(m.group(2)), int(m.group(3) or 0)) >= (4, 7):
        return _CLAUDE_NEW
    return _CLAUDE_OLD  # an older or unknown one: the stricter limit (it only costs detail)


def limit_of(p: Any) -> ImageLimit | None:
    """One seeing brain's picture limit: its own (`image_limit`), else what is known of its kind."""
    own = getattr(p, "image_limit", None)
    if isinstance(own, ImageLimit):
        return own
    if callable(own):
        got = own()
        if got is None or isinstance(got, ImageLimit):
            return got
    name, model = str(getattr(p, "name", "")), str(getattr(p, "model", ""))
    if name.startswith("claude") or model.startswith("claude"):
        return _claude_limit(model)
    for prefix, lim in IMAGE_LIMITS.items():
        if name.startswith(prefix):
            return lim
    return None


def image_limit(purpose: Purpose = "vision") -> ImageLimit | None:
    """The strictest picture limit among the brains that may see a picture sent for `purpose`
    (the chain's own seeing brains and its seeing fallbacks; a host's own providers included), so
    a screenshot is made to fit them all and no API shrinks it silently. None: no limit."""
    chain = brain(purpose)
    out: ImageLimit | None = None
    for p in chain._providers() + chain._eye_brains():
        if not getattr(p, "supports_vision", False):
            continue
        lim = limit_of(p)
        if lim is not None:
            out = lim.tighter(out)
    return out


def _hosts(purpose: Purpose) -> list[Provider]:
    """The host's own providers for this purpose — or, for a purpose it gave none for, the ones
    it gave for its agent (or any): a host that brought its brains gets no others."""
    ov = pool().overrides
    for p in (purpose, "agent", "fast", "light", "vision", "offline"):
        if ov.get(p):
            return ov[p]
    return []


def _key(n: BrainName | Provider) -> str:
    return n if isinstance(n, str) else str(getattr(n, "name", "") or type(n).__name__)


def _dedupe(order: Sequence[BrainName | None]) -> list[BrainName]:
    seen: list[BrainName] = []
    for n in order:
        if n is not None and n not in seen:  # None: no such brain (an embedded agent's offline one)
            seen.append(n)
    return seen
