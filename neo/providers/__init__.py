"""Brain factory + failover chain.

``brain("agent")`` returns the provider chain for agentic turns, ``brain("fast")`` for
chat-only turns. A chain tries providers in order and moves on when one is rate-limited or
unreachable, so a free-tier 429 degrades gracefully instead of failing the turn.
"""

from __future__ import annotations

import asyncio
import time as _time
from collections.abc import AsyncIterator
from typing import Literal

from neo.config import BrainName, settings
from neo.providers.base import (
    AllBrainsFailed,
    Message,
    Provider,
    ProviderError,
    RateLimited,
    ToolSpec,
    Turn,
    spoken_name,
)

__all__ = ["AllBrainsFailed", "Chain", "ProviderError", "RateLimited", "brain", "make"]

Purpose = Literal["agent", "fast", "vision", "offline", "light"]

_cache: dict[str, Provider] = {}


def make(name: BrainName) -> Provider:
    if name in _cache:
        return _cache[name]
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
    else:
        raise ProviderError(f"unknown brain {name}")
    _cache[name] = p
    return p


_DEMOTE_S = 300.0  # a brain that just failed goes to the back of the line for this long
# Every brain failed but one only asked for a short wait: wait, then ask it again. A per-minute
# limit is always clear within the minute, and on Groq's 8K a step of 4–6K tokens is ~30–45s of it:
# the one brain left carries a long task through that wait rather than ending it.
_WAIT_MAX_S = 65.0

# Shared across chains: once Gemini has fallen over in one request, the next request (a fresh
# Chain each time) should not pay the same failed round trips before reaching Groq.
_demoted: dict[str, float] = {}

# A request-shaped failure says nothing about the brain's health: the next request may be fine.
_NO_DEMOTE = ("bad_request", "too_large")


def _cooling(p: Provider) -> float:
    """When a brain that knows it's spent (every Gemini key out of quota) can be asked again."""
    try:
        return float(getattr(p, "cooling_until", lambda: 0.0)() or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


class Chain:
    """Ordered providers with automatic failover on RateLimited / ProviderError.

    A provider that fails is *demoted* for a few minutes (a spent quota: until it resets): still
    available, but tried after the ones that work. `sticky=False` keeps the given order regardless
    (the vision chain must not let a text-only brain answer first and silently drop the screenshot).
    When every brain fails, the error is an AllBrainsFailed that says, per brain, why — quota,
    too large, not running — so the task can end with that reason spoken."""

    def __init__(self, names: list[BrainName], *, sticky: bool = True) -> None:
        self.names = names
        self.sticky = sticky
        self.last_used: str = ""
        self.last_model: str = ""
        self._missing: list[tuple[str, ProviderError]] = []

    def _providers(self) -> list[Provider]:
        out: list[Provider] = []
        self._missing = []
        for n in self._order():
            try:
                out.append(make(n))
            except ProviderError as e:  # missing key → skip silently
                print(f"[brain] {n} unavailable: {e}")
                self._missing.append((n, e))
        if self.sticky:
            # A brain that knows it's spent (all its quota) goes last: the next one is asked at once.
            now = _time.time()
            out.sort(key=lambda p: _cooling(p) > now)
        return out

    def _order(self) -> list[BrainName]:
        if not self.sticky:
            return list(self.names)
        now = _time.time()
        healthy = [n for n in self.names if _demoted.get(n, 0.0) <= now]
        return healthy + [n for n in self.names if n not in healthy]

    @staticmethod
    def _demote(name: str, e: ProviderError | None = None) -> None:
        if e is not None and getattr(e, "kind", "") in _NO_DEMOTE:
            return
        until = _time.time() + _DEMOTE_S
        if e is not None and getattr(e, "kind", "") == "quota":
            until = max(until, float(getattr(e, "until", 0.0) or 0.0))
        _demoted[name] = until

    @property
    def name(self) -> str:
        return "+".join(self.names)

    @property
    def supports_vision(self) -> bool:
        return any(p.supports_vision for p in self._providers())

    supports_tools = True

    def _no_brain(self, failures: list[tuple[str, ProviderError]]) -> ProviderError:
        s = settings()
        if not (s.gemini_api_key or s.groq_api_key or s.anthropic_api_key):
            return ProviderError(
                "no API key set — add GEMINI_API_KEY to .env (free at aistudio.google.com/apikey)"
            )
        return AllBrainsFailed(failures + self._missing)

    async def _failed(self, p: Provider, e: ProviderError, failures: list[tuple[str, ProviderError]]) -> None:
        print(f"[brain] {p.name} failed ({type(e).__name__}: {str(e)[:80]}); trying next")
        failures.append((p.name, e))
        if self.sticky:
            self._demote(p.name, e)
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

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        effort="medium",
        max_tokens: int = 4096,
    ) -> Turn:
        failures: list[tuple[str, ProviderError]] = []
        providers = self._providers()
        for p in providers:
            try:
                turn = await p.complete(
                    messages, system=system, tools=tools, effort=effort, max_tokens=max_tokens
                )
                self.last_used, self.last_model = p.name, getattr(p, "model", "")
                return turn
            except ProviderError as e:
                await self._failed(p, e, failures)
        if (wait := self._short_wait(failures)) is not None:
            # Every brain is out, but one only for seconds (Groq's per-minute budget): a long
            # task is worth that pause rather than ending here.
            secs, name = wait
            p = next(p for p in providers if p.name == name)
            print(f"[brain] every brain failed; waiting {secs:.0f}s for {name}")
            from neo.events import bus

            await bus().note(f"Waiting {secs:.0f}s for {spoken_name(name)}'s rate limit")
            await asyncio.sleep(secs + 0.5)
            try:
                turn = await p.complete(
                    messages, system=system, tools=tools, effort=effort, max_tokens=max_tokens
                )
                self.last_used, self.last_model = p.name, getattr(p, "model", "")
                return turn
            except ProviderError as e:
                failures = [(n, x) for n, x in failures if n != name]
                await self._failed(p, e, failures)
        raise self._no_brain(failures)

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


def brain(purpose: Purpose = "agent") -> Chain:
    s = settings()
    if purpose == "fast":
        order: list[BrainName] = [s.fast_brain, s.brain, s.offline_brain]
    elif purpose == "vision":
        # Only brains that can see the screenshot come first; text-only ones would drop it silently.
        vision: list[BrainName] = [n for n in (s.brain, "gemini", "claude") if n not in ("groq", "local")]
        order = vision + [n for n in (s.brain, s.fast_brain, s.offline_brain) if n not in vision]
        return Chain(_dedupe(order), sticky=False)
    elif purpose == "offline":
        order = [s.offline_brain]
    elif purpose == "light":
        # One-tool actions that missed the regex fast path: cheapest capable brain first.
        order = ["gemini_lite", s.fast_brain, s.brain, s.offline_brain]
    else:
        order = [s.brain, s.fast_brain, s.offline_brain]
    return Chain(_dedupe(order))


def _dedupe(order: list[BrainName]) -> list[BrainName]:
    seen: list[BrainName] = []
    for n in order:
        if n not in seen:
            seen.append(n)
    return seen
