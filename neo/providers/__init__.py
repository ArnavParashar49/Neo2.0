"""Brain factory + failover chain.

``brain("agent")`` returns the provider chain for agentic turns, ``brain("fast")`` for
chat-only turns. A chain tries providers in order and moves on when one is rate-limited or
unreachable, so a free-tier 429 degrades gracefully instead of failing the turn.

The brains made so far, and which of them failed lately, live in a :class:`BrainPool`: NEO.app's
(the module's `_cache` and `_demoted`), or an embedded agent's own (neo.runtime) — built from
that agent's keys, and with the host's own providers for any purpose it supplied them for.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Literal

from neo import runtime
from neo.config import BrainName, settings
from neo.providers.base import Message, Provider, ProviderError, RateLimited, ToolSpec, Turn

Purpose = Literal["agent", "fast", "vision", "offline", "light"]

_cache: dict[str, Provider] = {}


class BrainPool:
    """One agent's brains: those made from its settings (by name), how recently each failed, and
    the host's own providers per purpose (`overrides`), used instead of the named brains."""

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

        p = GeminiProvider(model=settings().gemini_lite_model)
    elif name == "local":
        from neo.providers.openai_compat import local

        p = local()
    elif name == "claude":
        from neo.providers.claude import ClaudeProvider

        p = ClaudeProvider()
    else:
        raise ProviderError(f"unknown brain {name}")
    return p


_DEMOTE_S = 300.0  # a brain that just failed goes to the back of the line for this long

# Shared across chains: once Gemini has fallen over in one request, the next request (a fresh
# Chain each time) should not pay the same failed round trips before reaching Groq.
_demoted: dict[str, float] = {}


class Chain:
    """Ordered providers with automatic failover on RateLimited / ProviderError.

    A provider that fails is *demoted* for a few minutes: still available, but tried after the
    ones that work. `sticky=False` keeps the given order regardless (the vision chain must not
    let a text-only brain answer first and silently drop the screenshot)."""

    def __init__(self, names: Sequence[BrainName | Provider], *, sticky: bool = True) -> None:
        self.names = list(names)  # brain names, or provider instances (a host's own), in order
        self.sticky = sticky
        self.last_used: str = ""
        self.last_model: str = ""

    def _providers(self) -> list[Provider]:
        out: list[Provider] = []
        for n in self._order():
            if not isinstance(n, str):
                out.append(n)
                continue
            try:
                out.append(make(n))
            except ProviderError as e:  # missing key → skip silently
                print(f"[brain] {n} unavailable: {e}")
        return out

    def _order(self) -> list[BrainName | Provider]:
        import time as _time

        if not self.sticky:
            return list(self.names)
        now = _time.time()
        demoted = pool().demoted()
        healthy = [n for n in self.names if demoted.get(_key(n), 0.0) <= now]
        return healthy + [n for n in self.names if n not in healthy]

    @staticmethod
    def _demote(name: str) -> None:
        import time as _time

        pool().demoted()[name] = _time.time() + _DEMOTE_S

    @property
    def name(self) -> str:
        return "+".join(_key(n) for n in self.names)

    @property
    def supports_vision(self) -> bool:
        return any(p.supports_vision for p in self._providers())

    supports_tools = True

    def _no_brain(self, errors: list[str]) -> ProviderError:
        s = settings()
        if not pool().overrides and not (s.gemini_api_key or s.groq_api_key or s.anthropic_api_key):
            return ProviderError(
                "no API key set — add GEMINI_API_KEY to .env (free at aistudio.google.com/apikey)"
            )
        return ProviderError("all brains failed: " + "; ".join(errors)[:300])

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        effort="medium",
        max_tokens: int = 4096,
    ) -> Turn:
        errors: list[str] = []
        for p in self._providers():
            try:
                turn = await p.complete(
                    messages, system=system, tools=tools, effort=effort, max_tokens=max_tokens
                )
                self.last_used, self.last_model = p.name, getattr(p, "model", "")
                return turn
            except (RateLimited, ProviderError) as e:
                print(f"[brain] {p.name} failed ({type(e).__name__}: {str(e)[:80]}); trying next")
                errors.append(f"{p.name}: {str(e)[:80]}")
                if self.sticky:
                    self._demote(p.name)
                from neo.events import bus

                await bus().note(f"{p.name.capitalize()} unavailable, switching brain")
        raise self._no_brain(errors)

    async def stream(
        self, messages: list[Message], *, system: str = "", max_tokens: int = 2048
    ) -> AsyncIterator[str]:
        errors: list[str] = []
        for p in self._providers():
            produced = False
            try:
                async for chunk in p.stream(messages, system=system, max_tokens=max_tokens):
                    produced = True
                    yield chunk
                self.last_used, self.last_model = p.name, getattr(p, "model", "")
                return
            except (RateLimited, ProviderError) as e:
                if produced:  # can't restart a half-streamed answer
                    raise
                errors.append(f"{p.name}: {str(e)[:80]}")
                if self.sticky:
                    self._demote(p.name)
        raise self._no_brain(errors)


def brain(purpose: Purpose = "agent") -> Chain:
    if own := pool().overrides.get(purpose):  # the host's providers for this purpose
        return Chain(own, sticky=purpose != "vision")
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


def _key(n: BrainName | Provider) -> str:
    return n if isinstance(n, str) else str(getattr(n, "name", "") or type(n).__name__)


def _dedupe(order: list[BrainName]) -> list[BrainName]:
    seen: list[BrainName] = []
    for n in order:
        if n not in seen:
            seen.append(n)
    return seen
