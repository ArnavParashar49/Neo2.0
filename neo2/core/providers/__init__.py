"""Brain factory + failover chain.

``brain("agent")`` returns the provider chain for agentic turns, ``brain("fast")`` for
chat-only turns. A chain tries providers in order and moves on when one is rate-limited or
unreachable, so a free-tier 429 degrades gracefully instead of failing the turn.

The brains made so far, and which of them failed lately, live in a :class:`BrainPool`: NEO.app's
(the module's `_cache` and `_demoted`), or an embedded agent's own (neo2.core.runtime) — built from
that agent's keys, and with the host's own providers for any purpose it supplied them for.

A brain is named by a reference, ``<brain>[@<model>]`` (neo2.core.brainref): a bare name NEO always
had is built exactly as it always was; each other reference is its own brain (its own cache entry,
demotion and cooldowns). What one can do — sees, calls tools, only looks, runs on this Mac — is read
from the catalogue (neo2.core.providers.catalog.describe), never from a list of names. The roles
(Brain, Quick replies, Eyes, Backups) order the chains: see :func:`brain`.

A step that works from a picture (a screenshot the run took, a photo the user sent) goes only to
brains that can see it: the chain's own, then the seeing fallbacks (:data:`EYES`) — never to a
text-only cloud brain, which would act blind. When none can answer, the run stops and says so
(NoEyes). How hard each step thinks follows the run (neo2.core.providers.effort), and inside an agent run
the chain keeps to the time the run has left (neo2.core.providers.budget).
"""

from __future__ import annotations

import asyncio
import re
import time as _time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from neo2.core import brainref, logger, runtime
from neo2.core.config import BrainName, settings
from neo2.core.providers import budget, catalog
from neo2.core.providers.base import (
    BLIND_REFUSAL,
    AllBrainsFailed,
    Blind,
    Level,
    Message,
    NoBrain,
    NoEyes,
    NoTools,
    OutOfTime,
    Provider,
    ProviderError,
    QuotaExhausted,
    RateLimited,
    ToolSpec,
    TooSlow,
    Turn,
    Unavailable,
    spoken_name,
)
from neo2.core.providers.effort import Effort, for_brain

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
    "camera_order",
    "forget",
    "image_limit",
    "lens_order",
    "make",
    "pool",
    "rebuild",
]

log = logger(__name__)

Purpose = Literal["agent", "fast", "vision", "offline", "light"]

_cache: dict[str, Provider] = {}
# (brain, kind of error) already logged as unavailable. A chain is built for every step, so a brain
# with no key said so on every one (270 identical lines in one log): once per process tells as much.
_said_unavailable: set[tuple[str, str]] = set()


def _shown(name: str) -> str:
    """A brain's name as a log line may carry it: a placeholder that could be a pasted key never is."""
    return name if brainref.parse(name).valid or brainref.shown(name) else "(a brain NEO doesn't know)"


def _unavailable(name: str, e: BaseException) -> None:
    """Log that a brain isn't set up, once per process for each brain and kind of error (the chain
    still leaves it out every time, and keeps its reason for the "every brain failed" answer)."""
    key = (str(name), type(e).__name__)
    if key in _said_unavailable:
        return
    _said_unavailable.add(key)
    log.info(f"[brain] {_shown(str(name))} unavailable: {e}")


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

    def make(self, name: str) -> Provider:
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
    """The bound agent's brains (neo2.core.runtime), else NEO.app's."""
    rt = runtime.current_runtime()
    if rt is not None:
        return rt.get_or_create("brains", BrainPool)
    runtime.check_unbound("the brains")
    return _app_pool


def make(name: str) -> Provider:
    return pool().make(name)


def _build(name: str) -> Provider:
    """The brain a reference names. A bare name NEO always had: exactly the code it always ran.
    Another reference: its provider with that model, named by the reference. One that can't be read
    (or an endpoint that isn't set up): Unavailable — skipped like a brain with no key, never
    another brain in its place."""
    r = brainref.parse(name)
    if r.legacy:
        return _build_legacy(r.brain)  # type: ignore[arg-type]
    if not r.valid:
        raise Unavailable(f"{_shown(r.text) if r.text else 'an empty name'} isn't a brain NEO knows")
    p = _build_ref(r)
    p.name = r.text  # the reference is the brain: its cache entry, demotion and cooldowns
    return p


def _build_ref(r: brainref.Ref) -> Provider:
    s = settings()
    m = r.model
    if r.brain in ("gemini", "gemini_lite"):
        from neo2.core.providers.gemini import GeminiProvider

        return GeminiProvider(model=m, name=r.text)
    if r.brain in ("groq", "groq_vision"):
        # By what the model does, whichever name the reference starts with (catalog.describe):
        # groq@qwen/qwen3.8-27b is Groq's seeing brain, with its per-minute budget.
        if catalog.describe(r.text, s).sees:
            return _groq_vision(m)
        from neo2.core.providers.openai_compat import groq

        return groq(m)
    if r.brain == "local":
        from neo2.core.providers.openai_compat import local

        return local(m)
    if r.brain == "claude":
        from neo2.core.providers.claude import ClaudeProvider

        return ClaudeProvider(m)
    if r.brain in ("cerebras", "cerebras_vision"):
        from neo2.core.providers import cerebras

        return (cerebras.cerebras_vision if catalog.describe(r.text, s).sees else cerebras.cerebras)(m)
    if r.brain == "openai":
        if not s.openai_api_key:
            raise Unavailable("OPENAI_API_KEY is not set")
        try:
            from neo2.core.providers.openai_responses import openai as openai_brain
        except ImportError:
            raise Unavailable("OpenAI's brain isn't in this build of NEO") from None
        return openai_brain(m)
    if r.brain == "openrouter":
        from neo2.core.providers.openai_compat import openrouter

        return openrouter(m, name=r.text, vision=catalog.describe(r.text, s).sees)
    if r.custom:
        from neo2.core.providers.openai_compat import custom

        cp = catalog.custom_endpoint(r.brain, s)
        if cp is None:
            raise Unavailable(f"{r.brain} isn't set up")
        return custom(cp, s.custom_keys.get(cp.id, ""), model=m, name=r.text)
    raise Unavailable(f"{r.text} isn't a brain NEO can build")


def _build_legacy(name: BrainName) -> Provider:
    if name == "gemini":
        from neo2.core.providers.gemini import GeminiProvider

        p: Provider = GeminiProvider()
    elif name == "groq":
        from neo2.core.providers.openai_compat import groq

        p = groq()
    elif name == "gemini_lite":
        from neo2.core.providers.gemini import GeminiProvider

        p = GeminiProvider(model=settings().gemini_lite_model, name="gemini_lite")  # its own brain
    elif name == "local":
        from neo2.core.providers.openai_compat import local

        p = local()
    elif name == "claude":
        from neo2.core.providers.claude import ClaudeProvider

        p = ClaudeProvider()
    elif name == "groq_vision":
        p = _groq_vision()
    elif name == "cerebras":
        from neo2.core.providers.cerebras import cerebras

        p = cerebras()
    elif name == "cerebras_vision":
        from neo2.core.providers.cerebras import cerebras_vision

        p = cerebras_vision()
    else:
        raise ProviderError(f"unknown brain {name}")
    return p


# Groq's free tier, per model: tokens a minute (prompt and answer together) — one request's most.
_GROQ_VISION_TOKENS = 8000


def _groq_vision(model: str | None = None) -> Provider:
    """Groq's vision model (settings.groq_vision_model, or `model`): the last seeing fallback. Its
    own brain — its own rate limit and demotion — on the same key as the text one. It takes pictures
    (a few per request, ~2K tokens each), so its budget is a per-minute rate every step pays back."""
    from neo2.core.providers.openai_compat import OpenAICompatProvider

    s = settings()
    if not s.groq_api_key:
        raise Unavailable("GROQ_API_KEY is not set")
    if not (model or s.groq_vision_model):
        raise Unavailable("no Groq vision model is set (NEO_GROQ_VISION_MODEL)")
    return OpenAICompatProvider(
        name="groq_vision",
        base_url="https://api.groq.com/openai/v1",
        api_key=s.groq_api_key,
        model=model or s.groq_vision_model,
        supports_vision=True,
        # Not its reasoning switch: a vision model's values differ from gpt-oss's low/medium/high.
        reasoning_effort=False,
        token_budget=_GROQ_VISION_TOKENS,
        per_minute=True,
    )


_DEMOTE_S = 300.0  # a brain that just failed goes to the back of the line for this long
_KEY_REFUSED_S = 3600.0  # …one whose key was refused, this long (asking again only costs round trips)
# Every brain failed but one only asked for a short wait: wait, then ask it again. A per-minute
# limit is always clear within the minute, and on Groq's 8K a step of 4–6K tokens is ~30–45s of it:
# the one brain left carries a long task through that wait rather than ending it.
_WAIT_MAX_S = 65.0

# Shared across chains: once Gemini has fallen over in one request, the next request (a fresh
# Chain each time) should not pay the same failed round trips before reaching Groq. NEO.app's;
# an embedded agent's pool keeps its own.
_demoted: dict[str, float] = {}

# A request-shaped failure says nothing about the brain's health: the next request may be fine.
_NO_DEMOTE = ("bad_request", "too_large", "no_tools")


def _cooling(p: Provider) -> float:
    """When a brain that knows it's spent (every Gemini key out of quota) can be asked again."""
    try:
        return float(getattr(p, "cooling_until", lambda: 0.0)() or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


def _seen_by(turn: Turn, p: Provider) -> Turn:
    """The reply as written by `p`: a brain that doesn't declare vision saw no picture, whatever its
    reply says (a host's own provider, or any that never heard of Turn.sees, leaves it True)."""
    if not getattr(p, "supports_vision", False):
        turn.sees = False
    return turn


# A brain slower than this to answer one step: the user hears that NEO is waiting on it.
_PATIENCE_S = 20.0
# What a brain needs to answer once a rate-limit wait is over: a wait that leaves less of the run's
# time than this isn't worth starting.
_ANSWER_S = 3.0


def _too_late(p: Provider) -> ProviderError | None:
    """Within a run: the reason not to ask this brain now. None: ask it.

    The run's time is up: TooSlow (the step stops as out of time). The brain is cooling until after
    the time left (a spent quota, an overload, a per-minute wait): its own reason — the quota and
    when it resets, the overload — as it would have failed had it been asked. Whether the time is
    what stopped the step is decided once every brain had its turn (Chain._out_of_time)."""
    left = budget.left()
    if left is None:
        return None
    if left <= 0:
        return TooSlow("the run's time is up")
    until = _cooling(p)
    if until - _time.time() > left:
        return _cooling_error(p, until)
    return None


def _cooling_error(p: Provider, until: float) -> ProviderError:
    """Why a cooling brain can't be asked: its own words when it knows them (Gemini: quota or
    overload, `cooling_error()`), else the wait it asked for — a wait longer than any rate-limit
    pause is a spent quota until then."""
    # (Cerebras: cooling_reason() — out for the day, in its own words; None for a short wait)
    own = getattr(p, "cooling_error", None) or getattr(p, "cooling_reason", None)
    if callable(own):
        try:
            e = own()
        except Exception:  # noqa: BLE001 — can't say: the wait itself, below
            e = None
        if isinstance(e, ProviderError):
            return e
    wait = max(0.0, until - _time.time())
    if wait > _WAIT_MAX_S:
        return QuotaExhausted(f"{p.name}: can't be asked for {wait:.0f}s", until=until)
    return RateLimited(f"{p.name}: asked to wait {wait:.0f}s", retry_after=wait)


async def _patience(name: str) -> None:
    await asyncio.sleep(_PATIENCE_S)
    from neo2.core.events import bus

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
    Groq's vision model, not to Groq's text model). `lead`: the user's Eyes (settings.eyes_brain),
    the first a step working from a picture asks — ahead of the chain's own brains — and never one
    without a picture unless it is among them too. `effort`: the run's thinking policy — each step's
    level is read from the transcript (neo2.core.providers.effort); without one, the level the caller
    passes is used as it is."""

    def __init__(
        self,
        names: Sequence[str | Provider],
        *,
        sticky: bool = True,
        eyes: Sequence[str] = (),
        effort: Effort | None = None,
        lead: Sequence[str] = (),
    ) -> None:
        self.names = list(names)  # brain names, or provider instances (a host's own), in order
        self.sticky = sticky
        self.lead = [n for n in dict.fromkeys(lead) if n]
        self.eyes = [n for n in eyes if n not in self.names and n not in self.lead]
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
                _unavailable(n, e)
                self._missing.append((_shown(n), e))  # (said in the stop message: never a pasted key)
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
        if e is not None and getattr(e, "kind", "") == "key":
            until = _time.time() + _KEY_REFUSED_S
        self._demotions()[name] = until

    @property
    def name(self) -> str:
        return "+".join(_shown(_key(n)) if isinstance(n, str) else _key(n) for n in self.names)

    @property
    def supports_vision(self) -> bool:
        return (
            any(p.supports_vision for p in self._providers())
            or any(getattr(p, "supports_vision", False) for p in self._eye_brains())
            or bool(self._lead_brains())
        )

    @property
    def image_limit(self) -> tuple[int, int] | None:
        """The largest picture every brain here that can see takes as sent: (longest edge in
        pixels, total pixels), the strictest of those they declare (their own ``image_limit``; a
        chain inside a chain counts too). None: none declares one."""
        limits = [
            lim
            for p in self._providers()
            if getattr(p, "supports_vision", False) and (lim := getattr(p, "image_limit", None))
        ]
        if not limits:
            return None
        return min(int(edge) for edge, _ in limits), min(int(px) for _, px in limits)

    supports_tools = True

    def _eye_brains(self, names: Sequence[str] | None = None) -> list[Provider]:
        """The seeing fallbacks that are set up (a missing key: skipped, as in _providers)."""
        out: list[Provider] = []
        for n in self.eyes if names is None else names:
            try:
                p = make(n)
            except Exception as e:  # noqa: BLE001 — not set up (no key, no package): it can't look
                _unavailable(n, e)
                continue
            if getattr(p, "supports_vision", False):
                out.append(p)
        return out

    def _lead_brains(self) -> list[Provider]:
        """The user's Eyes that are set up and see (none: no Eyes brain named)."""
        return self._eye_brains(self.lead) if self.lead else []

    def _for_pixels(self, providers: list[Provider], failures: list[tuple[str, ProviderError]]) -> list[Provider]:
        """Who may answer a step working from a picture: the brains that can see it — the chain's
        own first, then its seeing fallbacks — and a text-only brain on this Mac only where the user
        put it ahead of a cloud brain (local-first: the screen stays on the Mac while a brain here
        can answer; the fallbacks, cloud brains all, come after it). Every other text-only brain is
        left out, as Blind: it would act on a picture it never saw. A host's own chain keeps its
        brains as the host gave them."""
        if self._hosts_only():
            return _eyes_first(providers)
        # The user's Eyes first (when they named one): their pick for every picture, wherever the
        # rest of the chain is. The rest as before, without them.
        lead = self._lead_brains()
        if lead:
            first = {p.name for p in lead}
            providers = [p for p in providers if p.name not in first]
            return lead + self._for_pixels_after(providers, failures, first)
        return self._for_pixels_after(providers, failures, set())

    def _for_pixels_after(
        self, providers: list[Provider], failures: list[tuple[str, ProviderError]], first: set[str]
    ) -> list[Provider]:
        have = {p.name for p in providers} | first
        extra = [p for p in self._eye_brains() if p.name not in have]

        def cloud(q: Provider) -> bool:
            return not getattr(q, "on_device", False)

        # A brain here guards the screen where the user put it before a cloud brain, and a cloud
        # brain that sees would come after it (one of the chain's own, or a fallback). Where the
        # user put it: the chain's order, not the one a failure's demotion left (a Gemini that just
        # failed, moved behind the local brain, doesn't make that brain a guard).
        at = {_key(n): i for i, n in enumerate(self.names)}
        placed = sorted(providers, key=lambda p: at.get(p.name, len(at)))
        guards = {
            p.name
            for i, p in enumerate(placed)
            if not cloud(p)
            and any(cloud(q) for q in placed[i + 1 :])
            and any(cloud(q) and q.supports_vision for q in placed[i + 1 :] + extra)
        }
        out: list[Provider] = []
        for p in _eyes_first(providers):
            if getattr(p, "supports_vision", False) or p.name in guards:
                out.append(p)
            else:
                failures.append((p.name, Blind(f"{p.name}: this model has no vision")))
        now = _time.time()
        return out + sorted(extra, key=lambda p: _cooling(p) > now)  # (stable: EYES order otherwise)

    @staticmethod
    def _out_of_time(failures: list[tuple[str, ProviderError]]) -> bool:
        """Did the run's time stop this step: it was up, or a call was cut off at it (TooSlow), or
        less is left than a brain needs to answer. A brain skipped for a quota or an overload that
        ends past the time left is not that: its own reason is what the user hears. Nor is a picture
        step no brain set up could see: more time wouldn't have helped (NoEyes says why)."""
        if any(getattr(e, "kind", "") == "budget" for _, e in failures):
            return True
        left = budget.left()
        if left is None or left > _ANSWER_S:
            return False
        return any(getattr(e, "kind", "") != "blind" for _, e in failures)

    def _no_brain(
        self,
        failures: list[tuple[str, ProviderError]],
        *,
        pixels: bool = False,
        seeing: Sequence[str] = (),
        photo: bool = False,
    ) -> ProviderError:
        """The error a step ends with when no brain answered. `seeing`: for a picture step, the
        brains that can see it (asked, or skipped for a cooldown); `photo`: the picture is one the
        user sent."""
        if self._out_of_time(failures):
            return OutOfTime(failures, limit_s=budget.limit())
        if self._hosts_only():  # no key of NEO's could have been missing
            return AllBrainsFailed(failures + self._missing)
        if pixels and any(n in seeing for n, _ in failures):
            return NoEyes(failures, seeing=seeing, photo=photo)  # a brain that sees couldn't answer
        if not pool().overrides and not catalog.any_key(settings()):
            # No key for any provider NEO may ask, and no model here answered: Brains & Keys is where
            # that changes (the session says so and opens it: neo2.core.onboarding). An embedded
            # agent has no such window: its words are the old ones.
            if runtime.library_bound() or runtime.library_mode():
                return NoBrain("no API key set — add GEMINI_API_KEY to .env (free at aistudio.google.com/apikey)")
            return NoBrain("no AI model is set up — connect one in Brains & Keys (or: python -m neo2 keys set gemini)")
        if pixels:  # none that sees is set up: the words say so, not that one is busy
            return NoEyes(failures or self._missing, seeing=(), photo=photo)
        return AllBrainsFailed(failures + self._missing)

    async def _failed(self, p: Provider, e: ProviderError, failures: list[tuple[str, ProviderError]]) -> None:
        log.warning(f"[brain] {p.name} failed ({type(e).__name__}: {str(e)[:80]}); trying next")
        failures.append((p.name, e))
        if self.sticky:
            self._demote(p.name, e)
        if self._alone():
            return  # no conversation to tell (and never NEO.app's)
        if getattr(e, "kind", "") == "key":
            _key_health(p.name, refused=True)
        from neo2.core.events import bus

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

    def _answered(self, p: Provider) -> None:
        """`p` answered: it's the one that did, and its key (NEO.app's) is known to work again."""
        self.last_used, self.last_model = p.name, getattr(p, "model", "")
        if not self._alone():
            _key_health(p.name, refused=False)

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
        """One brain's answer — within the run's time left, when a run is bound (neo2.core.providers.budget):
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
        seeing = [p.name for p in providers if getattr(p, "supports_vision", False)] if pixels else []
        photo = pixels and _photo_only(messages)

        def no_brain() -> ProviderError:
            return self._no_brain(failures, pixels=pixels, seeing=seeing, photo=photo)

        for p in providers:
            if (late := _too_late(p)) is not None:
                failures.append((p.name, late))
                continue
            if tools and getattr(p, "supports_tools", True) is False and not any(p is n for n in self.names):
                # (an endpoint saved with Tools off: a step with tools would come back as text that
                # only says it acted — the next brain takes it; it isn't demoted for chat turns. A
                # host's own provider object is the host's to send what it likes.)
                failures.append((p.name, NoTools(f"{p.name}: it can't call tools")))
                continue
            try:
                turn = _seen_by(await self._ask(p, messages, system, tools, level, max_tokens), p)
                self._answered(p)
                return turn
            except TooSlow as e:  # the run's time is up: no other brain could answer in none
                failures.append((p.name, e))
                raise no_brain() from None
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
            from neo2.core.events import bus

            await bus().note(f"Waiting {secs:.0f}s for {spoken_name(name)}'s rate limit")
            await asyncio.sleep(secs + 0.5)
            try:
                turn = _seen_by(await self._ask(p, messages, system, tools, level, max_tokens), p)
                self._answered(p)
                return turn
            except TooSlow as e:
                failures.append((p.name, e))
            except ProviderError as e:
                failures = [(n, x) for n, x in failures if n != name]
                await self._failed(p, e, failures)
        raise no_brain()

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
                self._answered(p)
                return
            except ProviderError as e:
                if produced:  # can't restart a half-streamed answer
                    raise
                failures.append((p.name, e))
                if self.sticky:
                    self._demote(p.name, e)
                if getattr(e, "kind", "") == "key" and not self._alone():
                    _key_health(p.name, refused=True)
        raise self._no_brain(failures)


def _key_health(name: str, *, refused: bool) -> None:
    """NEO.app's health list: "<provider> refused its API key" while a brain of it is refused (a 401,
    or a 403 about the key — KeyRefused); cleared when one answers, or the key changes (forget). An
    embedded agent's brains, and a host's own, never touch it."""
    if not isinstance(name, str) or not runtime.process_wide_ok():
        return
    pid = catalog.provider_of(name)
    if not pid:
        return
    from neo2.core import health

    if refused:
        health.report(f"key:{pid}", f"{catalog.provider_label(pid)} refused {_key_from(pid)}. Open Brains & Keys to replace it.")
    else:
        health.report(f"key:{pid}", None)


_KEY_FROM = {
    "saved": "the API key saved in NEO",
    "env": "the API key in the environment variable {env}",
    "dotenv": "the API key in ~/.neo/.env",
    "checkout_env": "the API key in the checkout's .env",
}


def _key_from(pid: str) -> str:
    """Which of the user's keys a provider refused, in words: "the API key saved in NEO" (where it
    comes from: keystore.sources — never the key)."""
    try:
        from neo2.core import keystore

        s = settings()
        src = keystore.sources(custom=list(getattr(s, "custom_providers", None) or ())).get(pid, {}).get("source", "")
        p = catalog.provider(pid)
        cp = catalog.custom_endpoint(pid, s) if pid.startswith("custom:") else None
        env = (p.env if p is not None else None) or (cp.env if cp is not None else None) or "its variable"
        return _KEY_FROM.get(src, "its API key").format(env=env)
    except Exception:  # noqa: BLE001 — where it came from is a detail
        return "its API key"


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
    from neo2.core.providers.history import request_start

    start = request_start(messages) if messages else 0
    return any(m.images or any(r.images for r in m.tool_results) for m in messages[start:])


def _photo_only(messages: list[Message]) -> bool:
    """The request's picture is one the user sent (a photo), and no screenshot was taken in it."""
    from neo2.core.providers.history import request_start

    start = request_start(messages) if messages else 0
    rest = messages[start:]
    return any(m.images for m in rest) and not any(r.images for m in rest for r in m.tool_results)


def _needs_eyes(messages: list[Message]) -> bool:
    """Is the request being worked on a pixel task: it carries a picture, or a brain that can't see
    already tried to act at screen coordinates in it and was refused (BLIND_REFUSAL) — the task
    needs a brain that sees, and another turn of a blind one would only repeat that."""
    from neo2.core.providers.history import request_start

    if _has_picture(messages):
        return True
    start = request_start(messages) if messages else 0
    return any(not r.ok and BLIND_REFUSAL in r.content for m in messages[start:] for r in m.tool_results)


# The brains that can see, in the order a step working from a picture asks them:
# 1. Gemini Flash (3.8 → 3.7 → 3.5, its own model walk);
# 2. Flash-Lite: a quota of its own, and it sees (in the logged board-game run every Flash model
#    was overloaded and it was never asked);
# 3. Claude, when set up;
# 4. last, Groq's vision model: 8K tokens a minute, ~30–40 s a picture step — the run's budget
#    (neo2.core.providers.budget) stops it rather than let it run past the task's limit.
# A local model joins only once it can see (supports_vision), wherever the user put it. A seeing
# fallback faster than Groq's goes in before "groq_vision" (and, when it should answer only the
# steps with a picture, is eyes_only in the catalogue).
# Which brains never see, only ever answer a step with a picture (Groq's vision model: its
# per-minute budget is kept for the pixel steps) or run on this Mac is read from the catalogue
# (catalog.describe: sees, eyes_only, on_device), never from a list of names here.
EYES: tuple[str, ...] = catalog.IMPLICIT_EYES


def _sees(n: str) -> bool:
    return catalog.describe(n).sees


def _eyes_only(n: str) -> bool:
    return catalog.describe(n).eyes_only


def _on_this_mac(n: Any) -> bool:
    return isinstance(n, str) and catalog.describe(n).on_device


def _cloud_seer(n: Any) -> bool:
    """`n` is a cloud brain that sees, as the user may name one. A bare old name: one of EYES,
    exactly as before."""
    if not isinstance(n, str) or not n:
        return False
    r = brainref.parse(n)
    if r.legacy:
        return n in EYES
    i = catalog.describe(n)
    return i.valid and i.sees and not i.on_device


def _backups(s: Any) -> list[str] | None:
    """The user's Backups: None is Automatic (today's chains)."""
    b = getattr(s, "backup_brains", None)
    return None if b is None else [n for n in b if n]


def _fallback_eyes(s: Any) -> tuple[str, ...]:
    """The seeing fallbacks the chains get: EYES — except for a user who put a brain on this Mac
    first and named no cloud brain that sees (NEO_BRAIN=local with only a Groq key: no picture ever
    left the Mac). A fallback in the cloud doesn't change that on its own: their picture steps stop
    and say so (DECISIONS 39), unless they opt in (NEO_CLOUD_EYES=on) or name one that sees (their
    Eyes or a Backup too). NEO_CLOUD_EYES=off: never — only the brains the user picked see a picture."""
    cloud_eyes = getattr(s, "cloud_eyes", "auto")
    if cloud_eyes == "off":
        return ()
    named = (s.brain, s.fast_brain, getattr(s, "offline_brain", None), getattr(s, "eyes_brain", ""), *(_backups(s) or ()))
    if cloud_eyes == "on" or not _on_this_mac(s.brain) or any(_cloud_seer(n) for n in named):
        return EYES
    return ()


def brain(purpose: Purpose = "agent", *, effort: Effort | None = None) -> Chain:
    """The chain for `purpose`, from the user's roles: Brain (settings.brain), Quick replies
    (fast_brain), Eyes (eyes_brain), Backups (backup_brains; None: Automatic) and light_brain.

    * agent: [brain, *backups, offline] — Automatic backups: [brain, fast_brain, offline], as before;
    * fast: [fast_brain, brain, *backups, offline];
    * light: [light_brain ("": fast_brain's), fast_brain, brain, *backups, offline];
    * vision: the brains that see — [eyes_brain, brain, *backups, the seeing fallbacks] — then the
      text ones, which take only the steps without a picture;
    * offline: [offline_brain].

    In every chain a step with a picture asks the user's Eyes first, then the chain's own brains
    that see, then the seeing fallbacks (cloud_eyes: "off" none). `effort`: the run's thinking
    policy (neo2.core.providers.effort). With the default settings, the chains NEO always had."""
    if own := _hosts(purpose):
        return Chain(own, sticky=purpose != "vision", effort=effort)
    from neo2.core.providers.cerebras_hooks import with_cerebras, with_cerebras_eyes

    s = settings()
    eyes = _fallback_eyes(s)
    # Cerebras (a paid key, DECISIONS 49): its seeing brain ahead of Groq's among the seeing
    # fallbacks — only where cloud fallbacks may see at all (a local-first user's pictures stay here)
    eyes = with_cerebras_eyes(eyes, s) if eyes else eyes
    look = getattr(s, "eyes_brain", "") or ""
    lead = [look] if look and brainref.parse(look).valid else []
    backups = _backups(s)
    more = backups or []
    if purpose == "fast":
        order: list[str | None] = [s.fast_brain, s.brain, *more, s.offline_brain]
    elif purpose == "vision":
        # The brains that can see the screenshot come first; text-only ones take only the steps that
        # carry no picture (an AX-only step): given one, they would act blind. An eyes-only fallback
        # is asked only for a picture, after these (a brain the user named is theirs to ask) — and
        # an eyes-only Eyes brain only as the chain's lead (for a picture).
        implicit = EYES if getattr(s, "cloud_eyes", "auto") != "off" else ()
        vision: list[str | None] = [
            n
            for n in dict.fromkeys((*lead, s.brain, *more, *implicit))
            if n and _sees(n) and (n == s.brain or not _eyes_only(n))
        ]
        order = vision + [n for n in (s.brain, s.fast_brain, *more, s.offline_brain) if n not in vision]
        # (Cerebras: its text brain ahead of Groq; its seeing one after the seeing brains, when enabled)
        return Chain(
            _dedupe(with_cerebras(_dedupe(order), purpose, s)), sticky=False, eyes=eyes, effort=effort, lead=lead
        )
    elif purpose == "offline":
        return Chain(_dedupe([s.offline_brain]), effort=effort)  # (no cloud eyes: offline means here)
    elif purpose == "light":
        # One-tool actions that missed the regex fast path: cheapest capable brain first.
        order = [getattr(s, "light_brain", "gemini_lite") or s.fast_brain, s.fast_brain, s.brain, *more, s.offline_brain]
    else:
        order = [s.brain, *([s.fast_brain] if backups is None else backups), s.offline_brain]
    return Chain(_dedupe(with_cerebras(_dedupe(order), purpose, s)), eyes=eyes, effort=effort, lead=lead)


def camera_order(s: Any = None) -> list[str]:
    """The brains a camera question's still may go to (all asked at once: neo2.core.vision.answer),
    in the user's order: Eyes (their pick for anything with a picture), Brain if it sees, Quick
    replies if it sees, then the camera's own fallbacks (Gemini, Cerebras's vision model) — those
    only where cloud fallbacks may see (cloud_eyes "off": only the brains the user picked). At most
    one brain of each provider: a paid one is never paid twice for one question. Default
    settings: Gemini, then Cerebras's vision model, as before."""
    s = s if s is not None else settings()
    roles = [getattr(s, "eyes_brain", ""), getattr(s, "brain", ""), getattr(s, "fast_brain", "")]
    if getattr(s, "cloud_eyes", "auto") != "off":
        roles += list(catalog.CAMERA_FALLBACKS)
    out: list[str] = []
    seen: set[str] = set()
    for n in dict.fromkeys(roles):
        if not n or not brainref.parse(n).valid or not catalog.describe(n, s).sees:
            continue
        pid = catalog.provider_of(n)
        if pid not in seen:
            seen.add(pid)
            out.append(n)
    return out


def lens_order(s: Any = None) -> list[str]:
    """The brains a Search Screen area may go to (neo2.core.lens.ask: asked one after another, the
    next only when the one before is slow), in the user's order: Eyes, Brain, Quick replies, then
    the Backups — each only if it sees — then the seeing fallbacks every chain gets (with Cerebras's
    seeing brain where it is enabled), at most one brain of each provider (a paid one is never paid
    twice for one area). A user who keeps their brains on this Mac and named no cloud brain that
    sees gets no cloud fallback (`_fallback_eyes`): their area stays on the Mac. camera_order is
    the camera's own and is not this."""
    s = s if s is not None else settings()
    roles = [getattr(s, "eyes_brain", ""), getattr(s, "brain", ""), getattr(s, "fast_brain", ""), *(_backups(s) or ())]
    from neo2.core.providers.cerebras_hooks import with_cerebras_eyes

    eyes = _fallback_eyes(s)
    eyes = with_cerebras_eyes(eyes, s) if eyes else eyes
    out: list[str] = []
    seen: set[str] = set()
    for n in dict.fromkeys((*roles, *eyes)):
        if not n or not brainref.parse(n).valid or not catalog.describe(n, s).sees:
            continue
        pid = catalog.provider_of(n)
        if pid not in seen:
            seen.add(pid)
            out.append(n)
    return out


def rebuild(provider: str, models: Sequence[str]) -> list[str]:
    """Drop NEO.app's built brains of `provider` on any of `models` (what its live list says of them
    changed: whether they see, call tools, what they cost), so the next chain builds them from what
    is known now. Their demotions and the health list stay. Returns the brains dropped."""
    ms = set(models)
    dropped = [k for k in list(_cache) if catalog.provider_of(k) == provider and catalog.describe(k).model in ms]
    for k in dropped:
        _cache.pop(k, None)
    return dropped


def forget(provider: str | None = None, *, keys: bool = True) -> list[str]:
    """Drop what NEO.app's brains of `provider` (a provider id: "anthropic", "custom:deepseek"; None:
    every one) have kept — the built clients, their demotions and "unavailable" notes, Cerebras's
    account, the camera's rests — so the next chain builds them again from the settings as they are
    now (a key just saved, replaced or removed; a daily cap changed). `keys` False (only its cap
    changed): what the health list says of its key, and the model lists kept for it, stay. Returns
    the brains dropped. An embedded agent's brains are its own, never touched here."""

    def mine(name: str) -> bool:
        return provider is None or catalog.provider_of(name) == provider

    dropped = [k for k in list(_cache) if mine(k)]
    for k in dropped:
        _cache.pop(k, None)
    for k in [k for k in list(_demoted) if mine(k)]:
        _demoted.pop(k, None)
    for key in [k for k in list(_said_unavailable) if mine(k[0])]:
        _said_unavailable.discard(key)
    if provider in (None, "cerebras"):
        from neo2.core.providers import cerebras_hooks

        with cerebras_hooks._accounts_lock:
            cerebras_hooks._accounts.pop(_app_pool, None)
    from neo2.core import health
    from neo2.core.providers import keycheck

    if keys:
        keycheck.forget(provider)  # (the model lists kept for its key)
        # (what the health list said of its key or a model: a new key is asked afresh)
        if provider:
            health.report(f"key:{provider}", None)
            health.clear(f"model:{provider}:")
        else:
            health.clear("key:")
            health.clear("model:")
    try:
        from neo2.core.vision import answer
    except Exception:  # noqa: BLE001 — the camera's module isn't there (a core install): nothing kept
        return dropped
    for rests in (answer._late_until, answer._late_seen):
        for k in [k for k in list(rests) if mine(k)]:
            rests.pop(k, None)
    return dropped


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

    def box(self, width: int, height: int) -> tuple[int, int]:
        """The size a width × height picture is scaled to, to fit (its own size when it does)."""
        s = self.scale(width, height)
        if s >= 1.0:
            return width, height
        return max(1, int(width * s)), max(1, int(height * s))

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
    """One seeing brain's picture limit: its own (`image_limit`), else what is known of its kind —
    and, for a brain that says how big one request may be (`max_request_bytes`, base64 pictures
    included), no picture larger than fits in it."""
    lim: ImageLimit | None = None
    own = getattr(p, "image_limit", None)
    if callable(own) and not isinstance(own, ImageLimit):
        try:
            own = own()
        except Exception:  # noqa: BLE001 — a limit it can't say: what is known of its kind
            own = None
    if isinstance(own, ImageLimit):
        lim = own
    elif isinstance(own, tuple) and len(own) == 2:
        # (longest edge, total pixels): a provider's own declaration (a host's, or a Chain's
        # strictest of its seeing brains — Chain.image_limit)
        lim = ImageLimit(int(own[0]), int(own[1]))
    else:
        name, model = str(getattr(p, "name", "")), str(getattr(p, "model", ""))
        if name.startswith("claude") or model.startswith("claude"):
            lim = _claude_limit(model)
        else:
            lim = next((v for k, v in IMAGE_LIMITS.items() if name.startswith(k)), None)
    per_request = getattr(p, "max_request_bytes", None)
    if isinstance(per_request, int) and not isinstance(per_request, bool) and per_request > 0:
        lim = ImageLimit(bytes=per_request * 3 // 4).tighter(lim)  # (base64: 4 bytes per 3)
    return lim


def image_limit(purpose: Purpose = "vision") -> ImageLimit | None:
    """The strictest picture limit among the brains that may see a picture sent for `purpose`
    (the user's Eyes, the chain's own seeing brains and its seeing fallbacks; a host's own providers included), so
    a screenshot is made to fit them all and no API shrinks it silently. None: no limit."""
    chain = brain(purpose)
    out: ImageLimit | None = None
    for p in chain._providers() + chain._eye_brains() + chain._lead_brains():
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


def _key(n: str | Provider) -> str:
    return n if isinstance(n, str) else str(getattr(n, "name", "") or type(n).__name__)


def _dedupe(order: Sequence[str | None]) -> list[str]:
    seen: list[str] = []
    for n in order:
        # None / "": no such brain (an embedded agent's offline one, a role left empty)
        if n and n not in seen:
            seen.append(n)
    return seen
