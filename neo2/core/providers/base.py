"""Provider-neutral message + tool types.

Every brain (Gemini, Groq, local MLX, Claude) translates to/from these. The agent loop
never touches a vendor SDK.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Collection
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["user", "assistant", "tool"]
# How hard a brain thinks for one reply. "minimal" only where the model takes it (a brain that
# doesn't gets the next level up: see neo2.core.providers.effort).
Level = Literal["minimal", "low", "medium", "high"]
# How much of a picture a brain looks at: "auto" is the brain's own default. Gemini bills a fixed
# number of tokens per level (neo2.core.providers.fit); the others size the picture themselves.
Detail = Literal["auto", "low", "medium", "high", "ultra"]


@dataclass
class ImagePart:
    data: bytes
    mime: str = "image/png"
    # How much of it the model should look at: "auto" (the provider's own choice), or "low" /
    # "medium" / "high" / "ultra" — mapped by each provider to its own sizing (Gemini's
    # media_resolution, …). A zoom or a board wants "high"; an after-look "medium".
    detail: Detail = "auto"


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]
    signature: bytes | None = None  # Gemini thought_signature — must be echoed back verbatim
    # The model that wrote this call. Its signature is only valid for that model: after a switch
    # (quota, overload) another one gets the documented bypass value instead.
    source: str = ""


@dataclass
class ToolResult:
    call_id: str
    name: str
    content: str
    ok: bool = True
    images: list[ImagePart] = field(default_factory=list)
    # What the action did, as NEO saw it (the tool's ToolOutput.effect: changed, scene_changed,
    # sheet, looked), when the tool reported it. Never sent to a brain — its words already say it;
    # the thinking level of the next step is read from it (neo2.core.providers.effort).
    effect: Any = None


@dataclass
class Message:
    role: Role
    text: str = ""
    images: list[ImagePart] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant only
    tool_results: list[ToolResult] = field(default_factory=list)  # tool only
    raw: Any = None  # provider-specific replay payload (e.g. Claude content blocks with thinking)

    @staticmethod
    def user(text: str, images: list[ImagePart] | None = None) -> Message:
        return Message(role="user", text=text, images=images or [])

    @staticmethod
    def assistant(text: str = "", tool_calls: list[ToolCall] | None = None, raw: Any = None) -> Message:
        return Message(role="assistant", text=text, tool_calls=tool_calls or [], raw=raw)

    @staticmethod
    def tool(results: list[ToolResult]) -> Message:
        return Message(role="tool", tool_results=results)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema (object)
    # Never dropped when a request is fitted into a small brain (neo2.core.providers.fit): `more_tools`,
    # which gets any other tool back, and every tool of a run fenced to its tools (nothing could).
    essential: bool = False


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Turn:
    """One model reply: text and/or tool calls."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    thinking: str = ""
    raw: Any = None
    # The tools the brain was actually shown for this reply, when fitting its request left some
    # out (None: all it was given). `more_tools` hands back from these, not from the full list.
    offered: list[str] | None = None
    # False: the brain that wrote this reply has no vision — no picture it was sent reached it, so
    # any screen coordinates in its calls were never seen (the agent loop refuses those).
    sees: bool = True


# What a text-only brain reads in place of a picture: plainly that it can't see it (a note like
# "[screenshot attached]" let one click at coordinates it made up — the Chess run).
BLIND_PICTURE = (
    "[You can't see this picture: this model has no vision. Don't click, drag or scroll at "
    "coordinates from it — use ax_tree element ids or click_text, or tell the user the screen "
    "can't be read right now.]"
)


# The words the input tools refuse a blind brain's coordinates with (neo2.core.tools.computer.pointer.blind):
# a request that holds that refusal is a pixel task — the chain then asks only brains that see.
BLIND_REFUSAL = "you can't see the screen"


class Provider(Protocol):
    name: str
    supports_vision: bool
    supports_tools: bool

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        effort: Level = "medium",
        max_tokens: int = 4096,
    ) -> Turn: ...

    def stream(
        self,
        messages: list[Message],
        *,
        system: str = "",
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        """Text-only streaming for chat turns (no tools)."""
        ...


class ProviderError(RuntimeError):
    """A brain couldn't answer. ``kind`` says why, in words the failover chain acts on and the
    user can be told: see :func:`failure_words`."""

    kind = "error"


class RateLimited(ProviderError):
    """429 — caller may fail over to another brain. ``retry_after``: seconds the brain asked us
    to wait, when it said (a per-minute limit clears in seconds; a quota does not)."""

    kind = "rate"

    def __init__(self, msg: str = "", *, retry_after: float | None = None) -> None:
        super().__init__(msg)
        self.retry_after = retry_after


class QuotaExhausted(RateLimited):
    """Every key/model of this brain is out of quota until ``until`` (epoch seconds): asking again
    before then only costs round trips."""

    kind = "quota"

    def __init__(self, msg: str = "", *, until: float = 0.0) -> None:
        import time as _time

        super().__init__(msg, retry_after=max(0.0, until - _time.time()) if until else None)
        self.until = until


class BadRequest(ProviderError):
    """The brain rejected *this request* (a 400): another brain may take it; this one is fine."""

    kind = "bad_request"


class NoTools(BadRequest):
    """Not asked: the step offers tools and this brain can't call them (an endpoint saved with
    Tools off). A chat turn without tools still goes to it."""

    kind = "no_tools"


class RequestTooLarge(BadRequest):
    """The request doesn't fit this brain's context or per-request token limit, even trimmed."""

    kind = "too_large"


class Overloaded(ProviderError):
    """The brain is up but too busy to answer (5xx); it usually clears within a minute or two."""

    kind = "busy"

    def __init__(self, msg: str = "", *, retry_after: float | None = None) -> None:
        super().__init__(msg)
        self.retry_after = retry_after


class Unavailable(ProviderError):
    """The brain isn't there at all: no key, the local server isn't running, no network."""

    kind = "offline"


class KeyRefused(Unavailable):
    """The provider refused the API key (a 401, or a 403 about the key): asking again before the
    user changes it only costs round trips. The chain demotes this brain for an hour."""

    kind = "key"


class NoBrain(ProviderError):
    """No brain is set up that could answer: no key for any provider NEO may ask, and no model on
    this Mac. Brains & Keys is where that changes."""

    kind = "no_brain"


class Blind(ProviderError):
    """Not asked: the step works from a picture and this brain has no vision (it would act blind)."""

    kind = "blind"


class TooSlow(ProviderError):
    """Not asked, or given up on: its answer (or the wait it asked for) would come after the run's
    time is up."""

    kind = "budget"


def failure_words(name: str, e: ProviderError) -> str:
    """One brain's failure as a short spoken clause: "Gemini is out of free quota until 9:00"."""
    import time as _time

    if isinstance(w := getattr(e, "words", None), str) and w:
        return w  # its own words (Cerebras: a spent budget, a refused key — not "out of free quota")
    who = spoken_name(name)
    kind = getattr(e, "kind", "error")
    if kind == "quota":
        until = getattr(e, "until", 0.0)
        when = ""
        if until and until - _time.time() < 36 * 3600:
            t = _time.localtime(until)
            when = f" until {t.tm_hour % 12 or 12}:{t.tm_min:02d} {'AM' if t.tm_hour < 12 else 'PM'}"
        return f"{who} is out of free quota{when}"
    if kind == "rate":
        return f"{who} is rate-limited right now"
    if kind == "busy":
        return f"{who} is overloaded right now"
    if kind == "too_large":
        return f"{who} couldn't take a request this long"
    if kind == "key":
        return f"{who} refused its API key"
    if kind == "no_brain":
        return "no AI model is set up"
    if kind == "offline":
        return f"{who} isn't available ({str(e)[:60]})" if name != "local" else f"{who} isn't running"
    if kind == "bad_request":
        return f"{who} rejected the request"
    if kind == "blind":
        return f"{who} can't see pictures"
    if kind == "no_tools":
        return f"{who} can't use tools"
    if kind == "budget":
        return f"{who} couldn't answer in the time left"
    return f"{who} failed ({str(e)[:60]})"


_SPOKEN_NAME = {
    "gemini": "Gemini",
    "gemini_lite": "Gemini Lite",
    "groq": "Groq",
    "groq_vision": "Groq's vision model",
    "cerebras": "Cerebras",
    "cerebras_vision": "Cerebras vision",
    "local": "the local model",
    "claude": "Claude",
}


def spoken_name(name: str) -> str:
    """A brain as the user hears it named ("gemini_lite" → "Gemini Lite"; a brain reference by its
    model's or provider's name from the catalogue: "Claude Haiku 5.5", "OpenAI", an endpoint's label)."""
    if name in _SPOKEN_NAME:
        return _SPOKEN_NAME[name]
    from neo2.core.brainref import parse

    if parse(name).valid:
        from neo2.core.providers.catalog import spoken

        return spoken(name)
    from neo2.core.brainref import shown

    # (a host's own provider's name, as before; never what could be a key pasted as a brain)
    return name.capitalize() if shown(name) or not name else "a brain NEO doesn't know"


class AllBrainsFailed(ProviderError):
    """Every brain in a chain failed; ``failures`` is (brain name, its error) in the order tried."""

    def __init__(self, failures: list[tuple[str, ProviderError]]) -> None:
        self.failures = failures
        super().__init__("all brains failed: " + "; ".join(f"{n}: {str(e)[:80]}" for n, e in failures)[:300])
        kinds = {getattr(e, "kind", "error") for _, e in failures}
        self.kind = kinds.pop() if len(kinds) == 1 else "error"

    def spoken(self) -> str:
        """Why the task stopped, for the user: which brain, and quota vs error."""
        said = self._said()
        if not said:
            return "I couldn't reach any model, so I stopped."
        return f"I had to stop: {said}."

    def _said(self, failures: list[tuple[str, ProviderError]] | None = None) -> str:
        # (each reason once: two brains on one account refused for one reason say it once)
        parts = list(
            dict.fromkeys(failure_words(n, e) for n, e in (self.failures if failures is None else failures))
        )
        if not parts:
            return ""
        return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]

    def _asked(self) -> list[tuple[str, ProviderError]]:
        """The failures of the brains that were asked: a brain left out of a picture step because
        it can't see (Blind) was never asked, and saying so first hides why the others failed."""
        asked = [(n, e) for n, e in self.failures if getattr(e, "kind", "") != "blind"]
        return asked or self.failures


class NoEyes(AllBrainsFailed):
    """A step working from a picture (a screenshot it took, a photo sent), and no brain that can
    see it could answer: the run stops and says so rather than hand the picture to a brain that
    would act blind (DECISIONS 39, 43).

    `seeing`: the brains that see which were asked and failed — none when no brain that sees is set
    up at all (then the words say that, not that one is busy). None: every failure that isn't Blind.
    `photo`: the picture is one the user sent, not a screenshot of the screen."""

    def __init__(
        self,
        failures: list[tuple[str, ProviderError]],
        *,
        seeing: Collection[str] | None = None,
        photo: bool = False,
    ) -> None:
        super().__init__(failures)
        self.kind = "no_eyes"
        self.photo = photo
        if seeing is None:
            seeing = {n for n, e in failures if getattr(e, "kind", "") != "blind"}
        self.seeing = frozenset(seeing)

    def spoken(self) -> str:
        what = "the picture" if self.photo else "the screen"
        # Why the brains that see couldn't answer; that the others can't see goes without saying.
        tried = [(n, e) for n, e in self.failures if n in self.seeing and getattr(e, "kind", "") != "blind"]
        if not tried:
            return (
                f"None of the models set up for me can see {what}, so I stopped rather than act blind. "
                "I can still press buttons and menu items by their names; to work from pictures I need a "
                "model that sees (a Gemini key, or Groq's vision model)."
            )
        return (
            f"The model that can see {what} is busy right now ({self._said(tried)}), so I stopped rather "
            "than act blind. I can still press buttons and menu items by their names, or you can try "
            "again in a minute."
        )


class OutOfTime(AllBrainsFailed):
    """The run's time is up before any brain could answer this step (or the wait one asked for would
    end past it): it stops now rather than overrun its limit."""

    def __init__(self, failures: list[tuple[str, ProviderError]], *, limit_s: float = 0.0) -> None:
        super().__init__(failures)
        self.kind = "budget"
        self.limit_s = limit_s

    def spoken(self) -> str:
        said = self._said(self._asked())
        limit = f" ({int(self.limit_s)}s)" if self.limit_s else ""
        head = f"I had to stop: {said}, and" if said else "I had to stop:"
        return f"{head} waiting any longer would run past my time limit{limit}."
