"""Provider-neutral message + tool types.

Every brain (Gemini, Groq, local MLX, Claude) translates to/from these. The agent loop
never touches a vendor SDK.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["user", "assistant", "tool"]


@dataclass
class ImagePart:
    data: bytes
    mime: str = "image/png"


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
        effort: Literal["low", "medium", "high"] = "medium",
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


def failure_words(name: str, e: ProviderError) -> str:
    """One brain's failure as a short spoken clause: "Gemini is out of free quota until 9:00"."""
    import time as _time

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
    if kind == "offline":
        return f"{who} isn't available ({str(e)[:60]})" if name != "local" else f"{who} isn't running"
    if kind == "bad_request":
        return f"{who} rejected the request"
    return f"{who} failed ({str(e)[:60]})"


_SPOKEN_NAME = {
    "gemini": "Gemini",
    "gemini_lite": "Gemini Lite",
    "groq": "Groq",
    "local": "the local model",
    "claude": "Claude",
}


def spoken_name(name: str) -> str:
    """A brain as the user hears it named ("gemini_lite" → "Gemini Lite")."""
    return _SPOKEN_NAME.get(name, name.capitalize())


class AllBrainsFailed(ProviderError):
    """Every brain in a chain failed; ``failures`` is (brain name, its error) in the order tried."""

    def __init__(self, failures: list[tuple[str, ProviderError]]) -> None:
        self.failures = failures
        super().__init__("all brains failed: " + "; ".join(f"{n}: {str(e)[:80]}" for n, e in failures)[:300])
        kinds = {getattr(e, "kind", "error") for _, e in failures}
        self.kind = kinds.pop() if len(kinds) == 1 else "error"

    def spoken(self) -> str:
        """Why the task stopped, for the user: which brain, and quota vs error."""
        parts = [failure_words(n, e) for n, e in self.failures]
        if not parts:
            return "I couldn't reach any model, so I stopped."
        said = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
        return f"I had to stop: {said}."
