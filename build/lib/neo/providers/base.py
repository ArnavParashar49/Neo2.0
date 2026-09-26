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
    pass


class RateLimited(ProviderError):
    """429 — caller may fail over to another brain."""
