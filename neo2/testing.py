"""Helpers for testing code that embeds NEO: a brain that plays back a script, no network.

    from neo2.testing import ScriptedBrain, Turn, ToolCall

    brain = ScriptedBrain([Turn(tool_calls=[ToolCall("1", "clock", {})]), "It's noon."])
    agent = neo2.AsyncAgent(neo2.AgentConfig(tools="web"), provider=brain)
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from neo2.core.providers.base import Message, ToolCall, ToolSpec, Turn

__all__ = ["ScriptedBrain", "ToolCall", "Turn"]


class ScriptedBrain:
    """A provider that answers with `turns` in order (a str is a text turn) and records what it
    was sent: `systems` (system prompts), `tools_seen` (tool names offered, per call), `inputs`
    (message texts) and `calls`. When the script runs out it answers "(no more turns)"."""

    supports_vision = True
    supports_tools = True

    def __init__(self, turns: Sequence[Turn | str] = (), *, name: str = "scripted", model: str = "script-1") -> None:
        self.turns: list[Turn] = [Turn(text=t) if isinstance(t, str) else t for t in turns]
        self.name, self.model = name, model
        self.systems: list[str] = []
        self.tools_seen: list[list[str]] = []
        self.inputs: list[str] = []
        self.calls = 0

    def add(self, *turns: Turn | str) -> None:
        self.turns.extend(Turn(text=t) if isinstance(t, str) else t for t in turns)

    def _next(self) -> Turn:
        return self.turns.pop(0) if self.turns else Turn(text="(no more turns)")

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        tools: list[ToolSpec] | None = None,
        effort: Any = "medium",
        max_tokens: int = 4096,
    ) -> Turn:
        self.calls += 1
        self.systems.append(system)
        self.inputs.extend(m.text for m in messages if m.text)
        self.tools_seen.append([t.name for t in tools or []])
        return self._next()

    async def stream(self, messages: list[Message], *, system: str = "", max_tokens: int = 2048) -> AsyncIterator[str]:
        self.calls += 1
        self.systems.append(system)
        self.inputs.extend(m.text for m in messages if m.text)
        yield self._next().text
