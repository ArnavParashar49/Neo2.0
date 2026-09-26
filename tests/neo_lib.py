"""Helpers for tests that embed NEO agents (neo.runtime) — built directly, as the SDK will."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from neo import runtime
from neo.agent.registry import catalog
from neo.config import LibrarySettings


def make_runtime(
    data_dir: Path, name: str = "agent", tools: Iterable[str] = (), *, library: bool = True, **settings: Any
) -> runtime.Runtime:
    """An agent with its own LibrarySettings (only `settings` passed in), a strict copy of these
    catalog tools, and its data dir created. Library mode comes on with it, as it does for the
    SDK; `library=False` turns it back off, for a test of NEO.app's own path next to an agent."""
    from neo.tools import load_all

    if not runtime.library_mode():  # NEO.app's loader (task modes read the app's settings)
        load_all()
    s = LibrarySettings(data_dir=data_dir, **settings)
    s.ensure_dirs()
    rt = runtime.Runtime(name, s, catalog().subset(tools))
    if not library:
        runtime.set_library_mode(False)
    return rt


class ScriptedBrain:
    """A fake provider that plays back turns (a str is a text turn) and records what it saw."""

    supports_vision = True
    supports_tools = True

    def __init__(self, turns, *, name: str = "scripted", model: str = "script-1") -> None:
        from neo.providers.base import Turn

        self.turns = [Turn(text=t) if isinstance(t, str) else t for t in turns]
        self.name, self.model = name, model
        self.systems: list[str] = []
        self.tools_seen: list[list[str]] = []
        self.inputs: list[str] = []  # every message text it was sent
        self.calls = 0

    async def complete(self, messages, *, system="", tools=None, effort="medium", max_tokens=4096):
        from neo.providers.base import Turn

        self.calls += 1
        self.systems.append(system)
        self.inputs.extend(m.text for m in messages if getattr(m, "text", ""))
        self.tools_seen.append([t.name for t in tools or []])
        return self.turns.pop(0) if self.turns else Turn(text="(no more turns)")

    async def stream(self, messages, *, system="", max_tokens=2048):
        self.calls += 1
        self.systems.append(system)
        t = self.turns.pop(0) if self.turns else None
        yield t.text if t else "(no more turns)"


def scripted(rt: runtime.Runtime, turns, **kw) -> ScriptedBrain:
    """Make `turns` the agent's only brain, for every purpose."""
    from neo.providers import BrainPool

    brain = ScriptedBrain(turns, **kw)
    rt.brains = BrainPool(overrides=dict.fromkeys(("agent", "fast", "vision", "light", "offline"), [brain]))
    return brain
