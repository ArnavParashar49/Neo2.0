"""Event bus + the single source of truth for NEO's visible state.

The UI's orb is a pure function of :class:`NeoState`. Every subsystem (voice, agent,
tools) publishes events here; the websocket server relays them to the overlay.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class NeoState(StrEnum):
    """Semantic states. The overlay maps each one to a thinking-orbs animation (user-configurable)."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    WORKING = "working"
    SEARCHING = "searching"
    SPEAKING = "speaking"
    CONNECTING = "connecting"
    CONFIRMING = "confirming"


@dataclass
class Event:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "data": self.data, "ts": self.ts}


Subscriber = Callable[[Event], Awaitable[None] | None]


class EventBus:
    """Tiny async pub/sub. Subscribers may be sync or async; errors never propagate."""

    def __init__(self) -> None:
        self._subs: list[Subscriber] = []
        self._state = NeoState.IDLE

    @property
    def state(self) -> NeoState:
        return self._state

    def subscribe(self, fn: Subscriber) -> Callable[[], None]:
        self._subs.append(fn)
        return lambda: self._subs.remove(fn)

    async def publish(self, type: str, **data: Any) -> None:
        ev = Event(type=type, data=data)
        for fn in list(self._subs):
            try:
                r = fn(ev)
                if asyncio.iscoroutine(r):
                    await r
            except Exception as e:  # noqa: BLE001 — a bad subscriber must not kill the bus
                print(f"[events] subscriber error on {type}: {e}")

    async def set_state(self, state: NeoState, **extra: Any) -> None:
        if state == self._state and not extra:
            return
        self._state = state
        await self.publish("state", state=state.value, **extra)

    # Convenience wrappers used all over the codebase -----------------------------------
    async def say(self, text: str, *, final: bool = True, role: str = "assistant") -> None:
        await self.publish("transcript", role=role, text=text, final=final)

    async def note(self, text: str) -> None:
        """Short progress line for the UI while a turn is in flight (provider fallbacks etc.)."""
        await self.publish("note", text=text)

    async def tool_start(self, name: str, args: dict[str, Any]) -> None:
        await self.publish("tool_start", name=name, args=args)

    async def tool_end(self, name: str, ok: bool, summary: str) -> None:
        await self.publish("tool_end", name=name, ok=ok, summary=summary[:300])


_bus: EventBus | None = None


def bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
