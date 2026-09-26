"""Event bus + the single source of truth for NEO's visible state.

Several things can be happening at once — a voice session listening, one agent job reading
mail, another opening an app. Each reports its own state under a *job* id; the orb shows the
aggregate (the most important thing going on). A job that finishes clears its entry, so the
state falls back to whatever else is active instead of flickering through idle.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class NeoState(StrEnum):
    """Semantic states. The overlay maps each one to a thinking-orbs animation."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    WORKING = "working"
    SEARCHING = "searching"
    SPEAKING = "speaking"
    CONNECTING = "connecting"
    CONFIRMING = "confirming"


# What wins when several things are active at once.
_PRIORITY = {
    NeoState.IDLE: 0,
    NeoState.LISTENING: 1,
    NeoState.THINKING: 2,
    NeoState.SEARCHING: 3,
    NeoState.WORKING: 4,
    NeoState.CONNECTING: 5,
    NeoState.SPEAKING: 6,
    NeoState.CONFIRMING: 7,
}


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
        self._base = NeoState.IDLE  # the "global" state (voice session / nothing running)
        self._jobs: dict[str, NeoState] = {}  # per-job states
        self._state = NeoState.IDLE

    @property
    def state(self) -> NeoState:
        return self._state

    @property
    def active_jobs(self) -> list[str]:
        return list(self._jobs)

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

    def _aggregate(self) -> NeoState:
        best = self._base
        for st in self._jobs.values():
            if _PRIORITY[st] > _PRIORITY[best]:
                best = st
        return best

    async def set_state(self, state: NeoState, *, job: str = "", **extra: Any) -> None:
        """Report a state. With a `job`, it's that job's state (IDLE ends the job); without,
        it's the base state. Publishes only when the aggregate actually changes."""
        if job:
            if state == NeoState.IDLE:
                self._jobs.pop(job, None)
            else:
                self._jobs[job] = state
        else:
            self._base = state
        agg = self._aggregate()
        if agg == self._state and not extra:
            return
        self._state = agg
        await self.publish("state", state=agg.value, job=job, **extra)

    async def end_job(self, job: str) -> None:
        await self.set_state(NeoState.IDLE, job=job)

    # Convenience wrappers used all over the codebase -----------------------------------
    async def say(self, text: str, *, final: bool = True, role: str = "assistant", job: str = "") -> None:
        await self.publish("transcript", role=role, text=text, final=final, job=job)

    async def note(self, text: str) -> None:
        """Short progress line for the UI while a turn is in flight (provider fallbacks etc.)."""
        await self.publish("note", text=text)

    async def tool_start(self, name: str, args: dict[str, Any], *, job: str = "") -> None:
        await self.publish("tool_start", name=name, args=args, job=job)

    async def tool_end(self, name: str, ok: bool, summary: str, *, job: str = "") -> None:
        await self.publish("tool_end", name=name, ok=ok, summary=summary[:300], job=job)


_bus: EventBus | None = None


def bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
