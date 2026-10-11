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
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from neo2.core import logger, runtime

log = logger(__name__)


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

# The job whose tool is running now: set as each tool call is announced (tool_start is awaited in
# the task that then runs the tool), so a tool can tell which job called it without a parameter.
current_job: ContextVar[str] = ContextVar("neo_current_job", default="")


class EventBus:
    """Tiny async pub/sub. Subscribers may be sync or async; errors never propagate."""

    def __init__(self) -> None:
        self._subs: list[Subscriber] = []
        self._base = NeoState.IDLE  # the "global" state (voice session / nothing running)
        self._jobs: dict[str, NeoState] = {}  # per-job states
        self._state = NeoState.IDLE
        self._tasks: dict[str, asyncio.Task] = {}  # the task each job runs in (for a scoped stop)
        self._job_end: list[Callable[[str], None]] = []
        # Bumps when the user starts a new sentence or sends a request (spoken or typed) — never for
        # words a model hands over. "Has the user said anything since?" is a comparison of two counts.
        self.user_turns = 0

    @property
    def state(self) -> NeoState:
        return self._state

    @property
    def active_jobs(self) -> list[str]:
        return list(self._jobs)

    def running_jobs(self, *, besides: str = "") -> list[str]:
        """Jobs still at work: their task hasn't finished (a job whose task ended without saying so,
        or that only waits for the user's yes, isn't doing anything)."""
        out = []
        for job, st in self._jobs.items():
            t = self._tasks.get(job)
            if job != besides and st != NeoState.CONFIRMING and (t is None or not t.done()):
                out.append(job)
        return out

    def new_user_turn(self) -> None:
        """The user started a new sentence, or sent a request."""
        self.user_turns += 1

    def subscribe(self, fn: Subscriber) -> Callable[[], None]:
        self._subs.append(fn)
        return lambda: self._subs.remove(fn)

    def on_job_end(self, fn: Callable[[str], None]) -> Callable[[], None]:
        """Call `fn(job)` whenever a job ends (its state goes back to idle)."""
        self._job_end.append(fn)
        return lambda: self._job_end.remove(fn)

    def cancel_job(self, job: str) -> bool:
        """Stop one job — the task it runs in — and nothing else. False when it isn't running."""
        t = self._tasks.get(job)
        if t is None or t.done():
            return False
        t.cancel()
        return True

    async def publish(self, type: str, **data: Any) -> None:
        ev = Event(type=type, data=data)
        for fn in list(self._subs):
            try:
                r = fn(ev)
                if asyncio.iscoroutine(r):
                    await r
            except Exception as e:  # noqa: BLE001 — a bad subscriber must not kill the bus
                log.warning(f"[events] subscriber error on {type}: {e}")

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
                ended = self._jobs.pop(job, None) is not None
                self._tasks.pop(job, None)
                if ended:
                    for fn in list(self._job_end):
                        try:
                            fn(job)
                        except Exception as e:  # noqa: BLE001 — a bad listener must not stop the job ending
                            log.warning(f"[events] job-end listener error: {type(e).__name__}")
            else:
                self._jobs[job] = state
                if job not in self._tasks and (task := asyncio.current_task()) is not None:
                    self._tasks[job] = task
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
        current_job.set(job)  # the caller's task runs the tool next: it can tell which job it is
        await self.publish("tool_start", name=name, args=args, job=job)

    async def tool_end(self, name: str, ok: bool, summary: str, *, job: str = "", kind: str = "") -> None:
        """`kind`: a failure's kind (neo2.core.agent.failures), when the agent loop knows it. Only a
        failure with one carries it: every other tool_end is the event it always was."""
        extra = {"kind": kind} if kind else {}
        await self.publish("tool_end", name=name, ok=ok, summary=summary[:300], job=job, **extra)


_bus: EventBus | None = None


def bus() -> EventBus:
    """The bound conversation's own bus (neo2.core.runtime), else NEO.app's."""
    sc = runtime.current_scope()
    if sc is not None:
        return sc.get_or_create("bus", EventBus)
    runtime.check_unbound("bus()")
    return app_bus()


def app_bus() -> EventBus:
    """NEO.app's own bus, whatever is bound: for NEO.app's parts that listen to it from import time
    on (the chat's browser card), which must never hook into an embedded agent's bus."""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
