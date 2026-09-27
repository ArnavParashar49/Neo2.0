"""Deadline waits instead of fixed sleeps: ask whether something happened until it has, or until
time runs out — done at the first change, never sooner than the evidence, never later than a cap.
A fixed ``sleep(0.7)`` after ⌘N waits 0.7 s whether the window took 40 ms or never came.

:func:`until` is the async form (the event loop runs between looks; None at the timeout);
:func:`until_sync` the one for code already on a worker thread. Both read time from one replaceable
clock, so a test runs them on a :class:`FakeClock` (``with wait.use_clock(FakeClock()):``) and never
really sleeps. The real clock reads this module's ``_now`` and ``_sleep`` at call time, which the
``fake_clock`` fixture in tests/conftest.py replaces instead: either way every wait is counted.

Portable code (the agent loop, the session) imports this lazily, where it is about the screen: the
package it lives in needs macOS.

:func:`in_thread` runs a blocking walk on a worker thread and tells it to stop (a
``threading.Event``) when the task awaiting it is cancelled — a tool timeout, the user stopping —
so no walk keeps running on after nobody wants its answer.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

__all__ = [
    "Clock",
    "FakeClock",
    "clock",
    "use_clock",
    "now",
    "sleep",
    "asleep",
    "precise_sleep",
    "until",
    "until_sync",
    "in_thread",
]


# The real clock's parts, looked up at call time (a test's fake_clock swaps _now and _sleep).
_now: Callable[[], float] = time.monotonic
_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
_sleep_sync: Callable[[float], None] = time.sleep

EVERY_S = 0.015  # a poll this often: well under a frame's worth of a person's patience


class Clock:
    """Real time: the monotonic clock and real sleeps (this module's _now, _sleep, _sleep_sync)."""

    def now(self) -> float:
        return _now()

    def sleep(self, s: float) -> None:
        if s > 0:
            _sleep_sync(s)

    async def asleep(self, s: float) -> None:
        await _sleep(max(0.0, s))


class FakeClock(Clock):
    """Virtual time for tests: sleeping advances it at once, and every sleep is recorded. Something
    that "takes time" (a fake app answering slowly) calls :meth:`advance`."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = float(start)
        self.slept: list[float] = []
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self.t

    def advance(self, s: float) -> None:
        with self._lock:
            self.t += max(0.0, float(s))

    def sleep(self, s: float) -> None:
        self.slept.append(max(0.0, s))
        self.advance(s)

    async def asleep(self, s: float) -> None:
        self.slept.append(max(0.0, s))
        self.advance(s)
        await asyncio.sleep(0)  # still a turn of the loop: other tasks run, as they would


clock: Clock = Clock()


@contextlib.contextmanager
def use_clock(c: Clock) -> Iterator[Clock]:
    """Run the block (and whatever it starts) on this clock."""
    global clock
    before = clock
    clock = c
    try:
        yield c
    finally:
        clock = before


def now() -> float:
    return clock.now()


def sleep(s: float) -> None:
    clock.sleep(s)


async def asleep(s: float) -> None:
    await clock.asleep(s)


_SPIN_S = 0.002  # the last moment of a precise sleep is spent yielding, not sleeping


def precise_sleep(s: float) -> None:
    """Sleep ``s`` seconds without the ~25% overshoot of a plain short sleep: most of it asleep,
    the last couple of milliseconds yielding until the deadline (an input's timing: a drag's steps)."""
    if s <= 0:
        return
    if type(clock) is Clock and _sleep_sync is time.sleep:  # real time (a fake clock just moves on)
        end = time.perf_counter() + s
        if s > _SPIN_S:
            time.sleep(s - _SPIN_S)
        while time.perf_counter() < end:
            time.sleep(0)
        return
    clock.sleep(s)


async def _ask(pred: Callable[[], Any], thread: bool) -> Any:
    if thread:
        v = await asyncio.to_thread(pred)
    else:
        v = pred()
    if inspect.isawaitable(v):
        v = await v
    return v


async def until(
    pred: Callable[[], Any] | Callable[[], Awaitable[Any]],
    timeout: float,
    every: float = EVERY_S,
    *,
    thread: bool = False,
) -> Any:
    """Ask ``pred()`` until it answers something truthy — returned at once — or ``timeout`` seconds
    pass: then None. It is asked once even with no time at all, and once more at the deadline, so a
    change at the last moment still counts. ``pred`` may be async; ``thread``: it blocks (an
    Accessibility or window-server call), so it is asked on a worker thread."""
    deadline = now() + max(0.0, float(timeout))
    step = max(0.001, float(every))
    while True:
        v = await _ask(pred, thread)
        if v:
            return v
        left = deadline - now()
        if left <= 0:
            return None
        await asleep(min(step, left))


def until_sync[T](
    pred: Callable[[], T], timeout: float, every: float = EVERY_S, *, stop: threading.Event | None = None
) -> T:
    """:func:`until` for code on a worker thread. ``stop`` set: the wait ends now (its last answer)."""
    deadline = now() + max(0.0, float(timeout))
    step = max(0.001, float(every))
    while True:
        v = pred()
        if v or (stop is not None and stop.is_set()):
            return v
        left = deadline - now()
        if left <= 0:
            return v
        sleep(min(step, left))


async def in_thread[T](fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """``fn(*args, cancel=<Event>, **kwargs)`` on a worker thread. If the task awaiting it is
    cancelled, the event is set, and a walk checking it ends where it is instead of running on."""
    cancel = threading.Event()
    try:
        return await asyncio.to_thread(fn, *args, cancel=cancel, **kwargs)
    except BaseException:
        cancel.set()
        raise
