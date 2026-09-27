"""Waiting for the screen instead of sleeping: poll until something is so, then go on at once.

A fixed ``sleep(0.7)`` after ⌘N waits 0.7 s whether the window took 40 ms or never came; ``until``
returns at the first poll that sees it, and says so (or None at the timeout). Its clock and sleep
are this module's ``_now`` and ``_sleep``, which a test's fake clock replaces (the ``fake_clock``
fixture in tests/conftest.py): every wait is counted, and nothing really sleeps.

Portable code (the agent loop, the session) imports this lazily, where it is about the screen:
the package it lives in needs macOS.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable

_now: Callable[[], float] = time.monotonic
_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep

EVERY_S = 0.015  # a poll this often: well under a frame's worth of a person's patience


async def until[T](pred: Callable[[], T | Awaitable[T]], timeout: float, every: float = EVERY_S) -> T | None:
    """``pred()``'s first truthy answer, polled now and then every ``every`` seconds until
    ``timeout`` has passed (None then). ``pred`` may be async; a blocking check (an AX read) is
    wrapped by the caller (``lambda: asyncio.to_thread(ax.x)``) so the loop never stalls."""
    end = _now() + max(0.0, timeout)
    while True:
        got = pred()
        if inspect.isawaitable(got):
            got = await got
        if got:
            return got
        left = end - _now()
        if left <= 0:
            return None
        await _sleep(min(max(every, 0.001), left))
