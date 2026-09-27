"""Autorelease pools for Cocoa calls made off the main thread.

NEO's core is long-running and calls AppKit/Foundation from worker threads (asyncio.to_thread),
which have no autorelease pool of their own: every autoreleased object (NSURL, NSRunningApplication,
AX values...) would pile up for the life of the process. Each entry point runs inside a pool.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable


def pool():
    try:
        import objc

        return objc.autorelease_pool()
    except Exception:  # noqa: BLE001 - no PyObjC (tests on other platforms): nothing to drain
        return contextlib.nullcontext()


def pooled[T](fn: Callable[..., T]) -> Callable[..., T]:
    @functools.wraps(fn)
    def wrapper(*a, **k) -> T:
        with pool():
            return fn(*a, **k)

    return wrapper
