"""Accessibility notifications for the app NEO works in: a count per app that moves whenever the
app tells Accessibility something changed — a window or sheet opened, the focus moved, a title or
a value changed, the layout changed, a menu opened or closed.

So a repeat look at a window nothing has changed in is the last look again (``ax.snapshot``,
``ax.window_elements``: no walk at all), and a wait for "something changed" ends at the first
notification (:func:`wait_change`) instead of polling.

One background thread runs a CFRunLoop holding one AXObserver, for the one app NEO is working in
(moved when NEO works in another). NEO.app only: never in library mode (the process is a host's),
never while real effects are off (a test gives it a fake backend), and not when NEO_AX_WATCH=0.
Chromium and Electron apps send few notifications, so they aren't watched: a count that doesn't
move proves nothing there, and callers keep polling.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from typing import Any

NOTIFICATIONS = (
    "AXFocusedWindowChanged",
    "AXFocusedUIElementChanged",
    "AXWindowCreated",
    "AXSheetCreated",
    "AXUIElementDestroyed",
    "AXTitleChanged",
    "AXValueChanged",
    "AXLayoutChanged",
    "AXMenuOpened",
    "AXMenuClosed",
    "AXSelectedChildrenChanged",
    "AXRowCountChanged",
    "AXWindowMiniaturized",
    "AXWindowMoved",
    "AXWindowResized",
)
# Without these the count can't be trusted to move when a window's contents change.
_NEEDED = ("AXFocusedWindowChanged", "AXUIElementDestroyed", "AXValueChanged", "AXTitleChanged")

OnNote = Callable[[int, str], None]


class Backend:
    """Where notifications come from: ``start(pid, on_note)`` begins delivering the app's
    notifications to ``on_note(pid, name)`` and returns a handle (None: it can't be watched);
    ``stop(handle)`` ends it."""

    def start(self, pid: int, on_note: OnNote) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def stop(self, handle: Any) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class _RealBackend(Backend):
    """An AXObserver on a CFRunLoop of its own thread."""

    def start(self, pid: int, on_note: OnNote) -> Any:
        box: dict[str, Any] = {"stop": False}
        ready = threading.Event()
        t = threading.Thread(target=self._run, args=(pid, on_note, box, ready), name=f"neo-axwatch-{pid}", daemon=True)
        t.start()
        if not ready.wait(1.0) or not box.get("ok"):
            box["stop"] = True
            return None
        box["thread"] = t
        return box

    def _run(self, pid: int, on_note: OnNote, box: dict[str, Any], ready: threading.Event) -> None:
        try:
            import ApplicationServices as AS
            import CoreFoundation as CF
            import objc

            @objc.callbackFor(AS.AXObserverCreate)
            def callback(_observer, _element, name, _refcon) -> None:
                try:
                    on_note(pid, str(name))
                except Exception:  # noqa: BLE001 - a notification must never take the thread down
                    pass

            box["callback"] = callback  # (kept alive as long as the observer)
            err, obs = AS.AXObserverCreate(pid, callback, None)
            if err != 0 or obs is None:
                ready.set()
                return
            app = AS.AXUIElementCreateApplication(pid)
            added = {n for n in NOTIFICATIONS if AS.AXObserverAddNotification(obs, app, n, None) == 0}
            loop = CF.CFRunLoopGetCurrent()
            source = AS.AXObserverGetRunLoopSource(obs)
            CF.CFRunLoopAddSource(loop, source, CF.kCFRunLoopDefaultMode)
            box.update(loop=loop, observer=obs, ok=all(n in added for n in _NEEDED))
            ready.set()
            while not box["stop"]:
                CF.CFRunLoopRunInMode(CF.kCFRunLoopDefaultMode, 0.5, False)
            for n in added:
                AS.AXObserverRemoveNotification(obs, app, n)
            CF.CFRunLoopRemoveSource(loop, source, CF.kCFRunLoopDefaultMode)
        except Exception:  # noqa: BLE001 - no watching: the callers poll
            box["ok"] = False
            ready.set()

    def stop(self, handle: Any) -> None:
        handle["stop"] = True
        try:
            import CoreFoundation as CF

            if handle.get("loop") is not None:
                CF.CFRunLoopStop(handle["loop"])
        except Exception:  # noqa: BLE001
            pass
        t = handle.get("thread")
        if t is not None and t is not threading.current_thread():
            t.join(1.0)


_cond = threading.Condition()
_gens: dict[int, int] = {}
_watched = 0
_handle: Any = None
_backend: Backend | None = None  # None: the real one (NEO.app); a test sets a fake
_last: dict[int, tuple[str, float]] = {}  # pid → (the last notification, when)


def _on_note(pid: int, name: str) -> None:
    with _cond:
        _gens[pid] = _gens.get(pid, 0) + 1
        _last[pid] = (name, time.monotonic())
        _cond.notify_all()


def enabled() -> bool:
    """May NEO watch notifications in this process now? NEO.app only (not a host's process), with
    real effects on — or a test's fake backend — and not switched off (NEO_AX_WATCH=0)."""
    if os.environ.get("NEO_AX_WATCH", "").strip().lower() in ("0", "off", "false", "no"):
        return False
    from neo.tools.computer import ax

    if ax._library():
        return False
    if _backend is not None:
        return True
    from neo import effects

    return effects.on()


def watch(pid: int) -> bool:
    """Watch this app's notifications from now on (and stop watching the one before). True when
    its count can be trusted to move."""
    global _watched, _handle
    if not pid or not enabled():
        return False
    with _cond:
        if _watched == pid and _handle is not None:
            return True
    stop()
    backend = _backend or _RealBackend()
    handle = backend.start(int(pid), _on_note)
    if handle is None:
        return False
    with _cond:
        _watched, _handle = int(pid), handle
        _gens.setdefault(int(pid), 0)
    return True


def stop() -> None:
    """Stop watching (the counts of apps no longer watched mean nothing)."""
    global _watched, _handle
    with _cond:
        handle, _handle = _handle, None
        _watched = 0
        _cond.notify_all()
    if handle is not None:
        try:
            (_backend or _RealBackend()).stop(handle)
        except Exception:  # noqa: BLE001
            pass


def watching() -> int:
    """The pid being watched (0: none)."""
    with _cond:
        return _watched if _handle is not None else 0


def generation(pid: int) -> int | None:
    """How many notifications the app has sent since NEO began watching it; None when it isn't
    watched (then nothing can be told from it)."""
    with _cond:
        if not pid or pid != _watched or _handle is None:
            return None
        return _gens.get(int(pid), 0)


def wait_change(pid: int, since: int, timeout: float) -> bool | None:
    """Block until the app's count moves past ``since`` (True), or ``timeout`` passes (False);
    None when the app isn't watched (the caller polls instead)."""
    end = time.monotonic() + max(0.0, timeout)
    with _cond:
        while True:
            if not pid or pid != _watched or _handle is None:
                return None
            if _gens.get(int(pid), 0) != since:
                return True
            left = end - time.monotonic()
            if left <= 0:
                return False
            _cond.wait(left)


async def wait_change_async(pid: int, since: int, timeout: float) -> bool | None:
    """:func:`wait_change` off the event loop."""
    import asyncio

    return await asyncio.to_thread(wait_change, pid, since, timeout)


def last(pid: int) -> tuple[str, float] | None:
    """(the app's last notification, when) — for a report of what changed."""
    with _cond:
        return _last.get(int(pid))


def use_backend(backend: Backend | None) -> None:
    """Tests: notifications come from ``backend`` (None: the real one again). Stops any watching."""
    global _backend
    stop()
    _backend = backend
    with _cond:
        _gens.clear()
        _last.clear()
