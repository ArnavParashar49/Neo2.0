"""Everything the controls self-test does to the Mac itself — outside NEO's own tools, which it
drives like a model would.

Starting and stopping the test window, reading the window list, asking which app is in front and
what is under a point, bringing the user's app back, and a listen-only event tap that notices the
user's own mouse and keys. One class, so a test swaps all of it for a fake; and while real effects
are off (neo2.core.effects: under pytest, or NEO_NO_EFFECTS=1) every method that would touch the real Mac
refuses — reading it included.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from neo2.core import effects

Rect = tuple[float, float, float, float]  # x, y, width, height: points, top-left origin (Quartz)


def _no_effects_here(what: str = "look at or act on the real Mac") -> None:
    effects.guard("selftest", f"{what} for the controls self-test")


class Child:
    """The test window's process: NEO's own child, fed commands on stdin."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self.proc = proc
        self.pid = int(proc.pid)

    def send(self, cmd: str) -> None:
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write((json.dumps({"cmd": cmd}) + "\n").encode())
            self.proc.stdin.flush()
        except (OSError, ValueError, AssertionError):
            pass  # gone already

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self, grace: float = 1.5) -> None:
        """Ask it to quit, then insist. Returns once the process is gone."""
        if self.alive():
            self.send("quit")
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(grace)
        if self.alive():
            self.proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(1.0)
        if self.alive():
            self.proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(1.0)
        with contextlib.suppress(OSError):
            if self.proc.stdin is not None:
                self.proc.stdin.close()


# The events the tap listens to: every way a person can use the mouse or the keyboard.
def _tap_types(keys: bool = True) -> tuple[int, ...]:
    import Quartz as Q

    mouse = (
        Q.kCGEventMouseMoved, Q.kCGEventLeftMouseDown, Q.kCGEventLeftMouseUp, Q.kCGEventRightMouseDown,
        Q.kCGEventRightMouseUp, Q.kCGEventLeftMouseDragged, Q.kCGEventRightMouseDragged,
        Q.kCGEventOtherMouseDown, Q.kCGEventOtherMouseUp, Q.kCGEventOtherMouseDragged, Q.kCGEventScrollWheel,
    )  # fmt: skip
    return (*mouse, Q.kCGEventKeyDown, Q.kCGEventKeyUp, Q.kCGEventFlagsChanged) if keys else mouse


class Tap:
    """A listen-only event tap on a thread of its own: it sees every mouse and key event in this
    login session (NEO's own too) and hands each to `on_event(type, event)`. It can't change or
    block anything. Listening to keys can take Input Monitoring on top of Accessibility: without
    it the tap listens to the mouse alone (`keys` is False)."""

    def __init__(self, on_event: Callable[[int, Any], None]) -> None:
        self._on_event = on_event
        self._tap = None
        self.keys = False
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, timeout: float = 1.0) -> bool:
        self._thread = threading.Thread(target=self._run, daemon=True, name="neo-selftest-tap")
        self._thread.start()
        self._ready.wait(timeout)
        return self._tap is not None

    def _run(self) -> None:
        import Quartz as Q

        def callback(proxy, typ, event, refcon):
            if typ in (Q.kCGEventTapDisabledByTimeout, Q.kCGEventTapDisabledByUserInput):
                if self._tap is not None:
                    Q.CGEventTapEnable(self._tap, True)
                return event
            try:
                self._on_event(int(typ), event)
            except Exception:  # noqa: BLE001 - a listener must never break the event stream
                pass
            return event

        self._callback = callback  # kept alive as long as the tap
        tap = None
        for keys in (True, False):
            mask = 0
            for t in _tap_types(keys):
                mask |= Q.CGEventMaskBit(t)
            tap = Q.CGEventTapCreate(
                Q.kCGSessionEventTap, Q.kCGHeadInsertEventTap, Q.kCGEventTapOptionListenOnly, mask, callback, None
            )
            if tap is not None:
                self.keys = keys
                break
        if tap is None:
            self._ready.set()
            return
        source = Q.CFMachPortCreateRunLoopSource(None, tap, 0)
        Q.CFRunLoopAddSource(Q.CFRunLoopGetCurrent(), source, Q.kCFRunLoopCommonModes)
        Q.CGEventTapEnable(tap, True)
        self._tap = tap
        self._ready.set()
        while not self._stop.is_set():
            Q.CFRunLoopRunInMode(Q.kCFRunLoopDefaultMode, 0.1, False)
        Q.CGEventTapEnable(tap, False)
        Q.CFMachPortInvalidate(tap)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(1.0)


class Mac:
    """The real Mac. Every method that touches it refuses while real effects are off (neo2.core.effects)."""

    def trusted(self) -> bool:
        from neo2.core.tools.computer import ax

        return bool(ax.is_trusted())

    def can_capture(self) -> bool:
        _no_effects_here()
        from neo2.core.tools.computer import screen

        return bool(screen.can_capture())

    def launch(self, log: Path, tag: str) -> Child:
        """Start the test window (neo2.core.selftest.testwin) as NEO's own child process."""
        _no_effects_here()
        cmd = [sys.executable, "-m", "neo2.core.selftest.testwin", "--log", str(log), "--tag", tag, "--parent", str(os.getpid())]
        with open(log.with_suffix(".stderr"), "wb") as err:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err, close_fds=True)
        return Child(proc)

    def windows(self) -> list[dict]:
        """The windows on screen, front to back: pid, owner name, number, layer, alpha, bounds."""
        _no_effects_here()
        import Quartz as Q

        opts = Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements
        return [_window(w) for w in (Q.CGWindowListCopyWindowInfo(opts, Q.kCGNullWindowID) or [])]

    def window(self, number: int) -> dict | None:
        """One window by its number (on screen or not)."""
        _no_effects_here()
        import Quartz as Q

        info = Q.CGWindowListCopyWindowInfo(Q.kCGWindowListOptionIncludingWindow, int(number)) or []
        return _window(info[0]) if len(info) else None

    def front(self) -> tuple[str, int]:
        _no_effects_here()
        from neo2.core.tools.computer.focus import front_app

        return front_app()

    def pointer(self) -> tuple[float, float]:
        _no_effects_here()
        from neo2.core.tools.computer import input as inp

        return inp.mouse_position()

    def hit_pid(self, x: float, y: float) -> int:
        """The pid of the app whose element is under a point (Accessibility's own hit test)."""
        _no_effects_here()
        import ApplicationServices as AS

        from neo2.core.tools.computer import ax

        try:
            # NEO's one system-wide element: its timeout is the process's default (2 s as NEO.app),
            # which a test must leave as it found it.
            err, el = AS.AXUIElementCopyElementAtPosition(ax._system_wide(), float(x), float(y), None)
            if err or el is None:
                return 0
            return self.element_pid(el)
        except Exception:  # noqa: BLE001
            return 0

    def element_pid(self, ref: Any) -> int:
        _no_effects_here()
        import ApplicationServices as AS

        try:
            err, pid = AS.AXUIElementGetPid(ref, None)
        except Exception:  # noqa: BLE001
            return 0
        return 0 if err else int(pid)

    def pids_named(self, name: str) -> list[int]:
        """Running apps called exactly `name` (LaunchServices, asked fresh)."""
        _no_effects_here()
        from neo2.core.tools.computer.focus import pid_running

        return [pid for n, pid in pid_running(name) if n == name]

    def activate(self, pid: int) -> bool:
        """Bring an app back to the front by its pid (the one the user was in before the test)."""
        _no_effects_here()
        from AppKit import NSRunningApplication

        from neo2.core.tools.computer.focus import force_front, front_app

        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(int(pid))
        if app is None or app.isTerminated():
            return False
        for _ in range(2):
            force_front(int(pid))
            for _ in range(8):
                if front_app()[1] == pid:
                    return True
                time.sleep(0.05)
        app.activateWithOptions_(1 << 1)
        time.sleep(0.2)
        return front_app()[1] == pid

    def buttons_down(self) -> bool:
        """Is any mouse button down right now — the user's, or one NEO pressed (Quartz's combined
        state of this login session)?"""
        _no_effects_here()
        import Quartz as Q

        state = Q.kCGEventSourceStateCombinedSessionState
        buttons = (Q.kCGMouseButtonLeft, Q.kCGMouseButtonRight, Q.kCGMouseButtonCenter)
        return any(bool(Q.CGEventSourceButtonState(state, b)) for b in buttons)

    def post(self, ev: Any) -> None:
        """Post an event without the watcher (only to let go of what NEO itself pressed)."""
        _no_effects_here()
        import Quartz as Q

        Q.CGEventPost(Q.kCGHIDEventTap, ev)

    def tap(self, on_event: Callable[[int, Any], None]) -> Tap | None:
        _no_effects_here()
        tap = Tap(on_event)
        return tap if tap.start() else None


def _window(w: Any) -> dict:
    b = w.get("kCGWindowBounds") or {}
    return {
        "pid": int(w.get("kCGWindowOwnerPID", 0)),
        "name": str(w.get("kCGWindowOwnerName", "") or ""),
        "number": int(w.get("kCGWindowNumber", 0)),
        "layer": int(w.get("kCGWindowLayer", 0)),
        "alpha": float(w.get("kCGWindowAlpha", 1.0)),
        "bounds": (float(b.get("X", 0)), float(b.get("Y", 0)), float(b.get("Width", 0)), float(b.get("Height", 0))),
    }


def contains(r: Rect, x: float, y: float, margin: float = 0.0) -> bool:
    return r[0] + margin <= x < r[0] + r[2] - margin and r[1] + margin <= y < r[1] + r[3] - margin
