"""The controls self-test's safety: nothing NEO sends during the test goes anywhere but the test
window, and the test stops the moment the user touches the mouse or the keyboard.

While a test runs, NEO's input module shows every event to the `Watch` just before posting it
(neo.tools.computer.input._watch). The watch:

* refuses (raises `Stop`, so the event is never posted) a mouse event anywhere but the test window:
  where a gesture starts (a move, a button going down, a scroll), the point must be inside the test
  window, the window on top there must be it, and Accessibility's own hit test must name its app;
  the rest of a gesture that began there (drags, the button going up) must stay inside it;
* refuses a key event unless the test window's app is the one in front (the keys go there);
* marks each event it lets through as NEO's own (kCGEventSourceUserData), so that
* a listen-only event tap (neo.selftest.mac.Tap) can tell the user's own events from NEO's: the
  first mouse move, click, scroll or key of theirs stops the run. Without a tap — or if the tap
  can't tell the two apart, which a calibration move shows — it falls back to checking that the
  pointer is still where NEO last put it before each mouse event (keys can't be watched then).

When it stops mid-gesture it lets go of whatever NEO pressed (a button still down, keys still
down), so nothing stays stuck.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from typing import Any

import Quartz as Q

from neo.selftest.mac import Rect, contains


class Stop(Exception):
    """The test must stop here: the user took over, or an event wouldn't reach the test window."""


def _kinds() -> tuple[dict[int, str], dict[int, str]]:
    mouse = {
        Q.kCGEventMouseMoved: "move",
        Q.kCGEventLeftMouseDown: "down",
        Q.kCGEventLeftMouseUp: "up",
        Q.kCGEventRightMouseDown: "rdown",
        Q.kCGEventRightMouseUp: "rup",
        Q.kCGEventLeftMouseDragged: "dragged",
        Q.kCGEventRightMouseDragged: "rdragged",
        Q.kCGEventOtherMouseDown: "odown",
        Q.kCGEventOtherMouseUp: "oup",
        Q.kCGEventOtherMouseDragged: "odragged",
        Q.kCGEventScrollWheel: "scroll",
    }
    keys = {Q.kCGEventKeyDown: "key_down", Q.kCGEventKeyUp: "key_up", Q.kCGEventFlagsChanged: "flags"}
    return mouse, keys


_DOWN_UP = {"down": "up", "rdown": "rup", "odown": "oup"}
_GESTURE_REST = {"up", "rup", "oup", "dragged", "rdragged", "odragged"}
_POINTER_SLACK = 3.0  # points: the pointer is "still where NEO put it"


class Watch:
    """See the module doc. `windows`: the test window's numbers (the main one, and the second
    display's if there is one); `bounds` their frames on screen as the window list reports them."""

    def __init__(self, mac: Any, test_pid: int, windows: dict[int, Rect], *, own_pid: int | None = None) -> None:
        self.mac = mac
        self.test_pid = int(test_pid)
        self.windows = dict(windows)
        self.own_pid = own_pid or os.getpid()
        self.tag = secrets.randbits(40) | 1  # this run's mark on NEO's own events
        self.stopped = ""  # what the user did: the run stops
        self.refused = ""  # an event that wouldn't have reached the test window: the run stops
        self.mode = "off"  # "tap": the user's own events are seen; "pointer": only a moved pointer is
        self.sent: list[dict] = []  # every event NEO posted during the run
        self.foreign: list[str] = []
        self._down: dict | None = None  # a button NEO pressed and hasn't let go of yet
        self._keys: set[int] = set()  # keys NEO pressed and hasn't let go of yet
        self._last: tuple[float, float] | None = None  # where NEO last put the pointer
        self._tap: Any = None
        self._ours_seen = 0
        self._calibrating = False
        self._lock = threading.RLock()
        self._mouse, self._keyk = _kinds()

    # ---- life ------------------------------------------------------------------------------
    def arm(self, start: tuple[float, float], *, calibrate_s: float = 0.6) -> str:
        """Install the watch and start listening. `start`: a point inside the test window — the
        calibration move goes there, and it's where the test begins anyway."""
        from neo.tools.computer import input as inp

        self._last = self.mac.pointer()
        inp._watch = self
        self._tap = self.mac.tap(self._seen)
        self._calibrating = self._tap is not None
        inp.move(*start)  # checked like any other event: it must land in the test window
        if self._tap is None:
            self.mode = "pointer"
            return self.mode
        deadline = time.monotonic() + calibrate_s
        while time.monotonic() < deadline and not self._ours_seen:
            time.sleep(0.02)
        self._calibrating = False
        if not self._ours_seen:
            # The tap saw nothing of NEO's (or couldn't tell it apart): watch the pointer instead.
            self._tap.stop()
            self._tap = None
            self.mode = "pointer"
        else:
            self.mode = "tap"
            if self.foreign:  # the user was using the mouse or keyboard while the test began
                self.stopped = self.foreign[0]
        return self.mode

    def disarm(self) -> None:
        """Let go of anything still pressed, stop listening and uninstall."""
        from neo.tools.computer import input as inp

        with self._lock:
            self.release()
            if inp._watch is self:
                inp._watch = None
        if self._tap is not None:
            self._tap.stop()
            self._tap = None

    # ---- the hook: every event NEO is about to post --------------------------------------------
    def __call__(self, ev: Any) -> None:
        with self._lock:
            if self.stopped or self.refused:
                raise Stop(self.stopped or self.refused)
            typ = int(Q.CGEventGetType(ev))
            kind = self._mouse.get(typ) or self._keyk.get(typ)
            x = y = 0.0
            if kind is None:
                self._refuse(f"an event of a kind the test doesn't send ({typ})")
            elif typ in self._mouse:
                if kind == "scroll":
                    x, y = self.mac.pointer()  # a scroll goes where the pointer is
                else:
                    loc = Q.CGEventGetLocation(ev)
                    x, y = float(loc.x), float(loc.y)
                if self.mode == "pointer":
                    self._pointer_still()
                if self._down is not None and kind in _GESTURE_REST:
                    if not self._inside(x, y):  # a drag that would leave the test window
                        self._refuse(f"a drag to ({x:.0f}, {y:.0f}) would have left the test window")
                else:
                    where = self.elsewhere(x, y)
                    if where:
                        self._refuse(f"a {_WHAT.get(kind, 'mouse event')} at ({x:.0f}, {y:.0f}) would have gone to {where}")
            else:
                code = int(Q.CGEventGetIntegerValueField(ev, Q.kCGKeyboardEventKeycode))
                if not (kind == "key_up" and code in self._keys):  # letting go of NEO's own key is fine
                    name, pid = self.mac.front()
                    if pid != self.test_pid:
                        self._refuse(f"keys would have gone to {name or 'another app'}, which is in front")
            Q.CGEventSetIntegerValueField(ev, Q.kCGEventSourceUserData, self.tag)
            self._record(kind or "?", typ, ev, x, y)

    # ---- where an event would go -----------------------------------------------------------
    def _inside(self, x: float, y: float) -> bool:
        return any(contains(b, x, y, 1.0) for b in self.windows.values())

    def elsewhere(self, x: float, y: float) -> str:
        """"" when the test window gets an event at this point, else where it would go instead."""
        top = self._top_window(x, y)
        if not self._inside(x, y):
            return (top["name"] or "another app") if top is not None else "outside the test window"
        if top is None:
            return "nothing (the test window isn't showing there)"
        if top["pid"] != self.test_pid or top["number"] not in self.windows:
            return f"{top['name'] or 'another window'} (it covers the test window there)"
        hit = self.mac.hit_pid(x, y)
        if hit != self.test_pid:
            return f"another app (pid {hit})" if hit else "nothing Accessibility can see"
        return ""

    def _top_window(self, x: float, y: float) -> dict | None:
        for w in self.mac.windows():  # front to back
            if w["alpha"] > 0.01 and w["bounds"][2] > 0 and contains(w["bounds"], x, y):
                return w
        return None

    def _pointer_still(self) -> None:
        """Without a tap: the pointer must be where NEO last put it — else the user moved it."""
        if self._last is None:
            return
        lx, ly = self._last
        for _ in range(4):
            x, y = self.mac.pointer()
            if abs(x - lx) <= _POINTER_SLACK and abs(y - ly) <= _POINTER_SLACK:
                return
            time.sleep(0.015)  # a move NEO just posted may take a moment to land
        self._stop("you moved the mouse")

    # ---- the user's own events (the tap's thread) -------------------------------------------
    def _seen(self, typ: int, ev: Any) -> None:
        pid = int(Q.CGEventGetIntegerValueField(ev, Q.kCGEventSourceUnixProcessID))
        tag = int(Q.CGEventGetIntegerValueField(ev, Q.kCGEventSourceUserData))
        if tag == self.tag or (pid and pid == self.own_pid):
            self._ours_seen += 1
            return
        kind = self._mouse.get(typ) or self._keyk.get(typ) or ""
        what = f"you {_USER_DID.get(kind, 'used the mouse or keyboard')}"
        self.foreign.append(what)
        if not self._calibrating and not self.stopped:
            self.stopped = what  # the next event NEO tries to post stops the run

    # ---- stopping --------------------------------------------------------------------------
    def _refuse(self, why: str) -> None:
        self.refused = why
        self.release()
        raise Stop(why)

    def _stop(self, why: str) -> None:
        self.stopped = why
        self.release()
        raise Stop(why)

    def release(self) -> None:
        """Let go of what NEO pressed and hasn't released (posted without the checks: it only
        finishes NEO's own gesture, in the test window)."""
        with self._lock:
            if self._down is not None:
                up = {"down": Q.kCGEventLeftMouseUp, "rdown": Q.kCGEventRightMouseUp, "odown": Q.kCGEventOtherMouseUp}
                button = {"down": Q.kCGMouseButtonLeft, "rdown": Q.kCGMouseButtonRight, "odown": Q.kCGMouseButtonCenter}
                d, self._down = self._down, None
                ev = Q.CGEventCreateMouseEvent(None, up[d["kind"]], tuple(d["at"]), button[d["kind"]])
                Q.CGEventSetIntegerValueField(ev, Q.kCGEventSourceUserData, self.tag)
                self.mac.post(ev)
                self.sent.append({"kind": _DOWN_UP[d["kind"]], "at": d["at"], "release": True})
            for code in sorted(self._keys):
                ev = Q.CGEventCreateKeyboardEvent(None, code, False)
                Q.CGEventSetIntegerValueField(ev, Q.kCGEventSourceUserData, self.tag)
                self.mac.post(ev)
                self.sent.append({"kind": "key_up", "code": code, "release": True})
            self._keys.clear()

    # ---- what NEO sent ---------------------------------------------------------------------
    def _record(self, kind: str, typ: int, ev: Any, x: float, y: float) -> None:
        entry: dict[str, Any] = {"kind": kind}
        if typ in self._mouse:
            entry["at"] = [round(x, 1), round(y, 1)]
            if kind in _DOWN_UP or kind in ("up", "rup", "oup"):
                entry["clicks"] = int(Q.CGEventGetIntegerValueField(ev, Q.kCGMouseEventClickState))
            if kind == "scroll":
                entry["dy"] = int(Q.CGEventGetIntegerValueField(ev, Q.kCGScrollWheelEventDeltaAxis1))
            if kind in _DOWN_UP:
                self._down = {"kind": kind, "at": entry["at"]}
            elif kind in ("up", "rup", "oup"):
                self._down = None
            if kind != "scroll":
                self._last = (x, y)
        else:
            code = int(Q.CGEventGetIntegerValueField(ev, Q.kCGKeyboardEventKeycode))
            entry["code"] = code
            entry["flags"] = int(Q.CGEventGetFlags(ev)) & 0xFFFF0000
            text = _unicode(ev)
            if text:
                entry["text"] = text
            if kind == "key_down":
                self._keys.add(code)
            elif kind == "key_up":
                self._keys.discard(code)
        self.sent.append(entry)


def _unicode(ev: Any) -> str:
    try:
        _, s = Q.CGEventKeyboardGetUnicodeString(ev, 64, None, None)
        return str(s or "")
    except Exception:  # noqa: BLE001
        return ""


_WHAT = {"move": "pointer move", "down": "click", "rdown": "right click", "odown": "click", "scroll": "scroll"}
_USER_DID = {
    "move": "moved the mouse",
    "down": "clicked",
    "up": "clicked",
    "rdown": "clicked",
    "rup": "clicked",
    "odown": "clicked",
    "oup": "clicked",
    "dragged": "dragged the mouse",
    "rdragged": "dragged the mouse",
    "odragged": "dragged the mouse",
    "scroll": "scrolled",
    "key_down": "pressed a key",
    "key_up": "pressed a key",
    "flags": "pressed a key",
}
