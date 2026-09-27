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
down), so nothing stays stuck. The rest of a gesture also stops when the test window is gone (its
app died mid-drag): nothing more of it lands on the app below.

A gesture it never saw begin — another NEO job's, whose button went down just before the test took
the mouse — is let through to its end (cutting it would leave that button stuck in that app), and
the test stops there.

After the checks it keeps listening while NEO cleans up (`disarm`, then `stop_listening`): the user
grabbing the mouse as the window closes is still seen, so the pointer and the front app are left to
them (`took_over`, `active_after_run`).

Accessibility actions (a press, a set value, focusing or raising) post no events, so the event hook
never sees them: `ax_hook` is shown each one instead, before it is made, and refuses (raises `Stop`)
any whose element isn't the test window's — or any once the user has taken over. It is installed at
NEO's one AX-action choke point when the code has it (``AX_CHOKE``), and around each function that
acts through Accessibility today (``AX_ACTIONS``), for as long as the watch is armed.

Bringing an app forward posts no event either, and happens before the click that needed it: the
WindowServer switch (with an AXRaise of the app's window), activation, an AppleScript activate. So
`bring_hook` is shown each way NEO does it (``BRINGS``: by pid, by name as NEO resolves it, by the
running app) before it happens, and refuses any app but the test window's — the user's app is never
activated to be refused only at the mouse-down after.
"""

from __future__ import annotations

import asyncio
import inspect
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


# Where Accessibility actions are made. AX_CHOKE: the module attribute NEO's AX-action choke point
# calls, when it has one, as hook(element, action, value). AX_ACTIONS: each function that acts
# through Accessibility today, and how to find what it acts on (an element, or a hit's pid).
AX_CHOKE = ("neo.tools.computer.ax", "_ax_watch")
AX_ACTIONS: dict[str, dict[str, tuple[str, str]]] = {
    "neo.tools.computer.ax": {
        "press_el": ("element", "press"),
        "press_element": ("element", "press"),
        "set_el_value": ("element", "set value"),
        "focus_el": ("ref", "focus"),
        "raise_window": ("hit", "raise"),
    },
    "neo.decide.media": {"_perform": ("ref", "perform"), "_set": ("ref", "set")},
}

# Each way NEO brings an app forward, and what names the app: "pid" (the argument), "name" (the app
# that name resolves to, as apps.bring_front resolves it), "app" (a running application's pid).
# apps._raise is where bring_front and bring_pid_front both end; force_front is the WindowServer
# switch itself (with its AXRaise), which apps._raise and the self-test's own clean-up call.
BRINGS: dict[str, dict[str, str]] = {
    "neo.tools.computer.apps": {"bring_pid_front": "pid", "bring_front": "name", "_raise": "app"},
    "neo.tools.computer.focus": {"force_front": "pid"},
}

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
        self.mode = "off"  # "tap": the user's events are seen ("tap-mouse": not keys); "pointer": a moved pointer
        self.sent: list[dict] = []  # every event NEO posted during the run
        self.foreign: list[str] = []
        self._down: dict | None = None  # a button NEO pressed and hasn't let go of yet
        self._keys: set[int] = set()  # keys NEO pressed and hasn't let go of yet
        self._last: tuple[float, float] | None = None  # where NEO last put the pointer
        self._tap: Any = None
        self._ours_seen = 0
        self._calibrating = False
        self._run_over: int | None = None  # len(self.foreign) when the checks ended (disarm)
        self._lock = threading.RLock()
        self._mouse, self._keyk = _kinds()
        self.ax_seen: list[dict] = []  # every Accessibility action NEO made during the run
        self.brought: list[dict] = []  # every time NEO brought an app (the test window's) forward
        self._ax_undo: list[tuple[Any, str, Any, bool]] = []

    # ---- life ------------------------------------------------------------------------------
    def arm(self, start: tuple[float, float], *, calibrate_s: float = 0.6) -> str:
        """Install the watch and start listening. `start`: a point inside the test window — the
        calibration move goes there, and it's where the test begins anyway."""
        from neo.tools.computer import input as inp

        self._last = self.mac.pointer()
        inp._watch = self
        self._install_ax()
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
            self.mode = "tap" if getattr(self._tap, "keys", True) else "tap-mouse"
            if self.foreign:  # the user was using the mouse or keyboard while the test began
                self.stopped = self.foreign[0]
        return self.mode

    def disarm(self) -> None:
        """The checks are over: let go of anything still pressed and uninstall. The tap keeps
        listening until `stop_listening`, so what the user does during the clean-up is seen."""
        from neo.tools.computer import input as inp

        with self._lock:
            self.release()
            if inp._watch is self:
                inp._watch = None
            self._uninstall_ax()
            self._run_over = len(self.foreign)

    def stop_listening(self) -> None:
        if self._tap is not None:
            self._tap.stop()
            self._tap = None

    # ---- after the run: is the Mac the user's now? -------------------------------------------
    def pointer_moved(self) -> bool:
        """Is the pointer somewhere other than where NEO last put it?"""
        if self._last is None:
            return False
        x, y = self.mac.pointer()
        return abs(x - self._last[0]) > _POINTER_SLACK or abs(y - self._last[1]) > _POINTER_SLACK

    def took_over(self) -> bool:
        """Did the user take the mouse or keyboard at any point (the clean-up included)?"""
        return bool(self.stopped) or self.pointer_moved()

    def active_after_run(self) -> bool:
        """Is the user using the Mac as NEO cleans up — has the tap heard them since the checks
        ended? (Without a tap nothing tells when a move happened: a pointer that isn't where NEO
        left it counts.)"""
        if self._tap is not None and self._run_over is not None:
            return len(self.foreign) > self._run_over
        return self.pointer_moved()

    # ---- Accessibility actions ---------------------------------------------------------------
    def ax_hook(self, target: Any, action: str = "", value: Any = None, *, pid: int | None = None) -> None:
        """Shown every Accessibility action before it is made: only the test window's elements,
        and nothing at all once the user has taken over. (Armed only: a watch left behind by a run
        that ended is no one's.)"""
        from neo.tools.computer import input as inp

        if inp._watch is not self:
            return
        with self._lock:
            if self.stopped or self.refused:
                raise Stop(self.stopped or self.refused)
            owner = int(pid) if pid is not None else int(self.mac.element_pid(target) or 0)
            if owner != self.test_pid:
                self._refuse(f"an Accessibility {action or 'action'} would have acted on another app (pid {owner or '?'})")
            self.ax_seen.append({"action": action, "pid": owner})

    def bring_hook(self, pid: int, via: str = "") -> None:
        """Shown every time NEO is about to bring an app forward: only the test window's app, and
        nothing at all once the user has taken over. (Armed only, like `ax_hook`.)"""
        from neo.tools.computer import input as inp

        if inp._watch is not self:
            return
        with self._lock:
            if self.stopped or self.refused:
                raise Stop(self.stopped or self.refused)
            if int(pid or 0) != self.test_pid:
                self._refuse(f"NEO would have brought another app forward (pid {pid or '?'}, {via})")
            self.brought.append({"via": via, "pid": int(pid)})

    def _install_ax(self) -> None:
        import importlib

        mod_name, attr = AX_CHOKE
        try:
            choke = importlib.import_module(mod_name)
        except ImportError:
            choke = None
        if choke is not None and hasattr(choke, attr):
            self._ax_undo.append((choke, attr, getattr(choke, attr), True))
            setattr(choke, attr, self.ax_hook)
        for mod_name, fns in AX_ACTIONS.items():
            try:
                mod = importlib.import_module(mod_name)
            except ImportError:
                continue
            for name, (kind, what) in fns.items():
                fn = getattr(mod, name, None)
                if fn is None:
                    continue
                self._ax_undo.append((mod, name, fn, True))
                setattr(mod, name, self._watched(fn, kind, what))
        for mod_name, fns in BRINGS.items():
            try:
                mod = importlib.import_module(mod_name)
            except ImportError:
                continue
            for name, kind in fns.items():
                fn = getattr(mod, name, None)
                if fn is None:
                    continue
                self._ax_undo.append((mod, name, fn, True))
                setattr(mod, name, self._bring_watched(fn, kind, f"{mod_name.rsplit('.', 1)[-1]}.{name}"))

    def _bring_watched(self, fn: Any, kind: str, via: str) -> Any:
        def pid_of(target: Any) -> int | None:
            """The pid this call would bring forward (None: a name no running app has — nothing to
            bring; apps._raise is watched should one appear)."""
            try:
                if kind == "pid":
                    return int(target or 0)
                if kind == "app":
                    return int(target.processIdentifier())
                from neo.tools.computer import apps

                a = apps._running(str(target))
                return int(a.processIdentifier()) if a is not None else None
            except Exception:  # noqa: BLE001 - an app that can't be named is not the test window
                return 0

        if inspect.iscoroutinefunction(fn):

            async def watched(target: Any, *a: Any, **k: Any) -> Any:
                pid = await asyncio.to_thread(pid_of, target) if kind == "name" else pid_of(target)
                if pid is not None:
                    self.bring_hook(pid, via)
                return await fn(target, *a, **k)

        else:

            def watched(target: Any, *a: Any, **k: Any) -> Any:  # type: ignore[misc]
                pid = pid_of(target)
                if pid is not None:
                    self.bring_hook(pid, via)
                return fn(target, *a, **k)

        watched.__wrapped__ = fn  # type: ignore[attr-defined]
        return watched

    def _watched(self, fn: Any, kind: str, what: str) -> Any:
        def watched(target: Any, *a: Any, **k: Any) -> Any:
            if kind == "hit":
                self.ax_hook(None, what, pid=int(getattr(target, "pid", 0) or 0))
            else:
                ref = getattr(target, "ref", None) if kind == "element" else target
                self.ax_hook(ref, (a[0] if a and isinstance(a[0], str) and kind == "ref" else what))
            return fn(target, *a, **k)

        watched.__wrapped__ = fn  # type: ignore[attr-defined]
        return watched

    def _uninstall_ax(self) -> None:
        while self._ax_undo:
            obj, name, old, _present = self._ax_undo.pop()
            setattr(obj, name, old)

    # ---- the hook: every event NEO is about to post --------------------------------------------
    def __call__(self, ev: Any) -> None:
        with self._lock:
            typ = int(Q.CGEventGetType(ev))
            kind = self._mouse.get(typ) or self._keyk.get(typ)
            if self._not_ours_to_cut(kind, ev):
                # The rest of a gesture begun before the watch (another NEO job's): its button or
                # key is already down in that app. It goes through, and the test stops here.
                if not self.stopped:
                    self.stopped = "another NEO job was using the mouse or keyboard"
                x, y = self._where(kind, ev) if typ in self._mouse else (0.0, 0.0)
                Q.CGEventSetIntegerValueField(ev, Q.kCGEventSourceUserData, self.tag)
                self._record(kind or "?", typ, ev, x, y)
                self.sent[-1]["theirs"] = True
                return
            if self.stopped or self.refused:
                self.release()  # (the tap saw the user between two of NEO's events)
                raise Stop(self.stopped or self.refused)
            x = y = 0.0
            if kind is None:
                self._refuse(f"an event of a kind the test doesn't send ({typ})")
            elif typ in self._mouse:
                x, y = self._where(kind, ev)
                if self.mode == "pointer":
                    self._pointer_still()
                if self._down is not None and kind in _GESTURE_REST:
                    if not self._inside(x, y):  # a drag that would leave the test window
                        self._refuse(f"a drag to ({x:.0f}, {y:.0f}) would have left the test window")
                    if not self._still_there(x, y):  # its app died mid-drag: the app below would get the rest
                        self._refuse("the test window closed in the middle of a drag")
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

    def _not_ours_to_cut(self, kind: str | None, ev: Any) -> bool:
        """Is this the rest of a gesture the watch never saw begin: a drag step or a button let go
        while a button is down that the test never pressed, or a key let go it never pressed?"""
        if kind in _GESTURE_REST:
            return self._down is None and bool(self.mac.buttons_down())
        if kind == "key_up":
            return int(Q.CGEventGetIntegerValueField(ev, Q.kCGKeyboardEventKeycode)) not in self._keys
        return False

    def _where(self, kind: str, ev: Any) -> tuple[float, float]:
        if kind == "scroll":
            return self.mac.pointer()  # a scroll goes where the pointer is
        loc = Q.CGEventGetLocation(ev)
        return float(loc.x), float(loc.y)

    # ---- where an event would go -----------------------------------------------------------
    def _inside(self, x: float, y: float) -> bool:
        return any(contains(b, x, y, 1.0) for b in self.windows.values())

    def _still_there(self, x: float, y: float) -> bool:
        """Is the test window under this point still on screen as the test app's own?"""
        for n, b in self.windows.items():
            if contains(b, x, y, 1.0):
                w = self.mac.window(n)
                return w is not None and int(w["pid"]) == self.test_pid
        return False

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
            elif self._down is not None and kind in _GESTURE_REST:
                self._down["at"] = entry["at"]  # a release lets go where the drag is now
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
