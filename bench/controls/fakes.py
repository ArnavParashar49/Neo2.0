"""A pretend Mac for the controls bench: NEO's real tools run against it, and it counts everything.

* ``FakeAS`` — Accessibility. Apps are trees of ``FakeEl``; every call is recorded (``calls``), a
  hung element costs the messaging timeout in virtual time, and pressing or setting an element
  changes the world (a menu item that opens a sheet, a sheet's Start that starts a new game).
* ``World`` — the window server: which app is in front, the windows front to back, what is under
  a point, and where every posted mouse and key event lands (a board that takes clicks and drags,
  fields that take typing, menu shortcuts, scroll areas). ``acted`` lists everything that was
  pressed, clicked or typed into, for the scenarios' checks.
* ``FakeScreen`` — ``screencapture`` answered with synthetic 3024×1964 Retina frames of the world
  (boards with pieces, sheets, small buttons), in the format asked for; every capture counted.
* ``FakeClock`` — every ``asyncio.sleep``, ``time.sleep`` (and so every ``wait.until`` poll) is
  counted and returns at once; ``time.monotonic`` moves on by what was slept, so deadlines hold.
* ``FakeBrain`` — the scripted model: records each request (effort, prompt tokens at fit's 3.4
  bytes per token, image tokens by ``ImagePart.detail``) and answers with its policy's next turn.
* ``Meter`` — the time spent inside fakes, so "Python ms" is NEO's own work only.

``install(world)`` puts it all in place at the lowest layers — ApplicationServices' and Quartz's
functions, AppKit's running-app classes, ``screencapture`` — and takes it all out again, so NEO's
own code above them (ax, pointer, input, screen, the tool handlers, the loop) is what's measured.
Anything the fakes miss is refused by NEO_NO_EFFECTS=1 instead of reaching the Mac.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import importlib
import json
import os
import subprocess
import sys
import threading
import time
import types
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

# ---- the cost model ---------------------------------------------------------------------------

AX_LATENCY_MS = (0.05, 0.15, 0.5, 0.9)  # per Accessibility call: measured spread across apps
HEADLINE_AX_MS = 0.15
CAPTURE_S = {"cli": 0.2, "sck": 0.1}  # one screencapture CLI run / one ScreenCaptureKit frame
MODEL_S = {"low": 1.5, "medium": 5.0, "high": 12.0}  # modeled time per model call, by effort
BYTES_PER_TOKEN = 3.4  # neo2.core.providers.fit's rule
# Gemini 3's tokens per image by media_resolution (docs, 2026-09); "auto" is the default (high).
IMG_TOKENS = {"low": 280, "medium": 560, "high": 1120, "auto": 1120}

W_PT, H_PT, SCALE = 1512, 982, 2  # the main display: points, and pixels per point
MENU_BAR_H = 24
DEFAULT_AX_TIMEOUT_S = 6.0  # the AX API's messaging timeout until one is set
NEO_PID = 900  # NEO's own overlay (its window floats bottom-right)


# ---- time spent in fakes ------------------------------------------------------------------------


class Meter:
    """Seconds spent inside fakes (their own bookkeeping, hit tests, rendering), subtracted from
    a primitive's wall time: what is left is NEO's own Python."""

    def __init__(self) -> None:
        self.fake_s = 0.0
        self._lock = threading.Lock()
        self._local = threading.local()

    def reset(self) -> None:
        with self._lock:
            self.fake_s = 0.0

    @contextlib.contextmanager
    def fake(self) -> Iterator[None]:
        if getattr(self._local, "depth", 0):
            self._local.depth += 1
            try:
                yield
            finally:
                self._local.depth -= 1
            return
        self._local.depth = 1
        t = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t
            self._local.depth = 0
            with self._lock:
                self.fake_s += dt


METER = Meter()


def faked(fn: Callable) -> Callable:
    """Time spent in ``fn`` is the fakes' own, not NEO's."""

    @functools.wraps(fn)
    def wrapper(*a: Any, **k: Any) -> Any:
        with METER.fake():
            return fn(*a, **k)

    return wrapper


# ---- the clock ----------------------------------------------------------------------------------

_REAL_ASLEEP = asyncio.sleep
_REAL_SLEEP = time.sleep
_REAL_MONOTONIC = time.monotonic


@dataclass
class Sleep:
    seconds: float
    kind: str  # "async" | "sync" | "ax_block" (a hung app's messaging timeout)
    where: str  # module:function that slept


class FakeClock:
    """Every sleep is counted and returns at once; ``time.monotonic`` runs ahead by what was slept
    (deadlines and ``wait.until`` loops end as they would), and so does the event loop's clock."""

    def __init__(self) -> None:
        self.offset = 0.0
        self.sleeps: list[Sleep] = []
        self.waits = 0  # wait.until calls (Step 0's seam), when the module exists
        self._lock = threading.Lock()

    def _record(self, seconds: float, kind: str, depth: int) -> None:
        s = max(0.0, float(seconds or 0.0))
        if s <= 0:
            return
        f = sys._getframe(depth)
        where = f"{f.f_globals.get('__name__', '?')}:{f.f_code.co_name}"
        with self._lock:
            self.offset += s
            self.sleeps.append(Sleep(s, kind, where))

    def block(self, seconds: float, where: str) -> None:
        """A call that blocks this long (a hung app answering after its timeout)."""
        with self._lock:
            self.offset += seconds
            self.sleeps.append(Sleep(seconds, "ax_block", where))

    async def asleep(self, delay: float = 0, result: Any = None) -> Any:
        self._record(delay, "async", 2)
        await _REAL_ASLEEP(0)
        return result

    def sleep(self, seconds: float) -> None:
        self._record(seconds, "sync", 2)

    def monotonic(self) -> float:
        return _REAL_MONOTONIC() + self.offset

    @property
    def slept(self) -> float:
        return sum(s.seconds for s in self.sleeps)

    def mark(self) -> int:
        return len(self.sleeps)

    def since(self, mark: int) -> list[Sleep]:
        return self.sleeps[mark:]


# ---- Accessibility --------------------------------------------------------------------------------

OK = 0
FAILURE = -25200
CANNOT_COMPLETE = -25204
ATTR_UNSUPPORTED = -25205
ACTION_UNSUPPORTED = -25206
NO_VALUE = -25212


class FakePoint:
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x, self.y = float(x), float(y)


class FakeSize:
    __slots__ = ("width", "height")

    def __init__(self, w: float, h: float) -> None:
        self.width, self.height = float(w), float(h)


class FakeAXError:
    """What CopyMultipleAttributeValues puts where an attribute has no value (an AXValue of the
    error type)."""

    __slots__ = ("code",)

    def __init__(self, code: int) -> None:
        self.code = code


class FakeEl:
    """One element: attributes, children, actions. ``on_press(world, el)`` runs on AXPress (and on
    a click that lands on it); ``commit`` marks a label that commits (sends, deletes, buys…) — the
    ground truth the scenarios check the gate against, never NEO's own opinion."""

    def __init__(
        self,
        role: str,
        title: str = "",
        value: Any = None,
        frame: tuple[float, float, float, float] = (0, 0, 0, 0),
        actions: tuple[str, ...] = (),
        children: tuple | list = (),
        *,
        on_press: Callable[[World, FakeEl], None] | None = None,
        on_click: Callable[[World, FakeEl, float, float, str], None] | None = None,
        commit: bool = False,
        first_mouse: bool = True,
        **extra: Any,
    ) -> None:
        self.attrs: dict[str, Any] = {"AXRole": role, "AXEnabled": True}
        if title:
            self.attrs["AXTitle"] = title
        if value is not None:
            self.attrs["AXValue"] = value
        self.set_frame(*frame)
        self.attrs.update(extra)
        self.children: list[FakeEl] = []
        self.parent: FakeEl | None = None
        self.actions = list(actions)
        self.on_press = on_press
        self.on_click = on_click
        self.commit = commit
        self.first_mouse = first_mouse
        self.pid = 0
        self.hung = False
        self.settable = {"AXValue", "AXFocused"} if role in _EDITABLE else {"AXFocused"}
        for c in children:
            self.add(c)

    # tree
    def add(self, child: FakeEl) -> FakeEl:
        child.parent = self
        child.set_pid(self.pid)
        self.children.append(child)
        return child

    def remove(self, child: FakeEl) -> None:
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def set_pid(self, pid: int) -> FakeEl:
        self.pid = pid
        for c in self.children:
            c.set_pid(pid)
        return self

    def walk(self) -> Iterator[FakeEl]:
        yield self
        for c in self.children:
            yield from c.walk()

    def find(self, pred: Callable[[FakeEl], bool]) -> FakeEl | None:
        return next((e for e in self.walk() if pred(e)), None)

    # geometry
    def set_frame(self, x: float, y: float, w: float, h: float) -> None:
        self.attrs["AXPosition"] = FakePoint(x, y)
        self.attrs["AXSize"] = FakeSize(w, h)

    @property
    def frame(self) -> tuple[float, float, float, float]:
        p, s = self.attrs["AXPosition"], self.attrs["AXSize"]
        return (p.x, p.y, s.width, s.height)

    def contains(self, x: float, y: float) -> bool:
        fx, fy, fw, fh = self.frame
        return fw > 0 and fh > 0 and fx <= x < fx + fw and fy <= y < fy + fh

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.frame
        return x + w / 2, y + h / 2

    # what it says
    @property
    def role(self) -> str:
        return str(self.attrs.get("AXRole", ""))

    @property
    def label(self) -> str:
        return str(self.attrs.get("AXTitle") or self.attrs.get("AXDescription") or self.attrs.get("AXValue") or "")

    def window(self) -> FakeEl | None:
        el: FakeEl | None = self
        while el is not None and el.role not in ("AXWindow", "AXSheet"):
            el = el.parent
        return el

    def __repr__(self) -> str:
        return f"<{self.role} {self.label[:30]!r}>"


_EDITABLE = {"AXTextField", "AXTextArea", "AXSearchField", "AXComboBox"}


class FakeAS:
    """ApplicationServices' Accessibility functions over the world's trees. Every call is recorded
    in ``calls`` as (kind, attribute-or-action); ``performed`` and ``sets`` list the actions."""

    kAXErrorSuccess = OK
    kAXErrorFailure = FAILURE
    kAXErrorCannotComplete = CANNOT_COMPLETE
    kAXErrorAttributeUnsupported = ATTR_UNSUPPORTED
    kAXErrorActionUnsupported = ACTION_UNSUPPORTED
    kAXErrorNoValue = NO_VALUE
    kAXValueCGPointType = 1
    kAXValueCGSizeType = 2
    kAXValueCGRectType = 3
    kAXValueCFRangeType = 4
    kAXValueAXErrorType = 5

    def __init__(self, world: World) -> None:
        self.world = world
        self.calls: list[tuple[str, str]] = []
        self.performed: list[tuple[FakeEl, str]] = []
        self.sets: list[tuple[FakeEl, str, Any]] = []
        self.timeouts: dict[int, float] = {}  # id(element) → its messaging timeout
        self.default_timeout = DEFAULT_AX_TIMEOUT_S  # set through the system-wide element
        self.system = FakeEl("AXSystemWide")
        self._lock = threading.Lock()

    # bookkeeping
    def _rec(self, kind: str, what: str = "") -> None:
        with self._lock:
            self.calls.append((kind, what))

    def count(self) -> int:
        return len(self.calls)

    def kinds(self) -> Counter:
        return Counter(k if k != "get" else f"get {w}" for k, w in self.calls)

    def _block(self, el: Any, what: str) -> bool:
        if getattr(el, "hung", False):
            t = self.timeouts.get(id(el), self.default_timeout)
            self.world.clock.block(t, f"ax:{what}")
            return True
        return False

    # values
    def _get(self, el: FakeEl, name: str) -> Any:
        """An attribute's value (None: no value), computed ones included."""
        w = self.world
        if name == "AXChildren":
            return list(el.children) or None
        if name == "AXParent":
            return el.parent
        if name in ("AXWindow", "AXTopLevelUIElement"):
            return el.window() if el.role not in ("AXWindow", "AXSheet") else None
        if name == "AXFocused":
            app = w.app_of(el)
            return bool(app is not None and app.focused is el)
        if el.role == "AXApplication":
            app = w.apps.get(el.pid)
            if app is not None:
                if name in ("AXFocusedWindow", "AXMainWindow"):
                    return app.focused_window()
                if name == "AXWindows":
                    return app.windows() or None
                if name == "AXFocusedUIElement":
                    return app.focused
                if name == "AXMenuBar":
                    return app.menubar
                if name == "AXFrontmost":
                    return w.front_pid == app.pid
        if el is self.system:
            front = w.apps.get(w.front_pid)
            if name == "AXFocusedUIElement":
                return front.focused if front is not None else None
            if name == "AXFocusedApplication":
                return front.root if front is not None else None
        return el.attrs.get(name)

    # ---- the API ----
    def AXIsProcessTrusted(self) -> bool:
        return True

    def AXIsProcessTrustedWithOptions(self, _opts: Any = None) -> bool:
        return True

    def AXAPIEnabled(self) -> bool:
        return True

    @faked
    def AXUIElementCreateApplication(self, pid: int) -> FakeEl:
        app = self.world.apps.get(int(pid))
        return app.root if app is not None else FakeEl("AXApplication").set_pid(int(pid))

    def AXUIElementCreateSystemWide(self) -> FakeEl:
        return self.system

    @faked
    def AXUIElementSetMessagingTimeout(self, el: Any, t: float) -> int:
        if el is self.system:
            self.default_timeout = float(t) if t else DEFAULT_AX_TIMEOUT_S
        else:
            self.timeouts[id(el)] = float(t) if t else self.default_timeout
        return OK

    @faked
    def AXUIElementCopyAttributeValue(self, el: Any, name: str, _out: Any) -> tuple[int, Any]:
        self._rec("get", str(name))
        if not isinstance(el, FakeEl):
            return FAILURE, None
        if self._block(el, str(name)):
            return CANNOT_COMPLETE, None
        v = self._get(el, str(name))
        if v is None:
            return (NO_VALUE, None) if str(name) in el.attrs or str(name) in _COMPUTED else (ATTR_UNSUPPORTED, None)
        return OK, v

    @faked
    def AXUIElementCopyMultipleAttributeValues(self, el: Any, names: Any, _options: Any, _out: Any) -> tuple[int, Any]:
        self._rec("multi", ",".join(str(n) for n in names))
        if not isinstance(el, FakeEl):
            return FAILURE, None
        if self._block(el, "multi"):
            return CANNOT_COMPLETE, None
        out = []
        for n in names:
            v = self._get(el, str(n))
            out.append(v if v is not None else FakeAXError(NO_VALUE))
        return OK, out

    @faked
    def AXUIElementCopyAttributeValues(self, el: Any, name: str, index: int, count: int, _out: Any) -> tuple[int, Any]:
        self._rec("values", str(name))
        if not isinstance(el, FakeEl):
            return FAILURE, None
        if self._block(el, str(name)):
            return CANNOT_COMPLETE, None
        v = self._get(el, str(name))
        if not isinstance(v, list):
            return (NO_VALUE if v is None else ATTR_UNSUPPORTED), None
        return OK, v[int(index) : int(index) + int(count)]

    @faked
    def AXUIElementGetAttributeValueCount(self, el: Any, name: str, _out: Any) -> tuple[int, int]:
        self._rec("count", str(name))
        if not isinstance(el, FakeEl):
            return FAILURE, 0
        if self._block(el, str(name)):
            return CANNOT_COMPLETE, 0
        v = self._get(el, str(name))
        return (OK, len(v)) if isinstance(v, list) else (NO_VALUE, 0)

    @faked
    def AXUIElementCopyAttributeNames(self, el: Any, _out: Any) -> tuple[int, Any]:
        self._rec("names")
        if not isinstance(el, FakeEl):
            return FAILURE, None
        return OK, sorted(set(el.attrs) | {"AXChildren", "AXParent", "AXWindow", "AXFocused"})

    @faked
    def AXUIElementCopyParameterizedAttributeNames(self, el: Any, _out: Any) -> tuple[int, Any]:
        self._rec("pnames")
        return OK, []

    @faked
    def AXUIElementCopyParameterizedAttributeValue(self, el: Any, name: str, param: Any, _out: Any) -> tuple[int, Any]:
        self._rec("pget", str(name))
        return ATTR_UNSUPPORTED, None

    @faked
    def AXUIElementCopyActionNames(self, el: Any, _out: Any) -> tuple[int, Any]:
        self._rec("actions")
        if not isinstance(el, FakeEl):
            return FAILURE, None
        if self._block(el, "actions"):
            return CANNOT_COMPLETE, None
        return OK, list(el.actions)

    @faked
    def AXUIElementCopyActionDescription(self, el: Any, action: str, _out: Any) -> tuple[int, Any]:
        self._rec("action_desc", str(action))
        return OK, str(action)

    @faked
    def AXUIElementPerformAction(self, el: Any, action: str) -> int:
        self._rec("perform", str(action))
        if not isinstance(el, FakeEl):
            return FAILURE
        self.performed.append((el, str(action)))
        if self._block(el, str(action)):
            return CANNOT_COMPLETE
        if str(action) not in el.actions and str(action) != "AXRaise":
            return ACTION_UNSUPPORTED
        self.world.perform(el, str(action), how="ax")
        return OK

    @faked
    def AXUIElementSetAttributeValue(self, el: Any, name: str, value: Any) -> int:
        self._rec("set", str(name))
        if not isinstance(el, FakeEl):
            return FAILURE
        self.sets.append((el, str(name), value))
        if self._block(el, str(name)):
            return CANNOT_COMPLETE
        return self.world.set_attr(el, str(name), value)

    @faked
    def AXUIElementIsAttributeSettable(self, el: Any, name: str, _out: Any) -> tuple[int, bool]:
        self._rec("settable", str(name))
        if not isinstance(el, FakeEl):
            return FAILURE, False
        if el.attrs.get("AXSubrole") == "AXSecureTextField" and name == "AXValue":
            return OK, True
        return OK, str(name) in el.settable

    @faked
    def AXUIElementCopyElementAtPosition(self, _sys: Any, x: float, y: float, _out: Any) -> tuple[int, Any]:
        self._rec("hit")
        el = self.world.element_at(float(x), float(y))
        if el is None:
            return NO_VALUE, None
        if self._block(el, "hit"):
            return CANNOT_COMPLETE, None
        return OK, el

    @faked
    def AXUIElementGetPid(self, el: Any, _out: Any) -> tuple[int, int]:
        self._rec("pid")
        if not isinstance(el, FakeEl):
            return FAILURE, 0
        return OK, int(el.pid)

    def AXUIElementPostKeyboardEvent(self, *a: Any) -> int:
        raise AssertionError("the bench never posts keys through Accessibility")

    def AXValueGetType(self, v: Any) -> int:
        if isinstance(v, FakePoint):
            return 1
        if isinstance(v, FakeSize):
            return 2
        if isinstance(v, FakeAXError):
            return 5
        return 0

    def AXValueGetValue(self, v: Any, kind: int, _out: Any) -> tuple[bool, Any]:
        if isinstance(v, FakeAXError):
            return True, v.code
        return True, v

    def AXValueCreate(self, kind: int, v: Any) -> Any:
        return v

    def AXUIElementGetTypeID(self) -> int:
        return 1

    def AXValueGetTypeID(self) -> int:
        return 2

    # Observers: the bench has no run loop to deliver notifications; creating one fails cleanly.
    def AXObserverCreate(self, pid: int, cb: Any, _out: Any) -> tuple[int, Any]:
        self._rec("observer")
        return FAILURE, None

    def AXObserverCreateWithInfoCallback(self, pid: int, cb: Any, _out: Any) -> tuple[int, Any]:
        self._rec("observer")
        return FAILURE, None

    def AXObserverAddNotification(self, *a: Any) -> int:
        return FAILURE

    def AXObserverRemoveNotification(self, *a: Any) -> int:
        return FAILURE

    def AXObserverGetRunLoopSource(self, *a: Any) -> Any:
        return None


_COMPUTED = {"AXChildren", "AXParent", "AXWindow", "AXTopLevelUIElement", "AXFocused", "AXFocusedWindow", "AXMainWindow",
             "AXWindows", "AXFocusedUIElement", "AXMenuBar", "AXFrontmost", "AXFocusedApplication"}  # fmt: skip


def ax_names() -> list[str]:
    """Every function FakeAS stands in for (the rest of ApplicationServices' AX* is refused)."""
    return [n for n in dir(FakeAS) if n.startswith("AX") and callable(getattr(FakeAS, n))]


# ---- apps and the world ---------------------------------------------------------------------------


@dataclass
class App:
    name: str
    pid: int
    root: FakeEl
    menubar: FakeEl | None = None
    bundle: str = ""
    regular: bool = True
    layer: int = 0  # the window server's layer (NEO's overlay floats above normal windows)
    alive: bool = True
    focused: FakeEl | None = None  # the keyboard focus
    _front_window: FakeEl | None = None

    def windows(self) -> list[FakeEl]:
        return [c for c in self.root.children if c.role == "AXWindow"]

    def focused_window(self) -> FakeEl | None:
        wins = self.windows()
        if self._front_window in wins:
            return self._front_window
        return wins[0] if wins else None


def app(name: str, pid: int, *windows: FakeEl, menus: list[FakeEl] | None = None, bundle: str = "",
        layer: int = 0, app_menu: bool = True) -> App:  # fmt: skip
    root = FakeEl("AXApplication", name)
    mb = None
    if menus is not None:
        mb = FakeEl("AXMenuBar", frame=(0, 0, W_PT, MENU_BAR_H))
        x = 10.0
        own = [menu_bar_item(name, [menu_item(f"Quit {name}", key="Q")])] if app_menu else []
        for m in [*own, *menus]:
            w = 12 + 8 * len(m.attrs.get("AXTitle", ""))
            m.set_frame(x, 0, w, MENU_BAR_H)
            mb.add(m)
            x += w
        root.add(mb)
    for w in windows:
        root.add(w)
    root.set_pid(pid)
    return App(name, pid, root, mb, bundle or f"com.example.{name.lower().replace(' ', '')}", layer=layer)


def window(title: str, frame: tuple[float, float, float, float], *kids: FakeEl, **extra: Any) -> FakeEl:
    extra.setdefault("AXSubrole", "AXStandardWindow")
    return FakeEl("AXWindow", title, frame=frame, children=kids, actions=("AXRaise",), AXModal=False, **extra)


def button(title: str, frame: tuple[float, float, float, float], *, on_press=None, commit: bool = False, **extra: Any) -> FakeEl:
    return FakeEl("AXButton", title, frame=frame, actions=("AXPress",), on_press=on_press, commit=commit, **extra)


def text(value: str, frame: tuple[float, float, float, float]) -> FakeEl:
    return FakeEl("AXStaticText", value=value, frame=frame)


def field_el(title: str, frame: tuple[float, float, float, float], *, role: str = "AXTextField", secure: bool = False,
             value: str = "") -> FakeEl:  # fmt: skip
    extra: dict[str, Any] = {"AXPlaceholderValue": title} if title else {}
    if secure:
        extra["AXSubrole"] = "AXSecureTextField"
    return FakeEl(role, title, value=value, frame=frame, actions=("AXConfirm",) if role != "AXTextArea" else (), **extra)


def menu_bar_item(title: str, items: list[FakeEl]) -> FakeEl:
    menu = FakeEl("AXMenu", frame=(0, 0, 0, 0), children=items)  # closed: no size
    return FakeEl("AXMenuBarItem", title, frame=(0, 0, 0, 0), actions=("AXPress", "AXCancel"), children=[menu])


def menu_item(title: str, *, key: str = "", on_press=None, commit: bool = False) -> FakeEl:
    extra: dict[str, Any] = {}
    if key:
        extra["AXMenuItemCmdChar"] = key.upper()
        extra["AXMenuItemCmdModifiers"] = 0  # ⌘ alone
    return FakeEl("AXMenuItem", title, frame=(0, 0, 0, 0), actions=("AXPress", "AXCancel"), on_press=on_press,
                  commit=commit, **extra)  # fmt: skip


# ---- a board --------------------------------------------------------------------------------------

FILES = "abcdefgh"
START = {**{f"{f}2": "P" for f in FILES}, **{f"{f}7": "p" for f in FILES},
         **{f"{f}1": p for f, p in zip(FILES, "RNBQKBNR", strict=True)},
         **{f"{f}8": p for f, p in zip(FILES, "rnbqkbnr", strict=True)}}  # fmt: skip


class Board:
    """An 8×8 board drawn in one view (nothing in it for Accessibility): white at the bottom. A
    piece moves by a drag, or by a click on it and then a click on the square it goes to."""

    def __init__(self, x: float, y: float, sq: float) -> None:
        self.x, self.y, self.sq = float(x), float(y), float(sq)
        self.pieces: dict[str, str] = dict(START)
        self.selected: str | None = None
        self.moves: list[tuple[str, str]] = []
        self.games = 1

    def square_at(self, px: float, py: float) -> str | None:
        c, r = int((px - self.x) // self.sq), int((py - self.y) // self.sq)
        return f"{FILES[c]}{8 - r}" if 0 <= c < 8 and 0 <= r < 8 else None

    def center(self, square: str) -> tuple[float, float]:
        c, rank = FILES.index(square[0]), int(square[1])
        return self.x + (c + 0.5) * self.sq, self.y + (8 - rank + 0.5) * self.sq

    def move(self, a: str, b: str) -> bool:
        if a == b or a not in self.pieces:
            return False
        self.pieces[b] = self.pieces.pop(a)
        self.moves.append((a, b))
        return True

    def reset(self) -> None:
        self.pieces, self.selected, self.moves = dict(START), None, []
        self.games += 1


class World:
    """The pretend Mac's state: apps, which is in front, windows front to back, the pointer, and
    everything input did."""

    def __init__(self) -> None:
        self.apps: dict[int, App] = {}
        self.front_pid = 0
        self.z: list[int] = []  # apps front to back (their windows in their own order)
        self.pointer = (700.0, 500.0)
        self.version = 0  # bumped by every visible change: the screen re-renders
        self.board: Board | None = None
        self.board_el: FakeEl | None = None
        self.posted: list[dict] = []  # every event posted (decoded)
        self.acted: list[dict] = []  # what input and AX did: presses, clicks, typing, scrolls
        self.brings: list[int] = []  # apps brought forward by NEO's tools
        self.scrolled: dict[int, float] = {}  # id(scroll area) → lines scrolled down
        self.clock = FakeClock()
        self.AS = FakeAS(self)
        self._down: dict | None = None  # a mouse button NEO holds: where, on what
        self._drag_moves = 0
        self.last_down: dict | None = None  # the press a click or drag that just ended began with
        self.window_numbers: dict[int, int] = {}  # id(window) → its number
        self.displays = [(1, 0.0, 0.0, float(W_PT), float(H_PT))]

    # ---- apps ----
    def add(self, a: App, *, front: bool = False) -> App:
        self.apps[a.pid] = a
        self.z.append(a.pid)
        for w in a.windows():
            self.window_numbers[id(w)] = 100 + len(self.window_numbers)
        if front or not self.front_pid:
            self.activate(a.pid, by_neo=False)
        return a

    def named(self, name: str) -> App:
        return next(a for a in self.apps.values() if a.name == name)

    def app_of(self, el: FakeEl | None) -> App | None:
        return self.apps.get(el.pid) if el is not None else None

    def activate(self, pid: int, *, by_neo: bool = True) -> bool:
        a = self.apps.get(pid)
        if a is None or not a.alive or (by_neo and not a.regular):
            return False
        if by_neo and pid != self.front_pid:
            self.brings.append(pid)
        self.front_pid = pid
        if pid in self.z:
            self.z.remove(pid)
        self.z.insert(0, pid)  # (a floating layer, NEO's overlay, still draws above it)
        self.version += 1
        return True

    def raise_window(self, win: FakeEl) -> None:
        a = self.app_of(win)
        if a is not None and win.role == "AXWindow":
            a._front_window = win
            a.root.children.remove(win)
            a.root.children.insert(1 if a.menubar is not None else 0, win)
            self.version += 1

    # ---- the window server ----
    def window_list(self) -> list[dict]:
        """Quartz's window list, front to back (higher layers first)."""
        out = []
        order = sorted(self.z, key=lambda p: -self.apps[p].layer)
        for pid in order:
            a = self.apps[pid]
            if not a.alive:
                continue
            for w in a.windows():
                x, y, ww, hh = w.frame
                out.append({
                    "kCGWindowOwnerPID": pid, "kCGWindowOwnerName": a.name, "kCGWindowNumber": self.window_numbers.get(id(w), 99),
                    "kCGWindowLayer": a.layer, "kCGWindowAlpha": 1.0, "kCGWindowName": w.attrs.get("AXTitle", ""),
                    "kCGWindowBounds": {"X": x, "Y": y, "Width": ww, "Height": hh}, "kCGWindowIsOnscreen": True,
                })  # fmt: skip
        return out

    def top_window(self, x: float, y: float) -> tuple[App, FakeEl] | None:
        for pid in sorted(self.z, key=lambda p: -self.apps[p].layer):
            a = self.apps[pid]
            for w in a.windows():
                if a.alive and w.contains(x, y):
                    return a, w
        return None

    def element_at(self, x: float, y: float) -> FakeEl | None:
        """What Accessibility's hit test finds: the deepest element under the point in the topmost
        window there (a sheet over its window first), or the front app's menu bar."""
        with METER.fake():
            if y < MENU_BAR_H:
                a = self.apps.get(self.front_pid)
                if a is not None and a.menubar is not None:
                    return next((m for m in a.menubar.children if m.contains(x, y)), a.menubar)
                return None
            hit = self.top_window(x, y)
            if hit is None:
                return None
            return _deepest(hit[1], x, y)

    # ---- AX actions ----
    def perform(self, el: FakeEl, action: str, how: str) -> None:
        if action == "AXRaise":
            self.raise_window(el)
            return
        self.acted.append(self._act("press", el, how))
        if el.on_press is not None:
            el.on_press(self, el)
            self.version += 1
        elif el.role in ("AXCheckBox",):
            el.attrs["AXValue"] = 0 if el.attrs.get("AXValue") else 1
            self.version += 1

    def set_attr(self, el: FakeEl, name: str, value: Any) -> int:
        a = self.app_of(el)
        if name == "AXFocused":
            if a is not None and value:
                a.focused = el
            return OK
        if name == "AXMain":
            if value:
                self.raise_window(el)
            return OK
        if name not in el.settable:
            return FAILURE
        if name == "AXValue" and el.attrs.get("AXSubrole") == "AXSecureTextField":
            return FAILURE
        el.attrs[name] = value
        self.acted.append(self._act("set", el, "ax", value=value))
        self.version += 1
        return OK

    def _act(self, kind: str, el: FakeEl | None, how: str, **more: Any) -> dict:
        a = self.app_of(el)
        return {"kind": kind, "how": how, "role": el.role if el else "", "label": el.label if el else "",
                "app": a.name if a else "", "pid": el.pid if el else 0, "commit": bool(el and el.commit),
                "neo": bool(a and a.pid == NEO_PID), **more}  # fmt: skip

    # ---- posted events (real CGEvents, decoded with Quartz's own getters) ----
    def post(self, ev: Any, to_pid: int = 0) -> None:
        import Quartz as Q

        with METER.fake():
            typ = int(Q.CGEventGetType(ev))
            loc = Q.CGEventGetLocation(ev)
            x, y = float(loc.x), float(loc.y)
            e: dict[str, Any] = {"type": typ, "at": (x, y)}
            if typ in (Q.kCGEventKeyDown, Q.kCGEventKeyUp):
                e["code"] = int(Q.CGEventGetIntegerValueField(ev, Q.kCGKeyboardEventKeycode))
                e["flags"] = int(Q.CGEventGetFlags(ev))
                try:
                    _n, s = Q.CGEventKeyboardGetUnicodeString(ev, 64, None, None)
                    e["text"] = str(s or "")
                except Exception:  # noqa: BLE001
                    e["text"] = ""
            elif typ == Q.kCGEventScrollWheel:
                e["dy"] = int(Q.CGEventGetIntegerValueField(ev, Q.kCGScrollWheelEventDeltaAxis1))
            else:
                e["clicks"] = int(Q.CGEventGetIntegerValueField(ev, Q.kCGMouseEventClickState))
            e["to_pid"] = to_pid
            self.posted.append(e)
            self._route(Q, e)

    def _route(self, Q: Any, e: dict) -> None:
        typ, (x, y) = e["type"], e["at"]
        if typ == Q.kCGEventMouseMoved:
            self.pointer = (x, y)
        elif typ in (Q.kCGEventLeftMouseDown, Q.kCGEventRightMouseDown, Q.kCGEventOtherMouseDown):
            self.pointer = (x, y)
            self._mouse_down(x, y, "right" if typ == Q.kCGEventRightMouseDown else "left", e.get("clicks", 1))
        elif typ in (Q.kCGEventLeftMouseDragged, Q.kCGEventRightMouseDragged, Q.kCGEventOtherMouseDragged):
            self.pointer = (x, y)
            if self._down is not None:
                self._drag_moves += 1
        elif typ in (Q.kCGEventLeftMouseUp, Q.kCGEventRightMouseUp, Q.kCGEventOtherMouseUp):
            self.pointer = (x, y)
            self._mouse_up(x, y)
        elif typ == Q.kCGEventScrollWheel:
            self._scroll(e.get("dy", 0))
        elif typ == Q.kCGEventKeyDown:
            self._key(e)

    def _mouse_down(self, x: float, y: float, button: str, clicks: int) -> None:
        hit = self.top_window(x, y) if y >= MENU_BAR_H else None
        el = self.element_at(x, y)
        a = self.app_of(el) if el is not None else (hit[0] if hit else None)
        self._down = {"at": (x, y), "el": el, "app": a, "button": button, "swallowed": False}
        self._drag_moves = 0
        if a is not None and a.pid != self.front_pid and a.layer == 0:
            # A click on a window in the background brings its app forward; a view that doesn't take
            # the first mouse (a board) only activates the window, and never sees the click.
            self.activate(a.pid, by_neo=False)
            if el is not None and not el.first_mouse:
                self._down["swallowed"] = True

    def _mouse_up(self, x: float, y: float) -> None:
        d, self._down = self._down, None
        if d is None or d["swallowed"]:
            return
        d["drags"] = self._drag_moves
        self.last_down = d
        el = d["el"]
        up_el = self.element_at(x, y)
        if el is not None and el.on_click is not None:
            el.on_click(self, el, x, y, d["button"])  # the board judges clicks and drags itself
            self.version += 1
            return
        if el is None or up_el is not el:
            return
        if el.role in _EDITABLE:
            a = self.app_of(el)
            if a is not None:
                a.focused = el
            self.acted.append(self._act("focus", el, "click"))
            return
        if "AXPress" in el.actions or el.role in ("AXButton", "AXCheckBox", "AXMenuItem", "AXLink", "AXRadioButton"):
            self.perform(el, "AXPress", how="click")
        else:
            self.acted.append(self._act("click", el, "click"))

    def _scroll(self, dy: int) -> None:
        el = self.element_at(*self.pointer)
        area = el
        while area is not None and area.role != "AXScrollArea":
            area = area.parent
        target = area or el
        if target is not None:
            self.scrolled[id(target)] = self.scrolled.get(id(target), 0.0) - dy  # dy < 0 scrolls down
            self.acted.append(self._act("scroll", target, "wheel", dy=dy))
            self.version += 1

    def _key(self, e: dict) -> None:
        from neo2.core.tools.computer import input as inp

        a = self.apps.get(self.front_pid)
        if a is None:
            return
        if e.get("to_pid") and e["to_pid"] != a.pid:
            a = self.apps.get(e["to_pid"]) or a
        import Quartz as Q

        cmd = bool(e["flags"] & Q.kCGEventFlagMaskCommand)
        if e.get("text") and e["code"] == 0 and not cmd:  # type_text: a Unicode string on key 0
            self._type(a, e["text"])
            return
        name = next((k for k, v in inp._KEYCODES.items() if v == e["code"]), "")
        if cmd and len(name) == 1 and a.menubar is not None:
            item = a.menubar.find(lambda m: m.role == "AXMenuItem" and m.attrs.get("AXMenuItemCmdChar") == name.upper())
            if item is not None:
                self.perform(item, "AXPress", how="key")
            return
        if name in ("return", "enter") and a.focused is not None:
            if a.focused.role == "AXTextArea":
                self._type(a, "\n")
            else:
                self.acted.append(self._act("return", a.focused, "key"))

    def _type(self, a: App, s: str) -> None:
        f = a.focused
        if f is None or f.role not in _EDITABLE or f.attrs.get("AXSubrole") == "AXSecureTextField":
            self.acted.append({"kind": "typed_nowhere", "how": "key", "app": a.name, "text": s, "pid": a.pid,
                               "neo": a.pid == NEO_PID, "commit": False, "label": "", "role": ""})  # fmt: skip
            return
        f.attrs["AXValue"] = str(f.attrs.get("AXValue") or "") + s
        self.acted.append(self._act("type", f, "key", text=s))
        self.version += 1

    # ---- checks ----
    def pressed(self, label: str) -> bool:
        return any(a["kind"] == "press" and a["label"] == label for a in self.acted)

    def commits_pressed(self) -> list[dict]:
        return [a for a in self.acted if a["commit"] and a["kind"] in ("press", "click")]

    def on_neo(self) -> list[dict]:
        acted = [a for a in self.acted if a.get("neo")]
        neo = self.apps.get(NEO_PID)
        if neo is not None:
            for e in self.posted:
                if e["type"] in (1, 3, 25) and any(w.contains(*e["at"]) for w in neo.windows()):
                    acted.append({"kind": "mouse_down_on_neo", "at": e["at"]})
        return acted


def _deepest(el: FakeEl, x: float, y: float) -> FakeEl:
    for c in reversed(el.children):  # the last child is drawn on top (a sheet over its window)
        if c.contains(x, y) and c.role not in ("AXMenu",):
            return _deepest(c, x, y)
    return el


# ---- screen ---------------------------------------------------------------------------------------


class FakeScreen:
    """``screencapture`` answered with synthetic Retina frames of the world. Every run is counted
    (``captures``); the file is written in the format asked for (``-t``), so NEO's own decode of it
    is measured as NEO's work."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.captures: list[dict] = []
        self._frame: tuple[int, Any] | None = None

    def frame(self) -> Any:
        """The whole main display at 2 px per point (cached until the world changes)."""
        if self._frame is not None and self._frame[0] == self.world.version:
            return self._frame[1]
        with METER.fake():
            img = render(self.world)
        self._frame = (self.world.version, img)
        return img

    def capture_in_process(self, region: tuple[float, float, float, float] | None = None) -> Any:
        """An in-process capture (ScreenCaptureKit, CGWindowList): counted like a screencapture run,
        marked ``in_process`` (charged ScreenCaptureKit's cost), answered with the region's picture —
        a PIL image at 2 px per point, the whole main display when no region is given."""
        self.captures.append({"region": tuple(region) if region else None, "fmt": "sck", "window": False, "in_process": True})
        with METER.fake():
            return self.grab(tuple(region) if region else None)

    def grab(self, region: tuple[float, float, float, float] | None = None) -> Any:
        img = self.frame()
        if region is None:
            return img.copy()
        x, y, w, h = region
        return img.crop((round(x * SCALE), round(y * SCALE), round((x + w) * SCALE), round((y + h) * SCALE)))

    def run(self, cmd: list[str], *a: Any, **k: Any) -> subprocess.CompletedProcess:
        """subprocess.run for NEO's computer modules: screencapture only."""
        if not cmd or os.path.basename(str(cmd[0])) != "screencapture":
            raise AssertionError(f"the bench runs no program but a fake screencapture ({cmd[:2]!r})")
        args = [str(c) for c in cmd[1:]]
        path, fmt, region, win = args[-1], "png", None, None
        if "-t" in args:
            fmt = args[args.index("-t") + 1].lower()
        if "-R" in args:
            region = tuple(float(v) for v in args[args.index("-R") + 1].split(","))
        if "-l" in args:
            num = int(args[args.index("-l") + 1])
            for w in self.world.window_list():
                if w["kCGWindowNumber"] == num:
                    b = w["kCGWindowBounds"]
                    win = (b["X"], b["Y"], b["Width"], b["Height"])
        self.captures.append({"region": region or win, "fmt": fmt, "window": win is not None, "in_process": False})
        with METER.fake():
            img = self.grab(region or win)
            pil_fmt = {"jpg": "JPEG", "jpeg": "JPEG", "bmp": "BMP", "tiff": "TIFF", "tif": "TIFF", "gif": "GIF"}.get(fmt, "PNG")
            img.save(path, format=pil_fmt)  # (screencapture's own encode: the CLI's time, not NEO's)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")


def _font(size: int) -> Any:
    from PIL import ImageFont

    for path in ("/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        with contextlib.suppress(Exception):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


_FONTS: dict[int, Any] = {}


def render(world: World) -> Any:
    """A 3024×1964 picture of the world: desktop, menu bar, every window back to front with its
    controls and texts, the board with its pieces, sheets on top of their windows."""
    from PIL import Image, ImageDraw

    k = SCALE
    img = Image.new("RGB", (W_PT * k, H_PT * k), (92, 110, 138))
    d = ImageDraw.Draw(img)
    f13 = _FONTS.setdefault(13, _font(13 * k))
    d.rectangle([0, 0, W_PT * k, MENU_BAR_H * k], fill=(246, 246, 246))
    front = world.apps.get(world.front_pid)
    if front is not None and front.menubar is not None:
        for m in front.menubar.children:
            x, y, _w, _h = m.frame
            d.text(((x + 4) * k, 4 * k), m.attrs.get("AXTitle", ""), fill=(20, 20, 20), font=f13)
    order = sorted(world.z, key=lambda p: -world.apps[p].layer)
    for pid in reversed(order):
        a = world.apps[pid]
        for w in reversed(a.windows()):
            _draw(d, w, world, f13)
    return img


_LINE_PT = 10  # one line of scrolling, in points


def _draw(d: Any, el: FakeEl, world: World, font: Any, dy: float = 0.0, clip: tuple | None = None) -> None:
    k = SCALE
    x, y, w, h = el.frame
    y -= dy
    if clip is not None and (y + h <= clip[1] or y >= clip[1] + clip[3]):
        return
    box = [x * k, y * k, (x + w) * k - 1, (y + h) * k - 1]
    role = el.role
    if w > 0 and h > 0:
        if role in ("AXWindow", "AXSheet"):
            d.rounded_rectangle(box, 10 * k if role == "AXWindow" else 4 * k, fill=(250, 250, 250), outline=(170, 170, 170))
            if role == "AXWindow":
                d.rectangle([x * k, y * k, (x + w) * k - 1, (y + 28) * k], fill=(232, 232, 232))
                d.text(((x + w / 2 - 60) * k, (y + 7) * k), el.attrs.get("AXTitle", "")[:40], fill=(60, 60, 60), font=font)
            clip = None
        elif el is world.board_el and world.board is not None:
            _draw_board(d, world.board)
        elif role in ("AXButton", "AXPopUpButton", "AXCheckBox"):
            fill = (120, 200, 255) if role == "AXCheckBox" and el.attrs.get("AXValue") else el.attrs.get("fill", (255, 255, 255))
            d.rounded_rectangle(box, 5 * k, fill=fill, outline=(150, 150, 150))
            d.text(((x + 6) * k, (y + 4) * k), el.label[:24], fill=(0, 0, 0), font=font)
        elif role in _EDITABLE:
            d.rectangle(box, fill=(255, 255, 255), outline=(120, 120, 120))
            v = "•" * len(str(el.attrs.get("AXValue") or "")) if el.attrs.get("AXSubrole") == "AXSecureTextField" else str(el.attrs.get("AXValue") or el.attrs.get("AXPlaceholderValue") or "")
            d.text(((x + 4) * k, (y + 3) * k), v[:60], fill=(30, 30, 30), font=font)
        elif role == "AXStaticText":
            t = str(el.attrs.get("AXValue") or "")
            if t:  # a line of text runs the width it's given (a page's paragraphs fill their column)
                t = ((t + " · ") * (1 + int(w / max(1, 7 * len(t) + 21))))[: max(1, int(w / 7))]
            d.text((x * k, y * k), t, fill=(30, 30, 30), font=font)
    if role == "AXScrollArea" and world.scrolled.get(id(el)):
        dy, clip = dy + world.scrolled[id(el)] * _LINE_PT, (x, y, w, h)
    for c in el.children:
        if c.role == "AXSheet":
            continue
        _draw(d, c, world, font, dy, clip)
    for c in el.children:
        if c.role == "AXSheet":
            _draw(d, c, world, font, dy, clip)


def _draw_board(d: Any, b: Board) -> None:
    """The squares, the selected piece's square lit (as a chess app shows a pick), and the pieces as
    discs — white ones white with a dark rim, black ones near-black — so the picture doesn't depend
    on a font having chess glyphs (without them a glyph draws as a striped box)."""
    k = SCALE
    for r in range(8):
        for c in range(8):
            x0, y0 = (b.x + c * b.sq) * k, (b.y + r * b.sq) * k
            fill = (238, 238, 210) if (r + c) % 2 == 0 else (118, 150, 86)
            if b.selected is not None and b.square_at(b.x + (c + 0.5) * b.sq, b.y + (r + 0.5) * b.sq) == b.selected:
                fill = (186, 202, 68)  # (a pick is plain to see: a different shade, not just a hue)
            d.rectangle([x0, y0, x0 + b.sq * k - 1, y0 + b.sq * k - 1], fill=fill)
    rad = b.sq * 0.34
    for sq, p in b.pieces.items():
        cx, cy = b.center(sq)
        white = p.isupper()
        d.ellipse([(cx - rad) * k, (cy - rad) * k, (cx + rad) * k, (cy + rad) * k],
                  fill=(255, 255, 255) if white else (24, 24, 24), outline=(30, 30, 30), width=k)  # fmt: skip


# ---- AppKit's running applications ---------------------------------------------------------------


class FakeRunningApp:
    def __init__(self, world: World, a: App) -> None:
        self._w, self._a = world, a

    def localizedName(self) -> str:
        return self._a.name

    def processIdentifier(self) -> int:
        return self._a.pid

    def bundleIdentifier(self) -> str:
        return self._a.bundle

    def activationPolicy(self) -> int:
        return 0 if self._a.regular else 1

    def isTerminated(self) -> bool:
        return not self._a.alive

    def isActive(self) -> bool:
        return self._w.front_pid == self._a.pid

    def isHidden(self) -> bool:
        return False

    def isFinishedLaunching(self) -> bool:
        return True

    def bundleURL(self) -> Any:
        return None

    def executableURL(self) -> Any:
        return None

    def ownsMenuBar(self) -> bool:
        return self._w.front_pid == self._a.pid

    def activateWithOptions_(self, _opts: int) -> bool:
        return self._w.activate(self._a.pid)

    def activateFromApplication_options_(self, _app: Any, _opts: int) -> bool:
        return self._w.activate(self._a.pid)


def fake_running_class(world: World) -> Any:
    class NSRunningApplication:  # noqa: N801 - AppKit's name
        @staticmethod
        def runningApplicationWithProcessIdentifier_(pid: int) -> FakeRunningApp | None:
            a = world.apps.get(int(pid))
            return FakeRunningApp(world, a) if a is not None and a.alive else None

        @staticmethod
        def runningApplicationsWithBundleIdentifier_(bid: str) -> list[FakeRunningApp]:
            return [FakeRunningApp(world, a) for a in world.apps.values() if a.bundle == bid and a.alive]

        @staticmethod
        def currentApplication() -> FakeRunningApp | None:
            a = world.apps.get(NEO_PID)
            return FakeRunningApp(world, a) if a else None

    return NSRunningApplication


def fake_workspace_class(world: World) -> Any:
    class _Shared:
        def frontmostApplication(self) -> FakeRunningApp | None:
            a = world.apps.get(world.front_pid)
            return FakeRunningApp(world, a) if a else None

        menuBarOwningApplication = frontmostApplication

        def runningApplications(self) -> list[FakeRunningApp]:
            return [FakeRunningApp(world, a) for a in world.apps.values() if a.alive]

        def URLForApplicationToOpenURL_(self, _url: Any) -> Any:
            return None

        def URLForApplicationWithBundleIdentifier_(self, _bid: str) -> Any:
            return None

        def notificationCenter(self) -> Any:
            raise AssertionError("the bench installs no workspace observers")

    shared = _Shared()

    class NSWorkspace:  # noqa: N801 - AppKit's name
        @staticmethod
        def sharedWorkspace() -> _Shared:
            return shared

    return NSWorkspace


# ---- installing it ----------------------------------------------------------------------------


class Patches:
    """setattr with undo."""

    def __init__(self) -> None:
        self._undo: list[tuple[Any, str, Any, bool]] = []

    def set(self, obj: Any, name: str, value: Any, *, only_if_present: bool = False) -> bool:
        present = hasattr(obj, name)
        if only_if_present and not present:
            return False
        self._undo.append((obj, name, getattr(obj, name, None), present))
        setattr(obj, name, value)
        return True

    def undo(self) -> None:
        for obj, name, old, present in reversed(self._undo):
            if present:
                setattr(obj, name, old)
            else:
                with contextlib.suppress(AttributeError):
                    delattr(obj, name)
        self._undo.clear()


def _refuse(what: str) -> Callable:
    def f(*a: Any, **k: Any) -> Any:
        raise AssertionError(f"the bench refused a real {what}")

    return f


@dataclass
class Env:
    """What ``install`` put in place, and the counters of one measurement."""

    world: World
    AS: FakeAS
    screen: FakeScreen
    clock: FakeClock
    patches: Patches = field(default_factory=Patches)

    def mark(self) -> dict:
        return {"ax": self.AS.count(), "captures": len(self.screen.captures), "sleeps": self.clock.mark(),
                "posted": len(self.world.posted), "fake_s": METER.fake_s, "t": time.perf_counter(),
                "brings": len(self.world.brings), "acted": len(self.world.acted)}  # fmt: skip

    def since(self, m: dict) -> dict:
        wall = time.perf_counter() - m["t"]
        fake = METER.fake_s - m["fake_s"]
        sleeps = self.clock.since(m["sleeps"])
        caps = self.screen.captures[m["captures"] :]
        return {
            "ax_calls": self.AS.count() - m["ax"],
            "captures": len(caps),
            "captures_in_process": sum(1 for c in caps if c.get("in_process")),
            "sleeps": len([s for s in sleeps if s.kind != "ax_block"]),
            "sleep_s": round(sum(s.seconds for s in sleeps if s.kind != "ax_block"), 4),
            "ax_block_s": round(sum(s.seconds for s in sleeps if s.kind == "ax_block"), 4),
            "posted": len(self.world.posted) - m["posted"],
            "brings": len(self.world.brings) - m["brings"],
            "acted": len(self.world.acted) - m["acted"],
            "py_ms": max(0.0, (wall - fake) * 1000),
            "wall_ms": wall * 1000,
            "fake_ms": fake * 1000,
            "slept_in": sorted({s.where for s in sleeps}),
        }


# What NEO's test suite swaps out around every test (tests/conftest.py: nothing under any point, no
# pictures, no waiting), as the code has it: the bench measures the real thing. Taken when this module
# is first imported — at collection time under pytest, before any fixture runs.
_PRISTINE_NAMES = {
    "neo2.core.tools.computer.ax": ("hit", "window_owner_at", "scene", "raise_window", "focused_element_text", "working_app"),
    "neo2.core.tools.computer.screen": ("sample_points",),
    "neo2.core.tools.computer.pointer": ("LOOK_S", "LOOK_AGAIN_S", "SETTLE_S", "HOVER_S", "POLL_S", "protected_in"),
    "neo2.core.agent.settle": ("probe",),
}


def _snapshot() -> dict[tuple[str, str], Any]:
    out: dict[tuple[str, str], Any] = {}
    for mod_name, names in _PRISTINE_NAMES.items():
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        for n in names:
            v = getattr(mod, n, None)
            if callable(v) and (getattr(v, "__name__", "") == "<lambda>" or not getattr(v, "__module__", "").startswith("neo2.core.")):
                continue  # (already a test's stand-in: not the real one)
            if v is not None:
                out[(mod_name, n)] = v
    return out


PRISTINE = _snapshot()


def _neo_modules() -> list[types.ModuleType]:
    return [m for n, m in list(sys.modules.items()) if (n == "neo" or n.startswith(("neo.", "neo2."))) and m is not None]  # neo2-codemod: keep (the repo's neo shim too)


@contextlib.contextmanager
def install(world: World, *, clock: bool = True) -> Iterator[Env]:
    """Put the pretend Mac under NEO's tools (see the module doc); everything comes back after."""
    import AppKit
    import ApplicationServices
    import Quartz

    from neo2.core.tools import computer  # noqa: F401 - the tools, registered
    from neo2.core.tools.computer import apps, ax, focus, pointer, screen  # noqa: F401
    from neo2.core.tools.computer import input as inp

    fs = FakeScreen(world)
    env = Env(world, world.AS, fs, world.clock)
    p = env.patches
    try:
        for (mod_name, n), v in PRISTINE.items():  # the real code, whatever a test swapped in
            p.set(sys.modules[mod_name], n, v)
        # Accessibility: every AX function, the ones the fake has and a refusal for the rest
        for name in dir(ApplicationServices):
            if name.startswith(("AXUIElement", "AXObserver", "AXValue", "AXIsProcessTrusted", "AXAPIEnabled", "AXMakeProcess")):
                fn = getattr(world.AS, name, None)
                if callable(getattr(ApplicationServices, name, None)) and not isinstance(getattr(ApplicationServices, name), type):
                    p.set(ApplicationServices, name, fn if fn is not None else _refuse(f"ApplicationServices.{name}"))
        # Quartz: posting goes to the world; its window list is the world's; taps and pictures are refused
        p.set(Quartz, "CGEventPost", lambda tap, ev: world.post(ev))
        p.set(Quartz, "CGEventPostToPid", lambda pid, ev: world.post(ev, to_pid=int(pid)))
        p.set(Quartz, "CGEventPostToPSN", _refuse("Quartz.CGEventPostToPSN"))
        p.set(Quartz, "CGWarpMouseCursorPosition", lambda pt: setattr(world, "pointer", (float(pt[0]), float(pt[1]))) or 0)

        def window_list(opts: int, number: int) -> list[dict]:
            wins = world.window_list()
            if number and opts & Quartz.kCGWindowListOptionIncludingWindow:
                return [w for w in wins if w["kCGWindowNumber"] == number]
            if number and opts & Quartz.kCGWindowListOptionOnScreenAboveWindow:  # (front to back: those before it)
                at = next((i for i, w in enumerate(wins) if w["kCGWindowNumber"] == number), len(wins))
                return wins[:at]
            return wins

        p.set(Quartz, "CGWindowListCopyWindowInfo", window_list)
        p.set(Quartz, "CGWindowListCreateImage", _refuse("Quartz.CGWindowListCreateImage"))
        p.set(Quartz, "CGEventTapCreate", _refuse("Quartz.CGEventTapCreate"))
        p.set(Quartz, "CGRequestScreenCaptureAccess", _refuse("Quartz.CGRequestScreenCaptureAccess"))
        p.set(Quartz, "CGPreflightScreenCaptureAccess", lambda: True)
        p.set(Quartz, "CGPreflightPostEventAccess", lambda: True)
        p.set(Quartz, "CGEventSourceButtonState", lambda state, button: False)
        real_create = Quartz.CGEventCreateMouseEvent
        p.set(Quartz, "CGEventCreate", lambda src: real_create(None, Quartz.kCGEventMouseMoved, world.pointer, 0))
        # AppKit: running apps and the workspace are the world's
        running, workspace = fake_running_class(world), fake_workspace_class(world)
        real_running, real_ws = AppKit.NSRunningApplication, AppKit.NSWorkspace
        p.set(AppKit, "NSRunningApplication", running)
        p.set(AppKit, "NSWorkspace", workspace)
        for m in _neo_modules():
            for attr, real, fake in (("NSRunningApplication", real_running, running), ("NSWorkspace", real_ws, workspace)):
                if getattr(m, attr, None) is real:
                    p.set(m, attr, fake)
        # the front app, running apps and bringing one forward (LaunchServices, activation)
        p.set(focus, "front_app", lambda: (world.apps[world.front_pid].name, world.front_pid) if world.front_pid else ("", 0))
        p.set(focus, "running", lambda: [(a.name, a.pid, "Foreground" if a.regular else "UIElement") for a in world.apps.values() if a.alive])
        p.set(focus, "pid_running", lambda name: [(a.name, a.pid) for a in world.apps.values() if a.alive and str(name).lower() in a.name.lower()])
        p.set(focus, "hosts_shell", lambda pid, depth=3: "")
        p.set(focus, "force_front", lambda pid: world.activate(int(pid)))

        async def bring_pid_front(pid: int) -> tuple[bool, str]:
            ok = world.activate(int(pid))
            a = world.apps.get(int(pid))
            return ok, a.name if a else ""

        async def bring_front(name: str, *, wait: float = 1.5) -> tuple[bool, str]:
            a = next((x for x in world.apps.values() if x.name.lower() == str(name).lower()), None)
            if a is None:
                return False, ""
            return world.activate(a.pid), a.name

        async def activate(name: str) -> str:
            ok, real = await bring_front(name)
            return f"Activated {real} (frontmost)" if ok else f"Error: {name} is not running"

        p.set(apps, "bring_pid_front", bring_pid_front)
        p.set(apps, "bring_front", bring_front)
        p.set(apps, "activate", activate)
        p.set(apps, "default_browser", lambda: "")
        p.set(apps, "osascript", _refuse("osascript"))
        # NEO's own processes: the overlay's pid
        from neo2.core.decide import media

        p.set(media, "_self_family", lambda: frozenset({NEO_PID}))
        p.set(media, "responsible_pid", lambda pid: pid)
        # the input module's own look at the pointer
        p.set(inp, "mouse_position", lambda: world.pointer)
        p.set(inp, "can_post", lambda: True)
        # Accessibility odds and ends that read the real Mac
        p.set(ax, "_expose_web", lambda pid, root: False, only_if_present=True)
        # a fresh process: NEO sets its messaging timeout on this Mac's Accessibility (not one it set
        # on another — the real one, or an earlier world's — in this process before)
        p.set(ax, "_timeout_for", None, only_if_present=True)
        p.set(computer, "_browser_names", lambda: set(), only_if_present=True)
        # the screen: one Retina display, screencapture answered by the world
        p.set(screen, "can_capture", lambda: True)
        p.set(screen, "displays", lambda: [screen.Display(i, x, y, w, h) for i, x, y, w, h in world.displays])
        p.set(screen, "display_points", lambda: (W_PT, H_PT), only_if_present=True)
        guarded = types.SimpleNamespace(**{k: getattr(subprocess, k) for k in dir(subprocess) if not k.startswith("__")})
        guarded.run = fs.run
        guarded.Popen = _refuse("subprocess.Popen")
        guarded.check_output = _refuse("subprocess.check_output")
        guarded.call = _refuse("subprocess.call")
        for m in _neo_modules():
            if m.__name__.startswith("neo2.core.tools.computer") and getattr(m, "subprocess", None) is subprocess:
                p.set(m, "subprocess", guarded)
        for hook in CAPTURE_HOOKS:  # in-process capture paths, when a workstream adds one
            mod_name, fn_name, *rest = hook
            with contextlib.suppress(ImportError):
                mod = importlib.import_module(mod_name)
                p.set(mod, fn_name, _captured(fs, rest[0] if rest else None, rest[1] if len(rest) > 1 else None),
                      only_if_present=True)  # fmt: skip
        with contextlib.suppress(ImportError):  # ScreenCaptureKit is "installed" here: sck_capture decides
            p.set(importlib.import_module("neo2.core.tools.computer.sck"), "enabled", lambda: True, only_if_present=True)
        # the clock
        if clock:
            p.set(asyncio, "sleep", world.clock.asleep)
            p.set(time, "sleep", world.clock.sleep)
            p.set(time, "monotonic", world.clock.monotonic)
            with contextlib.suppress(ImportError):
                wait = importlib.import_module("neo2.core.tools.computer.wait")
                # Step 0's wait keeps its own clock and sleep (bound at import): on the fake clock too
                p.set(wait, "_now", world.clock.monotonic, only_if_present=True)
                p.set(wait, "_sleep", world.clock.asleep, only_if_present=True)
                real_until = getattr(wait, "until", None)
                if real_until is not None:

                    async def until(*a: Any, **k: Any) -> Any:
                        world.clock.waits += 1
                        return await real_until(*a, **k)

                    p.set(wait, "until", until)
        yield env
    finally:
        p.undo()


# The functions that capture the screen in-process (ScreenCaptureKit, CGWindowList), as (module,
# function), (module, function, adapt) or (module, function, adapt, region_of). Each call is answered
# by FakeScreen.capture_in_process(region) — counted in ``captures`` (``in_process``: charged
# ScreenCaptureKit's cost, never free) — with a PIL image of the region (x, y, w, h in points, top-left
# origin; the ``region`` keyword or the first argument, or ``region_of(world, *args, **kwargs)``) at
# 2 px per point. When the real function returns something else (PNG or JPEG bytes, a CGImage…),
# ``adapt(image)`` turns the picture into that.


def _window_region(world: World, window_id: Any, *a: Any, **k: Any) -> tuple[float, float, float, float] | None:
    b = next((w["kCGWindowBounds"] for w in world.window_list() if w["kCGWindowNumber"] == int(window_id)), None)
    return (b["X"], b["Y"], b["Width"], b["Height"]) if b else None


CAPTURE_HOOKS: list[tuple] = [
    # eyes-capture's ScreenCaptureKit path (settings().sck_capture; off by default): a rectangle on a
    # display, and one window's own pixels
    ("neo2.core.tools.computer.sck", "capture_rect", None, lambda world, x, y, w, h, *a, **k: (x, y, w, h)),
    ("neo2.core.tools.computer.sck", "capture_window", None, _window_region),
]


def _captured(fs: FakeScreen, adapt: Callable[[Any], Any] | None, region_of: Callable[..., Any] | None = None) -> Callable:
    def capture(*a: Any, **k: Any) -> Any:
        region = region_of(fs.world, *a, **k) if region_of is not None else (k.get("region") or (a[0] if a else None))
        img = fs.capture_in_process(region)
        if adapt is None:
            return img
        with METER.fake():  # (the conversion stands in for the real function's own: not NEO's)
            return adapt(img)

    return capture


# ---- the scripted brain ---------------------------------------------------------------------------


def tokens(s: str) -> float:
    return len(str(s).encode("utf-8", "ignore")) / BYTES_PER_TOKEN


def spec_tokens(spec: Any) -> float:
    return tokens(spec.name) + tokens(spec.description) + tokens(json.dumps(spec.parameters, separators=(",", ":"))) + 12


@dataclass
class Call:
    """One request to the brain, as it would be billed."""

    effort: str
    system_tokens: float
    tool_tokens: float
    n_tools: int
    history_tokens: float
    images: int
    image_tokens: float
    tools: list[str]
    details: list[str]

    @property
    def prompt_tokens(self) -> float:
        return self.system_tokens + self.tool_tokens + self.history_tokens + self.image_tokens


def account(messages: list, system: str, tools: list | None, effort: str, sees: bool = True) -> Call:
    hist, images, img_tok, details = 0.0, 0, 0.0, []

    def image(i: Any) -> None:
        nonlocal images, img_tok
        detail = str(getattr(i, "detail", "auto") or "auto")
        details.append(detail)
        images += 1
        img_tok += IMG_TOKENS.get(detail, IMG_TOKENS["high"]) if sees else tokens("[You can't see this picture]")

    for m in messages:
        hist += 6 + tokens(m.text)
        for i in m.images:
            image(i)
        for c in m.tool_calls:
            hist += 6 + tokens(c.name) + tokens(json.dumps(c.args, ensure_ascii=False, default=str))
        for r in m.tool_results:
            hist += 6 + tokens(r.name) + tokens(r.content)
            for i in r.images:
                image(i)
    tools = list(tools or [])
    return Call(str(effort), tokens(system), sum(spec_tokens(s) for s in tools), len(tools), hist, images, img_tok,
                [s.name for s in tools], details)  # fmt: skip


class FakeBrain:
    """Stands in for ``neo2.core.providers.brain(purpose)``: every request is recorded (``calls``) and
    answered by ``policy(brain, messages, system, tools)``, which returns the next Turn."""

    name = "fake"

    def __init__(self, policy: Callable[..., Any], *, sees: bool = True) -> None:
        self.policy = policy
        self.sees = sees
        self.supports_vision = sees
        self.supports_tools = True
        self.calls: list[Call] = []
        self.issued: list[Any] = []  # every ToolCall it made
        self.per_turn: list[int] = []  # tool calls in each reply
        self.purposes: list[str] = []
        self.last_used, self.last_model = "fake", "fake-scripted"
        self.state: dict[str, Any] = {}
        # Everything it was handed, once each: (source, text) — the user's turns ("request": the words
        # and any early look NEO put with them) and every tool result (by tool name).
        self.handed: dict[tuple[str, str], None] = {}

    async def complete(self, messages: list, *, system: str = "", tools: list | None = None, effort: str = "medium",
                       max_tokens: int = 4096) -> Any:  # fmt: skip
        from neo2.core.providers.base import Turn

        self.calls.append(account(messages, system, tools, effort, self.sees))
        for m in messages:
            if m.role == "user" and m.text:
                self.handed.setdefault(("request", str(m.text)), None)
            for r in m.tool_results:
                self.handed.setdefault((str(r.name), str(r.content)), None)
        turn = self.policy(self, messages, system, list(tools or []))
        if not isinstance(turn, Turn):
            turn = Turn(text=str(turn or "Done."))
        turn.sees = self.sees
        self.issued.extend(turn.tool_calls)
        self.per_turn.append(len(turn.tool_calls))
        return turn

    async def stream(self, messages: list, *, system: str = "", max_tokens: int = 2048):
        self.calls.append(account(messages, system, None, "low", self.sees))
        yield "ok"
