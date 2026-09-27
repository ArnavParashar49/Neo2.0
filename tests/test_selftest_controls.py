"""The controls self-test, end to end, on a pretend Mac.

Nothing here posts a real event, performs a real AX action, activates or launches an app, captures
the screen or opens a window. Quartz's event calls, the AX functions NEO's tools use, the window
list, the front app, the event tap and `screencapture` are all fakes that feed a simulated test
window (`Desk`), which writes the same JSON log the real one (neo.selftest.testwin) does. The
runner (neo.selftest.controls.run) drives it through NEO's real tool handlers, so these tests
assert the exact events NEO would send and what the report makes of what arrived.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
import types
from pathlib import Path

import pytest
import Quartz as _RealQ
from PIL import Image, ImageDraw

from neo.selftest import controls, testwin
from neo.selftest import layout as L
from neo.selftest import watch as watch_mod

TEST_PID = 4242
USER_PID = 111
NEO_PID = os.getpid()
MAIN_WIN, SECOND_WIN = 777, 778

# ---- safety net: nothing real, whatever a test forgets ------------------------------------------


@pytest.fixture(autouse=True)
def _nothing_real(monkeypatch):
    from neo.tools import load_all

    load_all()

    def refuse(*a, **k):
        raise AssertionError("a real side effect was attempted")

    for n in ("CGEventPost", "CGEventPostToPid", "CGEventPostToPSN", "CGEventTapCreate", "CGWindowListCopyWindowInfo"):
        if hasattr(_RealQ, n):
            monkeypatch.setattr(_RealQ, n, refuse)
    import ApplicationServices as AS

    monkeypatch.setattr(AS, "AXUIElementPerformAction", refuse)
    monkeypatch.setattr(AS, "AXUIElementSetAttributeValue", refuse)
    monkeypatch.setattr(AS, "AXUIElementCopyElementAtPosition", refuse)
    from neo.tools.computer import apps, focus

    monkeypatch.setattr(focus, "force_front", refuse)
    monkeypatch.setattr(focus, "front_app", refuse)
    monkeypatch.setattr(apps, "osascript", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(subprocess, "run", refuse)
    yield


# ---- the pretend Mac --------------------------------------------------------------------------

CONTENT = (40.0, 92.0)  # the test window's content, top-left on screen (a 28 pt title bar above)
FRAME = [40.0, 64.0, 720.0, 358.0]


def _t(x, y, w, h):
    return [CONTENT[0] + x, CONTENT[1] + y, float(w), float(h)]


TARGETS = {  # where testwin puts them (its add() calls), on screen
    "board": _t(20, 40, 240, 240),
    "button_a": _t(280, 40, 110, 30),
    "button_b": _t(400, 40, 110, 30),
    "double": _t(520, 40, 180, 30),
    "field": _t(280, 84, 420, 24),
    "area": _t(280, 120, 260, 110),
    "list": _t(560, 120, 140, 160),
    "marker": _t(280, 244, 36, 36),
    "tile": _t(340, 244, 80, 36),
    "well": _t(440, 244, 100, 36),
}
MOUSE = {
    _RealQ.kCGEventMouseMoved: "move",
    _RealQ.kCGEventLeftMouseDown: "down",
    _RealQ.kCGEventLeftMouseUp: "up",
    _RealQ.kCGEventRightMouseDown: "rdown",
    _RealQ.kCGEventRightMouseUp: "rup",
    _RealQ.kCGEventLeftMouseDragged: "dragged",
    _RealQ.kCGEventScrollWheel: "scroll",
}
KEYS = {_RealQ.kCGEventKeyDown: "kdown", _RealQ.kCGEventKeyUp: "kup"}


def inside(r, x, y):
    return r[0] <= x < r[0] + r[2] and r[1] <= y < r[1] + r[3]


class Desk:
    """The user's Safari in front, NEO's test window (pid 4242) on screen behind the focus, and a
    window server that routes posted events: into the simulated window when they land on it,
    into `stray` when they would have gone anywhere else."""

    def __init__(self, tmp: Path, **opts):
        self.tmp = tmp
        self.opts = opts
        self.front = ("Safari", USER_PID)
        self.pointer = (900.0, 700.0)
        self.display = opts.get("display", (1512, 982))
        self.px = opts.get("px", 2)  # screenshot pixels per point
        self.posted: list[dict] = []
        self.stray: list[dict] = []
        self.taps: list[FakeTap] = []
        self.alive = False
        self.log = None
        self.name = ""
        self.active = False
        self.focus_el = ""
        self.field = ""
        self.area = ""
        self.list_off = 0.0
        self.piece = [1, 6]
        self.pressing = ""
        self.grab = None
        self.drags = 0
        self.dnd = False
        self.entered = False
        self.on_post = None  # scenario hook: (desk, event) after each posted event
        self.activations: list[int] = []
        self.commands: list[str] = []
        self._last_state = None
        self.cover = opts.get("cover")  # (name, pid, rect): a window on top of the test window

    # the window list
    def windows(self):
        out = []
        if self.cover:
            name, pid, rect = self.cover
            out.append({"pid": pid, "name": name, "number": 900, "layer": 0, "alpha": 1.0, "bounds": tuple(rect)})
        if self.alive:
            out.append({"pid": TEST_PID, "name": self.name, "number": MAIN_WIN, "layer": 0, "alpha": 1.0, "bounds": tuple(FRAME)})
        out.append({"pid": USER_PID, "name": "Safari", "number": 5, "layer": 0, "alpha": 1.0, "bounds": (0.0, 25.0, 1512.0, 957.0)})
        return out

    def top(self, x, y):
        return next((w for w in self.windows() if inside(w["bounds"], x, y)), None)

    def target_at(self, x, y):
        return next((n for n, r in TARGETS.items() if inside(r, x, y)), "")

    # the test window's log
    def write(self, ev, **data):
        self.log.write(ev, **data)

    def state(self):
        s = {"field": self.field, "area": self.area, "list": self.list_off, "piece": list(self.piece), "active": self.active, "key": self.active}
        if s != self._last_state:
            self._last_state = dict(s)
            self.write("state", **s)

    def start(self, log: Path, tag: str):
        self.alive = True
        self.name = f"{L.APP_NAME} {tag}"
        self.log = testwin.Log(str(log))
        self.write("launched", pid=TEST_PID, name=self.name)
        ready = {
            "pid": self.opts.get("ready_pid", TEST_PID),
            "tag": tag,
            "name": self.name,
            "window": MAIN_WIN,
            "frame": FRAME,
            "targets": {**TARGETS, **self.opts.get("targets", {})},
            "cell": L.CELL,
            "board_n": L.BOARD_N,
            "piece": list(self.piece),
            "screens": [[0, 0, *self.display]],
            "second": None,
            "active": False,
        }
        self.write("ready", **ready)
        self.state()

    def activate_test(self):
        if self.front[1] != TEST_PID:
            self.front = (self.name, TEST_PID)
            self.active = True
            self.write("focus", active=True)
            self.state()

    def command(self, cmd):
        self.commands.append(cmd)
        if cmd == "deactivate" and self.front[1] == TEST_PID:
            self.front = ("Safari", USER_PID)
            self.active = False
            self.write("focus", active=False)
            self.state()
        elif cmd == "quit":
            self.alive = False

    # the window server
    def post(self, ev):
        self.posted.append(ev)
        for tap in list(self.taps):
            tap.see(ev)
        kind = MOUSE.get(ev["type"]) or KEYS.get(ev["type"])
        if kind in ("move", "down", "rdown", "dragged"):
            self.pointer = ev["pos"]
        if kind in ("down", "rdown", "up", "rup", "dragged", "scroll"):
            x, y = self.pointer if kind == "scroll" else ev["pos"]
            top = self.top(x, y)
            if (top is None or top["pid"] != TEST_PID) and (kind != "up" or not self.pressing):
                self.stray.append(ev)
                return
            getattr(self, "_" + kind)(ev, x, y)
        elif kind in ("kdown", "kup"):
            if self.front[1] != TEST_PID:
                self.stray.append(ev)
                return
            if kind == "kdown":
                self._key(ev)
        if self.on_post is not None:
            self.on_post(self, ev)

    def _down(self, ev, x, y, button="left"):
        clicks = ev["fields"].get(_RealQ.kCGMouseEventClickState, 1)
        self.write("raw_ldown" if button == "left" else "raw_rdown", at=[x, y], clicks=clicks, window=MAIN_WIN)
        if not self.active:
            self.activate_test()
            if self.opts.get("swallow_first_click"):
                self.pressing = ""
                return  # like Chess's board: the click only brought the window forward
        target = self.target_at(x, y)
        self.pressing, self.drags, self.dnd, self.entered = target, 0, False, False
        if target == "board":
            bx, by, _, _ = TARGETS["board"]
            cell = testwin.cell_at(x - bx, y - by)
            self.grab = cell if (button == "left" and cell == self.piece) else None
            self.write("board_down", button=button, cell=cell, at=[x, y], clicks=clicks)
        elif target in ("double", "marker"):
            self.write(f"{target}_down", at=[x, y], clicks=clicks)
        elif target == "tile":
            self.write("tile_down", at=[x, y])
        elif target in ("field", "area"):
            self.focus_el = target

    def _rdown(self, ev, x, y):
        self._down(ev, x, y, "right")

    def _dragged(self, ev, x, y):
        self.drags += 1
        if self.pressing == "board" and self.drags == 1:
            bx, by, _, _ = TARGETS["board"]
            self.write("board_drag", cell=testwin.cell_at(x - bx, y - by), at=[x, y])
        if self.pressing == "tile":
            if self.drags == 1 and not self.opts.get("no_dnd"):
                self.dnd = True
                self.write("dnd_begin", at=[x, y])
            if self.dnd and not self.entered and self.target_at(x, y) == "well":
                self.entered = True
                self.write("dnd_enter")

    def _up(self, ev, x, y, button="left"):
        target = self.target_at(x, y)
        if self.pressing == "board":
            bx, by, _, _ = TARGETS["board"]
            cell = testwin.cell_at(x - bx, y - by)
            self.write("board_up", cell=cell, at=[x, y], drags=self.drags)
            if self.grab is not None and cell and cell != self.grab and self.drags:
                self.write("piece_moved", start=self.grab, end=cell)
                self.piece = cell
                self.state()
        elif self.pressing in ("button_a", "button_b") and target == self.pressing:
            self.write("button", name=L.BUTTON_A if target == "button_a" else L.BUTTON_B)
        elif self.pressing == "tile" and self.dnd:
            if target == "well":
                self.write("drop", payload=L.DRAG_PAYLOAD)
            self.write("dnd_end", operation=1 if target == "well" else 0, at=[x, y])
        self.pressing, self.grab = "", None

    def _rup(self, ev, x, y):
        self.pressing = ""

    def _scroll(self, ev, x, y):
        dy = ev["wheels"][0]
        self.write("raw_scroll", at=[x, y], dy=float(dy), precise=False)
        if self.target_at(x, y) == "list":
            self.list_off = max(0.0, min(L.LIST_ROWS * L.ROW_H - 160.0, self.list_off - dy * 10))
            self.write("scroll", target="list", offset=self.list_off)
            self.state()

    def _key(self, ev):
        flags = ev.get("flags") or 0
        if ev.get("uni"):
            text = ev["uni"][1]
            if self.opts.get("truncate20"):  # one key event carries at most 20 UTF-16 units
                text = text.encode("utf-16-le")[:40].decode("utf-16-le", errors="ignore")
            self.write("raw_key_down", chars=text, code=0, mods=0)
            self._insert(text)
            return
        code = ev["code"]
        self.write("raw_key_down", chars="", code=code, mods=flags & 0xFFFF0000)
        if code == 36 and self.focus_el == "field":
            self.write("field_return", value=self.field)
        elif code == 36 and self.focus_el == "area":
            self._insert("\n")
        elif code == 38 and flags & _RealQ.kCGEventFlagMaskCommand:
            self.write("menu", title=L.MENU_TITLE)

    def _insert(self, text):
        if self.focus_el == "field":
            self.field += text
            self.write("text", target="field", value=self.field)
        elif self.focus_el == "area":
            self.area += text
            self.write("text", target="area", value=self.area)
        self.state()

    # the user
    def user_moves(self, x, y):
        """A real mouse move: the tap sees it (no NEO tag, no pid), the pointer goes there."""
        ev = {"type": _RealQ.kCGEventMouseMoved, "pos": (x, y), "fields": {}, "src_pid": 0}
        self.pointer = (x, y)
        for tap in list(self.taps):
            tap.see(ev)

    # screenshots (the `screencapture` command)
    def screencapture(self, cmd):
        path = cmd[-1]
        w, h = self.display
        img = Image.new("RGB", (w * self.px, h * self.px), (120, 120, 120))
        d = ImageDraw.Draw(img)
        k = self.px
        fx, fy, fw, fh = FRAME
        d.rectangle([fx * k, fy * k, (fx + fw) * k - 1, (fy + fh) * k - 1], fill=(236, 236, 236))
        mx, my, mw, mh = TARGETS["marker"]
        d.rectangle([mx * k, my * k, (mx + mw) * k - 1, (my + mh) * k - 1], fill=L.MARKER_RGB)
        if "-R" in cmd:
            rx, ry, rw, rh = (int(v) for v in cmd[cmd.index("-R") + 1].split(","))
            img = img.crop((rx * k, ry * k, (rx + rw) * k, (ry + rh) * k))
        img.save(path)


class FakeTap:
    def __init__(self, desk, on_event, strip=False, keys=True):
        self.desk, self.on_event, self.strip, self.keys = desk, on_event, strip, keys
        desk.taps.append(self)

    def see(self, ev):
        seen = dict(ev, fields={}, src_pid=0) if self.strip else ev
        self.on_event(ev["type"], seen)

    def stop(self):
        if self in self.desk.taps:
            self.desk.taps.remove(self)


class FakeChild:
    pid = TEST_PID

    def __init__(self, desk):
        self.desk = desk

    def send(self, cmd):
        self.desk.command(cmd)

    def alive(self):
        return self.desk.alive

    def stop(self):
        was = self.desk.front[1] == TEST_PID
        self.desk.alive = False
        if was:
            self.desk.front = ("Finder", 1)  # macOS picks some other app when the test app quits


class FakeMac:
    def __init__(self, desk):
        self.desk = desk

    def trusted(self):
        return True

    def can_capture(self):
        return True

    def launch(self, log, tag):
        self.desk.start(log, tag)
        return FakeChild(self.desk)

    def windows(self):
        return self.desk.windows()

    def window(self, number):
        return next((w for w in self.desk.windows() if w["number"] == number), None)

    def front(self):
        return self.desk.front

    def pointer(self):
        return self.desk.pointer

    def hit_pid(self, x, y):
        top = self.desk.top(x, y)
        return top["pid"] if top else 0

    def element_pid(self, ref):
        return ref[0]

    def pids_named(self, name):
        return [TEST_PID] if name == self.desk.name else []

    def activate(self, pid):
        self.desk.activations.append(pid)
        self.desk.front = ("Safari", USER_PID) if pid == USER_PID else self.desk.front
        return self.desk.front[1] == pid

    def post(self, ev):
        Q = watch_mod.Q
        Q.CGEventPost(Q.kCGHIDEventTap, ev)

    def tap(self, on_event):
        if self.desk.opts.get("no_tap"):
            return None
        return FakeTap(self.desk, on_event, strip=bool(self.desk.opts.get("tap_blind")), keys=not self.desk.opts.get("tap_mouse_only"))


def fake_quartz(desk):
    Q = types.SimpleNamespace()
    for name in dir(_RealQ):
        if name.startswith("kCG"):
            setattr(Q, name, getattr(_RealQ, name))

    def pt(p):
        if not all(isinstance(v, (int, float)) for v in p):
            raise TypeError(f"depythonifying struct: {p!r}")
        return (float(p[0]), float(p[1]))

    Q.CGEventCreateMouseEvent = lambda src, typ, pos, btn: {"type": typ, "pos": pt(pos), "btn": btn, "fields": {}}
    Q.CGEventCreateKeyboardEvent = lambda src, code, down: {
        "type": _RealQ.kCGEventKeyDown if down else _RealQ.kCGEventKeyUp, "code": code, "fields": {}, "flags": 0, "uni": None,
    }  # fmt: skip
    Q.CGEventCreateScrollWheelEvent = lambda src, unit, count, *wheels: {"type": _RealQ.kCGEventScrollWheel, "wheels": wheels, "fields": {}}

    def get_field(ev, f):
        if f == _RealQ.kCGEventSourceUnixProcessID:
            return ev.get("src_pid", NEO_PID)
        if f == _RealQ.kCGKeyboardEventKeycode:
            return ev.get("code", 0)
        if f == _RealQ.kCGScrollWheelEventDeltaAxis1:
            return ev["wheels"][0]
        return ev["fields"].get(f, 0)

    def set_uni(ev, n, s):
        assert n == len(s.encode("utf-16-le")) // 2, "the UTF-16 length NEO passes must match the text"
        ev["uni"] = (n, s)

    Q.CGEventGetIntegerValueField = get_field
    Q.CGEventSetIntegerValueField = lambda ev, f, v: ev["fields"].__setitem__(f, v)
    Q.CGEventSetFlags = lambda ev, flags: ev.__setitem__("flags", flags)
    Q.CGEventGetFlags = lambda ev: ev.get("flags") or 0
    Q.CGEventKeyboardSetUnicodeString = set_uni
    Q.CGEventKeyboardGetUnicodeString = lambda ev, n, a, b: ((ev.get("uni") or (0, ""))[0], (ev.get("uni") or (0, ""))[1])
    Q.CGEventGetType = lambda ev: ev["type"]
    Q.CGEventCreate = lambda src: {"type": -1, "probe": True}
    Q.CGEventGetLocation = lambda ev: types.SimpleNamespace(
        x=(desk.pointer if ev.get("probe") else ev["pos"])[0], y=(desk.pointer if ev.get("probe") else ev["pos"])[1]
    )
    Q.CGEventPost = lambda tap, ev: desk.post(ev)
    return Q


def ax_elements(desk):
    from neo.tools.computer.ax import Element

    dx, dy = desk.opts.get("ax_shift", (0, 0))  # Accessibility reporting controls somewhere else

    def el(eid, role, title, value, name, frame=None, actions=()):
        x, y, w, h = frame or TARGETS[name]
        if name in ("button_a", "button_b", "field", "area"):
            x, y = x + dx, y + dy
        return Element(eid, role, title, value, x, y, w, h, actions=list(actions), ref=(TEST_PID, name))

    ax_x, ax_y, ax_w, ax_h = TARGETS["area"]
    return [
        el("e1", "AXWindow", f"{L.TITLE} ({desk.name[-4:]})", "", "window", frame=FRAME),
        el("e2", "AXStaticText", "", f"{L.TITLE}.", "label", frame=[60, 104, 680, 18]),
        el("e3", "AXButton", L.BUTTON_A, "", "button_a", actions=("AXPress",)),
        el("e4", "AXButton", L.BUTTON_B, "", "button_b", actions=("AXPress",)),
        el("e5", "AXTextField", L.FIELD_HINT, desk.field, "field"),
        el("e6", "AXTextArea", "", desk.area, "area", frame=[ax_x + 1, ax_y + 1, ax_w - 17, ax_h - 2]),
    ]


@pytest.fixture
def mac(monkeypatch, tmp_path):
    """A pretend Mac wired into NEO's own tools. Options go through `mac.set(...)` before the run."""
    from neo.tools.computer import apps, ax, focus, screen
    from neo.tools.computer import input as inp

    desk = Desk(tmp_path)
    fake = FakeMac(desk)
    for name in ("_WAIT_S", "_TEXT_WAIT_S", "_DRAG_WAIT_S"):  # the pretend window answers at once
        monkeypatch.setattr(controls, name, 0.3)
    Q = fake_quartz(desk)
    monkeypatch.setattr(inp, "Q", Q)
    monkeypatch.setattr(watch_mod, "Q", Q)
    monkeypatch.setattr(inp, "time", types.SimpleNamespace(sleep=lambda s: None))  # NEO's input pacing
    monkeypatch.setattr(inp, "_watch", None)
    monkeypatch.setattr(ax, "_last_tree", {})
    monkeypatch.setattr(ax, "_last_tree_ts", 0.0)
    monkeypatch.setattr(ax, "_last_tree_app", "")
    monkeypatch.setattr(apps, "_target", ("Notes", 1e18))  # an app NEO opened a moment ago

    def snapshot(app=None, *, max_lines=250):
        name, pid = desk.front
        from neo.tools.computer.ax import Element

        els = ax_elements(desk) if pid == TEST_PID else [Element("e1", "AXButton", "Reload", "", 5, 30, 20, 20, ref=(USER_PID, "x"))]
        ax._last_tree = {e.id: e for e in els}
        ax._last_tree_ts, ax._last_tree_app = time.time(), name.lower()
        return "\n".join([f"App: {name}"] + ["  " + e.line() for e in els])

    def press(eid):
        e = ax.get(eid)
        if e.ref[0] != TEST_PID:
            raise AssertionError("pressed something outside the test window")
        desk.write("button", name=e.title)
        return f"Pressed {eid} ({e.role.removeprefix('AX')} {e.title!r})"

    def set_value(eid, value):
        e = ax.get(eid)
        if e.ref != (TEST_PID, "area"):
            raise AssertionError("set a value outside the test window's text area")
        desk.area = value
        desk.write("text", target="area", value=value)
        desk.state()
        return f"Set {eid} value."

    def label_at(x, y):
        t = desk.target_at(x, y)
        return {"button_a": L.BUTTON_A, "button_b": L.BUTTON_B, "field": L.FIELD_HINT}.get(t, L.TITLE if inside(FRAME, x, y) else "Reload")

    async def activate(name):
        if name != desk.name:
            return f"Error: {name} is not running"
        desk.activate_test()
        return f"Activated {name} (frontmost)"

    async def bring_front(name, *, wait=1.5):
        raise AssertionError(f"bring_front({name!r}) — the self-test must not activate apps by name")

    main_area = lambda app=None: next(e for e in ax_elements(desk) if e.role == "AXTextArea")  # noqa: E731
    monkeypatch.setattr(ax, "snapshot", snapshot)
    monkeypatch.setattr(ax, "press", press)
    monkeypatch.setattr(ax, "set_value", set_value)
    monkeypatch.setattr(ax, "label_at", label_at)
    monkeypatch.setattr(ax, "app_at", lambda x, y: desk.name if inside(FRAME, x, y) else "Safari")
    monkeypatch.setattr(ax, "app_of", lambda ref: desk.name if ref[0] == TEST_PID else "Safari")
    monkeypatch.setattr(ax, "frontmost_app", lambda: desk.front)
    monkeypatch.setattr(ax, "focused_role", lambda: ({"field": "AXTextField", "area": "AXTextArea"}.get(desk.focus_el, ""), ""))
    monkeypatch.setattr(ax, "focused_is_secure", lambda: False)
    monkeypatch.setattr(ax, "focused_editable", lambda: desk.focus_el in ("field", "area") and desk.front[1] == TEST_PID)
    monkeypatch.setattr(ax, "main_text_area", main_area)
    monkeypatch.setattr(ax, "focused_element_text", lambda: "")
    monkeypatch.setattr(apps, "front_name", lambda: desk.front[0])
    monkeypatch.setattr(apps, "activate", activate)
    monkeypatch.setattr(apps, "bring_front", bring_front)
    monkeypatch.setattr(focus, "front_app", lambda: desk.front)
    monkeypatch.setattr(screen, "can_capture", lambda: True)
    monkeypatch.setattr(screen, "display_points", lambda: desk.display)

    def run(cmd, *a, **k):
        if cmd and cmd[0] == "screencapture":
            desk.screencapture(cmd)
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"refused real {cmd[0]}")

    monkeypatch.setattr(subprocess, "run", run)
    fake.set = lambda **opts: desk.opts.update(opts) or (setattr(desk, "cover", opts["cover"]) if "cover" in opts else None)
    fake.desk = desk
    return fake


def _run(fake, **kw):
    return asyncio.run(controls.run(mac=fake, **kw))


def _by_name(report):
    return {c.name: c for c in report.checks}


def _kinds(events):
    return [MOUSE.get(e["type"]) or KEYS.get(e["type"]) for e in events]


# ---- the checks, on a Mac where everything works --------------------------------------------------


def test_everything_arrives_and_the_report_says_so(mac):
    from neo.tools import computer
    from neo.tools.computer import apps, ax
    from neo.tools.computer import input as inp

    desk = mac.desk
    target_before = apps._target
    report = _run(mac)
    checks = _by_name(report)
    assert report.refused == "" and report.stopped == "", report.text()
    failed = {n: c.arrived for n, c in checks.items() if c.status == "fail"}
    assert failed == {}, report.text()
    assert checks["second_display"].status == "skip"  # one display
    passed = {n for n, c in checks.items() if c.status == "pass"}
    assert passed >= {
        "first_click", "click_point", "right_click", "double_click", "ax_tree", "click_id", "ax_press", "ax_set_value",
        "type_field", "hotkey_return", "type_area", "hotkey_menu", "scroll", "drag_board", "drag_drop",
        "screenshot_region", "screenshot_full", "activate", "feedback",
    }  # fmt: skip
    assert report.ok
    assert report.summary().startswith("Controls check: 19 of 19 passed")
    # nothing went anywhere but the test window
    assert desk.stray == []
    # every event NEO posted during the run carries this run's mark; the last one puts the pointer back
    *during, last = desk.posted
    tags = {e["fields"].get(_RealQ.kCGEventSourceUserData) for e in during}
    assert len(tags) == 1 and 0 not in tags and None not in tags
    assert MOUSE[last["type"]] == "move" and last["pos"] == (900.0, 700.0)
    assert _RealQ.kCGEventSourceUserData not in last["fields"]
    # cleaned up: the app that was in front is back, the window is gone, nothing is left installed
    assert desk.front == ("Safari", USER_PID) and desk.activations == [USER_PID]
    assert desk.alive is False
    assert inp._watch is None and computer._test_window_pid == 0
    assert apps._target == target_before
    assert ax._last_tree == {}  # no id from the test window outlives it
    assert report.seconds < 60


def test_the_exact_events_for_a_click_a_drag_and_typing(mac):
    desk = mac.desk
    marks: dict[str, int] = {}
    real = controls._one

    async def spy(fn, r):
        marks[fn.__name__] = len(desk.posted)
        return await real(fn, r)

    controls._one = spy
    try:
        _run(mac)
    finally:
        controls._one = real
    names = [f.__name__ for f in controls.CHECKS]

    def during(name):
        i = names.index(name)
        end = marks.get(names[i + 1], len(desk.posted)) if i + 1 < len(names) else len(desk.posted)
        return desk.posted[marks[name] : end]

    # a click: move there, press and let go, click count 1
    click = during("_first_click")
    x, y = TARGETS["board"][0] + 6.5 * L.CELL, TARGETS["board"][1] + 1.5 * L.CELL
    assert _kinds(click) == ["move", "down", "up"]
    assert all(e["pos"] == (x, y) for e in click)
    assert [e["fields"][_RealQ.kCGMouseEventClickState] for e in click[1:]] == [1, 1]
    # a double click: two presses, click counts 1 then 2
    dbl = during("_double_click")
    assert _kinds(dbl) == ["move", "down", "up", "down", "up"]
    assert [e["fields"][_RealQ.kCGMouseEventClickState] for e in dbl[1:]] == [1, 1, 2, 2]
    # a drag: move to the start, press, 12 drag steps, let go at the end
    drag = during("_drag_board")
    assert _kinds(drag) == ["move", "down", *["dragged"] * 12, "up"]
    assert drag[-1]["pos"] == (TARGETS["board"][0] + 4.5 * L.CELL, TARGETS["board"][1] + 3.5 * L.CELL)
    # typing: a click into the field, then the text in 20-character key events, down and up
    typed = during("_type_field")
    assert _kinds(typed[:3]) == ["move", "down", "up"]
    keys = typed[3:]
    assert _kinds(keys) == ["kdown", "kup"] * (len(keys) // 2)
    chunks = [e["uni"][1] for e in keys if KEYS[e["type"]] == "kdown"]
    assert "".join(chunks) == L.TYPE_TEXT and all(len(c) <= 20 for c in chunks)
    # a menu shortcut: one key with the Command flag
    menu = during("_hotkey_menu")
    assert _kinds(menu) == ["kdown", "kup"] and menu[0]["code"] == 38
    assert menu[0]["flags"] & _RealQ.kCGEventFlagMaskCommand
    # a scroll: the pointer to the list, then the wheel, down then up
    scroll = during("_scroll")
    assert _kinds(scroll) == ["move", "scroll", "move", "scroll"]
    assert [e["wheels"][0] for e in scroll if e["type"] == _RealQ.kCGEventScrollWheel] == [-5, 5]


# ---- what the report catches ------------------------------------------------------------------


def test_a_first_click_that_only_activates_the_window_is_caught(mac):
    """Chess: the board ignores the click that brings its window forward. NEO's click tool says it
    clicked; the report says what really happened, and that the tool's answer hid it."""
    mac.set(swallow_first_click=True)
    report = _run(mac)
    checks = _by_name(report)
    first = checks["first_click"]
    assert first.status == "fail"
    assert "only brought the window to the front" in first.arrived
    assert first.said.startswith("Left-clicked at")
    assert checks["click_point"].status == "pass"  # in front now, clicks land
    assert checks["feedback"].status == "fail" and "first click" in checks["feedback"].arrived
    reply = report.reply()
    assert "First click on a window that isn't in front" in reply and "Didn't work" in reply
    assert not report.ok


def test_characters_lost_between_key_events_are_shown(mac):
    mac.set(truncate20=True)
    report = _run(mac)
    typed = _by_name(report)["type_field"]
    assert typed.status == "fail"
    assert "missing" in typed.arrived
    assert "UTF-16 units" in typed.sent  # how NEO chunked it, so the cause is visible


def test_a_drag_and_drop_that_never_drops_is_described(mac):
    mac.set(no_dnd=True)
    report = _run(mac)
    dnd = _by_name(report)["drag_drop"]
    assert dnd.status == "fail"
    assert dnd.arrived.startswith("the tile was pressed") and "nothing was dropped" in dnd.arrived
    assert _by_name(report)["drag_board"].status == "pass"


def test_screenshot_coordinates_on_a_downscaled_screen_are_caught_without_clicking(mac):
    """A 3200-point-wide display: the screenshot is shrunk to 1600 px, so 1 image px is 2 pt, but
    the note says image coords are click coords. The check reports it and clicks nothing there."""
    mac.set()
    mac.desk.display, mac.desk.px = (3200, 1800), 1
    desk = mac.desk
    marks = {}
    real = controls._one

    async def spy(fn, r):
        marks[fn.__name__] = len(desk.posted)
        out = await real(fn, r)
        marks[fn.__name__ + ":end"] = len(desk.posted)
        return out

    controls._one = spy
    try:
        report = _run(mac)
    finally:
        controls._one = real
    checks = _by_name(report)
    full = checks["screenshot_full"]
    assert full.status == "fail"
    assert "a model following the note would click" in full.arrived and "2.00 pt" in full.arrived
    assert desk.posted[marks["_shot_full"] : marks["_shot_full:end"]] == []  # no click at the wrong point
    assert checks["screenshot_region"].status == "pass"  # a region shot of the window is 1:1


# ---- it stops, and never touches anything else ----------------------------------------------------


def test_the_user_moving_the_mouse_stops_the_run_at_once(mac):
    desk = mac.desk
    seen = {"n": 0}

    def hook(d, ev):
        if ev["type"] == _RealQ.kCGEventLeftMouseDragged:
            seen["n"] += 1
            if seen["n"] == 3 and seen.get("at") is None:
                seen["at"] = len(d.posted)
                d.user_moves(1300.0, 800.0)

    desk.on_post = hook
    report = _run(mac)
    assert "you moved the mouse" in report.stopped
    after = desk.posted[seen["at"] :]
    # only NEO letting go of the button it held (in the test window), then nothing at all
    assert _kinds(after) == ["up"] and after[0]["pos"] == desk.posted[seen["at"] - 1]["pos"]
    names = [c.name for c in report.checks]
    assert names[-1] == "feedback" and names[-2] == "drag_board"
    assert _by_name(report)["drag_board"].status == "stopped"
    # the pointer isn't moved back (the user has it), but their app comes back: the test window was in front
    assert desk.pointer == (1300.0, 800.0)
    assert desk.front == ("Safari", USER_PID)


def test_a_window_on_top_of_the_test_window_means_no_run_at_all(mac):
    mac.set(cover=("Notes", 222, (100.0, 150.0, 300.0, 200.0)))
    report = _run(mac)
    assert report.refused and "Notes" in report.refused and "never click anything but the test window" in report.refused
    assert mac.desk.posted == []  # not even a pointer move
    assert report.checks == []
    assert mac.desk.alive is False


def test_a_target_outside_the_window_is_refused_before_anything_moves(mac):
    mac.set(targets={"well": [900.0, 500.0, 100.0, 36.0]})
    report = _run(mac)
    assert report.refused and "Safari" in report.refused
    assert mac.desk.posted == []


def test_keys_are_never_sent_when_another_app_took_the_focus(mac):
    """Mail comes forward right after NEO clicks into the text field: the typing is refused, so
    Mail gets nothing, and the run stops."""
    desk = mac.desk
    state = {"clicked": False}

    def hook(d, ev):
        if ev["type"] == _RealQ.kCGEventLeftMouseUp and d.focus_el == "field" and not state["clicked"]:
            state["clicked"] = True
            d.front = ("Mail", 333)

    desk.on_post = hook
    report = _run(mac)
    assert "keys would have gone to Mail" in report.stopped
    assert not any(e["type"] in KEYS for e in desk.posted)
    assert desk.stray == []
    assert _by_name(report)["type_field"].status == "stopped"


def test_without_an_event_tap_the_pointer_is_watched(mac):
    mac.set(no_tap=True)
    report = _run(mac)
    assert report.info["watching"].startswith("pointer only")
    assert report.ok, report.text()


def test_without_a_tap_a_moved_pointer_still_stops_the_run(mac):
    mac.set(no_tap=True)
    desk = mac.desk

    def hook(d, ev):
        if ev["type"] == _RealQ.kCGEventRightMouseUp:
            d.pointer = (1400.0, 900.0)  # the user moved the mouse (no tap sees it)

    desk.on_post = hook
    report = _run(mac)
    assert "you moved the mouse" in report.stopped
    assert [c.name for c in report.checks][-2] == "double_click"
    assert _by_name(report)["double_click"].status == "stopped"


def test_a_tap_that_cannot_tell_neo_from_the_user_falls_back_to_the_pointer(mac):
    mac.set(tap_blind=True)
    report = _run(mac)
    assert report.info["watching"].startswith("pointer only")
    assert report.ok, report.text()
    assert mac.desk.taps == []


def test_a_window_that_is_not_neos_own_child_is_not_driven(mac):
    mac.set(ready_pid=999)
    report = _run(mac)
    assert "isn't the one NEO started" in report.refused
    assert mac.desk.posted == []


def test_the_report_is_saved_where_neo_keeps_its_data(mac):
    from neo.config import settings

    report = _run(mac)
    saved = controls.save(report)
    assert saved is not None and saved.parent == settings().data_dir / "selftest"
    assert "Controls check: 19 of 19 passed" in saved.read_text()
    assert saved.with_suffix(".json").exists()
    assert oct(saved.stat().st_mode & 0o777) == "0o600"


def test_accessibility_positions_off_the_window_are_reported_and_never_clicked(mac):
    """ax_tree says the buttons are 500 pt to the right — past the test window's edge, over Safari.
    The report says so, and the click by id that would have landed on Safari is never sent."""
    mac.set(ax_shift=(500, 0))
    report = _run(mac)
    checks = _by_name(report)
    assert checks["ax_tree"].status == "fail" and "is outside" in checks["ax_tree"].arrived
    assert "would have gone to Safari" in report.stopped
    assert checks["click_id"].status == "stopped"
    assert mac.desk.stray == []
    *during, last = mac.desk.posted
    assert all(inside(FRAME, *e["pos"]) for e in during if "pos" in e)
    assert MOUSE[last["type"]] == "move" and last["pos"] == (900.0, 700.0)  # only the pointer, put back


def test_a_tap_without_keys_still_watches_the_mouse_and_says_so(mac):
    mac.set(tap_mouse_only=True)
    report = _run(mac)
    assert report.info["watching"].startswith("your mouse (an event tap; keys")
    assert report.ok, report.text()


def test_a_window_that_never_starts_is_explained(mac, monkeypatch):
    def launch(log, tag):
        log.with_suffix(".stderr").write_text("Traceback ...\nImportError: no AppKit here\n")
        mac.desk.alive = False  # it died at once
        return FakeChild(mac.desk)

    monkeypatch.setattr(mac, "launch", launch)
    report = _run(mac)
    assert report.refused == "the test window quit before it was ready"
    assert "ImportError: no AppKit here" in report.info["test window error"]
    assert mac.desk.posted == []


def test_a_broken_step_is_a_report_not_a_crash(mac, monkeypatch):
    from neo.tools.computer import input as inp

    def broken(*a, **k):
        raise RuntimeError("the window server went away")

    monkeypatch.setattr(mac, "windows", broken)  # the preflight's own look at the screen fails
    report = _run(mac)
    assert report.refused.startswith("the test itself broke") and "window server went away" in report.refused
    assert mac.desk.posted == [] and mac.desk.alive is False and inp._watch is None
