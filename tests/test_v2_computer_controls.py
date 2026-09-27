"""Computer controls: a click, drag or scroll lands where the model aimed, in the app it aimed at,
never on NEO's own window, with events real apps accept — and its result says what it hit and
whether anything changed. (The Chess run: every click and drag reported "ok", the board never moved.)

Hermetic: Quartz (every event made and posted), Accessibility (what is under a point), app
switching, the displays and screencapture are fakes; nothing reaches the real Mac."""

from __future__ import annotations

import asyncio
import types

import pytest
import Quartz as RealQ
from PIL import Image

from neo.agent import confirm
from neo.agent.registry import ToolContext, ToolOutput, registry
from neo.tools.computer import apps, ax, focus, screen
from neo.tools.computer import input as inp

NEO, CHESS, DOCK, MAIL = 900, 4242, 77, 55


# ---- fakes -----------------------------------------------------------------------------------------


class FakeQ:
    """Quartz for neo.tools.computer.input: records every event posted (and every sleep), in order."""

    kCGHIDEventTap = "hid"
    kCGMouseEventClickState = "clickState"
    kCGScrollEventUnitLine = "line"
    kCGMouseButtonLeft, kCGMouseButtonRight, kCGMouseButtonCenter = "L", "R", "C"

    def __init__(self, log: list):
        self.log = log
        self.fail_on_dragged = 0  # raise while making the Nth dragged event (0: never)
        self._dragged = 0
        for k in ("MouseMoved", "LeftMouseDown", "LeftMouseUp", "RightMouseDown", "RightMouseUp",
                  "OtherMouseDown", "OtherMouseUp", "LeftMouseDragged"):  # fmt: skip
            setattr(self, f"kCGEvent{k}", k)
        for m in ("Command", "Shift", "Alternate", "Control", "SecondaryFn"):
            setattr(self, f"kCGEventFlagMask{m}", getattr(RealQ, f"kCGEventFlagMask{m}"))

    def CGEventCreateMouseEvent(self, src, kind, pt, btn):
        if kind == "LeftMouseDragged":
            self._dragged += 1
            if self.fail_on_dragged and self._dragged == self.fail_on_dragged:
                raise RuntimeError("the event couldn't be made")
        if not all(isinstance(v, (int, float)) for v in pt):
            raise ValueError("depythonifying 'double', got 'str'")  # what PyObjC says
        return {"kind": kind, "at": (float(pt[0]), float(pt[1])), "btn": btn}

    def CGEventCreateScrollWheelEvent(self, src, unit, n, dy, dx=0):
        return {"kind": "Scroll", "dy": dy, "dx": dx}

    def CGEventSetIntegerValueField(self, ev, f, v):
        ev[f] = v

    def CGEventSetFlags(self, ev, f):
        ev["flags"] = f

    def CGEventCreateKeyboardEvent(self, src, code, down):
        return {"kind": "KeyDown" if down else "KeyUp", "code": code}

    def CGEventKeyboardSetUnicodeString(self, ev, n, s):
        ev["text"] = s

    def CGEventCreate(self, src):
        return {}

    def CGEventGetLocation(self, ev):
        return types.SimpleNamespace(x=self.pointer[0], y=self.pointer[1])

    pointer = (700.0, 500.0)


class World:
    """What the fake Mac shows: windows (front-most first) owned by apps, which app is in front,
    and a board whose picture changes when a click reaches it."""

    names = {NEO: "NEO", CHESS: "Chess", DOCK: "Dock", MAIL: "Mail"}
    regular = {NEO: True, CHESS: True, DOCK: False, MAIL: True}

    def __init__(self, log: list):
        self.log = log
        self.front = ("NEO", NEO)  # the user just talked to NEO: its overlay is key
        self.regions: list[tuple] = [
            (1100, 380, 400, 600, NEO, "Stop", "AXButton", ""),  # NEO's card, bottom right, on top
            (0, 940, 60, 42, DOCK, "Finder", "AXDockItem", ""),
            (60, 940, 1452, 42, DOCK, "Trash", "AXDockItem", ""),
            (0, 25, 1512, 915, CHESS, "", "AXGroup", ""),  # the Chess window, under NEO's card
        ]
        self.bring_ok = True
        self.react = True  # the board redraws when a click reaches it
        self.version = 0
        self.sheet_after_keys = ""  # a sheet ⌘N opens
        self.sheet = ""
        self.samples = 0

    def hit(self, x, y):
        self.log.append(("hit", round(x), round(y)))
        for rx, ry, rw, rh, pid, label, role, sheet in self.regions:
            if rx <= x < rx + rw and ry <= y < ry + rh:
                return ax.Hit(pid=pid, app=self.names[pid], label=label, role=role, regular=self.regular[pid], sheet=sheet)
        return ax.Hit()

    async def bring(self, pid):
        self.log.append(("bring", pid))
        if self.bring_ok:
            self.front = (self.names[pid], pid)
        return self.bring_ok, self.names[pid]

    def scene(self, pid):
        return ax.Scene(front=self.front[0], front_pid=self.front[1], title="Chess", sheet=self.sheet, known=True)

    def sample(self, pts, r=60):
        self.samples += 1
        shade = 40 + 60 * self.version
        return [Image.new("L", (24, 24), shade) for _ in pts]

    def posted(self, ev):
        self.log.append(("post", ev["kind"], ev.get("at")))
        if ev["kind"] == "LeftMouseUp" and self.react and self.front[1] == CHESS:
            self.version += 1
        if ev["kind"] == "Scroll" and self.react:
            self.version += 1
        if ev["kind"] == "KeyUp" and self.sheet_after_keys:
            self.sheet = self.sheet_after_keys

    def events(self):
        return [e for e in self.log if e[0] == "post"]


@pytest.fixture
def world(monkeypatch):
    log: list = []
    w = World(log)
    q = FakeQ(log)
    monkeypatch.setattr(inp, "Q", q)
    monkeypatch.setattr(inp, "_post", w.posted)
    monkeypatch.setattr(inp, "time", types.SimpleNamespace(sleep=lambda s: log.append(("sleep", s))))
    monkeypatch.setattr(ax, "hit", w.hit)
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": pid == NEO)
    monkeypatch.setattr(ax, "scene", w.scene)
    monkeypatch.setattr(focus, "front_app", lambda: w.front)
    monkeypatch.setattr(apps, "bring_pid_front", w.bring)
    monkeypatch.setattr(apps, "_target", None)
    monkeypatch.setattr(screen, "sample_points", w.sample)
    w.q = q
    return w


def run(name, args, ctx=None) -> ToolOutput:
    return asyncio.run(registry().invoke(name, args, ctx or ToolContext(user_text="play chess")))


# ---- the target: bring its app forward first, never NEO's window -------------------------------------


def test_a_click_on_a_background_app_brings_it_forward_before_any_event(world):
    out = run("click", {"x": 612, "y": 540})
    kinds = [e[0] if e[0] != "post" else e[1] for e in world.log if e[0] in ("bring", "post")]
    assert kinds == ["bring", "MouseMoved", "LeftMouseDown", "LeftMouseUp"]  # Chess first, then the click
    assert world.log[world.log.index(("bring", CHESS)) + 1 :]  # …and the click after it
    assert out.ok and "in Chess" in out.text and "brought Chess to the front first" in out.text
    assert apps.target() == "Chess"  # "click the field, then type": the typing goes to Chess too


def test_a_click_is_not_sent_when_its_app_wont_come_forward(world):
    world.bring_ok = False
    out = run("click", {"x": 612, "y": 540})
    assert not out.ok and "didn't come to the front" in out.text and world.events() == []


def test_the_dock_and_other_non_apps_are_clicked_without_activating_them(world):
    out = run("click", {"x": 30, "y": 960}, ToolContext(user_text=""))
    assert ("bring", DOCK) not in world.log and [e[1] for e in world.events()] == ["MouseMoved", "LeftMouseDown", "LeftMouseUp"]
    head, fenced = out.text.split('<untrusted source="screen">')
    assert "in Dock" in head and "Finder" not in head and "Finder" in fenced  # screen words are fenced


def test_a_click_on_neos_own_window_is_refused(world):
    out = run("click", {"x": 1300, "y": 700})  # the model sees Chess there: NEO's card is invisible in screenshots
    assert not out.ok and "NEO's own window" in out.text and world.events() == []
    assert ("bring", NEO) not in world.log and ("bring", CHESS) not in world.log


def test_a_drag_that_ends_on_neos_window_is_refused(world):
    out = run("drag", {"x1": 300, "y1": 300, "x2": 1300, "y2": 700})
    assert not out.ok and "NEO's own window" in out.text and world.events() == []


def test_scroll_at_a_pointer_resting_on_neo_scrolls_the_app_behind_it(world, monkeypatch):
    world.q.pointer = (1300.0, 900.0)  # the user just clicked NEO's orb
    monkeypatch.setattr(ax, "work_window_frame", lambda: (0.0, 25.0, 1000.0, 915.0))
    out = run("scroll", {"at_pointer": True, "dy": -8})
    scrolls = [e for e in world.log if e[0] == "post"]
    assert out.ok and scrolls[0][1:] == ("MouseMoved", (500.0, 482.5)) and scrolls[1][1] == "Scroll"


def test_a_click_on_a_window_a_sheet_blocks_is_refused_and_says_what_the_sheet_wants(world):
    world.regions[3] = (0, 25, 1512, 915, CHESS, "", "AXGroup", "“Start a New Game?” (buttons: Start, Cancel)")
    out = run("click", {"x": 612, "y": 540})
    assert not out.ok and world.events() == []
    assert "sheet open" in out.text and "Start, Cancel" in out.text and '<untrusted source="screen">' in out.text


def test_a_point_off_every_display_is_refused(world):
    out = run("click", {"x": 5000, "y": 540})
    assert not out.ok and "off every screen" in out.text and world.events() == []


def test_coordinates_as_strings_are_taken_and_nonsense_is_refused_before_any_event(world):
    assert run("click", {"x": "612", "y": "540"}).ok
    world.log.clear()
    out = run("drag", {"x1": 300, "y1": 300, "x2": "e4", "y2": 500})
    assert not out.ok and "x2 must be a number" in out.text and world.events() == []


@pytest.mark.parametrize("args, why", [({"count": 0}, "count"), ({"count": 5}, "count"), ({"button": "back"}, "button"), ({"modifiers": ["hyper"]}, "modifier")])
def test_a_click_that_cannot_be_made_as_asked_is_an_error_not_a_different_click(world, args, why):
    out = run("click", {"x": 612, "y": 540, **args})
    assert not out.ok and why in out.text and world.events() == []


def test_triple_middle_and_modifier_clicks(world):
    run("click", {"x": 612, "y": 540, "count": 3, "button": "middle", "modifiers": ["cmd", "shift"]})
    downs = [e for e in world.log if e[0] == "post" and e[1] == "OtherMouseDown"]
    assert len(downs) == 3
    q_events = [e for e in world.log if e[0] == "post"]
    assert q_events[0][1] == "MouseMoved"


def test_modifier_flags_and_click_state_are_on_every_press(monkeypatch):
    posted: list = []
    q = FakeQ([])
    monkeypatch.setattr(inp, "Q", q)
    monkeypatch.setattr(inp, "_post", posted.append)
    monkeypatch.setattr(inp, "time", types.SimpleNamespace(sleep=lambda s: None))
    inp.click(10, 20, count=2, mods=["cmd"])
    presses = [e for e in posted if e["kind"] != "MouseMoved"]
    assert [e["clickState"] for e in presses] == [1, 1, 2, 2]
    assert all(e["flags"] == RealQ.kCGEventFlagMaskCommand for e in presses)


# ---- clicks by element id: where it is now, and what is really there -------------------------------------


def _element(monkeypatch, pid=CHESS, name="Chess", frame=(400.0, 300.0, 100.0, 40.0), title="New Game"):
    e = ax.Element("e7", "AXButton", title, "", 100, 100, 50, 50, actions=["AXPress"], ref=object())
    monkeypatch.setattr(ax, "_issued", {"e7": e})
    monkeypatch.setattr(ax, "pid_of", lambda el: pid)
    monkeypatch.setattr(ax, "app_of", lambda el: name)
    monkeypatch.setattr(ax, "visible_frame", lambda el: frame)
    monkeypatch.setattr(ax, "regular", lambda p: True)
    return e


def test_a_click_by_id_goes_where_the_element_is_now_not_where_it_was(world, monkeypatch):
    _element(monkeypatch)  # the tree said (100,100 50x50); the window moved since
    out = run("click", {"id": "e7"})
    moves = [e for e in world.log if e[0] == "post" and e[1] == "MouseMoved"]
    assert out.ok and moves[0][2] == (450.0, 320.0) and ("bring", CHESS) in world.log


def test_a_click_by_id_on_an_element_scrolled_out_of_view_is_refused(world, monkeypatch):
    _element(monkeypatch, frame=None)
    out = run("click", {"id": "e7"})
    assert not out.ok and "isn't on screen" in out.text and world.events() == []


def test_a_click_by_id_is_gated_on_what_is_really_under_the_point(world, monkeypatch):
    _element(monkeypatch)  # "New Game" in Chess — but Mail's Send button is on top there
    world.regions.insert(0, (380, 280, 200, 100, MAIL, "Send", "AXButton", ""))
    out = run("click", {"id": "e7"})
    assert out.halt and out.text.startswith(confirm.NEEDS_CONFIRM) and "Send" in out.text
    assert world.events() == [] and ("bring", CHESS) not in world.log
    confirm.cancel("t")


def test_a_click_by_id_on_a_covered_element_is_refused(world, monkeypatch):
    _element(monkeypatch)
    world.regions.insert(0, (380, 280, 200, 100, MAIL, "Inbox", "AXRow", ""))
    world.front = ("Chess", CHESS)  # Chess is in front, and still Mail's floating window covers it
    out = run("click", {"id": "e7"})
    assert not out.ok and "covers that element" in out.text and world.events() == []


def test_a_click_by_id_on_neos_own_element_is_refused(world, monkeypatch):
    _element(monkeypatch, pid=NEO, name="NEO", title="Yes, do it ⌘⏎")
    out = run("click", {"id": "e7"})
    assert not out.ok and "NEO's own window" in out.text and world.events() == []


# ---- drags real apps take -------------------------------------------------------------------------------


def test_a_drag_holds_the_button_moves_off_gently_and_lets_go_after_a_rest(world):
    world.front = ("Chess", CHESS)
    out = run("drag", {"x1": 300, "y1": 300, "x2": 500, "y2": 300})
    seq = [e for e in world.log if e[0] in ("post", "sleep")]
    kinds = [e[1] if e[0] == "post" else "sleep" for e in seq]
    down = kinds.index("LeftMouseDown")
    assert kinds[0] == "MouseMoved" and kinds[1] == "sleep" and kinds[down + 1] == "sleep"
    assert seq[down + 1][1] >= 0.1  # held a moment before moving
    drags = [e for e in seq if e[0] == "post" and e[1] == "LeftMouseDragged"]
    assert drags[0][2] == (304.0, 300.0)  # a small first step past the drag threshold
    assert drags[-1][2] == (500.0, 300.0) and kinds[-1] == "LeftMouseUp" and kinds[-2] == "sleep"
    assert sum(e[1] for e in seq if e[0] == "sleep") >= 0.4  # a human pace, not 0.18 s
    assert out.ok and "Changed" in out.text


def test_a_drag_that_fails_midway_still_lets_go_of_the_button(monkeypatch):
    posted: list = []
    q = FakeQ([])
    q.fail_on_dragged = 3
    monkeypatch.setattr(inp, "Q", q)
    monkeypatch.setattr(inp, "_post", posted.append)
    monkeypatch.setattr(inp, "time", types.SimpleNamespace(sleep=lambda s: None))
    with pytest.raises(RuntimeError):
        inp.drag(100, 100, 300, 100)
    assert posted[-1]["kind"] == "LeftMouseUp" and posted[-1]["clickState"] == 1


def test_a_drag_onto_the_trash_asks_first(world):
    out = run("drag", {"x1": 300, "y1": 300, "x2": 100, "y2": 960})
    assert out.halt and out.text.startswith(confirm.NEEDS_CONFIRM) and "Trash" in out.text and world.events() == []
    confirm.cancel("t")


# ---- what changed --------------------------------------------------------------------------------------


def test_a_click_after_which_nothing_changed_is_not_reported_as_done(world):
    world.react = False  # the click was swallowed
    out = run("click", {"x": 612, "y": 540})
    assert not out.ok and "Nothing visibly changed" in out.text and len(world.events()) == 3


def test_a_click_that_moved_something_says_so(world):
    out = run("click", {"x": 612, "y": 540})
    assert out.ok and "Changed: the screen changed around (612, 540)" in out.text


def test_a_command_the_user_gave_is_not_watched(world):
    ctx = ToolContext(user_text="scroll down", origin="user")
    run("scroll", {"at_pointer": True, "dy": -8}, ctx)
    assert world.samples == 0  # the user is looking at the screen themselves


def test_a_hotkey_that_opened_a_sheet_says_so(world, monkeypatch):
    world.front = ("Chess", CHESS)
    world.sheet_after_keys = "a sheet “Start a New Game?” (buttons: Start, Cancel)"
    monkeypatch.setattr(ax, "frontmost_app", lambda: ("Chess", CHESS))
    monkeypatch.setattr(apps, "front_name", lambda: "Chess")
    out = run("hotkey", {"keys": "cmd+n"})
    assert out.ok and "a sheet or dialog opened" in out.text and "Start, Cancel" in out.text


def test_no_event_is_sent_when_macos_wont_take_them(world, monkeypatch):
    monkeypatch.setattr(inp, "can_post", lambda: False)
    out = run("click", {"x": 612, "y": 540})
    assert out.halt and out.text.startswith(confirm.NEEDS_USER) and world.events() == []


# ---- a brain that can't see -------------------------------------------------------------------------------


def test_a_blind_brain_may_not_click_drag_or_scroll_at_coordinates(world):
    ctx = ToolContext(user_text="play chess", extras={"blind": True})
    for name, args in (("click", {"x": 612, "y": 540}), ("drag", {"x1": 1, "y1": 2, "x2": 3, "y2": 4}), ("scroll", {"x": 5, "y": 6})):
        out = run(name, args, ctx)
        assert not out.ok and "can't see the screen" in out.text
    assert world.events() == []
    # coordinates the user said themselves are theirs, not made up
    assert run("click", {"x": 612, "y": 540}, ToolContext(user_text="click at 612, 540", extras={"blind": True})).ok


# ---- screenshots: truthful geometry, and NEO maps pixels itself ------------------------------------------------


@pytest.fixture
def shots(monkeypatch):
    grabbed: list = []
    size = {"px": (3024, 1964)}

    def grab(extra):
        grabbed.append(extra)
        return Image.new("RGB", size["px"], (200, 200, 200))

    monkeypatch.setattr(screen, "can_capture", lambda: True)
    monkeypatch.setattr(screen, "_grab", grab)
    return types.SimpleNamespace(grabbed=grabbed, size=size)


def test_the_note_on_a_1_to_1_screenshot_says_pixels_are_points_and_gives_the_size(shots):
    out = run("screenshot", {})
    assert out.ok and out.images and not out.fence  # NEO's own words, not screen content
    assert "the image is 1512×982 px" in out.text and 'shot="s' in out.text
    assert "image pixels are also screen points" in out.text


def test_a_downscaled_screenshot_never_claims_pixels_are_click_points(shots, monkeypatch, world):
    monkeypatch.setattr(screen, "displays", lambda: [screen.Display(1, 0.0, 0.0, 1728.0, 1117.0)])
    shots.size["px"] = (3456, 2234)  # a 16" MacBook Pro at its default
    out = run("screenshot", {})
    assert "1600×1034 px" in out.text and "1.0800" in out.text and "also screen points" not in out.text
    sid = out.text.split('shot="')[1].split('"')[0]
    world.front = ("Chess", CHESS)
    got = run("click", {"x": 1500, "y": 900, "shot": sid})
    moved = [e for e in world.log if e[0] == "post" and e[1] == "MouseMoved"][0][2]
    assert got.ok and moved == pytest.approx((1620.0, 972.0))  # pixel × 1.08: NEO did the arithmetic
    assert not run("click", {"x": 1650, "y": 900, "shot": sid}).ok  # outside the 1600-px image: refused


def test_a_region_screenshot_maps_its_pixels_from_its_origin(shots, world):
    shots.size["px"] = (3024, 1888)
    out = run("screenshot", {"x": 0, "y": 38, "w": 1512, "h": 944})
    assert "1512×944 px" in out.text and "screen_y = 38 + image_y × 1.0000" in out.text
    sid = out.text.split('shot="')[1].split('"')[0]
    run("click", {"x": 612, "y": 502, "shot": sid})
    assert [e for e in world.log if e[0] == "post"][0][2] == (612.0, 540.0)  # the 38-pt origin added by NEO


def test_a_region_running_off_the_screen_is_cut_to_what_exists(shots):
    shots.size["px"] = (1024, 1000)
    out = run("screenshot", {"x": 1000, "y": 0, "w": 1000, "h": 500})
    assert shots.grabbed[-1] == ["-R", "1000,0,512,500"] and "region 512×500 pt" in out.text
    assert not run("screenshot", {"x": 3000, "y": 0, "w": 100, "h": 100}).ok


def test_other_displays_are_named_and_can_be_captured(shots, monkeypatch):
    two = [screen.Display(1, 0.0, 0.0, 1512.0, 982.0), screen.Display(2, 1512.0, -200.0, 1920.0, 1080.0)]
    monkeypatch.setattr(screen, "displays", lambda: two)
    out = run("screenshot", {})
    assert "display 2: 1920×1080 pt at (1512,-200)" in out.text and "display=2" in out.text
    shots.size["px"] = (1920, 1080)
    out = run("screenshot", {"display": 2})
    assert shots.grabbed[-1] == ["-D", "2"] and "screen_x = 1512 + image_x × 1.2000" in out.text


# ---- keys ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "combo, key, mods",
    [
        ("opt+left", "left", ["alt"]),
        ("ctl+c", "c", ["ctrl"]),
        ("⌘c", "c", ["cmd"]),
        ("cmd+opt+esc", "escape", ["cmd", "alt"]),
        ("cmd+shift+}", "]", ["cmd", "shift"]),
        ("cmd+?", "/", ["cmd", "shift"]),
        ("ctrl-shift-tab", "tab", ["ctrl", "shift"]),
        ("cmd+-", "-", ["cmd"]),
    ],
)
def test_every_way_of_writing_a_chord_presses_that_chord(combo, key, mods):
    flags, k = inp.parse_chord(combo)
    assert k == key and flags == sum(inp._MODS[m] for m in mods)


def test_an_unknown_modifier_or_key_presses_nothing(world, monkeypatch):
    monkeypatch.setattr(ax, "frontmost_app", lambda: ("Chess", CHESS))
    monkeypatch.setattr(apps, "front_name", lambda: "Chess")
    for combo in ("hyper+c", "cmd+a+b", "cmd+foo"):
        out = run("hotkey", {"keys": combo})
        assert not out.ok and world.events() == []
    assert inp.hotkey("hyper+c").startswith("Error")
