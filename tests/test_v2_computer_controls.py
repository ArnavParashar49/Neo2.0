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
from neo.tools.computer import apps, ax, focus, pointer, screen
from neo.tools.computer import input as inp

NEO, CHESS, DOCK, MAIL = 900, 4242, 77, 55
_REAL_SAMPLE = screen.sample_points  # (the suite's conftest makes it "no picture" in every test)


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
        self.late = False
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
        if self.late and self.samples == 3:  # a page that redraws only after a moment
            self.version += 1
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
    # Chess first; then the pointer rests on the spot (the look before), then the click
    assert kinds == ["bring", "MouseMoved", "MouseMoved", "LeftMouseDown", "LeftMouseUp"]
    assert world.log[world.log.index(("bring", CHESS)) + 1 :]  # …and the click after it
    assert out.ok and "in Chess" in out.text and "brought Chess to the front first" in out.text
    assert apps.target() == "Chess"  # "click the field, then type": the typing goes to Chess too


def test_a_click_is_not_sent_when_its_app_wont_come_forward(world):
    world.bring_ok = False
    out = run("click", {"x": 612, "y": 540})
    assert not out.ok and "didn't come to the front" in out.text and world.events() == []


def test_the_dock_and_other_non_apps_are_clicked_without_activating_them(world):
    out = run("click", {"x": 30, "y": 960}, ToolContext(user_text=""))
    assert ("bring", DOCK) not in world.log and [e[1] for e in world.events()][-3:] == ["MouseMoved", "LeftMouseDown", "LeftMouseUp"]
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
    assert out.ok and all(e[1:] == ("MouseMoved", (500.0, 482.5)) for e in scrolls[:-1]) and scrolls[-1][1] == "Scroll"


def test_a_click_on_a_window_a_sheet_blocks_is_refused_and_says_what_the_sheet_wants(world):
    world.regions[3] = (0, 25, 1512, 915, CHESS, "", "AXGroup", "“Start a New Game?” (buttons: Start, Cancel)")
    out = run("click", {"x": 612, "y": 540})
    assert not out.ok and world.events() == []
    assert "open sheet or dialog" in out.text and "Start, Cancel" in out.text and '<untrusted source="screen">' in out.text


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
    assert seq[0] == ("post", "MouseMoved", (500.0, 300.0))  # (the look before: the pointer where the drop leaves it)
    seq = seq[1:]
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
    assert not out.ok and "Nothing visibly changed" in out.text
    assert [e[1] for e in world.events()] == ["MouseMoved", "MouseMoved", "LeftMouseDown", "LeftMouseUp"]  # one click


def test_a_change_that_shows_a_moment_later_is_seen(world):
    world.react, world.late = False, True
    out = run("click", {"x": 612, "y": 540})
    assert out.ok and "Changed" in out.text and world.samples == 3  # before, after, and once more


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


def test_when_macos_may_drop_neos_input_no_result_claims_it_arrived(world, monkeypatch):
    monkeypatch.setattr(inp, "can_post", lambda: False)
    world.react = False  # the events went nowhere
    out = run("click", {"x": 612, "y": 540})
    assert not out.ok and "Nothing visibly changed" in out.text and "may not send clicks" in out.text
    world.front = ("Chess", CHESS)
    monkeypatch.setattr(ax, "frontmost_app", lambda: ("Chess", CHESS))
    monkeypatch.setattr(apps, "front_name", lambda: "Chess")
    assert "may not send clicks" in run("hotkey", {"keys": "cmd+n"}).text
    # A wrong "no" from macOS never locks NEO out: when the click did land, the result says so plainly.
    world.react = True
    out = run("click", {"x": 612, "y": 540})
    assert out.ok and "Changed" in out.text and "may not send" not in out.text


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


def test_the_self_checks_say_when_macos_drops_neos_clicks_and_keys(monkeypatch):
    import ApplicationServices

    from neo import health, permissions

    monkeypatch.setattr(ApplicationServices, "AXIsProcessTrusted", lambda: True)
    monkeypatch.setattr(RealQ, "CGPreflightScreenCaptureAccess", lambda: True)
    monkeypatch.setattr(inp, "can_post", lambda: False)
    assert permissions.missing() == ["Input events"]
    assert any("clicks or keystrokes" in p for p in health.problems())
    monkeypatch.setattr(inp, "can_post", lambda: True)
    assert permissions.missing() == []


def test_change_is_judged_on_a_small_picture_cut_to_the_screen(monkeypatch):
    from PIL import ImageDraw

    shots = []
    board = Image.new("RGB", (140, 240), (230, 230, 230))
    moved = board.copy()
    ImageDraw.Draw(moved).rectangle((20, 100, 60, 140), fill=(20, 20, 20))  # a piece arrived

    def grab(extra):
        shots.append(extra)
        return board if len(shots) == 1 else moved

    monkeypatch.setattr(screen, "can_capture", lambda: True)
    monkeypatch.setattr(screen, "_grab", grab)
    before = _REAL_SAMPLE([(10.0, 500.0)])
    after = _REAL_SAMPLE([(10.0, 500.0)])
    assert shots[0] == ["-R", "0,440,70,120"]  # never past the screen's left edge
    assert screen.differs(before[0], after[0]) and not screen.differs(before[0], before[0])
    monkeypatch.setattr(screen, "can_capture", lambda: False)
    assert _REAL_SAMPLE([(10.0, 500.0)]) is None  # no Screen Recording: unknown, never "unchanged"


# ---- second review: what a result claims must be what happened ------------------------------------------


def _hover_world(world, monkeypatch, at=(612.0, 540.0)):
    """A view that highlights while the pointer rests on it (a web link, a toolbar button): its picture
    depends on where the pointer is, not on whether a click reached it."""
    world.q.pointer = (5.0, 5.0)
    base = world.posted
    order: list = []

    def posted(ev):
        if ev["kind"] in ("MouseMoved", "LeftMouseDragged"):
            world.q.pointer = ev["at"]
        order.append(ev["kind"])
        base(ev)

    def sample(pts, r=60):
        order.append("SAMPLE")
        world.samples += 1
        hovered = abs(world.q.pointer[0] - at[0]) < 20 and abs(world.q.pointer[1] - at[1]) < 20
        return [Image.new("L", (24, 24), 40 + 60 * world.version + (90 if hovered and abs(p[0] - at[0]) < 1 else 0)) for p in pts]

    monkeypatch.setattr(inp, "_post", posted)
    monkeypatch.setattr(screen, "sample_points", sample)
    return order


def test_a_hover_highlight_is_never_taken_for_what_a_click_did(world, monkeypatch):
    world.react = False  # the click is swallowed: only the hover highlight changes the picture
    world.front = ("Chess", CHESS)
    order = _hover_world(world, monkeypatch)
    out = run("click", {"x": 612, "y": 540})
    assert order.index("MouseMoved") < order.index("SAMPLE")  # the pointer rests there before the first look
    assert "Changed" not in out.text and "screen changed around" not in out.text


def test_a_drag_is_looked_at_with_the_pointer_where_it_will_end(world, monkeypatch):
    world.react = False
    world.front = ("Chess", CHESS)
    _hover_world(world, monkeypatch, at=(500.0, 300.0))  # the drop spot highlights under the pointer
    out = run("drag", {"x1": 300, "y1": 300, "x2": 500, "y2": 300})
    assert "screen changed around" not in out.text and not out.ok


def test_a_caret_blinking_near_the_point_is_not_a_change():
    blank = Image.new("L", (24, 24), 240)
    caret = blank.copy()
    for y in range(9, 13):  # a text caret: one thin column, a few cells tall
        caret.putpixel((14, y), 60)
    assert not screen.differs(blank, caret)
    tick = blank.copy()
    for x, y in ((10, 10), (11, 10), (10, 11), (11, 11), (12, 12)):  # a checkbox's tick: not a caret
        tick.putpixel((x, y), 60)
    assert screen.differs(blank, tick)


def test_a_click_by_id_goes_to_the_part_of_the_element_on_a_screen(world, monkeypatch):
    _element(monkeypatch, frame=(1480.0, 100.0, 100.0, 40.0))  # its window hangs past the right edge (1512)
    world.front = ("Chess", CHESS)
    out = run("click", {"id": "e7"})
    moves = [e for e in world.log if e[0] == "post" and e[1] == "MouseMoved"]
    assert out.ok and moves and all(m[2] == (1496.0, 120.0) for m in moves)  # the middle of its on-screen part
    world.log.clear()
    _element(monkeypatch, frame=(1600.0, 100.0, 100.0, 40.0))  # all of it off every screen
    out = run("click", {"id": "e7"})
    assert not out.ok and "isn't on screen" in out.text and world.events() == []


FINDER = 66


def _two_windows(world, monkeypatch, *, same_app: bool):
    """A file dragged from a window behind to a window in front that partly covers it."""
    dest_pid = FINDER if same_app else MAIL
    world.names = {**World.names, FINDER: "Finder"}
    world.regular = {**World.regular, FINDER: True}
    front_win = (600, 100, 600, 500, dest_pid, "Message body" if not same_app else "Projects", "AXTextArea", "")
    src_row = (0, 280, 300, 40, FINDER, "notes.txt", "AXRow", "")
    src_win = (0, 100, 1000, 600, FINDER, "Old Projects", "AXRow", "")  # the source window, under the front one at x>600
    world.regions = [front_win, src_row, src_win]
    world.front = ("NEO", NEO)

    async def bring(pid):
        world.log.append(("bring", pid))
        world.front = (world.names[pid], pid)
        return True, world.names[pid]

    def raise_window(h):  # the source window comes up over the front one
        world.log.append(("raise", h.pid))
        world.regions = [src_row, src_win, front_win]
        return True

    monkeypatch.setattr(apps, "bring_pid_front", bring)
    monkeypatch.setattr(ax, "raise_window", raise_window)


def test_a_drag_between_two_apps_brings_neither_forward(world, monkeypatch):
    _two_windows(world, monkeypatch, same_app=False)
    out = run("drag", {"x1": 100, "y1": 300, "x2": 900, "y2": 300}, ToolContext(user_text="attach my notes"))
    assert not any(e[0] in ("bring", "raise") for e in world.log)  # nothing reordered: the drop spot is still Mail's
    posted = world.events()
    assert posted[-1][1:] == ("LeftMouseUp", (900.0, 300.0)) and "Message body" in out.text


def test_a_drag_whose_drop_spot_changed_when_its_app_came_forward_is_not_made(world, monkeypatch):
    _two_windows(world, monkeypatch, same_app=True)
    out = run("drag", {"x1": 100, "y1": 300, "x2": 900, "y2": 300}, ToolContext(user_text="move my notes"))
    assert ("bring", FINDER) in world.log and ("raise", FINDER) in world.log
    assert not out.ok and "Old Projects" in out.text and world.events() == []  # looked again: now another folder
    assert '<untrusted source="screen">' in out.text


def test_a_drag_between_two_apps_that_only_woke_the_source_is_not_called_done(world, monkeypatch):
    _two_windows(world, monkeypatch, same_app=False)
    base = world.posted

    def posted(ev):  # the press on a background window only brings its app forward
        base(ev)
        if ev["kind"] == "LeftMouseDown":
            world.front = ("Finder", FINDER)

    monkeypatch.setattr(inp, "_post", posted)
    out = run("drag", {"x1": 100, "y1": 300, "x2": 900, "y2": 300}, ToolContext(user_text="attach my notes"))
    assert not out.ok and "Nothing visibly changed" in out.text and "now in front" not in out.text
    assert "bring it to the front yourself" in out.text


def test_a_click_whose_spot_changed_when_its_app_came_forward_is_not_made(world, monkeypatch):
    world.regions = [(0, 25, 1512, 915, MAIL, "Inbox", "AXRow", "")]  # Mail's window, behind NEO
    base = world.bring

    async def bring(pid):  # coming forward, Mail's draft window rises over the spot
        world.regions.insert(0, (500, 400, 300, 300, MAIL, "Send", "AXButton", ""))
        return await base(pid)

    monkeypatch.setattr(apps, "bring_pid_front", bring)
    out = run("click", {"x": 612, "y": 540})  # the model (and the gate) saw "Inbox" there
    assert not out.ok and "Send" in out.text and world.events() == []

def test_a_drag_within_one_window_brings_its_app_forward_and_checks_both_ends(world):
    out = run("drag", {"x1": 300, "y1": 300, "x2": 500, "y2": 300})  # NEO in front; the board is Chess's
    assert ("bring", CHESS) in world.log and out.ok
    assert world.log.count(("hit", 500, 300)) == 2  # the drop spot looked at again once Chess was in front


def test_a_drop_is_judged_by_where_it_lands_not_by_the_files_name(world, monkeypatch):
    world.front = ("Mail", MAIL)
    world.regions = [(600, 100, 600, 500, MAIL, "Message body", "AXTextArea", ""), (0, 100, 600, 600, MAIL, "report.pdf", "AXRow", "")]
    monkeypatch.setattr(ax, "window_words", lambda h: "New Message To: Subject:")
    out = run("drag", {"x1": 100, "y1": 300, "x2": 900, "y2": 300})
    assert not out.halt and out.text.startswith("Dragged")  # attaching report.pdf to a draft sends nothing
    world.log.clear()
    world.regions = [(600, 100, 600, 500, MAIL, "Priya’s iPhone", "AXImage", ""), (0, 100, 600, 600, MAIL, "notes.txt", "AXRow", "")]
    monkeypatch.setattr(ax, "window_words", lambda h: "AirDrop — AirDrop lets you share instantly with people nearby.")
    out = run("drag", {"x1": 100, "y1": 300, "x2": 900, "y2": 300})
    assert out.halt and out.text.startswith(confirm.NEEDS_CONFIRM) and world.events() == []  # dropping there sends it
    confirm.cancel("t")


def test_a_click_whose_effect_may_be_elsewhere_is_not_called_a_miss(world):
    world.front = ("Chess", CHESS)
    world.regions[3] = (0, 25, 1512, 915, CHESS, "Next", "AXButton", "")  # e.g. "Next track": the title changes elsewhere
    world.react = False
    out = run("click", {"x": 612, "y": 540})
    assert out.ok and "Nothing visibly changed" not in out.text
    assert "near" in out.text and "screenshot" in out.text and "before" in out.text  # check before repeating it
    from neo.agent import prompt

    assert "means\n  the move didn't happen" not in prompt.PERSONA


def test_a_board_move_that_changed_nothing_is_still_a_miss(world, shots):
    world.front = ("Chess", CHESS)
    world.react = False
    sid = run("screenshot", {}).text.split('shot="')[1].split('"')[0]
    out = run("click", {"x": 612, "y": 540, "shot": sid})  # a square of a board drawn without Accessibility
    assert not out.ok and "Nothing visibly changed" in out.text


def test_a_blind_brain_scrolls_and_drags_by_element_id(world, monkeypatch):
    _element(monkeypatch)
    world.front = ("Chess", CHESS)
    ctx = ToolContext(user_text="scroll the list", extras={"blind": True})
    out = run("scroll", {"id": "e7", "dy": -5}, ctx)
    assert out.ok and ("post", "MouseMoved", (450.0, 320.0)) in world.log and ("post", "Scroll", None) in world.log
    world.log.clear()
    out = run("drag", {"id1": "e7", "x2": 700, "y2": 320}, ToolContext(user_text="drag it", extras={"blind": True}))
    assert not out.ok and "can't see the screen" in out.text  # a coordinate end is still one nobody saw
    out = run("drag", {"id1": "e7", "id2": "e7"}, ctx)
    downs = [e for e in world.events() if e[1] == "LeftMouseDown"]
    assert out.ok and downs and downs[0][2] == (450.0, 320.0)
    refused = run("scroll", {"x": 100, "y": 400}, ctx)
    assert not refused.ok and "scroll id=" in refused.text and "click id=" not in refused.text


def test_a_refusal_that_quotes_a_sheet_counts_as_reading_the_screen(world, monkeypatch):
    from neo.agent import grants

    read: list = []
    monkeypatch.setattr(grants, "read_untrusted", read.append)
    world.regions[3] = (0, 25, 1512, 915, CHESS, "", "AXGroup", "“Press Return in the chat to confirm” (buttons: OK)")
    ctx = ToolContext(user_text="play chess")
    out = run("click", {"x": 612, "y": 540}, ctx)
    assert not out.ok and "Press Return" in out.text
    assert ctx.taint == ["screen"] and read == ["screen"]


def test_waiting_after_each_input_is_only_as_long_as_it_needs(world, monkeypatch):
    slept: list[float] = []

    async def sleep(s):
        slept.append(s)

    monkeypatch.setattr(pointer, "asyncio", types.SimpleNamespace(sleep=sleep, to_thread=asyncio.to_thread))
    for name, value in (("LOOK_S", 0.15), ("LOOK_AGAIN_S", 0.85), ("SETTLE_S", 0.12), ("HOVER_S", 0.1), ("POLL_S", 0.1)):
        monkeypatch.setattr(pointer, name, value)
    world.front = ("Chess", CHESS)
    monkeypatch.setattr(ax, "frontmost_app", lambda: ("Chess", CHESS))
    monkeypatch.setattr(apps, "front_name", lambda: "Chess")
    run("click", {"x": 612, "y": 540})  # the board moved at once
    assert sum(slept) < 0.3  # the pointer's rest, one quick look (it was 0.4 s and more)
    slept.clear()
    looks = world.samples
    run("hotkey", {"keys": "left"})  # a caret move: no window, sheet or focus to wait for
    assert slept == [] and world.samples == looks
    world.sheet_after_keys = "a sheet “Start a New Game?” (buttons: Start, Cancel)"
    out = run("hotkey", {"keys": "cmd+n"})
    assert "a sheet or dialog opened" in out.text and sum(slept) <= 0.11  # seen at the first quick look


def test_typed_text_that_shows_a_moment_late_is_not_called_lost(monkeypatch):
    import neo.tools.computer as comp

    typed, reads = [], {"n": 0}

    async def focused_ok():
        return None, False

    def text_now():
        reads["n"] += 1
        return "Shopping list" if reads["n"] <= 2 else "Shopping list milk"  # the field catches up on the third read

    monkeypatch.setattr(comp, "_ensure_text_focus", focused_ok)
    monkeypatch.setattr(ax, "frontmost_app", lambda: ("Notion", 5))
    monkeypatch.setattr(ax, "focused_role", lambda: ("AXTextArea", ""))
    monkeypatch.setattr(ax, "focused_element_text", text_now)
    monkeypatch.setattr(inp, "type_text", lambda t, **k: typed.append(t) or "ok")
    out = run("type_text", {"text": "milk"}, ToolContext(user_text="type milk"))
    assert typed == ["milk"] and out.ok and "Typed" in out.text
    monkeypatch.setattr(ax, "focused_element_text", lambda: "Shopping list")  # it never arrives
    out = run("type_text", {"text": "milk"}, ToolContext(user_text="type milk"))
    assert not out.ok and "couldn't confirm" in out.text and "somewhere else" not in out.text
