"""The review of eyes-marks, fixed: a board square NEO didn't find itself is never a blind brain's to
act on; the marks walk, the look and a screenshot's marks keep their deadlines on a hung app; the
waits after an input keep theirs by the clock; a snap never carries a click off something pressable
onto the control around it; a change seen in the scene is confirmed by a picture (after it settles,
for a brain that gets the picture); the approved click is the click that was gated; a screenshot in
use outlives after-looks; a brain that can't see gets no coordinates in any result; a helper's mark
words reach the click gate.

Hermetic: the fake Mac of test_v2_computer_controls, fake Accessibility trees and a fake clock."""

from __future__ import annotations

import asyncio
import time
import types

import pytest
import test_eyes_coords as ec
import test_v2_computer_controls as base
from PIL import Image
from test_v2_computer_controls import CHESS, run

from neo2.core.agent import confirm
from neo2.core.agent.registry import ToolContext
from neo2.core.tools.computer import ax, marks, observe, pointer, screen

world = base.world
new_tools = ec.new_tools
captures = ec.captures
clock = ec.clock
_no_picture = ec._no_picture
_no_card_left = ec._no_card_left

BLIND = {"blind": True}


def _blind(text="play e4") -> ToolContext:
    return ToolContext(user_text=text, extras=dict(BLIND))


# ---- 1. a board's squares: the blind rule goes through the seam ----------------------------------------


class _BoardError(Exception):
    """Like the board helper's own error: not a ValueError, words with no coordinates."""


def _board(source: str, asked: list | None = None):
    def resolve(cell, shot, *, blind=False):
        if asked is not None:
            asked.append((cell, blind))
        if blind and source != "neo":
            raise _BoardError(
                "Error: you can't see the screen, and this board's squares were mapped from a seeing model's corners"
            )
        return (612.0, 540.0)

    return resolve


def test_a_blind_brain_is_refused_squares_a_seeing_model_mapped(world, new_tools):
    world.front = ("Chess", CHESS)
    asked: list = []
    pointer.set_cell_resolver(_board("vision", asked))
    try:
        out = run("click", {"cell": "e4"}, _blind())
        assert not out.ok and "seeing model" in out.text and "Error: Error" not in out.text
        out = run("drag", {"from": "e2", "to": "e4"}, _blind())
        assert not out.ok and world.events() == []
        assert asked and all(b for _, b in asked)  # the brain's blindness reached the board helper
        assert run("click", {"cell": "e4"}, ToolContext(user_text="play e4")).ok  # a seeing brain may
        assert asked[-1] == ("e4", False)
    finally:
        pointer.set_cell_resolver(None)


def test_a_board_found_by_neo_still_serves_a_blind_brain(world, new_tools):
    world.front = ("Chess", CHESS)
    pointer.set_cell_resolver(_board("neo"))
    try:
        out = run("click", {"cell": "e4"}, _blind())
        assert out.ok and ec._moved(world) == (612.0, 540.0)
    finally:
        pointer.set_cell_resolver(None)


def test_a_cell_resolver_that_cant_hear_the_blind_flag_is_not_taken():
    before = pointer._cell_resolver  # (the board helper's, registered at import)
    with pytest.raises(TypeError):
        pointer.set_cell_resolver(lambda cell, shot: (1.0, 2.0))
    assert pointer._cell_resolver is before
    pointer.set_cell_resolver(lambda cell, shot, **kw: (1.0, 2.0))  # (a catch-all takes it)
    pointer.set_cell_resolver(None)


# ---- 2. the marks walk, a screenshot's marks and the look keep their deadlines -----------------------------

APP, PID = "Busy", 4343
SLOW = 0.25  # one Accessibility answer from a hung app (the real messaging timeout is 2.0 s)


def _hung_app(monkeypatch, *, has_window: bool = True):
    root, win = object(), object()
    asked: list = []

    def attr(el, name, default=None):
        asked.append(name)
        time.sleep(SLOW)
        if el is root and name in ("AXFocusedWindow", "AXMainWindow"):
            return win if has_window else default
        return default

    monkeypatch.setattr(ax, "_app_el", lambda pid: root)
    monkeypatch.setattr(ax, "_attr", attr)
    monkeypatch.setattr(ax, "_expose_web", lambda pid, r: False)
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": False)
    monkeypatch.setattr(ax, "app_names", lambda pid: {APP})
    monkeypatch.setattr(ax, "working_app", lambda: (APP, PID))
    monkeypatch.setattr(ax, "pid_for_app", lambda n: PID)
    monkeypatch.setattr(ax, "_point_size", lambda el: (attr(el, "AXPosition"), attr(el, "AXSize"), (0.0, 0.0, 800.0, 600.0))[-1])
    monkeypatch.setattr(ax, "scene", lambda pid: ax.Scene())
    return asked


@pytest.mark.parametrize("has_window", [False, True])
def test_the_marks_walk_ends_by_its_deadline_on_a_hung_app(monkeypatch, has_window):
    _hung_app(monkeypatch, has_window=has_window)
    t0 = time.monotonic()
    got = marks.collect_sync(None, pid=PID, name=APP, deadline_s=0.2)
    took = time.monotonic() - t0
    assert got.marks == [] and took < 0.2 + SLOW + 0.1  # at most the one answer it was waiting on


def test_an_app_that_cant_answer_is_left_at_its_first_question(monkeypatch):
    root, asked = object(), []

    def copy(el, name, out):
        asked.append(name)
        return ax._CANNOT_COMPLETE, None

    monkeypatch.setattr(ax, "AS", types.SimpleNamespace(AXUIElementCopyAttributeValue=copy))
    monkeypatch.setattr(ax, "_app_el", lambda pid: root)
    monkeypatch.setattr(ax, "_attr", lambda el, name, default=None: asked.append(name) or default)
    title, frame, els, blocked = marks.walk_window(PID, time.monotonic() + 5.0)
    assert asked == ["AXRole"] and els == [] and frame is None  # kAXErrorCannotComplete: nothing more asked


def test_a_screenshots_marks_never_hold_it_past_their_deadline(monkeypatch):
    def slow_walk(area, *, pid=0, name="", deadline_s=0.8, limit=marks.MAX_MARKS):
        time.sleep(1.0)  # a walk stuck on one answer
        return marks.Collected(marks=[marks.Mark("e1", "AXButton", "Late", (0.0, 0.0, 10.0, 10.0))])

    monkeypatch.setattr(marks, "collect_sync", slow_walk)

    async def screenshot_like():
        t0 = time.monotonic()
        walk = asyncio.create_task(marks.collect((0.0, 0.0, 800.0, 600.0), ToolContext(), pid=PID, deadline_s=0.2))
        await asyncio.sleep(0.05)  # the capture
        got = await walk
        return got, time.monotonic() - t0

    got, took = asyncio.run(screenshot_like())
    assert took < 0.2 + marks.GRACE_S + 0.1 and got.marks == [] and got.refused == "slow"


def test_a_look_ends_by_its_deadline_on_a_hung_app(monkeypatch):
    _hung_app(monkeypatch)
    monkeypatch.setattr(pointer, "protected_in", lambda area: False)

    async def go():
        t0 = time.monotonic()
        out = await observe.observe(APP, image=True, deadline_s=0.2, ctx=ToolContext())
        return out, time.monotonic() - t0

    out, took = asyncio.run(go())
    assert took < 0.2 + marks.GRACE_S + 0.1
    assert out.ok and not out.images and "in time" in out.text


# ---- 3. the waits after an input: bounded by the clock --------------------------------------------------------


def _scenes(monkeypatch, t, cost, *, focus_at=None, pixels_at=None):
    grey, dark = Image.new("L", (24, 24), 40), Image.new("L", (24, 24), 200)
    t.setdefault("scenes", 0)

    def scene(pid):
        t["now"] += cost
        t["scenes"] += 1
        focus = "Group “board”" if focus_at is not None and t["now"] >= focus_at else ""
        return ax.Scene(front="Chess", front_pid=CHESS, title="Chess", focus=focus, known=True)

    def sample(pts, r=60, frame=None, like=None):
        t["samples"] += 1
        return [dark if pixels_at is not None and t["now"] >= pixels_at else grey for _ in pts]

    monkeypatch.setattr(ax, "scene", scene)
    monkeypatch.setattr(screen, "sample_points", sample)
    return pointer.Look(ax.Scene(front="Chess", front_pid=CHESS, title="Chess", known=True), [grey])


@pytest.mark.parametrize("cost", [0.0, 0.01, 0.025, 0.3])
def test_a_key_or_a_press_waits_at_most_its_deadline_by_the_clock(monkeypatch, clock, cost):
    before = _scenes(monkeypatch, clock, cost)
    out = asyncio.run(pointer.scene_change(pointer.Look(before.scene, None), CHESS))
    assert out.changed is not True
    assert clock["now"] <= pointer.SCENE_WAIT_S + 1e-6  # (it was 3.4 s at 0.3 s a look)
    assert clock["scenes"] <= pointer.SCENE_WAIT_S / pointer.POLL_S + 1


def test_a_key_that_moves_the_focus_is_seen_at_the_next_look(monkeypatch, clock):
    before = _scenes(monkeypatch, clock, 0.01, focus_at=0.1)
    out = asyncio.run(pointer.scene_change(pointer.Look(before.scene, None), CHESS))
    assert out.changed is True and clock["now"] <= 0.1 + pointer.POLL_S + 0.011


@pytest.mark.parametrize("cost, bound", [(0.0, 1.0), (0.025, 1.05), (0.3, 1.6)])
def test_nothing_changed_is_said_no_later_than_before(monkeypatch, clock, cost, bound):
    """The old bound (LOOK_S + LOOK_AGAIN_S and two looks) holds however slow the app's answers."""
    before = _scenes(monkeypatch, clock, cost)
    out = asyncio.run(pointer.outcome(before, CHESS, [(612.0, 540.0)], ["(612, 540)"]))
    assert out.changed is False and clock["now"] <= bound + 1e-6 and clock["samples"] == 2


# ---- 4. a snap never moves a click off something pressable ----------------------------------------------------


class Node:
    def __init__(self, **attrs):
        self.attrs = attrs


def _tree(monkeypatch, frames: dict):
    monkeypatch.setattr(ax, "_attr", lambda el, name, default=None: el.attrs.get(name, default) if isinstance(el, Node) else default)
    monkeypatch.setattr(ax, "visible_frame", lambda el: frames.get(el))
    monkeypatch.setattr(
        ax, "AS", types.SimpleNamespace(AXUIElementCopyActionNames=lambda el, out: (0, list(el.attrs.get("actions", ()))))
    )


def _hit(el) -> ax.Hit:
    return ax.Hit(pid=CHESS, app="Browser", label="", role=el.attrs["AXRole"], regular=True, el=el)


def test_a_buttons_own_text_snaps_to_the_button(monkeypatch):
    win = Node(AXRole="AXWindow")
    button = Node(AXRole="AXButton", AXTitle="Start", AXParent=win, actions=["AXPress"])
    text = Node(AXRole="AXStaticText", AXValue="Start", AXParent=button, actions=["AXShowMenu"])
    _tree(monkeypatch, {button: (600.0, 530.0, 80.0, 24.0)})
    got = pointer.control_of(_hit(text))
    assert got is not None and got.role == "AXButton" and got.label == "Start" and got.frame == (600.0, 530.0, 80.0, 24.0)


def test_a_pressable_icon_inside_a_link_is_clicked_where_the_model_aimed(monkeypatch):
    web = Node(AXRole="AXWebArea")
    link = Node(AXRole="AXLink", AXTitle="Invoice from Bob", AXParent=web, actions=["AXPress"])
    star = Node(AXRole="AXImage", AXDescription="Star this message", AXParent=link, actions=["AXPress"])
    span = Node(AXRole="AXGroup", AXParent=link, actions=["AXPress"])  # a span with its own click handler
    inner = Node(AXRole="AXStaticText", AXValue="Archive", AXParent=span)
    frames = {link: (100.0, 200.0, 600.0, 40.0), star: (110.0, 210.0, 16.0, 16.0), span: (600.0, 205.0, 60.0, 30.0)}
    _tree(monkeypatch, frames)
    assert pointer.control_of(_hit(star)) is None
    assert pointer.control_of(_hit(inner)) is None  # its span has an action of its own
    monkeypatch.setattr(ax, "hit", lambda x, y: _hit(star))
    assert asyncio.run(pointer.snap((118.0, 218.0), _hit(star))) is None  # the star, not the link's middle


def test_a_field_or_a_web_page_is_never_snapped_out_of(monkeypatch):
    web = Node(AXRole="AXWebArea")
    loose = Node(AXRole="AXStaticText", AXValue="Terms", AXParent=web)
    button = Node(AXRole="AXButton", AXTitle="Search", actions=["AXPress"])
    field = Node(AXRole="AXSearchField", AXParent=button)  # (a field inside a control: a caret's place matters)
    _tree(monkeypatch, {button: (0.0, 0.0, 200.0, 30.0)})
    assert pointer.control_of(_hit(loose)) is None
    assert pointer.control_of(_hit(field)) is None


# ---- 5. a change seen in the scene is confirmed by a picture ------------------------------------------------


def test_a_change_seen_in_the_scene_is_confirmed_by_one_picture(monkeypatch, clock):
    before = _scenes(monkeypatch, clock, 0.0, focus_at=0.04, pixels_at=0.12)
    out = asyncio.run(pointer.outcome(before, CHESS, [(612.0, 540.0)], ["(612, 540)"]))
    assert out.changed is True and "focus moved" in out.said
    assert clock["samples"] == 1 and clock["now"] < pointer.LOOK_S  # at once, one picture


def test_a_brain_that_gets_the_picture_gets_it_once_the_change_settled(monkeypatch, clock):
    before = _scenes(monkeypatch, clock, 0.0, focus_at=0.04, pixels_at=0.12)
    out = asyncio.run(pointer.outcome(before, CHESS, [(612.0, 540.0)], ["(612, 540)"], settle=True))
    assert out.changed is True and "the screen changed around (612, 540)" in out.said and "focus moved" in out.said
    assert clock["now"] == pytest.approx(0.04 + pointer.LOOK_S) and clock["samples"] == 1


def test_the_after_look_is_checked_for_a_protected_window_after_the_capture(world, new_tools, captures, monkeypatch):
    world.front = ("Chess", CHESS)
    calls: list = []

    def protected_in(area):
        calls.append(len(captures))
        return len(captures) > 0  # a protected app's window came up while the picture was taken

    monkeypatch.setattr(pointer, "protected_in", protected_in)
    out = run("click", {"x": 612, "y": 540, "shot": ec._shot().id})
    assert out.ok and not out.images and not out.effect.looked and calls == [0, 1]


# ---- 6. the approved click is the click that was gated --------------------------------------------------------


def test_the_approved_click_lands_where_the_gate_judged(world, new_tools, monkeypatch):
    world.front = ("Chess", CHESS)
    ec._controls(world, monkeypatch, {"del": ("AXButton", "Delete All", (600.0, 530.0, 80.0, 24.0))})
    first = run("click", {"x": 602, "y": 551})
    assert first.text.startswith("NEEDS_CONFIRM") and first.staged
    with confirm.approved(first.staged):
        out = run("click", {"x": 602, "y": 551}, ToolContext(user_text="yes", origin="approval"))
    assert out.ok and ec._moved(world) == (640.0, 542.0)  # the control's middle, as the agent's run


# ---- 7. a screenshot in use outlives after-looks ----------------------------------------------------------------


def test_a_screenshot_the_model_keeps_using_is_never_pushed_out(world, new_tools, captures):
    world.front = ("Chess", CHESS)
    s = pointer.remember(0, 0, 1.0, 1512, 982)
    for i in range(pointer._KEEP_SHOTS + 4):
        out = run("click", {"x": 300 + i, "y": 500, "shot": s.id}, ToolContext(user_text="play"))
        assert "any more" not in out.text and out.effect.looked, (i, out.text)  # each one got its after-look
    assert pointer.get_shot(s.id) is s


# ---- 8. a brain that can't see gets no coordinates in any result ------------------------------------------------

_COORDS = r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)"


def _no_coords(text: str) -> None:
    import re

    assert not re.search(_COORDS, text), text


def test_a_blind_brains_results_name_squares_marks_and_ids_not_points(world, new_tools, monkeypatch):
    world.front = ("Chess", CHESS)
    pointer.set_cell_resolver(_board("neo"))
    pointer.set_id_resolver("t", lambda eid: (612.0, 540.0))
    e = ax.Element("e777", "AXButton", "Start", "", 600.0, 530.0, 80.0, 24.0, actions=["AXPress"], ref=object())
    monkeypatch.setattr(ax, "get", lambda eid: e if eid == "e777" else None)
    monkeypatch.setattr(ax, "visible_frame", lambda ref: (600.0, 530.0, 80.0, 24.0))
    monkeypatch.setattr(ax, "pid_of", lambda ref: CHESS)
    monkeypatch.setattr(ax, "app_of", lambda ref: "Chess")
    try:
        out = run("click", {"cell": "e4"}, _blind())
        assert out.ok and "square e4" in out.text
        _no_coords(out.text)
        out = run("click", {"id": "t3"}, _blind())
        assert out.ok and "mark t3" in out.text
        _no_coords(out.text)
        out = run("click", {"id": "e777"}, _blind())
        assert out.ok and "e777" in out.text
        _no_coords(out.text)
        out = run("drag", {"from": "e2", "to": "e4"}, _blind())
        assert out.ok and "square e2 → square e4" in out.text
        _no_coords(out.text)
        out = run("scroll", {"cell": "e4", "dy": -3}, _blind())
        assert "Scrolled down 3 at square e4" in out.text  # (the fake board's shade no longer changes by now)
        _no_coords(out.text)
        world.react = False  # nothing changes: the "look before repeating" words name no point either
        out = run("click", {"cell": "e4"}, _blind())
        _no_coords(out.text)
        out = run("click", {"x": 612, "y": 540}, _blind("click at 612, 540"))  # the user's own numbers
        _no_coords(out.text)
    finally:
        pointer.set_cell_resolver(None)
        pointer.set_id_resolver("t", None)


def test_a_blind_brains_refusals_name_no_point(world, new_tools, monkeypatch):
    world.front = ("Chess", CHESS)
    pointer.set_cell_resolver(lambda cell, shot, *, blind=False: (5000.0, 5000.0))  # off every screen
    try:
        out = run("click", {"cell": "e4"}, _blind())
        assert not out.ok and "off every screen" in out.text
        _no_coords(out.text)
        seeing = run("click", {"cell": "e4"}, ToolContext(user_text="play"))
        assert not seeing.ok and "(5000, 5000)" in seeing.text  # a brain that can see is told where
    finally:
        pointer.set_cell_resolver(None)


def test_a_spot_that_changed_on_the_way_is_refused_without_its_point_for_a_blind_brain(world, new_tools, monkeypatch):
    pointer.set_cell_resolver(_board("neo"))
    real = world.hit

    def hit(x, y):  # once Chess is in front, something else is at the square
        h = real(x, y)
        if world.front[1] == CHESS and h.pid == CHESS:
            h.label = "Resign"
        return h

    monkeypatch.setattr(ax, "hit", hit)
    try:
        out = run("click", {"cell": "e4"}, _blind())
        assert not out.ok and "changed what's at" in out.text and world.events() == []
        _no_coords(out.text)
    finally:
        pointer.set_cell_resolver(None)


# ---- 9. a helper's mark words reach the click gate --------------------------------------------------------------


def test_a_helpers_mark_words_are_judged_by_the_click_gate(world, new_tools):
    world.front = ("Chess", CHESS)
    pointer.set_id_resolver("t", lambda eid: (612.0, 540.0), labels=lambda eid: ["Delete account"])
    try:
        out = run("click", {"id": "t3"}, ToolContext(user_text="tidy up"))
        assert out.text.startswith("NEEDS_CONFIRM") and "Delete account" in out.text and world.events() == []
    finally:
        pointer.set_id_resolver("t", None)
    assert pointer._id_labels == {}


def test_the_walk_hands_its_deadline_to_ax_cores_walk_when_there_is_one(monkeypatch):
    made: list = []

    class Walk:  # (ax-core's _Walk: a deadline, and an end at an app that stopped answering)
        def __init__(self, deadline_s=None, cancel=None):
            made.append(deadline_s)

    monkeypatch.setattr(ax, "_Walk", Walk, raising=False)
    got = marks._walk_limits(time.monotonic() + 0.5)
    assert isinstance(got["w"], Walk) and 0.4 < made[0] <= 0.5
    monkeypatch.delattr(ax, "_Walk")
    assert marks._walk_limits(time.monotonic() + 0.5) == {}


def test_a_drag_from_a_helpers_mark_is_judged_by_its_words_too(world, new_tools):
    world.front = ("Chess", CHESS)
    pointer.set_id_resolver("t", lambda eid: (612.0, 540.0), labels=lambda eid: ["Delete account"])
    try:
        out = run("drag", {"id1": "t3", "x2": 300, "y2": 300}, ToolContext(user_text="tidy up", extras={"blind": False}))
        assert out.text.startswith("NEEDS_CONFIRM") and "Delete account" in out.text and world.events() == []
    finally:
        pointer.set_id_resolver("t", None)
