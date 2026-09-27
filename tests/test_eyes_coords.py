"""Exact pixels: every way a model may say where (screenshot pixels, Gemini's own 0–1000 points and
boxes, element ids, board squares) goes through one checked function — NEO does all the arithmetic,
a brain that saw nothing is refused every numeric form — a point inside a control is snapped to its
middle and gated again, and every pointer result says what it did (Effect) and, for a brain that
can see, shows the picture after it.

Hermetic: the fake Mac of test_v2_computer_controls (Quartz, Accessibility, app switching, the
displays and screencapture are fakes); nothing reaches the real Mac."""

from __future__ import annotations

import inspect
import io

import pytest
import test_v2_computer_controls as base
from PIL import Image
from test_v2_computer_controls import CHESS, run

from neo.agent import confirm
from neo.agent.registry import ToolContext, registry
from neo.providers.base import ImagePart
from neo.tools.computer import ax, marks, pointer, screen

# The fake Mac of the pointer tests (its fixtures, by the same names).
world = base.world
shots = base.shots
_REAL_PROTECTED_IN = pointer.protected_in  # (the suite fakes it: no protected window anywhere)


@pytest.fixture(autouse=True)
def _no_card_left():
    """A question a test staged (the gate asked before a click) never outlives it."""
    yield
    confirm.cancel("t")


@pytest.fixture(autouse=True)
def _no_picture(monkeypatch):
    """As the suite's _no_real_screen says: no picture can be taken (a test that wants one fakes
    the capture itself — ``shots``, ``captures``). The after-look never reaches the real screen, and
    never asks for Screen Recording."""
    monkeypatch.setattr(screen, "can_capture", lambda: False)
    monkeypatch.setattr(pointer, "protected_in", lambda area: False)  # (the window server: no protected window)


@pytest.fixture
def new_tools(monkeypatch):
    """The pointer tools as this workstream builds them (pointer.click_tool …), in the registry."""
    reg = registry()
    for name, fn in (("click", pointer.click_tool), ("drag", pointer.drag_tool), ("scroll", pointer.scroll_tool)):
        monkeypatch.setattr(reg.get(name), "handler", fn)


@pytest.fixture
def captures(monkeypatch):
    """screen.capture as a fake: every area asked for, a grey picture at 1 px per point."""
    asked: list = []

    def capture(region=None, display=None):
        asked.append(region)
        w, h = (region[2], region[3]) if region else (1512, 982)
        buf = io.BytesIO()
        Image.new("RGB", (int(w), int(h)), (90, 90, 90)).save(buf, format="PNG")
        return ImagePart(buf.getvalue(), "image/png"), 1.0

    monkeypatch.setattr(screen, "can_capture", lambda: True)
    monkeypatch.setattr(screen, "capture", capture)
    monkeypatch.setattr(screen, "shot", None, raising=False)  # (the capture pipeline's own shot: not here)
    return asked


def _ctx(text="play chess", **extras) -> ToolContext:
    return ToolContext(user_text=text, extras=extras)


def _moved(w) -> tuple[float, float]:
    return [e for e in w.log if e[0] == "post" and e[1] == "MouseMoved"][-1][2]


# ---- the same results as before, on the new handlers ------------------------------------------------

_BASE = [
    n
    for n, fn in inspect.getmembers(base, inspect.isfunction)
    if n.startswith("test_") and set(inspect.signature(fn).parameters) <= {"world", "monkeypatch", "shots"}
]


@pytest.mark.parametrize("name", _BASE)
def test_every_existing_pointer_check_holds_on_the_new_handlers(name, request, new_tools):
    """NEO's window refused, a sheet refused, the owner brought forward first, a drag onto the Trash
    asks, a blind brain's coordinates refused, hover-aware looks — all unchanged."""
    fn = getattr(base, name)
    fn(**{p: request.getfixturevalue(p) for p in inspect.signature(fn).parameters})


# ---- coordinate forms: NEO does the arithmetic ---------------------------------------------------------


def _shot(x0=0.0, y0=0.0, ppp=1.0, w=1512, h=982) -> pointer.Shot:
    return pointer.remember(x0, y0, ppp, w, h)


@pytest.mark.parametrize(
    "shot, nx, ny, want",
    [
        ((0, 0, 1.0, 1512, 982), 500, 500, (756.0, 491.0)),
        ((0, 0, 2.0, 756, 491), 500, 500, (756.0, 491.0)),  # a half-size image: the same place
        ((0, 0, 1.08, 1600, 1034), 250, 750, (432.0, 837.54)),
        ((40, 64, 0.5, 1440, 716), 0, 0, (40.0, 64.0)),  # a Retina region (native pixels)
        ((40, 64, 0.5, 1440, 716), 1000, 1000, (40 + 1439.5 * 0.5, 64 + 715.5 * 0.5)),  # the last pixel, never past it
    ],
)
def test_nx_ny_on_0_to_1000_map_through_the_shot(shot, nx, ny, want):
    s = _shot(*shot)
    got = pointer.numeric({"nx": nx, "ny": ny, "shot": s.id}, _ctx())
    assert got.form == "normalized" and got.shot is s
    assert got.point == pytest.approx(want, abs=0.01)


def test_a_gemini_box_is_acted_on_at_its_middle():
    s = _shot(0, 25, 1.0, 1512, 915)  # a window's region
    for box in ([88, 370, 150, 470], "88,370,150,470", "[88, 370, 150, 470]", {"ymin": 88, "xmin": 370, "ymax": 150, "xmax": 470}):
        got = pointer.numeric({"box_2d": box, "shot": s.id}, _ctx())
        assert got.form == "box" and got.point == pytest.approx((0.42 * 1512, 25 + 0.119 * 915), abs=0.01)
        assert "box_2d [88, 370, 150, 470]" in got.said and s.id in got.said


@pytest.mark.parametrize(
    "args, says",
    [
        ({"nx": 1200, "ny": 10}, "0–1000"),
        ({"nx": -1, "ny": 10}, "0–1000"),
        ({"box_2d": [10, 20, 1001, 40]}, "0–1000"),
        ({"box_2d": [300, 20, 100, 40]}, "y first"),  # [ymin, xmin, ymax, xmax]: the min past the max
        ({"box_2d": [1, 2, 3]}, "4 numbers"),
        ({"box_2d": ["a", 2, 3, 4]}, "must be a number"),
        ({"x": 2000, "y": 10}, "outside screenshot"),
    ],
)
def test_out_of_range_numbers_are_refused_with_a_hint(args, says):
    s = _shot()
    with pytest.raises(pointer.Refused) as r:
        pointer.numeric({**args, "shot": s.id}, _ctx())
    assert says in str(r.value)


def test_0_to_1000_forms_need_the_shot_they_are_on():
    for args in ({"nx": 10, "ny": 10}, {"box_2d": [1, 2, 3, 4]}):
        with pytest.raises(pointer.Refused) as r:
            pointer.numeric(args, _ctx())
        assert "shot id" in str(r.value)
    with pytest.raises(pointer.Refused) as r:
        pointer.numeric({"nx": 10, "ny": 10, "shot": "s999-0000"}, _ctx())
    assert "take a new one" in str(r.value)


def test_two_ways_at_once_are_refused_not_guessed():
    s = _shot()
    with pytest.raises(pointer.Refused) as r:
        pointer.numeric({"x": 5, "y": 5, "box_2d": [1, 2, 3, 4], "shot": s.id}, _ctx())
    assert "one way only" in str(r.value)


def test_a_point_off_every_screen_is_refused_in_every_form(monkeypatch):
    s = _shot(1600, 0, 1.0, 400, 300)  # a region NEO once saw on a display that is gone now
    for args in ({"nx": 500, "ny": 500}, {"box_2d": [0, 0, 10, 10]}, {"x": 5, "y": 5}):
        with pytest.raises(pointer.Refused) as r:
            pointer.numeric({**args, "shot": s.id}, _ctx())
        assert "off every screen" in str(r.value)


def test_the_note_gives_the_image_size_and_every_form():
    note = pointer.note(_shot(0, 0, 1.08, 1600, 1034), "all of display 1", [])
    assert "the image is 1600×1034 px" in note and 'shot="s' in note
    for form in ("x,y", "nx,ny", "box_2d [ymin, xmin, ymax, xmax]", "0–1000", "element id"):
        assert form in note


def test_clicks_in_the_new_forms_land_where_neo_computed(world, new_tools):
    s = _shot()
    world.front = ("Chess", CHESS)
    out = run("click", {"nx": 405, "ny": 550, "shot": s.id})
    assert out.ok and _moved(world) == pytest.approx((612.36, 540.1), abs=0.01)
    assert f"(405, 550) of 1000 in {s.id}" in out.text
    out = run("click", {"box_2d": [540, 400, 560, 410], "shot": s.id})
    assert out.ok and _moved(world) == pytest.approx((612.36, 540.1), abs=0.01)
    out = run("drag", {"box_2d1": [300, 200, 310, 210], "nx2": 300, "ny2": 700, "shot": s.id})
    downs = [e[2] for e in world.log if e[0] == "post" and e[1] in ("LeftMouseDown", "LeftMouseUp")]
    assert out.ok and downs[-2:] == [pytest.approx((309.96, 299.51), abs=0.01), pytest.approx((453.6, 687.4), abs=0.01)]
    out = run("scroll", {"nx": 500, "ny": 500, "shot": s.id, "dy": -3})
    assert out.ok and [e for e in world.log if e[0] == "post"][-1][1] == "Scroll"


def test_saying_nowhere_is_an_error_that_lists_the_forms(world, new_tools):
    out = run("click", {})
    assert not out.ok and "nx,ny" in out.text and "box_2d" in out.text and world.events() == []


# ---- a brain that saw nothing: no numbers in any form ---------------------------------------------------


@pytest.mark.parametrize(
    "name, args",
    [
        ("click", {"nx": 405, "ny": 550}),
        ("click", {"box_2d": [540, 400, 560, 410]}),
        ("drag", {"box_2d1": [300, 200, 310, 210], "box_2d2": [700, 200, 710, 210]}),
        ("drag", {"nx1": 1, "ny1": 2, "nx2": 3, "ny2": 4}),
        ("scroll", {"nx": 500, "ny": 500}),
        ("click", {"x": 612, "y": 540}),
    ],
)
def test_a_blind_brain_is_refused_every_numeric_form(world, new_tools, name, args):
    s = _shot()
    # even when the user's own words happen to hold the same numbers: 0–1000 of a picture is never theirs
    ctx = _ctx("405 550 540 400 560 410 300 200 310 210 700 1 2 3 4 500", blind=True)
    out = run(name, {**args, "shot": s.id}, ctx)
    assert not out.ok and "can't see the screen" in out.text and "any form" in out.text
    assert world.events() == []


def test_a_blind_brain_may_use_a_square_neo_found(world, new_tools, monkeypatch):
    world.front = ("Chess", CHESS)
    monkeypatch.setattr(pointer, "_cell_resolver", None)
    out = run("click", {"cell": "e4"}, _ctx(blind=True))
    assert not out.ok and "no board" in out.text and world.events() == []
    asked = []

    def board(cell, shot, *, blind=False):  # (a board NEO found itself: a blind brain may use its squares)
        asked.append((cell, shot, blind))
        if cell == "z9":
            raise ValueError("there's no square z9 on the 8×8 board")
        return (612.0, 540.0)

    pointer.set_cell_resolver(board)
    try:
        out = run("click", {"cell": "e4"}, _ctx(blind=True))
        assert out.ok and _moved(world) == (612.0, 540.0) and "square e4" in out.text and asked[-1] == ("e4", None, True)
        out = run("click", {"cell": "z9"}, _ctx(blind=True))
        assert not out.ok and "no square z9" in out.text
        out = run("drag", {"from": "e2", "to": "e4"}, _ctx(blind=True))
        assert out.ok and "square e2 → square e4" in out.text
    finally:
        pointer.set_cell_resolver(None)


def test_the_old_points_function_keeps_its_contract():
    pts, shot = pointer.points({"x": 612, "y": 540}, ("x", "y"), _ctx())
    assert pts == [(612.0, 540.0)] and shot is None
    with pytest.raises(pointer.Refused) as r:
        pointer.points({"x": 612}, ("x", "y"), _ctx())
    assert "are needed" in str(r.value)
    s = _shot()
    pts, shot = pointer.points({"nx1": 0, "ny1": 0, "x2": 5, "y2": 6, "shot": s.id}, ("x1", "y1", "x2", "y2"), _ctx())
    assert pts == [(0.0, 0.0), (5.0, 6.0)] and shot is s


# ---- snapping onto the control a point is inside ---------------------------------------------------------


class _El:
    def __init__(self, name):
        self.name = name


def _controls(world, monkeypatch, frames: dict):
    """Controls in the fake world: name → (role, label, frame). ax.hit carries the element under the
    point; control_of answers from these frames."""
    els = {n: _El(n) for n in frames}
    real_hit = world.hit

    def hit(x, y):
        h = real_hit(x, y)
        for n, (_role, _label, (fx, fy, fw, fh)) in frames.items():
            if fx <= x <= fx + fw and fy <= y <= fy + fh:
                h.el, h.label, h.role = els[n], frames[n][1], frames[n][0]
                break
        return h

    def control_of(h):
        for n, el in els.items():
            if h.el is el:
                role, label, frame = frames[n]
                return pointer.Control(role, label, frame, el) if role in pointer._SNAP_ROLES else None
        return None

    monkeypatch.setattr(ax, "hit", hit)
    monkeypatch.setattr(pointer, "control_of", control_of)
    return els


def test_a_point_inside_a_button_is_clicked_at_its_middle_and_named(world, new_tools, monkeypatch):
    world.front = ("Chess", CHESS)
    _controls(world, monkeypatch, {"start": ("AXButton", "Start", (600.0, 530.0, 80.0, 24.0))})
    s = _shot()
    out = run("click", {"x": 602, "y": 551, "shot": s.id})  # the button's bottom-left corner
    assert out.ok and _moved(world) == (640.0, 542.0)
    head, fenced = out.text.split('<untrusted source="screen">')
    assert "moved from (602, 551) to the middle of the control there" in head and "Start" not in head
    assert "snapped to Button “Start”" in fenced  # the screen's words stay fenced


def test_a_near_miss_beside_a_button_is_never_snapped_onto_it(world, new_tools, monkeypatch):
    world.front = ("Chess", CHESS)
    _controls(world, monkeypatch, {"start": ("AXButton", "Start", (600.0, 530.0, 80.0, 24.0))})
    out = run("click", {"x": 597, "y": 540})  # 3 pt left of it: the board beside it, not the button
    assert out.ok and _moved(world) == (597.0, 540.0) and "snapped" not in out.text


@pytest.mark.parametrize("why", ["grid", "surface", "blind", "user", "text", "middle-elsewhere", "id"])
def test_no_snap_where_it_could_press_something_else(world, new_tools, monkeypatch, why):
    world.front = ("Chess", CHESS)
    frames = {"start": ("AXButton", "Start", (600.0, 530.0, 80.0, 24.0)), "board": ("AXGroup", "", (0.0, 25.0, 590.0, 900.0))}
    if why == "text":
        frames["start"] = ("AXTextField", "Name", (600.0, 530.0, 80.0, 24.0))  # a caret's place matters
    els = _controls(world, monkeypatch, frames)
    ctx = _ctx("click at 602, 551") if why != "blind" else _ctx("click at 602, 551", blind=True)
    args = {"x": 602, "y": 551}
    if why == "grid":
        pointer.set_grid_check(lambda x, y: x < 700)
    if why == "surface":
        args = {"x": 300, "y": 300}
    if why == "user":
        ctx = ToolContext(user_text="click at 602, 551", origin="user")
    if why == "middle-elsewhere":
        real = ax.hit

        def hit(x, y):  # something else covers the middle (a badge, a wrapped link's gap)
            h = real(x, y)
            if (x, y) == (640.0, 542.0):
                h.el = _El("badge")
            return h

        monkeypatch.setattr(ax, "hit", hit)
    if why == "id":
        e = ax.Element("e901", "AXButton", "Start", "", 600.0, 530.0, 80.0, 24.0, actions=["AXPress"])
        monkeypatch.setattr(ax, "get", lambda eid: e if eid == "e901" else None)
        monkeypatch.setattr(ax, "visible_frame", lambda ref: (600.0, 530.0, 80.0, 24.0))
        monkeypatch.setattr(ax, "pid_of", lambda ref: CHESS)
        monkeypatch.setattr(ax, "app_of", lambda ref: "Chess")
        e.ref = els["start"]
        args = {"id": "e901"}
    try:
        out = run("click", args, ctx)
    finally:
        pointer.set_grid_check(None)
    assert out.ok and "snapped" not in out.text
    want = {"surface": (300.0, 300.0), "id": (640.0, 542.0)}.get(why, (602.0, 551.0))
    assert _moved(world) == want


def test_a_snapped_control_is_gated_again_before_anything_is_posted(world, new_tools, monkeypatch):
    world.front = ("Chess", CHESS)
    # The model aimed at a bare spot of a button whose own label says what it does.
    _controls(world, monkeypatch, {"del": ("AXButton", "Delete All", (600.0, 530.0, 80.0, 24.0))})
    out = run("click", {"x": 602, "y": 551})
    assert out.text.startswith("NEEDS_CONFIRM") and "Delete All" in out.text and world.events() == []


def test_a_control_whose_middle_is_under_neos_card_is_not_snapped_there(world, new_tools, monkeypatch):
    world.front = ("Chess", CHESS)
    # A control whose middle lies under NEO's card (it floats on top, invisible in screenshots).
    _controls(world, monkeypatch, {"b": ("AXButton", "Go", (1060.0, 400.0, 100.0, 40.0))})
    out = run("click", {"x": 1065, "y": 405})
    assert out.ok and "snapped" not in out.text and _moved(world) == (1065.0, 405.0)
    assert not [e for e in world.events() if e[2] and e[2][0] >= 1100]  # nothing near NEO's window


def test_the_snapped_spot_is_probed_again_before_the_gate(world, new_tools, monkeypatch):
    world.front = ("Chess", CHESS)
    _controls(world, monkeypatch, {"start": ("AXButton", "Start", (600.0, 530.0, 80.0, 24.0))})
    probed = []
    real = pointer.probe

    async def probe(pts):
        probed.append(list(pts))
        if pts == [(640.0, 542.0)]:
            raise pointer.Refused("Error: that spot is NEO's own window or an app NEO never touches.", protected=True)
        return await real(pts)

    monkeypatch.setattr(pointer, "probe", probe)
    out = run("click", {"x": 602, "y": 551})
    assert probed == [[(602.0, 551.0)], [(640.0, 542.0)]]
    assert not out.ok and "NEO's own window" in out.text and world.events() == []


# ---- what an input did: Effect, and the picture after it ---------------------------------------------------


def test_every_pointer_result_carries_its_effect(world, new_tools):
    world.front = ("Chess", CHESS)
    out = run("click", {"x": 612, "y": 540})
    assert out.effect.changed is True and out.effect.looked is False  # (no picture: no Screen Recording faked)
    world.react = False
    out = run("click", {"x": 612, "y": 540})
    assert out.effect.changed is False and not out.ok
    world.react = True
    world.sheet = "a sheet “Start a New Game?” (buttons: Start, Cancel)"
    out = run("drag", {"x1": 300, "y1": 300, "x2": 500, "y2": 300})
    assert out.effect.changed is True and out.effect.sheet.startswith("a sheet")
    out = run("scroll", {"x": 300, "y": 300, "dy": -3})
    assert out.effect.changed is True
    # the user's own command: no look before, so nothing is claimed either way
    out = run("click", {"x": 612, "y": 540}, ToolContext(user_text="click at 612 540", origin="user"))
    assert out.effect.changed is None


def test_a_sheet_that_opens_is_a_scene_change(world, new_tools, monkeypatch):
    world.front = ("Chess", CHESS)
    base_posted = world.posted

    def posted(ev):
        base_posted(ev)
        if ev["kind"] == "LeftMouseUp":
            world.sheet = "a sheet “Start a New Game?”"

    monkeypatch.setattr(pointer.screen, "sample_points", lambda pts, r=60: None)
    import neo.tools.computer.input as inp

    monkeypatch.setattr(inp, "_post", posted)
    out = run("click", {"x": 612, "y": 540})
    assert out.effect.changed is True and out.effect.scene_changed is True and "Start a New Game" in out.effect.sheet


def test_a_seeing_brain_gets_the_picture_after_a_pixel_click(world, new_tools, captures):
    world.front = ("Chess", CHESS)
    s = _shot(0, 25, 1.0, 1512, 915)  # the model's window shot: the after-look is that area again
    ctx = _ctx()
    out = run("click", {"x": 612, "y": 515, "shot": s.id}, ctx)
    assert out.ok and len(out.images) == 1 and out.effect.looked is True
    assert captures[-1] == (0, 25, 1512, 915)
    sid = out.text.split("After-look ")[1].split(" ")[0]
    new = pointer.get_shot(sid)
    assert new is not None and new.id != s.id and (new.x0, new.y0, new.w, new.h) == (0, 25, 1512, 915)
    assert "screen" in ctx.taint  # a picture of the screen went into the result
    if hasattr(out.images[0], "detail"):
        assert out.images[0].detail == "medium"
    # The next click may use the after-look's own shot.
    assert run("click", {"nx": 500, "ny": 500, "shot": sid}, _ctx()).ok


def test_the_after_look_of_a_full_screen_shot_is_a_box_around_the_point(world, new_tools, captures):
    world.front = ("Chess", CHESS)
    s = _shot()  # all of display 1: the window isn't known here, so a box around the point
    out = run("click", {"x": 20, "y": 500, "shot": s.id})
    assert out.effect.looked and captures[-1] == (0, 340, 260, 320)  # cut to the screen's left edge


def test_no_picture_for_a_blind_brain_the_users_own_click_or_an_ax_tree_id(world, new_tools, captures, monkeypatch):
    world.front = ("Chess", CHESS)
    out = run("click", {"x": 612, "y": 540}, _ctx("click at 612, 540", blind=True))
    assert out.ok and not out.images and not out.effect.looked
    out = run("click", {"x": 612, "y": 540}, ToolContext(user_text="click at 612, 540", origin="user"))
    assert out.ok and not out.images
    e = ax.Element("e777", "AXButton", "Start", "", 600.0, 530.0, 80.0, 24.0, actions=["AXPress"], ref=object())
    monkeypatch.setattr(ax, "get", lambda eid: e if eid == "e777" else None)
    monkeypatch.setattr(ax, "visible_frame", lambda ref: (600.0, 530.0, 80.0, 24.0))
    monkeypatch.setattr(ax, "pid_of", lambda ref: CHESS)
    monkeypatch.setattr(ax, "app_of", lambda ref: "Chess")
    out = run("click", {"id": "e777"}, _ctx())
    assert out.ok and not out.images  # an id from ax_tree: the text outcome is enough
    marks._remember(["e777"])  # …but an id from a picture's marks gets the picture after it
    out = run("click", {"id": "e777"}, _ctx())
    assert out.ok and len(out.images) == 1 and out.effect.looked
    assert captures == [captures[0]]  # one capture: only the marked click


def test_no_picture_when_the_capture_fails(world, new_tools, monkeypatch):
    world.front = ("Chess", CHESS)

    def refused(*a, **k):
        raise RuntimeError("screencapture failed")

    monkeypatch.setattr(screen, "can_capture", lambda: True)
    monkeypatch.setattr(screen, "capture", refused)
    monkeypatch.setattr(screen, "shot", None, raising=False)
    out = run("click", {"x": 612, "y": 540, "shot": _shot().id})
    assert out.ok and not out.images and out.effect.looked is False


def test_a_refused_click_posts_nothing_and_takes_no_picture(world, new_tools, captures):
    out = run("click", {"x": 1300, "y": 700, "shot": _shot().id})  # NEO's own card
    assert not out.ok and world.events() == [] and captures == []


def test_the_look_area_holds_every_point_and_stays_on_one_display(monkeypatch):
    two = [screen.Display(1, 0.0, 0.0, 1512.0, 982.0), screen.Display(2, 1512.0, 0.0, 1920.0, 1080.0)]
    monkeypatch.setattr(screen, "displays", lambda: two)
    assert pointer.look_area([(1500.0, 10.0)], None, None) == (1260, 0, 252, 170)  # cut at display 1's edges
    assert pointer.look_area([(100.0, 100.0), (900.0, 700.0)], None, (50.0, 50.0, 400.0, 300.0)) == (50, 50, 874, 674)
    region = pointer.remember(1600, 100, 0.5, 400, 200)
    assert pointer.look_area([(1650.0, 150.0)], region, (0.0, 0.0, 800.0, 600.0)) == (1600, 100, 200, 100)
    assert pointer.look_area([(5000.0, 10.0)], None, None) is None


def test_a_confirmation_is_still_a_confirmation_on_the_new_handlers(world, new_tools, monkeypatch):
    world.front = ("Chess", CHESS)
    world.regions.insert(0, (600, 530, 80, 24, CHESS, "Send", "AXButton", ""))
    out = run("click", {"x": 610, "y": 540})
    assert out.halt and out.text.startswith("NEEDS_CONFIRM") and world.events() == []


# ---- the outcome, without fixed sleeps: a scene change ends the wait at once ---------------------------------


@pytest.fixture
def clock(monkeypatch):
    """A fake clock: pointer's sleeps advance it (no real waiting), its monotonic reads it."""
    import asyncio
    import types

    t = {"now": 0.0, "samples": 0}

    async def sleep(s):
        t["now"] += s

    monkeypatch.setattr(pointer, "asyncio", types.SimpleNamespace(sleep=sleep, to_thread=asyncio.to_thread))
    monkeypatch.setattr(pointer, "time", types.SimpleNamespace(monotonic=lambda: t["now"]))
    monkeypatch.setattr(pointer, "LOOK_S", 0.15)
    monkeypatch.setattr(pointer, "LOOK_AGAIN_S", 0.85)
    return t


def _outcome_world(monkeypatch, t, *, sheet_at=None, pixels_at=None):
    grey = Image.new("L", (24, 24), 40)
    dark = Image.new("L", (24, 24), 200)

    def scene(pid):
        sheet = "a sheet “Start a New Game?”" if sheet_at is not None and t["now"] >= sheet_at else ""
        return ax.Scene(front="Chess", front_pid=CHESS, title="Chess", sheet=sheet, known=True)

    def sample(pts, r=60):
        t["samples"] += 1
        return [dark if pixels_at is not None and t["now"] >= pixels_at else grey for _ in pts]

    monkeypatch.setattr(ax, "scene", scene)
    monkeypatch.setattr(screen, "sample_points", sample)
    return pointer.Look(scene(CHESS), [grey])


def _run_outcome(before):
    import asyncio

    return asyncio.run(pointer.outcome(before, CHESS, [(612.0, 540.0)], ["(612, 540)"]))


def test_a_sheet_that_opens_at_60ms_ends_the_wait_before_the_first_picture(monkeypatch, clock):
    before = _outcome_world(monkeypatch, clock, sheet_at=0.06)
    out = _run_outcome(before)
    assert out.changed is True and out.scene_changed and "sheet" in out.said
    assert clock["now"] <= 0.081 and clock["samples"] == 1  # seen at the 2nd look at the scene, confirmed by one picture


def test_a_board_that_redraws_at_once_is_seen_at_the_first_picture(monkeypatch, clock):
    before = _outcome_world(monkeypatch, clock, pixels_at=0.0)
    out = _run_outcome(before)
    assert out.changed is True and not out.scene_changed and clock["now"] == pytest.approx(0.15) and clock["samples"] == 1


def test_nothing_changed_is_said_by_one_second_with_two_pictures(monkeypatch, clock):
    before = _outcome_world(monkeypatch, clock)
    out = _run_outcome(before)
    assert out.changed is False and clock["now"] == pytest.approx(1.0) and clock["samples"] == 2


def test_a_late_sheet_is_caught_between_the_two_pictures(monkeypatch, clock):
    before = _outcome_world(monkeypatch, clock, sheet_at=0.4)
    out = _run_outcome(before)
    assert out.changed is True and clock["now"] == pytest.approx(0.43) and clock["samples"] == 2  # (was 1.0 s; the 2nd picture confirms it)


def test_a_slow_app_cannot_stretch_the_wait(monkeypatch, clock):
    before = _outcome_world(monkeypatch, clock)

    def scene(pid):  # every answer takes 0.3 s of the clock (a busy app)
        clock["now"] += 0.3
        return before.scene

    monkeypatch.setattr(ax, "scene", scene)
    _run_outcome(before)
    # the clock bounds the watching, not only the sleeps: no later than before the scene was watched
    # (LOOK_S + LOOK_AGAIN_S and two looks: 1.6 s)
    assert clock["now"] <= 1.0 + 2 * 0.3 + 1e-6


def test_no_picture_is_taken_over_a_protected_apps_window(world, new_tools, captures, monkeypatch):
    world.front = ("Chess", CHESS)
    seen = []

    def protected_in(area):
        seen.append(area)
        return True

    monkeypatch.setattr(pointer, "protected_in", protected_in)
    out = run("click", {"x": 612, "y": 540, "shot": _shot().id})
    assert out.ok and not out.images and not out.effect.looked and seen and captures == []


def test_the_window_server_decides_what_is_protected_and_fails_closed(monkeypatch):
    import Quartz as Q

    from neo.config import settings

    wins = [
        {"kCGWindowBounds": {"X": 0, "Y": 0, "Width": 300, "Height": 300}, "kCGWindowOwnerPID": 900, "kCGWindowOwnerName": "NEO"},
        {"kCGWindowBounds": {"X": 500, "Y": 500, "Width": 100, "Height": 100}, "kCGWindowOwnerPID": 66, "kCGWindowOwnerName": "Vault"},
        {"kCGWindowBounds": {"X": 0, "Y": 0, "Width": 1512, "Height": 982}, "kCGWindowOwnerPID": CHESS, "kCGWindowOwnerName": "Chess"},
    ]
    monkeypatch.setattr(Q, "CGWindowListCopyWindowInfo", lambda opts, wid: wins)
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": pid == 900)
    monkeypatch.setattr(ax, "app_names", lambda pid: {"Vault", "com.example.vault"} if pid == 66 else {"Chess"})
    monkeypatch.setattr(settings(), "protected_apps", ["com.example.vault"])
    assert _REAL_PROTECTED_IN((0.0, 0.0, 400.0, 400.0)) is False  # NEO's overlay never shows in a picture
    assert _REAL_PROTECTED_IN((450.0, 450.0, 100.0, 100.0)) is True
    monkeypatch.setattr(Q, "CGWindowListCopyWindowInfo", lambda opts, wid: None)
    assert _REAL_PROTECTED_IN((0.0, 0.0, 10.0, 10.0)) is True  # the window server can't say: no picture


def test_the_tool_words_name_every_form_and_no_app():
    import re

    for text in (pointer.CLICK_DESCRIPTION, pointer.DRAG_DESCRIPTION, pointer.SCROLL_DESCRIPTION, pointer.SCREENSHOT_DESCRIPTION):
        assert "0–1000" in text and "shot id" in text
        assert not re.search(r"\b(Chess|Safari|Chrome|Finder|Mail|Notes)\b", text)
    assert {"nx", "ny", "box_2d"} <= set(pointer.FORM_ARGS)
    assert {"nx1", "ny1", "nx2", "ny2", "box_2d1", "box_2d2"} <= set(pointer.DRAG_FORM_ARGS)
    # the whole schemas keep every argument the tools took before, and add the forms
    reg = registry()
    for name, params in (("click", pointer.CLICK_PARAMETERS), ("drag", pointer.DRAG_PARAMETERS), ("scroll", pointer.SCROLL_PARAMETERS)):
        assert set(reg.get(name).parameters["properties"]) < set(params["properties"])


def test_a_helpers_own_marks_are_clicked_by_id_even_by_a_blind_brain(world, new_tools):
    world.front = ("Chess", CHESS)

    def text_marks(eid):
        if eid != "t3":
            raise KeyError(f"there's no text mark {eid} any more — look again")
        return (612.0, 540.0)

    pointer.set_id_resolver("t", text_marks)
    try:
        out = run("click", {"id": "t3"}, _ctx(blind=True))
        assert out.ok and _moved(world) == (612.0, 540.0) and "mark t3" in out.text
        out = run("click", {"id": "t9"}, _ctx(blind=True))
        assert not out.ok and "no text mark t9" in out.text
        assert not run("click", {"id": "e5"}).ok  # Accessibility's ids still go to ax (unknown here)
        with pytest.raises(ValueError):
            pointer.set_id_resolver("e", text_marks)
    finally:
        pointer.set_id_resolver("t", None)


def test_pictures_come_from_the_capture_pipelines_own_shot_when_it_has_one(monkeypatch):
    import types

    asked = []

    def shot(region=None, display=None, *, kind=None, detail=None, fmt=None):
        asked.append((region, kind))
        return types.SimpleNamespace(part=ImagePart(b"jpeg", "image/jpeg"), x0=float(region[0]), y0=float(region[1]), ppp=0.5, w=region[2] * 2, h=region[3] * 2)

    monkeypatch.setattr(screen, "shot", shot, raising=False)
    img, x0, y0, ppp, w, h = pointer.capture_area((10, 20, 300, 200), "after")
    assert asked == [((10, 20, 300, 200), "after")] and (x0, y0, ppp, w, h) == (10.0, 20.0, 0.5, 600, 400)
    assert img.mime == "image/jpeg"
