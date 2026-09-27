"""The tools group's wiring after the split: the eyes (capture, marks, OCR, board) and the commands
(resolver, menus) reach the handlers in tools_*.py through the frozen gates. Hermetic: every capture,
Accessibility read and event is faked."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("Quartz")

from neo2.core.agent.registry import ToolContext  # noqa: E402
from neo2.core.providers.base import ImagePart  # noqa: E402
from neo2.core.tools.computer import apps, ax, focus, gates, pointer, screen, tools_pointer  # noqa: E402


def _cap(kind="full", *, clickable=True, x0=0.0, y0=0.0, ppp=1.0, w=100, h=80):
    return SimpleNamespace(
        part=ImagePart(b"x", "image/jpeg"), x0=x0, y0=y0, ppp=ppp, w=w, h=h, kind=kind, clickable=clickable,
        covered=(), frame=None, window_id=0, pid=0,
    )  # fmt: skip


class _App:
    def __init__(self, pid, name):
        self._pid, self._name = pid, name

    def processIdentifier(self):  # noqa: N802 - AppKit's name
        return self._pid

    def localizedName(self):  # noqa: N802
        return self._name


@pytest.fixture
def shots(monkeypatch):
    took: list[tuple] = []
    monkeypatch.setattr(screen, "shot", lambda region=None, display=None, **k: took.append(("shot", region)) or _cap())
    monkeypatch.setattr(screen, "zoom", lambda x, y, w, h, **k: took.append(("zoom", x, y, w, h)) or _cap("zoom", ppp=0.5))
    monkeypatch.setattr(screen, "window_shot", lambda wid, *a, **k: took.append(("window", wid)) or _cap("window", x0=10, y0=20))
    monkeypatch.setattr(apps, "_running", lambda name: _App(4321, "Pages") if name.lower() == "pages" else None)
    monkeypatch.setattr(focus, "front_window_id", lambda pid: 77 if pid == 4321 else 0)
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": False)
    return took


def _run(a: dict):
    return asyncio.run(tools_pointer.screenshot(a, ToolContext(origin="agent")))


def test_a_screenshot_takes_the_working_apps_front_window(shots, monkeypatch):
    monkeypatch.setattr(apps, "target", lambda: "Pages")
    out = _run({})
    assert shots == [("window", 77)] and "Pages's front window" in out.text and out.images
    assert pointer.get_shot(out.text.split()[1]).x0 == 10  # clicks map through the window's own origin


def test_the_whole_display_when_asked_or_when_no_app_is_known(shots, monkeypatch):
    monkeypatch.setattr(apps, "target", lambda: None)
    _run({})
    monkeypatch.setattr(apps, "target", lambda: "Pages")
    _run({"display": 1})
    _run({"x": 0, "y": 0, "w": 50, "h": 50})
    assert [s[0] for s in shots] == ["shot", "shot", "shot"]


def test_never_a_protected_apps_or_neos_window(shots, monkeypatch):
    monkeypatch.setattr(apps, "target", lambda: "Pages")
    monkeypatch.setattr(gates, "_protected", lambda name, pid=0: name == "Pages")
    _run({})
    monkeypatch.setattr(gates, "_protected", lambda name, pid=0: False)
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": True)
    _run({"app": "Pages"})
    assert [s[0] for s in shots] == ["shot", "shot"]


def test_a_window_picture_that_cannot_be_mapped_is_for_looking_only(shots, monkeypatch):
    monkeypatch.setattr(screen, "window_shot", lambda wid, *a, **k: _cap("window", clickable=False))
    out = _run({"app": "Pages"})
    assert "for looking only" in out.text and 'shot="' not in out.text


def test_a_zoom_around_a_point(shots):
    out = _run({"zoom": True, "x": 300, "y": 200})
    assert shots == [("zoom", 300.0, 200.0, float(screen.ZOOM_W), float(screen.ZOOM_H))] and "zoom" in out.text
    assert not _run({"zoom": True, "x": 300}).ok  # a zoom needs its point


def test_a_click_on_a_part_of_a_window_picture_another_window_covered_is_refused():
    s = pointer.remember(100.0, 100.0, 1.0, 400, 300, covered=((150.0, 150.0, 50.0, 50.0),))
    with pytest.raises(pointer.Refused, match="under another window"):
        pointer.numeric({"x": 60, "y": 60, "shot": s.id}, ToolContext(), "click")
    assert pointer.numeric({"x": 10, "y": 10, "shot": s.id}, ToolContext(), "click").point == (110.0, 110.0)


def test_a_screenshot_lists_the_controls_in_it(shots, monkeypatch):
    from neo2.core.tools.computer import marks

    seen = []

    async def collect(area, ctx=None, *, pid=0, deadline_s=0.8):
        seen.append((area, pid))
        return marks.Collected(app="Pages", pid=pid)

    monkeypatch.setattr(marks, "collect", collect)
    monkeypatch.setattr(marks, "for_shot", lambda col, s, c: "\n(marks of " + col.app + ")")
    monkeypatch.setattr(apps, "target", lambda: "Pages")
    out = _run({})
    assert seen == [(None, 4321)] and out.text.endswith("(marks of Pages)")  # the window's own app, walked alongside
    _run({"x": 0, "y": 0, "w": 50, "h": 40})
    assert seen[-1] == ((0.0, 0.0, 50.0, 40.0), 0)


def test_a_blind_brains_screenshot_says_no_numbers(shots, monkeypatch):
    import re

    monkeypatch.setattr(apps, "target", lambda: None)
    for a in ({}, {"x": 10, "y": 20, "w": 300, "h": 200}, {"zoom": True, "x": 300, "y": 200}):
        out = asyncio.run(tools_pointer.screenshot(a, ToolContext(origin="agent", extras={"blind": True})))
        assert out.ok and not re.search(r"\d", out.text), out.text


# ---- eyes-ocr -------------------------------------------------------------------------------------------


def test_click_text_reads_the_screen_when_accessibility_names_nothing(monkeypatch):
    from neo2.core.agent.registry import ToolOutput
    from neo2.core.tools.computer import ocr, tools_label

    async def no_guard(want=""):
        return None

    monkeypatch.setattr(gates, "_guard_front", no_guard)
    monkeypatch.setattr(ax, "window_elements", lambda app=None: ("Game", (0.0, 0.0, 800.0, 600.0), []))
    asked = []

    async def fallback(a, c, *, app, window, els=None, pid=0):
        asked.append((a["label"], app, window))
        return ToolOutput("Clicked the words 'Hint'")

    monkeypatch.setattr(ocr, "click_text_fallback", fallback)
    out = asyncio.run(tools_label.click_text({"label": "Hint"}, ToolContext(origin="agent")))
    assert asked == [("Hint", "Game", (0.0, 0.0, 800.0, 600.0))] and out.text.startswith("Clicked the words")

    async def unread(a, c, **k):
        return None  # the screen couldn't be read: the tool's own answer

    monkeypatch.setattr(ocr, "click_text_fallback", unread)
    assert "is called 'Hint'" in asyncio.run(tools_label.click_text({"label": "Hint"}, ToolContext(origin="agent")))


def test_an_outcome_says_which_words_changed(monkeypatch):
    from neo2.core.tools.computer import ocr

    monkeypatch.setattr(ocr, "sample_change", lambda b, a: "“Start” → “Stop”")
    before = pointer.Look(ax.Scene(), ["b"])
    out = pointer.Outcome("Changed: the screen changed around (1, 2).", "", True)
    got = asyncio.run(pointer._words(out, before, pointer.Look(ax.Scene(), ["a"])))
    assert got.details == "“Start” → “Stop”"
    same = pointer.Outcome("Nothing visibly changed.", "", False)
    assert asyncio.run(pointer._words(same, before, pointer.Look(ax.Scene(), ["a"]))).details == ""


def test_text_marks_are_clicked_by_id_through_the_pointer_with_their_words_gated():
    import inspect

    from neo2.core.tools.computer import ocr

    assert "ocr.install()" in inspect.getsource(tools_pointer)  # (at import: text marks work from the start)
    assert ocr.install()
    assert pointer._id_resolvers["t"] is ocr.resolve_mark and pointer._id_labels["t"] is ocr.mark_labels


# ---- commands-resolver --------------------------------------------------------------------------------


def _session_world(monkeypatch, intent="agent_task"):
    from neo2.core import config, events
    from neo2.core.agent import fastpath
    from neo2.core.agent import session as sess_mod
    from neo2.core.agent.session import Session
    from neo2.core.events import EventBus
    from neo2.core.reflex.schema import Decision
    from neo2.core.tools import load_all

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(events, "_bus", EventBus())
    load_all()

    class Reflex:
        async def decide(self, text):
            return Decision(intent, 0.9, False, 0.0, False, 0.0, "fake")

    from neo2.core.agent.registry import Registry

    ran: list[tuple[str, dict, str]] = []
    real = Registry.invoke

    async def invoke(self, name, args, ctx):
        if name == "click_text":
            ran.append((name, dict(args), ctx.origin))
            from neo2.core.agent.registry import ToolOutput

            return ToolOutput("Clicked 'Start'")
        return await real(self, name, args, ctx)

    # (on the class: an instance attribute would outlive the test, shadowing later patches of the class)
    monkeypatch.setattr(Registry, "invoke", invoke)
    monkeypatch.setattr(fastpath, "plan", lambda text, *a, **k: None)  # (no regex fast path covers it)
    agent_runs: list[str] = []

    async def agent(self, text, d, images, job, **kw):
        agent_runs.append(text)
        return sess_mod.Reply("agent", "agent", d, job=job)

    monkeypatch.setattr(Session, "_agent", agent)
    return Session(reflex=Reflex()), ran, agent_runs


def test_the_users_words_pointing_at_a_control_take_the_by_utterance_lane(monkeypatch):
    s, ran, agent_runs = _session_world(monkeypatch)
    r = asyncio.run(s.handle("start wala button dabao", from_user=True))
    assert ran and ran[0][0] == "click_text" and ran[0][1]["utterance"] and ran[0][2] == "user"
    assert r.route == "quick" and agent_runs == []
    asyncio.run(s.handle("start wala button dabao"))  # Live's relay, not the user's own words: no lane
    assert len(ran) == 1 and agent_runs == ["start wala button dabao"]


def test_the_lane_is_shut_when_the_resolver_is_off(monkeypatch):
    s, ran, agent_runs = _session_world(monkeypatch)
    monkeypatch.setenv("NEO_RESOLVER", "off")
    from neo2.core import config

    monkeypatch.setattr(config, "_settings", None)
    asyncio.run(s.handle("start wala button dabao", from_user=True))
    assert ran == [] and agent_runs == ["start wala button dabao"]


# ---- commands-menus ----------------------------------------------------------------------------------


def test_menu_command_is_registered_before_click_text_as_an_input_tool():
    from neo2.core.agent.registry import catalog
    from neo2.core.tools import PRESETS, load_all
    from neo2.mcp.server import INPUT_TOOLS

    load_all()
    names = [t.name for t in catalog().all()]
    assert names.index("menu_command") < names.index("click_text")  # (its fast paths win "… in the File menu")
    t = catalog().get("menu_command")
    assert t.by_utterance and not t.read_only and "menu_command" in INPUT_TOOLS
    assert "menu_command" in PRESETS["mac"] and "menu_command" not in PRESETS["mac_read"]
    assert "look" in PRESETS["mac_read"] and catalog().get("look").read_only


def test_typing_into_a_named_field_goes_through_fill_field(monkeypatch):
    from neo2.core.agent.registry import ToolOutput
    from neo2.core.tools.computer import menus, tools_keys

    got = []

    async def fill(text, field, c, *, app="", replace=False):
        got.append((text, field, app, replace))
        return ToolOutput("Typed")

    monkeypatch.setattr(menus, "fill_field", fill)
    out = asyncio.run(tools_keys.type_text({"text": "cats", "field": "search"}, ToolContext(origin="user")))
    assert out.text == "Typed" and got == [("cats", "search", "", False)]


def test_the_users_words_naming_no_control_may_name_a_menu_command(monkeypatch):
    from neo2.core.agent.registry import ToolOutput
    from neo2.core.tools.computer import menus, ocr, tools_label

    async def no_guard(want=""):
        return None

    monkeypatch.setattr(gates, "_guard_front", no_guard)
    monkeypatch.setattr(ax, "window_elements", lambda app=None: ("Game", (0.0, 0.0, 800.0, 600.0), []))
    asked = []

    async def by_utterance(text, c):
        asked.append((text, c.origin))
        return ToolOutput("Chose Game › New Game")

    async def no_ocr(*a, **k):
        return None

    monkeypatch.setattr(menus, "by_utterance", by_utterance)
    monkeypatch.setattr(ocr, "click_text_fallback", no_ocr)
    a = {"label": "start a new game", "utterance": "start a new game"}
    out = asyncio.run(tools_label.click_text(a, ToolContext(origin="user")))
    assert out.text == "Chose Game › New Game" and asked == [("start a new game", "user")]
    out = asyncio.run(tools_label.click_text(a, ToolContext(origin="agent")))  # a model's label: no menu guess
    assert len(asked) == 1 and "is called" in out


# ---- eyes-board ----------------------------------------------------------------------------------------


class _Board:
    window = None

    def parse(self, name):
        return ("abcdefgh".index(name[0]), 8 - int(name[1]))

    def name(self, col, row):
        return f"{'abcdefgh'[col]}{8 - row}"


def test_squares_go_through_the_board_helper_and_its_checks(monkeypatch):
    from neo2.core.tools.computer import board

    b = _Board()
    monkeypatch.setattr(pointer, "_cell_resolver", tools_pointer._cell_point)
    monkeypatch.setattr(board, "points_for", lambda a, keys, **k: ([(100.0, 200.0)], b, [a["cell"]]))
    grabbed = []

    def picture(bb, frame=None, backend=""):
        grabbed.append((frame, backend))
        return ("pic" if frame is None else "cut"), "cli"

    monkeypatch.setattr(tools_pointer, "_board_picture", picture)
    seen = []
    monkeypatch.setattr(board, "before_move", lambda bb, pic, **k: seen.append(("before", pic)))
    monkeypatch.setattr(board, "after_move", lambda bb, pic, frm="", to="", **k: board.Move(f"On the board: {frm}→{to}.", True))
    aims = [pointer.Aim((1.0, 2.0), "cell", None, "square e2", cell="e2"), pointer.Aim((3.0, 4.0), "cell", None, "square e4", cell="e4")]
    state = asyncio.run(pointer._cells_before(aims))
    assert seen == [("before", "pic")] and state == (b, "e2", "e4", "cli")
    text, ok = asyncio.run(pointer._cells_after(state, "Dragged e2 → e4.", True))
    assert ok and text.endswith("On the board: e2→e4.")
    # the pointer's last look after the input (its frame) is handed on: the board is looked at in it,
    # taken the same way as the look before (no picture of its own)
    grabbed.clear()
    asyncio.run(pointer._cells_after(state, "Dragged e2 → e4.", True, frame="the last look"))
    assert grabbed == [("the last look", "cli")]

    def stale(bb, pic, **k):
        raise board.BoardError("Error: the board isn't where NEO last saw it — take a new screenshot first; nothing was done.")

    monkeypatch.setattr(board, "before_move", stale)
    with pytest.raises(pointer.Refused, match="isn't where NEO last saw it"):
        asyncio.run(pointer._cells_before(aims[:1]))  # (nothing is posted)
    monkeypatch.setattr(board, "after_move", lambda bb, pic, frm="", to="", **k: board.Move("Nothing changed on the board.", False))
    assert asyncio.run(pointer._cells_after((b, "", "e4", "cli"), "Clicked.", True)) == ("Clicked.\nNothing changed on the board.", False)


def test_a_board_that_cannot_be_looked_at_is_never_acted_on(monkeypatch):
    from neo2.core.tools.computer import board

    monkeypatch.setattr(pointer, "_cell_resolver", tools_pointer._cell_point)
    monkeypatch.setattr(board, "points_for", lambda a, keys, **k: ([(100.0, 200.0)], _Board(), [a["cell"]]))

    def no_picture(b, frame=None, backend=""):
        raise PermissionError("no Screen Recording")

    monkeypatch.setattr(tools_pointer, "_board_picture", no_picture)
    with pytest.raises(pointer.Refused, match="nothing was done"):
        asyncio.run(pointer._cells_before([pointer.Aim((1.0, 2.0), "cell", None, "square e2", cell="e2")]))


def test_the_screenshot_note_names_a_boards_squares(shots, monkeypatch):
    from neo2.core.tools.computer import marks

    async def collect(area, ctx=None, *, pid=0, deadline_s=0.8):
        return marks.Collected(app="Pages", pid=pid, window=(10.0, 20.0, 800.0, 600.0), frames=[(10.0, 60.0, 800.0, 560.0, "AXGroup")])

    looked = []
    monkeypatch.setattr(marks, "collect", collect)
    monkeypatch.setattr(tools_pointer, "_board_note", lambda cap, shot, col, blind, region=None: looked.append((col.frames, blind)) or "Board: squares a1–h8.")
    monkeypatch.setattr(apps, "target", lambda: "Pages")
    out = _run({})
    assert out.text.endswith("Board: squares a1–h8.") and looked == [([(10.0, 60.0, 800.0, 560.0, "AXGroup")], False)]
    out = asyncio.run(tools_pointer.screenshot({}, ToolContext(origin="agent", extras={"blind": True})))
    assert out.text.endswith("Board: squares a1–h8.") and looked[-1][1] is True


def test_map_board_is_a_read_only_tool_that_acts_on_nothing():
    from neo2.core.agent.registry import catalog
    from neo2.core.tools import PRESETS, load_all
    from neo2.mcp.server import INPUT_TOOLS

    load_all()
    t = catalog().get("map_board")
    assert t is not None and t.read_only and "map_board" not in INPUT_TOOLS
    assert "map_board" in PRESETS["mac"] and "map_board" in PRESETS["mac_read"]
    for name, key in (("click", "cell"), ("scroll", "cell"), ("drag", "from"), ("drag", "to")):
        assert key in catalog().get(name).parameters["properties"], (name, key)


def test_a_real_flat_board_in_the_picture_the_model_gets_is_named_without_coordinates():
    import io
    import re

    from eyes_board_synth import flat_screen

    from neo2.core.tools.computer import board, marks

    board.forget()
    img, truth = flat_screen()
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)  # (as the model gets it)
    cap = SimpleNamespace(part=ImagePart(buf.getvalue(), "image/jpeg"))
    shot = pointer.remember(0.0, 0.0, 1.0, img.width, img.height)
    col = marks.Collected(app="Game", pid=4242, window=(0.0, 0.0, 1512.0, 982.0), frames=[])
    try:
        for blind in (False, True):
            note = tools_pointer._board_note(cap, shot, col, blind)
            assert "8×8" in note and 'cell="<square>"' in note, note
            assert not re.search(r"\(\d+(?:\.\d+)?,\s*\d+", note)  # no points in any form
        (x, y) = tools_pointer._cell_point("e2", shot, blind=True)
        bx, by, bw, bh = truth["board"]
        assert bx <= x <= bx + bw and by <= y <= by + bh  # the square's point, computed by NEO
    finally:
        board.forget()


def test_the_board_after_a_move_is_cut_from_the_pointers_last_look(monkeypatch):
    """The look at the board after a move comes from the frame the pointer's last look took (the
    after-look's capture) when it shows the whole board, as the screen showed it, the same way as
    the look before — else NEO takes its own picture of the board."""
    from PIL import Image

    from neo2.core.tools.computer import board

    grabs = []

    def grab(region=None, display=None):
        grabs.append(region)
        return screen.Frame(Image.new("RGB", (int(region[2]), int(region[3]))), float(region[0]), float(region[1]), 1.0)

    monkeypatch.setattr(screen, "grab", grab)
    monkeypatch.setattr(board, "region", lambda b, margin=0.03: (100, 100, 200, 200))
    big = screen.Frame(Image.new("RGB", (800, 800)), 0.0, 0.0, 1.0)
    pic, how = tools_pointer._board_picture(_Board(), big, "cli")
    assert grabs == [] and how == "cli" and pic.rgb.shape[:2] == (200, 200) and (pic.x0, pic.y0) == (100.0, 100.0)
    tools_pointer._board_picture(_Board(), big, "sck")  # (taken another way: colours don't compare)
    tools_pointer._board_picture(_Board(), screen.Frame(Image.new("RGB", (150, 150)), 0.0, 0.0, 1.0), "cli")  # (too small)
    tools_pointer._board_picture(_Board(), screen.Frame(Image.new("RGB", (800, 800)), 0.0, 0.0, 1.0, window_id=7), "cli")
    assert grabs == [(100, 100, 200, 200)] * 3


def test_a_region_screenshot_is_its_own_evidence_for_a_board(monkeypatch):
    """A close picture of a board (a region screenshot) is judged by itself: a board that is a small
    share of a busy window — the self-test's tilted one — is looked for there, at an angle too. The
    window around it, full of controls, would say there is no large area drawn in one piece."""
    import io

    import numpy as np
    from eyes_board_synth import board_to_image, tilted_screen

    from neo2.core.tools.computer import board, marks

    board.forget()
    img, truth = tilted_screen(focal=600.0)  # (a board a small share of the window, as the self-test's)
    corners = board_to_image(truth["H"], [(0, 0), (8, 0), (8, 8), (0, 8)])
    x0, y0 = (int(v) - 20 for v in np.min(corners, axis=0))
    x1, y1 = (int(v) + 20 for v in np.max(corners, axis=0))
    region = (x0, y0, x1 - x0, y1 - y0)
    crop = img.crop((x0, y0, x1, y1))
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=85)
    cap = SimpleNamespace(part=ImagePart(buf.getvalue(), "image/jpeg"))
    shot = pointer.remember(float(x0), float(y0), 1.0, crop.width, crop.height)
    window = (0.0, 0.0, 1512.0, 982.0)
    busy = [(x, y, 40.0, 20.0, "AXButton") for x in range(0, 1512, 60) for y in range(0, 982, 40)
            if not (x0 - 40 <= x <= x1 and y0 - 20 <= y <= y1)]  # fmt: skip
    col = marks.Collected(app="Game", pid=4242, window=window, frames=busy)
    try:
        assert board.canvas_area(window, busy) is None  # (the window as a whole: no evidence)
        note = tools_pointer._board_note(cap, shot, col, False, region)
        assert "8×8" in note, note
    finally:
        board.forget()


def test_a_look_names_a_boards_squares_and_never_hints():
    """The look tool's picture gets the board's words too (observe.add_note, with the look's own
    walk as the evidence) — only a board NEO found: never the "give its corners" hint."""
    import io

    from eyes_board_synth import flat_screen
    from PIL import Image

    from neo2.core.tools.computer import board, marks, observe

    assert tools_pointer._look_note in observe._notes and pointer._hears(tools_pointer._look_note, "collected")
    board.forget()
    img, truth = flat_screen()
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    part = ImagePart(buf.getvalue(), "image/jpeg")
    shot = pointer.remember(0.0, 0.0, 1.0, img.width, img.height)
    col = marks.Collected(app="Game", pid=4242, window=(0.0, 0.0, 1512.0, 982.0), frames=[])
    ctx = ToolContext()
    try:
        note = tools_pointer._look_note(shot, part, ctx, collected=col)
        assert "8×8" in note and 'cell="<square>"' in note, note
        board.forget()
        blank = Image.new("RGB", (1512, 982), (230, 230, 230))
        buf = io.BytesIO()
        blank.save(buf, format="JPEG", quality=85)
        assert tools_pointer._look_note(shot, ImagePart(buf.getvalue(), "image/jpeg"), ctx, collected=col) == ""
        assert tools_pointer._look_note(shot, None, ctx, collected=col) == ""  # (no picture: a blind brain's look)
    finally:
        board.forget()


# ---- the baseline's gaps: a scroll under another window, a click_text with NEO's window in front ----------


_REAL_OWNER_AT = ax.window_owner_at  # (the suite's conftest makes it "no window there" in every test)


def _win(pid, name, x, y, w, h):
    return {"kCGWindowOwnerPID": pid, "kCGWindowOwnerName": name, "kCGWindowAlpha": 1.0,
            "kCGWindowBounds": {"X": x, "Y": y, "Width": w, "Height": h}}  # fmt: skip


def test_a_scroll_at_an_element_goes_where_its_app_shows(monkeypatch):
    """A scroll goes to the window under the pointer: aimed at an element (by id) whose middle
    another app's window covers, it goes to a part of the element that shows — or, when none does,
    nothing is done and the answer names the window in the way (never a place)."""
    import Quartz

    el = ax.Element("e8", "AXScrollArea", "", "", 350.0, 122.0, 870.0, 748.0)
    where = pointer.Aim((785.0, 496.0), "id", element=el, pid=501, owner="Safari")
    wins = [_win(505, "Notes", 700, 110, 700, 700), _win(501, "Safari", 120, 70, 1100, 800)]
    monkeypatch.setattr(Quartz, "CGWindowListCopyWindowInfo", lambda opts, n: list(wins))
    monkeypatch.setattr(ax, "window_owner_at", _REAL_OWNER_AT)  # (the window server: the list above)
    x, y = pointer._clear_point(where)
    assert 350 <= x < 700 and 122 <= y < 870  # Safari's own part of its page, not Notes
    wins.insert(0, _win(505, "Notes", 0, 0, 1512, 982))  # (Notes over all of it)
    with pytest.raises(pointer.Refused) as got:
        pointer._clear_point(where)
    assert "Notes covers Safari's ScrollArea e8" in str(got.value) and "(" not in str(got.value)
    wins[:] = [_win(501, "Safari", 120, 70, 1100, 800)]
    assert pointer._clear_point(where) == (785.0, 496.0)  # (nothing over it: its own middle)


def test_click_text_with_neos_own_window_in_front_uses_the_app_behind_it(monkeypatch):
    """The user just talked to NEO (its window in front) and NEO hasn't worked in an app lately:
    "click Start" means the app whose window is in front behind NEO's — not a question back."""
    from neo2.core.agent.registry import registry
    from neo2.core.tools import load_all
    from neo2.core.tools.computer import tools_label

    load_all()
    asked = []
    monkeypatch.setattr(apps, "_target", None)
    monkeypatch.setattr(tools_label, "_neo_in_front", lambda: True)
    monkeypatch.setattr(apps, "front_name", lambda: "NEO")
    monkeypatch.setattr(ax, "working_app", lambda: ("Chess", 502))

    def elements(app=None, **k):
        asked.append(app)
        return app or "NEO", None, []

    monkeypatch.setattr(ax, "window_elements", elements)
    out = asyncio.run(registry().invoke("click_text", {"label": "Start"}, ToolContext(origin="user")))
    assert asked == ["Chess"] and "Chess has no window in front" in out.text  # (the app behind it was looked at)
    monkeypatch.setattr(ax, "working_app", lambda: ("NEO", 900))  # (nothing behind NEO's own window)
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": pid == 900)
    out = asyncio.run(registry().invoke("click_text", {"label": "Start"}, ToolContext(origin="user")))
    assert out.text.startswith("NEEDS_USER: click where") and asked == ["Chess"]  # (as before: the user is asked)
