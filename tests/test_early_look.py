"""The early first look (decision 41) and the resolver prewarm (neo2.core.agent.prefetch).

Hermetic: a made-up Mac — which app is in front, what NEO last worked in, what the installed apps
are called, what the front window holds and what a picture of it looks like — all faked. Nothing
is read from the real screen, no picture is taken, no model is loaded, nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
import re
import threading
import time
import types

import pytest
from PIL import Image

import neo2.core.reflex
from neo2.core.agent import grants, prefetch
from neo2.core.agent.registry import ToolContext, registry
from neo2.core.decide import inventory as inv
from neo2.core.providers.base import ImagePart, Message, ToolCall, ToolResult, Turn
from neo2.core.reflex import zeroshot
from neo2.core.reflex.schema import Decision
from neo2.core.tools.computer import apps, ax, focus, pointer, screen

FIX_WORDS = __import__("pathlib").Path(__file__).parent / "fixtures" / "decide"
# Any position or size, however spelled: what a brain that can't see must never be handed.
COORDS = re.compile(r"@\(|\(\s*\d+\s*,\s*\d+|\b[xy]\s*[=:]\s*\d|\d+\s*[x×]\s*\d+\)")


def _decision(*, screen_p: float = 0.0, tool: str = "", intent: str = "agent_task") -> Decision:
    return Decision(intent, 0.95, False, 0.0, screen_p >= 0.5, screen_p, source="laya+head", tool=tool, tool_p=0.6 if tool else 0.0)


def _env(monkeypatch, key: str, value: str | None) -> None:
    """Set (or clear) a NEO_ setting for this test, whether neo2.core.config has the key yet or not."""
    from neo2.core import config

    if value is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(config, "_settings", None)


def _png(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 200, 200)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def world(monkeypatch):
    """A Mac with four made-up apps. ``world.front``: the app in front; its window holds three
    elements; a picture of it is 800×600 px for a 400×300 pt window at (100, 50)."""
    from neo2.core import config
    from neo2.core.tools import load_all

    load_all()
    monkeypatch.setenv("NEO_PREFETCH_LOOK", "1")  # (whatever a suite-wide default says)
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(inv, "_WORDS_FILE", str(FIX_WORDS / "words.txt"))
    monkeypatch.setattr(inv, "_CONNECTIVES_FILE", str(FIX_WORDS / "connectives.txt"))
    monkeypatch.setattr(inv, "_words_known", {})
    monkeypatch.setattr(inv, "_connectives", None)
    A = inv.App
    installed = [
        A("test.quillpad", "Quillpad", "/A/Quillpad.app"),
        A("test.wayfarer", "Wayfarer", "/A/Wayfarer.app"),
        A("test.vaultkeeper", "Vaultkeeper", "/A/Vaultkeeper.app"),
        A("test.tablero", "Tablero", "/A/Tablero.app"),
    ]
    pids = {"Quillpad": 101, "Wayfarer": 102, "Vaultkeeper": 103, "Tablero": 104, "NEO": 99}
    w = types.SimpleNamespace(
        front=("Quillpad", 101), installed=installed, scanned=True, walks=0, shots=0, walk_s=0.0,
        switch_to=None, capture=_png(800, 600), trees={}, above=[], region_shots=[], web=False, deadlines=[],
    )

    def snapshot(app=None, *, max_lines=250, deadline_s=None):
        w.walks += 1
        w.deadlines.append(deadline_s)
        pid = pids.get(app) if app else w.front[1]
        if w.web and pid not in ax._web_held:
            # a Chromium or Electron app: its web tree is switched on by this walk and held (ax._expose_web),
            # and needs a moment to fill in first
            ax._web_held[pid] = (None, [], None)
            time.sleep(ax._WEB_SETTLE_S)
        if w.walk_s:
            time.sleep(w.walk_s)
        els = [
            ax.Element("e1", "AXButton", "New Page", "", 120, 60, 80, 22),
            ax.Element("e2", "AXTextField", "Title", "Ideas for (3, 4)", 120, 90, 300, 22, focused=True),
            ax.Element("e3", "AXStaticText", "", "x=5 is the answer", 120, 130, 300, 18),
        ]
        for e in els:
            ax._issued[e.id] = e
        if w.switch_to:
            w.front = w.switch_to  # the user brought another app forward meanwhile
        return "\n".join([f'App: {app} — window "Draft"'] + ["  " + e.line() for e in els] + ["  … 12 more elements (use ax_find to search)"])

    monkeypatch.setattr(ax, "_issued", {})
    monkeypatch.setattr(ax, "working_app", lambda: w.front)
    monkeypatch.setattr(ax, "app_names", lambda pid: {n for n, p in pids.items() if p == pid} | {f"test.{n.lower()}" for n, p in pids.items() if p == pid})
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": pid == 99 or name.lower() == "neo")
    monkeypatch.setattr(ax, "pid_for_app", lambda name: pids.get(name))
    monkeypatch.setattr(ax, "snapshot", snapshot)
    monkeypatch.setattr(inv, "snapshot", lambda: list(w.installed) if w.scanned else None)
    by_pid = {pids[a.name]: a for a in installed}
    monkeypatch.setattr(inv, "app_for_pid", lambda pid: by_pid.get(pid) if w.scanned else None)

    def no_scan(refresh=False):
        raise AssertionError("the early look must never start a scan of the installed apps")

    monkeypatch.setattr(inv, "inventory", no_scan)
    monkeypatch.setattr(screen, "can_capture", lambda: True)
    monkeypatch.setattr(focus, "front_window_id", lambda pid: 7000 + pid)
    monkeypatch.setattr(prefetch, "_window_bounds", lambda wid: (100.0, 50.0, 400.0, 300.0))

    def capture_window(wid, region=None):
        w.shots += 1
        return ImagePart(w.capture, "image/png")

    monkeypatch.setattr(screen, "capture_window", capture_window)

    def capture(region=None, display=None):  # a screenshot of a part of the screen: 2 px per point
        w.region_shots.append(region)
        return ImagePart(w.capture, "image/png"), 0.5

    monkeypatch.setattr(screen, "capture", capture)
    # the windows in front of the pictured one, as the window server lists them (none by default)
    monkeypatch.setattr(prefetch, "_windows_above", lambda wid: [dict(x) for x in w.above], raising=False)
    monkeypatch.setattr(ax, "_web_held", {})
    monkeypatch.setattr(prefetch, "_last_warm", float("-inf"))
    monkeypatch.setattr(prefetch, "_slow_until", {})
    return w


def _window(pid: int, owner: str, x: float, y: float, w: float, h: float, *, alpha: float = 1.0, number: int = 9) -> dict:
    """One window as CGWindowListCopyWindowInfo describes it."""
    return {
        "kCGWindowNumber": number, "kCGWindowOwnerPID": pid, "kCGWindowOwnerName": owner, "kCGWindowAlpha": alpha,
        "kCGWindowLayer": 3, "kCGWindowBounds": {"X": x, "Y": y, "Width": w, "Height": h},
    }  # fmt: skip


def _look(text, d, *, game=None, sees=True, images=None):
    """The first look as NEO.app takes it (prefetch.start, then collect), from a test with no loop."""
    assert prefetch.wanted(d, game=game, images=images) is not None

    async def go():
        p = prefetch.start(text, d, game=game, images=images, sees=sees)
        assert p is not None
        return await prefetch.collect(p, ["ax_tree", "click"])

    return asyncio.run(go())


# ---- when -----------------------------------------------------------------------------------------


def test_the_requests_own_evidence_decides_whether_a_look_is_worth_taking(world):
    assert prefetch.wanted(_decision(screen_p=0.2)) is None  # nothing says it will look
    assert prefetch.wanted(_decision(screen_p=0.9)) == prefetch.Want(True, "needs_screen")
    got = prefetch.wanted(_decision(screen_p=0.1, tool="click_text"))
    assert got == prefetch.Want(False, "screen_tool")  # the elements, no picture
    assert prefetch.wanted(_decision(tool="weather")) is None  # a tool that doesn't work the screen
    game = world.installed[3]
    assert prefetch.wanted(_decision(), game=game).image
    # the one-tool lane's tool finds its target itself; a picture the caller brought is enough
    assert prefetch.wanted(_decision(screen_p=0.9), only=["click_text"]) is None
    assert prefetch.wanted(_decision(screen_p=0.9), images=[ImagePart(b"x")]).image is False


def test_a_brain_that_cannot_be_asked_counts_as_blind():
    class Broken:
        @property
        def supports_vision(self):
            raise RuntimeError("no keys")

    assert prefetch.sees(types.SimpleNamespace(supports_vision=True))
    assert not prefetch.sees(Broken()) and not prefetch.sees(object())


def test_only_a_run_given_the_screen_tools_can_use_it():
    assert prefetch.uses_screen(["memory", "ax_tree"])
    assert prefetch.uses_screen([registry().get("click").spec()])
    assert not prefetch.uses_screen(["memory", "weather", "open_app"])


# ---- which app ------------------------------------------------------------------------------------


def test_the_named_app_in_front_is_looked_at(world):
    t = prefetch.target("in quillpad, press new page")
    assert t == prefetch.Target("Quillpad", 101, "named")


def test_a_named_app_that_is_not_in_front_is_not(world):
    assert prefetch.target("in wayfarer, open the downloads") is None
    assert world.walks == 0


def test_an_app_that_only_shares_the_named_apps_name_is_not(world, monkeypatch):
    # another vendor's "Quillpad" is in front: the bundle id tells them apart
    other = inv.App("test.other.quillpad", "Quillpad", "/B/Quillpad.app")
    monkeypatch.setattr(inv, "app_for_pid", lambda pid: other)
    assert prefetch.target("in quillpad, press new page") is None
    monkeypatch.setattr(inv, "app_for_pid", lambda pid: None)  # not in the scan: its names decide
    assert prefetch.target("in quillpad, press new page") is not None


def test_a_request_about_two_apps_is_not(world):
    assert prefetch.target("copy this from quillpad into wayfarer") is None


def test_the_front_app_is_not_even_read_when_nothing_points_at_one(world, monkeypatch):
    monkeypatch.setattr(ax, "working_app", lambda: pytest.fail("read the front app"))
    assert prefetch.target("press the new page button") is None  # none named, NEO worked in none
    world.scanned = False
    assert prefetch.target("in quillpad, press new page") is None


def test_no_app_named_only_the_one_neo_last_worked_in(world, monkeypatch):
    assert prefetch.target("press the new page button") is None  # NEO never worked in it
    apps.set_target("Wayfarer")
    assert prefetch.target("press the new page button") is None  # it worked in another one
    apps.set_target("Quillpad")
    assert prefetch.target("press the new page button") == prefetch.Target("Quillpad", 101, "last")


def test_never_a_protected_app_or_neos_own_window(world, monkeypatch):
    from neo2.core import config

    world.front = ("Vaultkeeper", 103)
    assert prefetch.target("in vaultkeeper, press new page") is not None  # not protected yet
    monkeypatch.setenv("NEO_PROTECTED_APPS", '["test.vaultkeeper"]')  # by its bundle id
    monkeypatch.setattr(config, "_settings", None)
    assert prefetch.target("in vaultkeeper, press new page") is None
    world.front = ("NEO", 99)
    apps.set_target("NEO")
    assert prefetch.target("press the new page button") is None


def test_the_gate_helpers_are_asked_at_call_time_and_any_doubt_protects(world, monkeypatch):
    from neo2.core.tools import computer

    monkeypatch.setattr(computer, "_protected", lambda owner, pid=0: True)
    assert prefetch.target("in quillpad, press new page") is None

    def broken(owner, pid=0):
        raise RuntimeError("AppKit unavailable")

    monkeypatch.setattr(computer, "_protected", broken)
    assert prefetch.target("in quillpad, press new page") is None


def test_nothing_is_looked_at_before_the_installed_apps_are_known(world):
    world.scanned = False  # and no scan is started for it (the fixture's inventory() would fail)
    assert prefetch.target("in quillpad, press new page") is None


def test_nor_without_accessibility(world, monkeypatch):
    monkeypatch.setattr(ax, "is_trusted", lambda: False)
    assert prefetch.target("in quillpad, press new page") is None


def test_the_game_being_played_counts_as_named(world):
    tablero = world.installed[3]
    assert prefetch.target("make the first move", game=tablero) is None  # not in front
    world.front = ("Tablero", 104)
    assert prefetch.target("make the first move", game=tablero) == prefetch.Target("Tablero", 104, "game")


# ---- what -----------------------------------------------------------------------------------------


def test_the_list_carries_ids_and_labels_and_no_coordinates_in_any_form(world):
    look = _look("in quillpad, press new page", _decision(tool="click_text"))
    assert look is not None and look.picture is None and world.shots == 0
    assert 'e1 Button "New Page"' in look.elements and "e2 TextField" in look.elements and "(focused)" in look.elements
    assert "… 12 more elements" in look.elements
    assert not COORDS.search(look.elements), look.elements
    block = look.block()
    assert block.startswith(prefetch.MARK) and '<untrusted source="ax_tree">' in block
    assert not COORDS.search(block)


def test_an_unknown_line_is_scrubbed_too():
    raw = 'App: X\n  q9 Button "Go" @(10,20 30x40)\n  pos (12, 34) and x=5 y: 6 and @(1,2)'
    out = prefetch.compact(raw)
    assert not COORDS.search(out) and 'Button "Go"' in out


def test_a_picture_of_that_window_alone_only_when_the_screen_is_needed_and_the_brain_sees(world):
    look = _look("in quillpad, what does this page say", _decision(screen_p=0.9))
    assert look.picture is not None and world.shots == 1
    shot = pointer.get_shot(look.picture.shot)
    # its pixels map onto that window: 800 px across a 400 pt window at (100, 50)
    assert (shot.x0, shot.y0, shot.ppp) == (100.0, 50.0, 0.5) and look.picture.size == (800, 600)
    assert shot.to_screen(400, 300) == (300.0, 200.0)
    assert f'shot="{look.picture.shot}"' in look.block() and "(100" not in look.block()
    assert look.taint == ("ax_tree", "screenshot")
    blind = _look("in quillpad, what does this page say", _decision(screen_p=0.9), sees=False)
    assert blind.picture is None and world.shots == 1 and blind.taint == ("ax_tree",)


def test_a_board_in_the_first_looks_picture_is_named_by_its_squares(world):
    """A board NEO finds in the first look's picture comes with its squares' names, so the first
    model call can name a square (click cell=…) without a screenshot of its own. Only a board NEO
    found — never a hint, never a coordinate; a window with none says nothing about boards."""
    import io
    import re

    from PIL import Image, ImageDraw

    from neo2.core.tools.computer import board

    img = Image.new("RGB", (800, 600), (200, 200, 205))
    d = ImageDraw.Draw(img)
    for r in range(8):
        for c in range(8):
            d.rectangle([160 + 60 * c, 60 + 60 * r, 219 + 60 * c, 119 + 60 * r],
                        fill=(238, 238, 210) if (r + c) % 2 == 0 else (118, 150, 86))  # fmt: skip
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    world.capture = buf.getvalue()
    board.forget()
    try:
        look = _look("in quillpad, what does this page say", _decision(screen_p=0.9))
        assert look.picture is not None and "8×8" in look.board and 'cell="<square>"' in look.board
        assert f"[{look.board}]" in look.block()
        assert not re.search(r"\(\d+(?:\.\d+)?,\s*\d+", look.board)  # no points in any form
        (x, y) = board.points_for({"cell": "r1c1"}, ("cell",))[0][0]
        assert abs(x - (100 + 190 * 0.5)) <= 3 and abs(y - (50 + 90 * 0.5)) <= 3  # the square's middle, on the window
        board.forget()
        world.capture = _png(800, 600)
        plain = _look("in quillpad, what does this page say", _decision(screen_p=0.9))
        assert plain.picture is not None and plain.board == "" and "board" not in plain.block().lower()
    finally:
        board.forget()


def test_screen_recording_is_never_asked_for(world, monkeypatch):
    monkeypatch.setattr(screen, "can_capture", lambda: False)
    monkeypatch.setattr(screen, "_request_access", lambda: pytest.fail("asked for Screen Recording"))
    look = _look("in quillpad, what does this page say", _decision(screen_p=0.9))
    assert look is not None and look.picture is None


def test_an_app_that_changes_under_the_walk_is_dropped(world):
    world.switch_to = ("Wayfarer", 102)
    assert _look("in quillpad, press new page", _decision(tool="click_text")) is None


def test_a_name_another_process_answers_to_is_not_walked(world, monkeypatch):
    monkeypatch.setattr(ax, "pid_for_app", lambda name: 555)
    assert _look("in quillpad, press new page", _decision(tool="click_text")) is None
    assert world.walks == 0


# ---- the picture shows what a click at its pixels hits ----------------------------------------------


def _screen_png(w: int, h: int, floaty: tuple[int, int, int, int] | None) -> Image.Image:
    """The screen, 1 px per point: the window grey, another app's floating window red over part of it."""
    img = Image.new("RGB", (w, h), (30, 30, 30))
    img.paste((200, 200, 200), (100, 50, 500, 350))  # Quillpad's window at (100, 50), 400×300 pt
    if floaty:
        x, y, fw, fh = floaty
        img.paste((220, 0, 0), (x, y, x + fw, y + fh))
    return img


def _bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_a_window_another_app_covers_is_pictured_as_the_screen_shows_it(world, monkeypatch):
    """The window's own pixels (screencapture -l) include the part another window covers, but a click
    there lands in that other window: a picture whose pixels are clicked by must show what is really on
    screen. Every sampled pixel of the attached picture shows the app a click at it would hit."""
    floaty = (250, 150, 200, 120)
    world.above = [_window(555, "Floaty", *floaty)]
    screen_img = _screen_png(1512, 982, floaty)
    own = _bytes(Image.new("RGB", (400, 300), (200, 200, 200)))  # the window's own pixels: nothing over it
    monkeypatch.setattr(screen, "capture_window", lambda wid, region=None: ImagePart(own))

    def capture(region=None, display=None):
        world.region_shots.append(region)
        x, y, w, h = region
        return ImagePart(_bytes(screen_img.crop((x, y, x + w, y + h)))), 1.0

    monkeypatch.setattr(screen, "capture", capture)
    look = _look("in quillpad, what does this page say", _decision(screen_p=0.9))
    assert look is not None and look.picture is not None
    assert world.region_shots == [(100, 50, 400, 300)]  # that window's part of the screen
    pic = Image.open(io.BytesIO(look.picture.image.data)).convert("RGB")

    def hit(x, y):
        fx, fy, fw, fh = floaty
        return "Floaty" if fx <= x < fx + fw and fy <= y < fy + fh else "Quillpad"

    ctx = ToolContext(user_text="in quillpad, what does this page say")
    for px in range(10, 400, 37):
        for py in range(10, 300, 29):
            (sx, sy), = pointer.points({"x": px, "y": py, "shot": look.picture.shot}, ("x", "y"), ctx)[0]
            shows = "Floaty" if pic.getpixel((px, py)) == (220, 0, 0) else "Quillpad"
            assert shows == hit(sx, sy), (px, py, sx, sy)
    assert "in front of part of it" in look.block()


def test_a_protected_app_over_the_window_keeps_the_picture_out(world, monkeypatch):
    from neo2.core import config

    monkeypatch.setenv("NEO_PROTECTED_APPS", '["test.vaultkeeper"]')
    monkeypatch.setattr(config, "_settings", None)
    world.above = [_window(103, "Vaultkeeper", 150, 100, 120, 80)]
    look = _look("in quillpad, what does this page say", _decision(screen_p=0.9))
    assert look is not None and look.picture is None  # the list still goes; no pixels of that app do
    assert world.shots == 0 and world.region_shots == []
    world.above = [_window(0, "", 150, 100, 120, 80)]  # a window the window server names no owner for
    look = _look("in quillpad, what does this page say", _decision(screen_p=0.9))
    assert look is not None and look.picture is None and world.shots == 0 and world.region_shots == []


def test_neos_own_window_a_clear_one_or_one_beside_it_is_no_cover(world):
    world.above = [
        _window(99, "NEO", 0, 0, 1512, 982),  # NEO's own overlay: no screenshot shows it, no click lands on it
        _window(555, "Floaty", 150, 100, 120, 80, alpha=0.0),  # fully transparent: no screenshot shows it either
        _window(555, "Floaty", 600, 400, 120, 80),  # beside the window, not over it
        _window(555, "Floaty", 500, 50, 100, 100),  # touching its edge only
    ]
    look = _look("in quillpad, what does this page say", _decision(screen_p=0.9))
    assert look.picture is not None and world.shots == 1 and world.region_shots == []
    assert "in front of part of it" not in look.block()


def test_when_the_window_server_cant_say_what_is_over_it_no_picture(world, monkeypatch):
    def broken(wid):
        raise RuntimeError("window server")

    monkeypatch.setattr(prefetch, "_windows_above", broken)
    look = _look("in quillpad, what does this page say", _decision(screen_p=0.9))
    assert look is not None and look.picture is None and world.shots == 0 and world.region_shots == []


# ---- start, collect, attach -----------------------------------------------------------------------


async def test_it_runs_beside_routing_and_is_attached_with_its_taint(world):
    ctx = ToolContext(user_text="x")
    task = grants.new_task()
    with grants.task(task):
        p = prefetch.start("in quillpad, what does this page say", _decision(screen_p=0.9), sees=True)
        assert p is not None
        await asyncio.sleep(0)  # (playbook recall and tool selection happen here)
        look = await prefetch.collect(p, ["ax_tree", "click"])
        assert look is not None and ctx.taint == []  # read, but not attached yet: nothing tainted
        goal, images = look.attach("in quillpad, what does this page say", [], ctx)
        assert ctx.taint == ["ax_tree", "screenshot"]
        assert grants._book().reads[task] == 2  # a yes "for this task" given before ends here
    assert goal.startswith("in quillpad, what does this page say\n\n" + prefetch.MARK)
    assert images == look.images and len(images) == 1


async def test_a_run_without_the_screen_tools_gets_nothing_and_nothing_is_tainted(world):
    p = prefetch.start("in quillpad, press new page", _decision(tool="click_text"))
    assert await prefetch.collect(p, ["memory", "open_app"]) is None
    await asyncio.gather(p.task, return_exceptions=True)
    assert p.task.cancelled()


async def test_a_look_not_ready_by_its_deadline_is_dropped(world, monkeypatch):
    _env(monkeypatch, "NEO_PREFETCH_LOOK_S", "0.05")
    world.walk_s = 0.3
    t = time.monotonic()
    p = prefetch.start("in quillpad, press new page", _decision(tool="click_text"))
    assert await prefetch.collect(p, ["ax_tree"]) is None
    assert time.monotonic() - t < 0.25  # the first call never waits for a slow app
    assert world.deadlines[-1] == 0.05  # (and the walk itself was told to end by then: the AX core stops it)
    # …and that app's next requests don't wait for it at all, for a while
    walks = world.walks
    t = time.monotonic()
    p = prefetch.start("in quillpad, press new page", _decision(tool="click_text"))
    assert await prefetch.collect(p, ["ax_tree"]) is None and world.walks == walks
    assert time.monotonic() - t < 0.04
    monkeypatch.setattr(prefetch, "_slow_until", {})
    world.walk_s = 0.0
    p = prefetch.start("in quillpad, press new page", _decision(tool="click_text"))
    assert await prefetch.collect(p, ["ax_tree"]) is not None


async def test_a_web_apps_one_time_settle_is_waited_out_and_never_held_against_it(world, monkeypatch):
    """A Chromium or Electron app's first look in a while switches its web tree on, which then needs a
    moment (ax._WEB_SETTLE_S at most) before the walk: once per hold, on top of a fast walk. That look is
    still taken, and the app's next requests get theirs as fast as any."""
    _env(monkeypatch, "NEO_PREFETCH_LOOK_S", "0.2")
    monkeypatch.setattr(ax, "_WEB_SETTLE_S", 0.3)
    world.web, world.walk_s = True, 0.02
    d = _decision(tool="click_text")
    t = time.monotonic()
    first = await prefetch.collect(prefetch.start("in quillpad, press new page", d), ["ax_tree"])
    assert first is not None and 101 in ax._web_held
    assert time.monotonic() - t < 0.2 + 0.3 + 0.15  # the deadline, plus that settle once at most
    assert 101 not in prefetch._slow_until
    t = time.monotonic()
    again = await prefetch.collect(prefetch.start("in quillpad, press new page", d), ["ax_tree"])
    assert again is not None and world.walks == 2 and time.monotonic() - t < 0.2


async def test_a_web_app_slow_beyond_its_settle_is_still_skipped(world, monkeypatch):
    _env(monkeypatch, "NEO_PREFETCH_LOOK_S", "0.1")
    monkeypatch.setattr(ax, "_WEB_SETTLE_S", 0.1)
    world.web, world.walk_s = True, 0.35
    d = _decision(tool="click_text")
    p = prefetch.start("in quillpad, press new page", d)
    assert await prefetch.collect(p, ["ax_tree"]) is None
    await asyncio.gather(p.task, return_exceptions=True)
    p = prefetch.start("in quillpad, press new page", d)
    assert await prefetch.collect(p, ["ax_tree"]) is None and world.walks == 1  # not waited for again


async def test_the_setting_turns_it_off(world, monkeypatch):
    _env(monkeypatch, "NEO_PREFETCH_LOOK", "0")
    assert prefetch.start("in quillpad, what does this page say", _decision(screen_p=0.9)) is None
    assert world.walks == 0


async def test_an_embedded_agent_gets_it_only_when_its_host_opts_in(world, monkeypatch):
    from neo2.core import runtime

    host = types.SimpleNamespace(reflex="off")  # an embedded agent's settings, without the key yet
    monkeypatch.setattr(prefetch, "settings", lambda: host)
    monkeypatch.setattr(runtime, "library_bound", lambda: True)
    _env(monkeypatch, "NEO_PREFETCH_LOOK", "1")
    assert not prefetch.enabled()  # off, and never read from the environment there
    host.prefetch_look = True  # the host turned it on (a LibrarySettings key)
    assert prefetch.enabled()
    host.prefetch_look = False
    monkeypatch.setattr(runtime, "library_bound", lambda: False)
    assert not prefetch.enabled()  # the setting, once neo2.core.config has it, wins everywhere


def test_the_suite_takes_no_first_look_unless_a_test_opts_in():
    """tests/conftest.py turns it off for every test (this Mac's real front app is never read by a test
    that reaches Session._agent, the game route included); the ones here opt in (the world fixture)."""
    assert not prefetch.enabled()


def test_the_settings_in_neos_env_file_count_before_neo_config_has_the_keys(world, monkeypatch, tmp_path):
    """NEO.app's settings come from ~/.neo/.env through neo2.core.config, which never puts them into the
    environment: read the way Settings reads every other key (here a made-up .env file)."""
    from neo2.core import config, effects

    env = tmp_path / "user.env"
    env.write_text("NEO_PREFETCH_LOOK=0\nNEO_PREFETCH_LOOK_S=0.3\nNEO_PREWARM_RESOLVER=off\n")
    for k in ("NEO_PREFETCH_LOOK", "NEO_PREFETCH_LOOK_S", "NEO_PREWARM_RESOLVER"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setitem(config.Settings.model_config, "env_file", str(env))
    monkeypatch.setattr(effects, "why_off", lambda: "")  # (effects off, Settings reads no .env file at all)
    monkeypatch.setattr(config, "_settings", None)
    assert not prefetch.enabled()
    assert prefetch._deadline_s() == 0.3
    assert not prefetch._setting("prewarm_resolver", True, False)
    env.write_text("NEO_PREFETCH_LOOK=1\n")
    monkeypatch.setattr(config, "_settings", None)  # (a new Settings: NEO.app reloads its settings)
    assert prefetch.enabled() and prefetch._deadline_s() == prefetch.LOOK_S


async def test_no_screen_tools_in_this_agent_no_look(world, monkeypatch):
    real = registry().get
    monkeypatch.setattr(registry(), "get", lambda n: None if n == "ax_tree" else real(n))
    assert prefetch.start("in quillpad, what does this page say", _decision(screen_p=0.9)) is None


async def test_a_failed_walk_leaves_the_run_to_look_for_itself(world, monkeypatch):
    def broken(app=None, *, max_lines=250, deadline_s=None):
        raise RuntimeError("AX went away")

    monkeypatch.setattr(ax, "snapshot", broken)
    p = prefetch.start("in quillpad, press new page", _decision(tool="click_text"))
    assert await prefetch.collect(p, ["ax_tree"]) is None


async def test_a_cancelled_request_cancels_its_look(world):
    world.walk_s = 0.2
    p = prefetch.start("in quillpad, press new page", _decision(tool="click_text"))
    job = asyncio.ensure_future(prefetch.collect(p, ["ax_tree"]))
    await asyncio.sleep(0.02)
    job.cancel()
    with pytest.raises(asyncio.CancelledError):
        await job
    await asyncio.gather(p.task, return_exceptions=True)
    assert p.task.cancelled()


# ---- the history ----------------------------------------------------------------------------------


def test_a_finished_run_keeps_only_a_stub_and_the_users_own_picture(world):
    look = _look("in quillpad, what does this page say", _decision(screen_p=0.9))
    theirs = ImagePart(b"the user's own picture")
    goal, images = look.attach("in quillpad, what does this page say", [theirs], ToolContext())
    msgs = [Message.user(goal, images=images), Message.assistant("It says hello.")]
    prefetch.forget(msgs)
    assert msgs[0].text == f"in quillpad, what does this page say\n\n{prefetch.STUB}"
    assert msgs[0].images == [theirs]
    assert pointer.get_shot(look.picture.shot) is None  # nothing is clicked by a picture nobody has any more


_NO_CHANGE = "Clicked New Page. Nothing visibly changed there within 1s — look before trying again."


@pytest.mark.parametrize(
    "newer",
    [
        ToolResult("c2", "ax_tree", 'App: Quillpad — window "Draft"'),
        ToolResult("c2", "ax_find", 'e7 Button "Share"'),
        ToolResult("c2", "screenshot", "Screenshot s9 attached (all of display 1)"),
        ToolResult("c2", "ax_press", "Pressed New Page. Changed: a sheet or dialog opened."),
        ToolResult("c2", "click", "Clicked (300, 200) in Quillpad. Changed: the screen changed around the point."),
    ],
    ids=lambda r: r.name,
)
def test_in_a_run_a_newer_look_or_a_change_on_screen_makes_it_stale(world, newer):
    look = _look("in quillpad, what does this page say", _decision(screen_p=0.9))
    goal, images = look.attach("in quillpad, what does this page say", [], ToolContext())
    msgs = [Message.user(goal, images=images), Message.assistant("", [ToolCall("c1", "click", {"id": "e1"})]),
            Message.tool([ToolResult("c1", "click", _NO_CHANGE)]),
            Message.assistant("", [ToolCall("c3", "ax_tree", {})]),
            Message.tool([ToolResult("c3", "ax_tree", "Error: Quillpad isn't responding", ok=False)])]  # fmt: skip
    prefetch.supersede(msgs)
    # an action that changed nothing, or a look that failed, leaves it standing
    assert prefetch.MARK in msgs[0].text and pointer.get_shot(look.picture.shot) is not None
    msgs += [Message.assistant("", [ToolCall("c2", newer.name, {})]), Message.tool([newer])]
    prefetch.supersede(msgs)
    assert prefetch.MARK not in msgs[0].text and msgs[0].images == []
    assert pointer.get_shot(look.picture.shot) is None  # its pixels can't be clicked by after it's gone


def test_a_stale_first_look_keeps_how_its_boards_squares_are_named(world):
    """Mid-run, the first look goes stale once a move changes the screen — but the board is still
    there: NEO's words on how its squares are named stay (not the pieces as they stood). A finished
    run keeps none of it."""
    import dataclasses

    look = _look("in quillpad, what does this page say", _decision(screen_p=0.9))
    words = (
        "NEO found a board of 8×8 squares. Squares are named a1–h8: White at the bottom. To act on a square, "
        'name it — click cell="<square>" — and NEO works out the points. Pieces now, top row first: 8/8/8/8/8/8/8/8.'
    )
    look = dataclasses.replace(look, board=words)
    goal, images = look.attach("make the first move e4", [], ToolContext())
    moved = ToolResult("c1", "click", "Left-clicked (square e2) in Quillpad. Changed: the screen changed around e2.")
    msgs = [Message.user(goal, images=images), Message.assistant("", [ToolCall("c1", "click", {"cell": "e2"})]),
            Message.tool([moved])]  # fmt: skip
    prefetch.supersede(msgs)
    assert prefetch.MARK not in msgs[0].text and msgs[0].images == []
    assert msgs[0].text.endswith("and NEO works out the points.]") and "Pieces now" not in msgs[0].text
    assert msgs[0].text.startswith(f"make the first move e4\n\n{prefetch.STUB}\n[NEO found a board of 8×8")
    done = [Message.user(look.attach("make the first move e4", [], ToolContext())[0])]
    prefetch.forget(done)
    assert done[0].text == f"make the first move e4\n\n{prefetch.STUB}"


def _loop_wired():
    import inspect

    from neo2.core.agent import loop

    if "prefetch.supersede" not in inspect.getsource(loop.run_agent):
        pytest.skip("loop.py doesn't call prefetch.supersede yet (required wiring: the integrator adds it)")


async def test_the_loop_stubs_the_first_look_once_the_model_changed_the_screen(world, monkeypatch):
    from neo2.core.agent.loop import run_agent
    from neo2.core.events import EventBus

    _loop_wired()
    text = "in quillpad, what does this page say"
    look = await prefetch.collect(prefetch.start(text, _decision(screen_p=0.9), sees=True), ["ax_tree", "ax_press"])
    goal, images = look.attach(text, None, ToolContext())

    async def press(a, c):
        return "Pressed New Page. Changed: a sheet or dialog opened."

    monkeypatch.setattr(registry().get("ax_press"), "handler", press)
    chain = _Chain(then=[Turn(tool_calls=[ToolCall("c1", "ax_press", {"id": "e1"})])])
    tools = [registry().get(n).spec() for n in ("ax_tree", "ax_press")]
    await run_agent(goal, provider=chain, system="s", tools=tools, ev=EventBus(), images=images, verify=False)
    first, second = chain.requests[0][-1], chain.requests[1][0]
    assert prefetch.MARK in first.text and first.images
    assert prefetch.STUB in second.text and prefetch.MARK not in second.text and second.images == []


# ---- through the agent loop: one round trip less, nothing blind -----------------------------------


class _Brain:
    """Acts on the element it needs once it has seen the window — in the goal or in a look it asked
    for — and looks first otherwise. Counts round trips."""

    supports_vision, supports_tools, name = True, True, "fake"

    def __init__(self):
        self.calls = 0
        self.first_request = ""

    async def complete(self, messages, *, system="", tools=None, effort="medium", **kw):
        self.calls += 1
        if self.calls == 1:
            self.first_request = messages[-1].text
        seen = " ".join([m.text for m in messages if m.role == "user"] + [r.content for m in messages for r in m.tool_results])
        pressed = any(c.name == "ax_press" for m in messages for c in m.tool_calls)
        if pressed:
            return Turn(text="Done.")
        m = re.search(r'(e\d+) Button "New Page"', seen)
        if m:
            return Turn(tool_calls=[ToolCall(f"c{self.calls}", "ax_press", {"id": m.group(1)})])
        return Turn(tool_calls=[ToolCall(f"c{self.calls}", "ax_tree", {"app": "Quillpad"})])


async def _run(monkeypatch, goal, images=None):
    from neo2.core.agent.loop import run_agent
    from neo2.core.events import EventBus

    async def tree(a, c):
        return ax.snapshot(a.get("app"))

    async def press(a, c):
        return f"Pressed {a['id']}."

    monkeypatch.setattr(registry().get("ax_tree"), "handler", tree)
    monkeypatch.setattr(registry().get("ax_press"), "handler", press)
    brain = _Brain()
    tools = [registry().get(n).spec() for n in ("ax_tree", "ax_press")]
    res = await run_agent(goal, provider=brain, system="s", tools=tools, ev=EventBus(), images=images, verify=False)
    return brain, res


async def test_with_the_first_look_the_run_acts_one_round_trip_sooner(world, monkeypatch):
    text = "in quillpad, press new page"
    before, _ = await _run(monkeypatch, text)
    p = prefetch.start(text, _decision(tool="click_text"))
    look = await prefetch.collect(p, ["ax_tree", "ax_press"])
    goal, images = look.attach(text, None, ToolContext())
    after, res = await _run(monkeypatch, goal, images)
    assert (before.calls, after.calls) == (3, 2)
    assert [s.tool for s in res.steps] == ["ax_press"] and res.steps[0].args == {"id": "e1"}


def test_a_brain_that_cannot_see_gets_no_picture_and_no_coordinates(world):
    from neo2.core.providers.openai_compat import _to_messages

    look = _look("in quillpad, what does this page say", _decision(screen_p=0.9))
    goal, images = look.attach("in quillpad, what does this page say", [], ToolContext())
    sent = _to_messages([Message.user(goal, images=images)], "sys", vision=False)
    body = sent[-1]["content"]
    assert isinstance(body, str) and "no vision" in body
    assert not COORDS.search(body)
    # (a chain that could see when the look was taken, then fell to a blind brain): never told to click
    # by pixel as if it could see, but what to use instead
    assert "If you can't see it, use the element ids above" in body
    for m in re.finditer("pixel position", body):
        assert body[: m.start()].rstrip().endswith("If you can see it: to act on something only the picture shows, give its"), body


# ---- prewarm --------------------------------------------------------------------------------------


class _FakeLaya:
    def __init__(self):
        self.calls = []

    def system_one(self, state, questions):
        time.sleep(0.005)  # a warm pass
        self.calls.append(state)
        answer = {"type": "choice", "choice": "A", "probabilities": {"A": 0.9, "B": 0.1}}
        return {"answers": {q: answer for q in questions}}

    def __exit__(self, *exc):
        pass


@pytest.fixture
def laya(monkeypatch):
    """Laya on, the zero-shot checkpoint faked: a load takes ``laya.load_s``."""
    from neo2.core import config

    monkeypatch.setenv("NEO_REFLEX", "laya")
    monkeypatch.setattr(config, "_settings", None)
    state = types.SimpleNamespace(load_s=0.15, opened=[])

    def open_(path):
        time.sleep(state.load_s)
        agent = _FakeLaya()
        state.opened.append(agent)
        return agent

    monkeypatch.setattr(zeroshot, "_find", lambda: "/nowhere/typed-decisions")
    monkeypatch.setattr(zeroshot, "_open", open_)
    for k, v in (("_agent", None), ("_loading", None), ("_fetching", None), ("_failed_at", float("-inf")), ("_used", 0.0)):
        monkeypatch.setattr(zeroshot, k, v)
    monkeypatch.setattr(prefetch, "_last_warm", float("-inf"))
    yield state
    if zeroshot._timer is not None:
        zeroshot._timer.cancel()


def _until(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while not cond() and time.monotonic() < end:
        time.sleep(0.005)
    return cond()


Q = {"app": {"type": "choice", "instructions": "Which?", "criteria": {"A": "a", "B": "b"}}}


def test_the_wake_word_loads_the_resolvers_laya_while_the_user_speaks(laya):
    t = time.perf_counter()
    assert prefetch.prewarm("wake")
    assert time.perf_counter() - t < 0.05  # never waited for
    assert _until(zeroshot.loaded)
    t = time.perf_counter()
    assert zeroshot.ask({"request": "x"}, Q, wait_s=1.0) is not None
    assert time.perf_counter() - t < laya.load_s / 2  # the resolver's question finds it warm
    assert len(laya.opened) == 1


def test_a_cold_resolver_question_pays_the_load(laya):
    t = time.perf_counter()
    assert zeroshot.ask({"request": "x"}, Q, wait_s=1.0) is not None
    assert time.perf_counter() - t >= laya.load_s  # (what the prewarm takes off the request)


def test_a_loaded_model_is_kept_from_being_let_go(laya):
    zeroshot.ask({"request": "x"}, Q, wait_s=1.0)
    zeroshot._used = time.monotonic() - zeroshot.IDLE_S + 1  # a second from being unloaded
    assert prefetch.prewarm("wake")
    assert _until(lambda: time.monotonic() - zeroshot._used < 5)
    assert len(laya.opened) == 1


def test_prewarm_is_rate_limited_and_never_blocks_on_a_busy_laya(laya, monkeypatch):
    assert prefetch.prewarm("wake")
    assert not prefetch.prewarm("speech")  # the words right after the wake word: already warming
    assert _until(zeroshot.loaded)
    agent = laya.opened[0]
    n = len(agent.calls)  # (its load's own warm-up pass)
    prefetch._last_warm = float("-inf")
    done = threading.Event()
    real = prefetch._warm
    monkeypatch.setattr(prefetch, "_warm", lambda reason: (real(reason), done.set()))
    with neo2.core.reflex.laya_lock():  # the reflex is deciding: the prewarm doesn't queue behind it
        assert prefetch.prewarm("wake")
        assert done.wait(1.0)  # it gave up at once, with Laya still busy
        assert len(agent.calls) == n
    time.sleep(0.05)
    assert len(agent.calls) == n  # and nothing it left waiting ran once Laya was free again
    assert all(c == {"text": "warm up"} for c in agent.calls)


def test_no_prewarm_with_laya_off_the_setting_off_or_embedded(laya, monkeypatch):
    from neo2.core import config, runtime

    _env(monkeypatch, "NEO_PREWARM_RESOLVER", "0")
    assert not prefetch.prewarm("wake")
    _env(monkeypatch, "NEO_PREWARM_RESOLVER", None)
    monkeypatch.setattr(runtime, "library_bound", lambda: True)
    assert not prefetch.prewarm("wake")
    monkeypatch.setattr(runtime, "library_bound", lambda: False)
    monkeypatch.setenv("NEO_REFLEX", "off")
    monkeypatch.setattr(config, "_settings", None)
    assert not prefetch.prewarm("wake")
    assert not laya.opened


def test_other_warmers_run_in_the_prewarm_thread_and_their_errors_stay_there(laya, monkeypatch):
    ran = threading.Event()
    monkeypatch.setattr(prefetch, "_warmers", [])

    def boom():
        raise RuntimeError("index missing")

    prefetch.register_warmer(boom)
    prefetch.register_warmer(ran.set)
    prefetch.register_warmer(ran.set)
    assert len(prefetch._warmers) == 2
    assert prefetch.prewarm("wake") and ran.wait(2)


def test_the_early_actor_prewarms_on_listening_and_once_per_utterance(monkeypatch):
    from neo2.core.agent.early import EarlyActor

    seen: list[str] = []
    monkeypatch.setattr(prefetch, "prewarm", lambda reason="wake": seen.append(reason) or True)
    ea = EarlyActor()
    assert ea.listening("wake")
    ea.feed("in quill")
    ea.feed("in quillpad press new page")
    assert seen == ["wake"]  # the words after a wake word it was told about: nothing more
    ea.new_utterance()
    ea.feed("and then")
    ea.feed("and then save it")
    assert seen == ["wake", "speech"]  # an utterance nobody announced warms on its first words


# ---- through the session (after the integrator wires neo2/core/agent/session.py) ------------------------


def _wired():
    import inspect

    from neo2.core.agent import session as sess

    if "prefetch.start" not in inspect.getsource(sess.Session._agent_task):
        pytest.skip("session.py isn't wired to neo2.core.agent.prefetch yet (the integrator does it after Step 0)")
    return sess


def test_the_settings_are_neo_config_keys_once_the_session_is_wired():
    """Required with the session wiring: the keys in neo2.core.config (an embedded agent's host can only
    turn the first look on through a LibrarySettings key; it is off there by default)."""
    _wired()
    from neo2.core.config import LibrarySettings, Settings

    for key in ("prefetch_look", "prefetch_look_s", "prewarm_resolver"):
        assert key in Settings.model_fields and key in LibrarySettings.model_fields, key
    assert Settings.model_fields["prefetch_look"].default is True
    assert LibrarySettings.model_fields["prefetch_look"].default is False


class _Chain:
    supports_vision, supports_tools, name, last_used, last_model = True, True, "fake", "fake", "fake"

    def __init__(self, then: list[Turn] | None = None):
        self.requests: list[list[Message]] = []
        self.then = list(then or [])

    async def complete(self, messages, *, system="", tools=None, effort="medium", **kw):
        # as sent: the session stubs the look afterwards, in these very messages
        self.requests.append([dataclasses.replace(m, images=list(m.images)) for m in messages])
        return self.then.pop(0) if self.then else Turn(text="Done.")


async def _session_run(monkeypatch, text, d, **kw):
    sess = _wired()
    chain = _Chain()
    monkeypatch.setattr(sess, "brain", lambda purpose="agent": chain)
    s = sess.Session()
    reply = await s._agent(text, d, None, "jlook", **kw)
    return s, chain, reply


async def test_the_session_hands_the_first_look_to_the_first_call_and_keeps_a_stub(world, monkeypatch):
    s, chain, reply = await _session_run(monkeypatch, "in quillpad, what does this page say", _decision(screen_p=0.9))
    goal = chain.requests[0][-1]
    assert goal.role == "user" and prefetch.MARK in goal.text and len(goal.images) == 1
    assert 'e1 Button "New Page"' in goal.text and not COORDS.search(goal.text)
    assert set(reply.taint) == {"ax_tree", "screenshot"}  # the request read the screen: said so
    kept = next(m for m in s.history if m.role == "user")
    assert prefetch.STUB in kept.text and prefetch.MARK not in kept.text and kept.images == []


async def test_the_session_takes_no_look_at_a_protected_app_and_taints_nothing(world, monkeypatch):
    from neo2.core import config

    monkeypatch.setenv("NEO_PROTECTED_APPS", '["Quillpad"]')
    monkeypatch.setattr(config, "_settings", None)
    s, chain, reply = await _session_run(monkeypatch, "in quillpad, what does this page say", _decision(screen_p=0.9))
    assert prefetch.MARK not in chain.requests[0][-1].text and not chain.requests[0][-1].images
    assert reply.taint == () and world.walks == 0 and world.shots == 0


async def test_the_session_takes_no_look_with_the_setting_off(world, monkeypatch):
    _env(monkeypatch, "NEO_PREFETCH_LOOK", "off")
    s, chain, reply = await _session_run(monkeypatch, "in quillpad, what does this page say", _decision(screen_p=0.9))
    assert prefetch.MARK not in chain.requests[0][-1].text and reply.taint == ()


async def test_the_session_looks_at_the_game_being_played(world, monkeypatch):
    world.front = ("Tablero", 104)
    d = _decision(screen_p=0.95)
    s, chain, reply = await _session_run(monkeypatch, "make the first move", d, game=world.installed[3])
    assert prefetch.MARK in chain.requests[0][-1].text and chain.requests[0][-1].images
