"""A first click on a window whose app isn't the active one — the controls self-test of 2026-09-27:
another app (Claude) was active, the test window belonged to NEO's child python process (no Dock
icon: not a "regular" app, though its window is a normal one), and the click only activated that
window: its board never saw it, and the click tool still answered "Left-clicked at (255, 186) in
python." — it sounded like success.

So the app that owns the normal window at the point comes forward (active, that window key) before
the click, whatever its activation policy; nothing that isn't an app window is ever activated; and
a click that itself brought its app forward, after which nothing changed at the point, says it
only brought the app forward instead of calling it done.

Hermetic: what is under a point, the window server's list, which app is active and whose window is
key, the activation calls, the pointer events and the screen pictures are all fakes."""

from __future__ import annotations

import asyncio
import types

import pytest
from PIL import Image

from neo2.core.agent.registry import ToolContext, ToolOutput, registry
from neo2.core.tools.computer import apps, ax, focus, pointer, screen
from neo2.core.tools.computer import input as inp

TEST, CLAUDE = 97767, 629  # the self-test's window (a python process), the app that was active
FRAME = (40.0, 73.0, 720.0, 358.0)  # the test window on screen (points, top-left origin)
TITLE_BAR = 28.0  # its title bar: drawn anew when the window becomes key
BOARD = (255.0, 186.0)  # a board cell deep in the content (the self-test's cell 6,1)
UNDER_TITLE = (400.0, 110.0)  # content just under the title bar: a look around it takes the title bar in
Win = types.SimpleNamespace  # a window as the window server has it: pid, owner, layer, frame
_REAL = {n: getattr(ax, n, None) for n in ("windows_at", "key_window", "is_key")}  # (the suite fakes them in every test)


def inside(r, x, y):
    return r[0] <= x < r[0] + r[2] and r[1] <= y < r[1] + r[3]


class Desk:
    """Claude active; the test window behind it, owned by a process that isn't a regular app. Its
    view doesn't take the first mouse: a click while its app isn't active with that window key only
    activates it (the Chess board does the same)."""

    def __init__(self):
        self.front = ("Claude", CLAUDE)  # what LaunchServices calls frontmost
        self.active = False  # the test app really active, its window key
        self.bring_activates = True  # bring_pid_front makes it active (else: frontmost by LaunchServices' word only)
        self.force_activates = True
        self.first_mouse = False  # the view takes the click that activates its window
        self.lag, self.pending = 0, 0  # a busy app: activated by a click, it shows so only at the lag-th look after
        self.layer, self.role, self.owner = 0, "AXGroup", TEST  # the window server's word on the window at the point
        self.win, self.claude_win = object(), object()  # the two windows' AX elements
        self.clicks: list[tuple[float, float]] = []  # clicks the test window's view got
        self.log: list[tuple] = []

    # what is where
    def hit(self, x, y):
        if inside(FRAME, x, y):
            return ax.Hit(pid=TEST, app="python", label="", role=self.role, regular=False, window=self.win)
        return ax.Hit(pid=CLAUDE, app="Claude", label="", role="AXGroup", regular=True, window=self.claude_win)

    def windows_at(self, x, y):
        out = []
        if inside(FRAME, x, y):
            out.append(Win(pid=self.owner, owner="python", layer=self.layer, frame=FRAME))
        out.append(Win(pid=CLAUDE, owner="Claude", layer=0, frame=(0.0, 25.0, 1512.0, 957.0)))
        return out

    # who is active
    def key_window(self, pid):
        if pid == TEST:
            return self.win if self.active else None
        return self.claude_win if self.front[1] == CLAUDE else None

    def is_key(self, h):
        return h.window is not None and self.key_window(h.pid) is h.window

    def _activate(self):
        self.front, self.active = ("python", TEST), True

    async def bring(self, pid):
        self.log.append(("bring", pid))
        self.front = ("python", pid)
        if self.bring_activates:
            self.active = True
        return True, "python"

    def force(self, pid):
        self.log.append(("force", pid))
        if self.force_activates:
            self._activate()
        return True

    # the pointer
    def move(self, x, y):
        self.log.append(("move", round(x), round(y)))

    def click(self, x, y, button="left", count=1, mods=()):
        self.log.append(("click", round(x), round(y)))
        if inside(FRAME, x, y):
            if not self.active:
                self._wake()  # the click is spent activating the window…
                if not self.first_mouse:
                    return  # …and the view never sees it
            self.clicks.append((x, y))

    def _wake(self):
        if self.lag:
            self.pending = self.lag
        else:
            self._activate()

    # what NEO looks at
    def scene(self, pid, front=None):
        if self.pending:
            self.pending -= 1
            if not self.pending:
                self._activate()
        mine = pid == TEST and self.active
        return ax.Scene(
            front=self.front[0], front_pid=self.front[1], title="NEO controls self-test" if mine else "",
            focus="Group" if mine else "", known=True,
        )  # fmt: skip

    def sample(self, pts, r=60, frame=None, like=None):
        g = screen._SAMPLE_GRID
        step = 2 * r / g
        out = []
        for px, py in pts:
            img = Image.new("L", (g, g), 200)
            for j in range(g):
                cy = py - r + (j + 0.5) * step
                for i in range(g):
                    cx = px - r + (i + 0.5) * step
                    if not inside(FRAME, cx, cy):
                        continue
                    if cy < FRAME[1] + TITLE_BAR:
                        img.putpixel((i, j), 90 if self.active else 160)  # the title bar: darker once key
                    elif self.clicks:
                        img.putpixel((i, j), 40)  # the content answered a click
                    else:
                        img.putpixel((i, j), 230)
            out.append(img)
        return out

    def order(self):
        return [e[0] for e in self.log if e[0] in ("bring", "force", "click")]


@pytest.fixture
def desk(monkeypatch):
    d = Desk()
    monkeypatch.setattr(ax, "hit", d.hit)
    monkeypatch.setattr(ax, "windows_at", d.windows_at, raising=False)
    monkeypatch.setattr(ax, "key_window", d.key_window, raising=False)
    monkeypatch.setattr(ax, "is_key", d.is_key, raising=False)
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": False)
    monkeypatch.setattr(ax, "app_names", lambda pid: set())
    monkeypatch.setattr(ax, "scene", d.scene)
    monkeypatch.setattr(focus, "front_app", lambda: d.front)
    monkeypatch.setattr(focus, "force_front", d.force)
    monkeypatch.setattr(apps, "bring_pid_front", d.bring)
    monkeypatch.setattr(inp, "click", d.click)
    monkeypatch.setattr(inp, "move", d.move)
    monkeypatch.setattr(screen, "sample_points", d.sample)
    return d


def click(at, origin="agent") -> ToolOutput:
    ctx = ToolContext(user_text="play on the board", origin=origin)
    return asyncio.run(registry().invoke("click", {"x": at[0], "y": at[1]}, ctx))


# ---- the owner of the window at the point comes forward first, whatever its activation policy ----------


def test_a_non_regular_apps_normal_window_is_brought_forward_before_the_first_click(desk):
    out = click(BOARD)
    assert desk.order() == ["bring", "click"] and ("bring", TEST) in desk.log
    assert desk.clicks == [BOARD]  # the board got the click
    assert out.ok and "brought python to the front first" in out.text
    assert apps.target() == "python"  # "click the field, then type": the typing goes there too


def test_an_app_frontmost_by_launchservices_word_only_is_made_active_before_the_click(desk):
    """The self-test's report said the window "was already in front" — LaunchServices called its app
    frontmost — yet the click only activated it: it wasn't active, no window of it was key."""
    desk.front = ("python", TEST)
    out = click(BOARD)
    assert desk.order() == ["force", "click"] and desk.clicks == [BOARD]
    assert out.ok and "only brought" not in out.text


def test_an_app_already_active_with_its_window_key_is_left_alone(desk):
    desk._activate()
    out = click(BOARD)
    assert desk.order() == ["click"] and desk.clicks == [BOARD] and out.ok
    assert "brought" not in out.text
    assert apps.target() == "python"  # its own normal window: the typing goes there next


@pytest.mark.parametrize(
    "layer, role, owner",
    [
        (20, "AXDockItem", TEST),  # the Dock
        (24, "AXMenuBarItem", TEST),  # the menu bar
        (25, "AXMenuBarItem", TEST),  # a status item
        (101, "AXMenuItem", TEST),  # a menu
        (0, "AXPopover", TEST),  # a popover, whatever its layer
        (0, "AXMenuItem", TEST),  # a menu, whatever its layer
        (23, "AXGroup", TEST),  # Notification Center, Control Center, Spotlight: over every normal window
        (3, "AXGroup", TEST),  # a floating panel: not a normal window either
        (0, "AXGroup", 4040),  # a normal window, but another process's (a view service drawing in it)
    ],
)
def test_nothing_but_the_owners_own_normal_window_is_ever_activated(desk, layer, role, owner):
    desk.layer, desk.role, desk.owner = layer, role, owner
    click(BOARD)
    assert desk.order() == ["click"]  # clicked as it is: nothing brought forward, nothing forced


def test_nothing_is_activated_when_the_window_server_cant_say(desk, monkeypatch):
    monkeypatch.setattr(ax, "windows_at", lambda x, y: None, raising=False)
    click(BOARD)
    assert desk.order() == ["click"]


def test_a_protected_app_is_still_never_brought_forward(desk, monkeypatch):
    from neo2.core import config

    monkeypatch.setenv("NEO_PROTECTED_APPS", '["python"]')
    monkeypatch.setattr(config, "_settings", None)
    out = click(BOARD)
    assert not out.ok and desk.order() == []


def test_a_window_that_covers_the_point_once_the_app_is_in_front_is_still_refused(desk, monkeypatch):
    base = desk.bring

    async def bring(pid):  # coming forward, a panel of Claude's is still over the spot
        monkeypatch.setattr(ax, "hit", lambda x, y: ax.Hit(pid=CLAUDE, app="Claude", label="", role="AXGroup", regular=True))
        return await base(pid)

    monkeypatch.setattr(apps, "bring_pid_front", bring)
    out = click(BOARD)
    assert not out.ok and "still over that spot" in out.text and desk.clicks == []


def test_another_window_of_the_app_that_comes_over_the_point_is_not_clicked(desk, monkeypatch):
    """Coming forward, the app put another of its own windows over the point (its main window, the
    one activation brings forward; a space switch). Neither has a label — a board, a canvas — so only
    the window tells them apart: that one isn't clicked, nor raised."""
    other = object()
    base = desk.bring

    async def bring(pid):
        monkeypatch.setattr(ax, "hit", lambda x, y: ax.Hit(pid=TEST, app="python", label="", role="AXGroup", regular=False, window=other))
        return await base(pid)

    raised = []
    monkeypatch.setattr(apps, "bring_pid_front", bring)
    monkeypatch.setattr(ax, "raise_window", lambda h: raised.append(h) or True)
    out = click(BOARD)
    assert not out.ok and desk.order() == ["bring"] and raised == []
    assert "another of its windows" in out.text


def test_an_overlay_of_the_app_in_front_is_neither_raised_nor_typed_into_next(desk, monkeypatch):
    """The app in front is active, but what is at the point is an overlay of it (Spotlight's panels
    are on layer 23), not a normal window: it isn't raised, and "click, then type" doesn't make it
    the app to type in — the next typing would bring it forward by name."""
    desk._activate()
    desk.layer = 23
    raised = []
    monkeypatch.setattr(ax, "raise_window", lambda h: raised.append(h) or True)
    click(BOARD)
    assert desk.clicks == [BOARD]
    assert raised == [] and apps.target() is None


def test_the_wait_for_the_app_is_a_deadline_not_a_fixed_sleep(desk, monkeypatch, fake_clock):
    monkeypatch.setattr(pointer, "KEY_S", 1.0, raising=False)
    desk.bring_activates = False
    polls = {"n": 0}

    def key_window(pid):  # it becomes active on the third look
        polls["n"] += 1
        if polls["n"] >= 3:
            desk.active = True
        return desk.win if desk.active else None

    monkeypatch.setattr(ax, "key_window", key_window, raising=False)
    click(BOARD)
    assert desk.clicks == [BOARD]
    assert 0 < sum(fake_clock.sleeps) < 0.2  # done at the first look that saw it, not after a set time


# ---- a click that only brought its app forward says so -----------------------------------------------


def test_a_click_that_only_activated_the_window_is_not_called_done(desk):
    desk.bring_activates = False  # frontmost by LaunchServices' word, never active: the click activates it
    out = click(BOARD)
    assert desk.clicks == []  # (the board never saw it)
    assert "only brought python to the front" in out.text
    assert "Changed" not in out.text and "now in front" not in out.text
    assert "screenshot" in out.text and "twice" in out.text  # look before clicking again; no second click
    assert [e for e in desk.log if e[0] == "click"] == [("click", 255, 186)]


def test_the_title_bar_redrawing_as_the_window_becomes_key_is_no_effect_of_the_click(desk):
    desk.bring_activates = False
    out = click(UNDER_TITLE)  # the look around the point takes in the title bar, which darkens
    assert "only brought python to the front" in out.text and "the screen changed" not in out.text


@pytest.mark.parametrize(
    "case",
    [
        "frontmost by LaunchServices' word, the switch didn't take",  # bring() knew: t.cold
        "a floating panel of an app in the background",  # never brought forward (layer 3): the click activates it
    ],
)
def test_an_activation_that_shows_only_at_the_second_look_is_still_no_success(desk, case):
    """A busy app: the click only activated it, and that shows only at the second look."""
    if case.startswith("frontmost"):
        desk.front, desk.force_activates = ("python", TEST), False
    else:
        desk.layer = 3
    desk.lag = 2
    out = click(BOARD)
    assert desk.clicks == [] and desk.active  # (it did come forward, at the second look)
    assert "only brought python to the front" in out.text
    assert "Changed" not in out.text and "brought python to the front first" not in out.text


def test_without_pictures_a_click_that_woke_its_app_isnt_said_to_have_changed_nothing_at_the_point(desk, monkeypatch):
    """No Screen Recording: nothing shows whether the view took the click (here it did), so the
    answer is that the click may have only brought the app forward — not that nothing changed there."""
    desk.bring_activates, desk.first_mouse = False, True
    monkeypatch.setattr(screen, "sample_points", lambda pts, r=60, frame=None, like=None: None)
    out = click(BOARD)
    assert desk.clicks == [BOARD]
    assert "may have only brought python to the front" in out.text
    assert "nothing else changed" not in out.text


@pytest.mark.parametrize("origin, says", [("agent", "The drag only brought"), ("user", "The drag may have only brought")])
def test_a_drag_whose_mouse_down_only_activated_the_window_is_not_called_done(desk, monkeypatch, origin, says):
    """The same for a drag on the board: its button press only activated the window. The answer
    doesn't say the app was brought forward first (it wasn't seen to be), nor that anything changed."""
    desk.bring_activates = False
    monkeypatch.setattr(inp, "drag", lambda x1, y1, x2, y2, **kw: desk.click(x1, y1))
    monkeypatch.setattr(ax, "window_words", lambda h: "", raising=False)
    ctx = ToolContext(user_text="play on the board", origin=origin)
    out = asyncio.run(registry().invoke("drag", {"x1": BOARD[0], "y1": BOARD[1], "x2": 300, "y2": 250}, ctx))
    text = out.text if isinstance(out, ToolOutput) else str(out)
    assert desk.clicks == []
    assert "brought python to the front first" not in text
    assert f"{says} python to the front" in text and "Changed" not in text
    assert "before dragging again" in text


def test_a_first_click_that_did_register_is_reported_as_its_change(desk):
    desk.bring_activates, desk.first_mouse = False, True  # the view takes the activating click
    out = click(BOARD)
    assert desk.clicks == [BOARD]
    assert out.ok and "the screen changed around (255, 186)" in out.text and "only brought" not in out.text


def test_the_users_own_click_says_so_too_though_nothing_is_looked_at(desk):
    """A click the user asked for (the self-test's are) gets no before-and-after look — but a click
    that itself brought its app forward still says it may only have done that."""
    desk.bring_activates = False
    out = click(BOARD, origin="user")
    text = out.text if isinstance(out, ToolOutput) else str(out)
    assert text.startswith("Left-clicked at (255, 186) in python") and "may have only brought python to the front" in text
    assert "nothing else changed" not in text  # (nothing was looked at)


def test_a_users_own_click_that_did_not_wake_anything_is_said_as_before(desk):
    desk._activate()
    out = click(BOARD, origin="user")
    assert out.text == "Left-clicked at (255, 186) in python."


# ---- the window server's and Accessibility's word ---------------------------------------------------------


def test_the_window_server_names_every_window_under_a_point_with_its_layer(monkeypatch):
    import Quartz

    listed = [
        {"kCGWindowOwnerPID": 1, "kCGWindowOwnerName": "Window Server", "kCGWindowLayer": 24,
         "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 1512, "Height": 25}},  # the menu bar
        {"kCGWindowOwnerPID": 9, "kCGWindowOwnerName": "Ghost", "kCGWindowLayer": 3, "kCGWindowAlpha": 0,
         "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 1512, "Height": 982}},  # invisible: not there
        {"kCGWindowOwnerPID": TEST, "kCGWindowOwnerName": "python", "kCGWindowLayer": 0,
         "kCGWindowBounds": {"X": 40, "Y": 73, "Width": 720, "Height": 358}},
        {"kCGWindowOwnerPID": 7, "kCGWindowOwnerName": "NoLayer",
         "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 1512, "Height": 982}},  # no layer given: not a normal one
    ]  # fmt: skip
    monkeypatch.setattr(Quartz, "CGWindowListCopyWindowInfo", lambda opts, wid: listed)
    got = _REAL["windows_at"](255.0, 186.0)
    assert [(w.pid, w.layer) for w in got] == [(TEST, 0), (7, 1)]
    assert got[0].frame == (40.0, 73.0, 720.0, 358.0)
    assert [(w.pid, w.layer) for w in _REAL["windows_at"](10.0, 10.0)] == [(1, 24), (7, 1)]
    monkeypatch.setattr(Quartz, "CGWindowListCopyWindowInfo", lambda opts, wid: None)
    assert _REAL["windows_at"](255.0, 186.0) is None  # it couldn't say


def test_a_window_is_key_only_while_its_app_is_active_and_a_sheet_counts_as_its_window(monkeypatch):
    attrs = {
        ("app", "AXFrontmost"): True,
        ("app", "AXFocusedWindow"): "win",
        ("sheet", "AXRole"): "AXSheet",
        ("sheet", "AXParent"): "win",
        ("win", "AXRole"): "AXWindow",
        ("other", "AXRole"): "AXWindow",
    }
    monkeypatch.setattr(ax, "_app_el", lambda pid: "app")
    monkeypatch.setattr(ax, "_attr", lambda el, name, default=None: attrs.get((el, name), default))
    monkeypatch.setattr(ax, "key_window", _REAL["key_window"])
    is_key = _REAL["is_key"]
    assert is_key(ax.Hit(pid=TEST, window="win")) and is_key(ax.Hit(pid=TEST, window="sheet"))
    assert not is_key(ax.Hit(pid=TEST, window="other")) and not is_key(ax.Hit(pid=TEST))
    attrs[("app", "AXFrontmost")] = False  # frontmost by LaunchServices' word, perhaps — not active by its own
    assert ax.key_window(TEST) is None and not is_key(ax.Hit(pid=TEST, window="win"))


def test_two_points_are_in_the_same_window_when_one_is_on_a_sheet_of_it(monkeypatch):
    attrs = {("sheet", "AXRole"): "AXSheet", ("sheet", "AXParent"): "win", ("win", "AXRole"): "AXWindow", ("other", "AXRole"): "AXWindow"}
    monkeypatch.setattr(ax, "_attr", lambda el, name, default=None: attrs.get((el, name), default))
    assert ax.same_window(ax.Hit(pid=TEST, window="win"), ax.Hit(pid=TEST, window="sheet")) is True
    assert ax.same_window(ax.Hit(pid=TEST, window="win"), ax.Hit(pid=TEST, window="other")) is False
    assert ax.same_window(ax.Hit(pid=TEST, window="win"), ax.Hit(pid=TEST)) is None  # (nothing to tell by)
