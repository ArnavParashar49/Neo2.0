"""The Accessibility side of the computer controls: ids that never silently mean another element,
trees that show what's in view, presses and value sets that do only what they say, and NEO's own
window off-limits. Hermetic: ApplicationServices is a fake tree of elements."""

from __future__ import annotations

import asyncio
import types

import pytest

import neo.tools.computer as comp
from neo.agent import confirm
from neo.agent.registry import ToolContext, registry
from neo.tools.computer import apps, ax, focus, pointer, policy

OK, NO_VALUE, UNSUPPORTED, CANNOT_COMPLETE = 0, -25212, -25205, -25204
_REAL_HIT = ax.hit  # (the suite's conftest swaps it for "nothing there" in every test)
_REAL_HOSTS_SHELL = focus.hosts_shell  # (and this for "no shell")
_REAL_FOCUSED_TEXT = ax.focused_element_text  # (and this for "no text")


class El:
    def __init__(self, role, title="", value=None, frame=(0, 0, 0, 0), actions=(), children=(), pid=0, **extra):
        self.attrs = {"AXRole": role, "AXEnabled": True, "AXFocused": False, **extra}
        if title:
            self.attrs["AXTitle"] = title
        if value is not None:
            self.attrs["AXValue"] = value
        self.frame(*frame)
        self.children, self.actions, self.pid, self.parent = list(children), list(actions), pid, None
        self.hung, self.ignore_set, self.act_err = False, False, OK
        for c in self.children:
            c.parent = self

    def frame(self, x, y, w, h):
        self.attrs["AXPosition"] = types.SimpleNamespace(x=x, y=y)
        self.attrs["AXSize"] = types.SimpleNamespace(width=w, height=h)

    def set_pid(self, pid):
        self.pid = pid
        for c in self.children:
            c.set_pid(pid)
        return self


class FakeAS:
    kAXRoleAttribute, kAXSubroleAttribute, kAXTitleAttribute = "AXRole", "AXSubrole", "AXTitle"
    kAXDescriptionAttribute, kAXValueAttribute = "AXDescription", "AXValue"
    kAXPositionAttribute, kAXSizeAttribute, kAXEnabledAttribute = "AXPosition", "AXSize", "AXEnabled"
    kAXFocusedAttribute, kAXChildrenAttribute = "AXFocused", "AXChildren"
    kAXFocusedWindowAttribute, kAXMainWindowAttribute = "AXFocusedWindow", "AXMainWindow"
    kAXFocusedUIElementAttribute = "AXFocusedUIElement"
    kAXValueCGPointType, kAXValueCGSizeType = 1, 2

    def __init__(self):
        self.apps: dict[int, El] = {}
        self.performed: list = []
        self.sets: list = []
        self.timeouts: list = []
        self.system = El("AXSystemWide")

    def add(self, pid, root):
        self.apps[pid] = root.set_pid(pid)
        return root

    def AXIsProcessTrusted(self):
        return True

    def AXUIElementCreateApplication(self, pid):
        return self.apps.get(pid) or El("AXApplication", pid=pid)

    def AXUIElementCreateSystemWide(self):
        return self.system

    def AXUIElementSetMessagingTimeout(self, el, t):
        self.timeouts.append((el, t))
        return OK

    def AXUIElementCopyAttributeValue(self, el, name, _):
        if el.hung:
            return CANNOT_COMPLETE, None
        if name == "AXChildren":
            return (OK, list(el.children)) if el.children else (NO_VALUE, None)
        if name == "AXParent":
            return (OK, el.parent) if el.parent is not None else (NO_VALUE, None)
        if name == "AXFocusedWindow" and "AXFocusedWindow" not in el.attrs:
            win = next((c for c in el.children if c.attrs["AXRole"] == "AXWindow"), None)
            return (OK, win) if win is not None else (NO_VALUE, None)
        v = el.attrs.get(name)
        return (OK, v) if v is not None else (UNSUPPORTED, None)

    def AXValueGetValue(self, v, kind, _):
        if isinstance(v, str):
            raise TypeError("not an AXValue")  # a Java/Qt oddity: a string where a point should be
        return True, v

    def AXUIElementCopyActionNames(self, el, _):
        return OK, list(el.actions)

    def AXUIElementPerformAction(self, el, action):
        self.performed.append((el, action))
        return el.act_err

    def AXUIElementSetAttributeValue(self, el, name, value):
        self.sets.append((el, name, value))
        if not el.ignore_set:
            el.attrs[name] = value
        return OK

    def AXUIElementIsAttributeSettable(self, el, name, _):
        return OK, True

    def AXUIElementGetPid(self, el, _):
        return OK, el.pid

    def AXUIElementCopyElementAtPosition(self, el, x, y, _):
        self.asked_at = (el, x, y)
        return (OK, self.at_point) if getattr(self, "at_point", None) is not None else (NO_VALUE, None)


@pytest.fixture
def fake(monkeypatch):
    f = FakeAS()
    monkeypatch.setattr(ax, "AS", f)
    names: dict[int, str] = {}
    f.names = names

    def add(name, pid, root):
        names[pid] = name
        return f.add(pid, root)

    f.add_app = add
    monkeypatch.setattr(ax, "pid_for_app", lambda n: next((p for p, x in names.items() if x.lower() == n.lower()), None))
    monkeypatch.setattr(ax, "working_app", lambda: next(((x, p) for p, x in names.items()), ("", 0)))
    monkeypatch.setattr(ax, "frontmost_app", lambda: next(((x, p) for p, x in names.items()), ("", 0)))
    monkeypatch.setattr(ax, "app_of", lambda el: names.get(getattr(el, "pid", 0), ""))
    monkeypatch.setattr(ax, "_expose_web", lambda pid, root: False)
    monkeypatch.setattr(ax, "_last_tree", {})
    monkeypatch.setattr(ax, "_issued", {})
    monkeypatch.setattr(ax, "_last_tree_app", "")
    monkeypatch.setattr(ax, "_last_tree_pid", 0)
    monkeypatch.setattr(ax, "_last_tree_ts", 0.0)
    return f


def window(title, *kids, frame=(0, 25, 1512, 957)):
    return El("AXWindow", title, frame=frame, children=kids)


def button(title, frame=(10, 60, 80, 24), actions=("AXPress",), **extra):
    return El("AXButton", title, frame=frame, actions=actions, **extra)


def _ids(tree: str) -> dict[str, str]:
    """title → id in a printed tree."""
    out = {}
    for line in tree.splitlines()[1:]:
        bits = line.split()
        if '"' in line:
            out[line.split('"')[1]] = bits[0]
    return out


# ---- ids -----------------------------------------------------------------------------------------------


def test_an_id_from_an_earlier_look_still_means_its_own_element(fake):
    reply, shuffle = button("Reply"), button("Shuffle")
    fake.add_app("Mail", 1, El("AXApplication", children=[window("Inbox", reply)]))
    fake.add_app("Music", 2, El("AXApplication", children=[window("Music", shuffle)]))
    mail = _ids(ax.snapshot("Mail"))
    music = _ids(ax.snapshot("Music"))  # a second look (another app, a parallel call) …
    assert set(mail.values()).isdisjoint(music.values())  # … hands out new ids,
    assert ax.get(mail["Reply"]).ref is reply  # and Mail's "Reply" is still Reply, never Shuffle


def test_a_slow_walk_that_finishes_late_never_replaces_the_newer_tree(fake, monkeypatch):
    fake.add_app("Busy", 1, El("AXApplication", children=[window("Busy", button("Delete All"))]))
    fake.add_app("Fine", 2, El("AXApplication", children=[window("Fine", button("Save"))]))
    real_walk = ax._walk
    order: list[str] = []

    def slow_first(el, *a, **k):
        if el.attrs.get("AXTitle") == "Busy" and not order:
            order.append("busy")
            ax.snapshot("Fine")  # the newer look finishes while the slow one is still walking
        return real_walk(el, *a, **k)

    monkeypatch.setattr(ax, "_walk", slow_first)
    busy = _ids(ax.snapshot("Busy"))
    assert ax._last_tree_app == "fine" and "Delete All" not in {e.title for e in ax._last_tree.values()}
    assert ax.get(busy["Delete All"]).title == "Delete All"  # its own ids still resolve to its own elements


def test_find_for_an_app_that_is_not_running_says_so(fake):
    fake.add_app("Music", 2, El("AXApplication", children=[window("Music", button("Shuffle"))]))
    ax.snapshot("Music")
    out = ax.find("reply", app="Mail")
    assert out.startswith("Error") and "not running" in out and "Shuffle" not in out


# ---- what a tree shows -------------------------------------------------------------------------------------


def test_the_tree_shows_the_focused_window_first_and_only_what_is_in_view(fake):
    rows = [El("AXRow", f"Row {i}", frame=(10, 100 + 30 * i, 300, 28)) for i in range(40)]
    scroll = El("AXScrollArea", frame=(0, 90, 400, 300), children=rows)  # 10 rows fit in view
    main = window("Doc", scroll, button("Place order", frame=(500, 800, 100, 30)))
    side = window("Palette", *[button(f"Tool {i}") for i in range(5)], frame=(1200, 25, 300, 600))
    fake.add_app("Shop", 1, El("AXApplication", children=[side, main], AXFocusedWindow=main))
    tree = ax.snapshot("Shop")
    lines = tree.splitlines()
    assert lines[1].split()[1] == "Window" and '"Doc"' in lines[1]  # the focused window first
    shown = [x for x in lines if "Row " in x]
    assert 9 <= len(shown) <= 11 and "Row 30" not in tree  # rows scrolled out of view aren't listed
    assert "Place order" in tree and tree.count('Window "Doc"') == 1  # the window isn't walked twice


def test_an_app_whose_frames_dont_add_up_is_still_shown(fake):
    kids = [button(f"Tool {i}", frame=(5000 + i, 5000, 40, 20)) for i in range(3)]  # all "outside" the window
    fake.add_app("Odd", 1, El("AXApplication", children=[window("Odd", *kids)]))
    tree = ax.snapshot("Odd")
    assert all(f"Tool {i}" in tree for i in range(3))


def test_a_closed_sheet_left_in_the_tree_blocks_nothing(fake, monkeypatch):
    closed = El("AXSheet", frame=(0, 0, 0, 0), children=[button("Start")])
    board = El("AXGroup", frame=(0, 60, 1512, 900))
    win = window("Chess", board, closed)
    fake.add_app("Chess", 42, El("AXApplication", children=[win]))
    board.attrs["AXWindow"] = win
    fake.AXUIElementCopyElementAtPosition = lambda sys, x, y, _: (OK, board)
    monkeypatch.setattr(ax, "_running", lambda pid: None)
    assert _REAL_HIT(300.0, 500.0).sheet == ""


def test_one_odd_element_never_fails_the_whole_tree(fake):
    odd = button("Odd")
    odd.attrs["AXPosition"] = "not a point"
    fake.add_app("Java", 1, El("AXApplication", children=[window("W", odd, button("Fine"))]))
    tree = ax.snapshot("Java")
    assert not tree.startswith("Error") and "Fine" in tree


def test_closed_menus_are_not_listed_and_menu_bar_items_are(fake):
    bar = El("AXMenuBar", frame=(0, 0, 1512, 24), children=[
        El("AXMenuBarItem", "Game", frame=(60, 0, 50, 24), actions=["AXPress"], children=[El("AXMenu", frame=(0, 0, 0, 0))]),
    ])  # fmt: skip
    fake.add_app("Chess", 1, El("AXApplication", children=[window("Chess"), bar]))
    tree = ax.snapshot("Chess")
    assert "MenuBarItem \"Game\"" in tree and "Menu @(0,0 0x0)" not in tree


def test_an_open_menu_below_its_window_is_listed_and_counts_as_on_screen(fake):
    item = El("AXMenuItem", "Large", frame=(100, 950, 120, 20), actions=["AXPress"])
    menu = El("AXMenu", frame=(100, 900, 140, 80), children=[item])
    popup = El("AXPopUpButton", "Size", frame=(100, 880, 120, 20), actions=["AXPress"], children=[menu])
    fake.add_app("Pages", 1, El("AXApplication", children=[window("Doc", popup, frame=(0, 25, 800, 880))]))
    tree = ax.snapshot("Pages")
    assert '"Large"' in tree and ax.visible_frame(item) == (100, 950, 120, 20)


def test_typing_that_never_reached_the_field_is_not_reported_as_typed(monkeypatch):
    events = _typing(monkeypatch, focus_takes=True, visible=(300, 100, 800, 800))
    monkeypatch.setattr(ax, "focused_element_text", lambda: "Dear Sam,")  # the same before and after
    out = asyncio.run(registry().invoke("type_text", {"text": "hello"}, ToolContext()))
    assert not out.ok and "couldn't confirm" in out.text and ("type", "hello") in events


def test_a_hung_app_is_asked_briefly_and_said_to_be_busy(fake):
    root = fake.add_app("Stuck", 1, El("AXApplication", children=[window("W")]))
    root.hung = True
    out = ax.snapshot("Stuck")
    assert out.startswith("Error") and "isn't responding" in out
    assert fake.timeouts and fake.timeouts[0][1] <= 2.0


def test_a_big_tree_keeps_what_can_be_acted_on_within_what_the_model_is_shown(fake, monkeypatch):
    monkeypatch.setattr(ax, "_lines_budget", lambda: 1500)
    texts = [El("AXStaticText", value=f"paragraph {i} " + "words " * 8, frame=(10, 60 + i, 400, 1)) for i in range(60)]
    fake.add_app("Page", 1, El("AXApplication", children=[window("Page", *texts, button("Buy now", frame=(10, 900, 90, 30)))]))
    tree = ax.snapshot("Page")
    assert "Buy now" in tree and len(tree) <= 1700 and "more elements" in tree


# ---- pressing and setting --------------------------------------------------------------------------------------


def _one(fake, el, name="Chat", pid=3):
    fake.add_app(name, pid, El("AXApplication", children=[window(name, el)]))
    return _ids(ax.snapshot(name))


def test_press_never_performs_a_non_press_action(fake):
    field = El("AXTextArea", "Message", frame=(10, 60, 300, 40), actions=["AXConfirm", "AXShowMenu"])
    eid = _one(fake, field)["Message"]
    out = asyncio.run(registry().invoke("ax_press", {"id": eid}, ToolContext()))
    assert not out.ok and "can't be pressed" in out.text and fake.performed == []  # never AXConfirm (a send)


def test_a_press_the_app_was_too_busy_to_confirm_is_not_called_a_failure(fake):
    b = button("Add")
    b.act_err = CANNOT_COMPLETE
    eid = _one(fake, b)["Add"]
    out = asyncio.run(registry().invoke("ax_press", {"id": eid}, ToolContext()))
    assert out.ok and "didn't answer in time" in out.text and len(fake.performed) == 1


def test_press_acts_on_the_element_it_asked_about(fake, monkeypatch):
    keep, delete = button("Keep"), button("Delete All")
    eid = _one(fake, keep)["Keep"]

    async def another_job_looks_meanwhile(*a, **k):
        ax._last_tree[eid] = ax.Element(eid, "AXButton", "Delete All", "", 0, 0, 1, 1, actions=["AXPress"], ref=delete)
        return None

    monkeypatch.setattr(comp, "_ask_first", another_job_looks_meanwhile)
    asyncio.run(registry().invoke("ax_press", {"id": eid}, ToolContext()))
    assert [a for _el, a in fake.performed] == ["AXPress"] and fake.performed[0][0] is keep


def test_neos_own_buttons_are_never_pressed_or_filled(fake, monkeypatch):
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": name == "NEO")
    ids = _one(fake, button("Hand back to NEO"), name="NEO", pid=9)
    assert not asyncio.run(registry().invoke("ax_press", {"id": ids["Hand back to NEO"]}, ToolContext())).ok
    field = El("AXTextField", "Ask NEO", frame=(10, 60, 300, 24))
    ids = _one(fake, field, name="NEO", pid=9)
    out = asyncio.run(registry().invoke("ax_set_value", {"id": ids["Ask NEO"], "value": "hi"}, ToolContext()))
    assert not out.ok and fake.performed == [] and fake.sets == []


def test_set_value_never_fills_a_password_field_or_flips_a_checkbox(fake):
    pw = El("AXTextField", "Password", frame=(10, 60, 200, 24), AXSubrole="AXSecureTextField")
    box = El("AXCheckBox", "Allow full disk access", value=0, frame=(10, 90, 200, 24))
    fake.add_app("Settings", 4, El("AXApplication", children=[window("Settings", pw, box)]))
    ids = _ids(ax.snapshot("Settings"))
    out = asyncio.run(registry().invoke("ax_set_value", {"id": ids["Password"], "value": "hunter2"}, ToolContext()))
    assert not out.ok and "password" in out.text
    out = asyncio.run(registry().invoke("ax_set_value", {"id": ids["Allow full disk access"], "value": "1"}, ToolContext()))
    assert not out.ok and "not a text field" in out.text and fake.sets == []


def test_set_value_reads_back_and_is_honest_when_the_field_ignores_it(fake, monkeypatch):
    from neo.decide import axkit

    field = El("AXTextField", "Search", value="", frame=(10, 60, 200, 24))
    ids = _one(fake, field, name="Web", pid=5)
    monkeypatch.setattr(axkit, "_put", lambda el, text, newline: False)  # set, read back, typed: still empty
    out = asyncio.run(registry().invoke("ax_set_value", {"id": ids["Search"], "value": "shoes"}, ToolContext()))
    assert not out.ok and "didn't take the text" in out.text
    monkeypatch.setattr(axkit, "_put", lambda el, text, newline: True)
    out = asyncio.run(registry().invoke("ax_set_value", {"id": ids["Search"], "value": "shoes"}, ToolContext()))
    assert out.ok and "shoes" in out.text


# ---- click_text after a press that failed ---------------------------------------------------------------------


@pytest.mark.parametrize("err, clicks", [("Error: AX action failed (-25204)", 0), ("Error: AX action failed (-25206)", 1)])
def test_click_text_clicks_with_the_mouse_only_when_the_press_is_unsupported(monkeypatch, err, clicks):
    els = [ax.Element("w1", "AXButton", "Add to cart", "", 200, 200, 80, 24, actions=["AXPress"])]
    monkeypatch.setattr(ax, "window_elements", lambda app=None: ("Shop", (0, 0, 1000, 800), els))
    monkeypatch.setattr(comp.apps, "_target", None)
    monkeypatch.setattr(comp.apps, "front_name", lambda: "Shop")
    monkeypatch.setattr(ax, "press_element", lambda e: err)
    clicked = []
    monkeypatch.setattr(comp.inp, "click", lambda x, y, **kw: clicked.append((x, y)) or "clicked")
    out = asyncio.run(registry().invoke("click_text", {"label": "add to cart"}, ToolContext()))
    assert out.ok and len(clicked) == clicks


# ---- typing: focus without clicking off-screen ----------------------------------------------------------------


def _typing(monkeypatch, focus_takes: bool, visible):
    area = ax.Element("t0", "AXTextArea", "", "long text", 300, 100, 800, 6000, ref=object())
    monkeypatch.setattr(comp.apps, "_target", None)
    monkeypatch.setattr(comp.apps, "front_name", lambda: "Editor")
    monkeypatch.setattr(ax, "frontmost_app", lambda: ("Editor", 8))
    monkeypatch.setattr(ax, "focused_is_secure", lambda: False)
    monkeypatch.setattr(ax, "focused_role", lambda: ("AXTextArea", ""))
    state = {"focused": False}
    monkeypatch.setattr(ax, "focused_editable", lambda: state["focused"])
    monkeypatch.setattr(ax, "main_text_area", lambda app=None: area)

    def focus_el(el):
        state["focused"] = focus_takes
        return focus_takes

    monkeypatch.setattr(ax, "focus_el", focus_el)
    monkeypatch.setattr(ax, "visible_frame", lambda el: visible)
    events: list = []

    def click(x, y, **kw):
        events.append(("click", x, y))
        state["focused"] = True
        return "clicked"

    monkeypatch.setattr(comp.inp, "click", click)
    monkeypatch.setattr(comp.inp, "hotkey", lambda k: events.append(("key", k)) or f"Pressed {k}")
    monkeypatch.setattr(comp.inp, "type_text", lambda s, **kw: events.append(("type", s)) or "Typed")
    monkeypatch.setattr(comp, "_doc_like", lambda name, pid: True)
    return events


def test_typing_into_a_long_document_focuses_it_without_clicking(monkeypatch):
    events = _typing(monkeypatch, focus_takes=True, visible=(300, 100, 800, 800))
    out = asyncio.run(registry().invoke("type_text", {"text": "hello"}, ToolContext()))
    assert out.ok and events == [("key", "cmd+down"), ("type", "hello")]


def test_when_focus_must_be_clicked_it_is_clicked_where_the_text_is_on_screen(monkeypatch):
    events = _typing(monkeypatch, focus_takes=False, visible=(300, 100, 800, 800))
    out = asyncio.run(registry().invoke("type_text", {"text": "hello"}, ToolContext()))
    assert out.ok and events[0] == ("click", 700.0, 500.0)  # not (700, 3100): the document's middle


def test_a_text_area_with_nothing_on_screen_is_not_clicked(monkeypatch):
    events = _typing(monkeypatch, focus_takes=False, visible=None)
    out = asyncio.run(registry().invoke("type_text", {"text": "hello"}, ToolContext()))
    assert not out.ok and events == []


# ---- NEO's own processes, the app behind it, shells, web trees ----------------------------------------------------


def test_neo_is_told_by_its_process_family_and_its_overlay_by_name(monkeypatch):
    from neo.decide import media

    monkeypatch.setattr(media, "_self_family", lambda: frozenset({500, 501}))
    monkeypatch.setattr(media, "responsible_pid", lambda pid: {777: 501}.get(pid, pid))
    assert ax.is_neo(500) and ax.is_neo(777)  # a helper NEO.app is responsible for
    assert ax.is_neo(0, "neo-ui") and ax.is_neo(0, "NEO")
    assert not ax.is_neo(4242, "Chess")


def test_with_neo_in_front_a_look_is_at_the_app_behind_it(monkeypatch):
    import Quartz

    monkeypatch.setattr(ax, "frontmost_app", lambda: ("NEO", 900))
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": pid == 900)
    wins = [
        {"kCGWindowOwnerPID": 900, "kCGWindowOwnerName": "NEO", "kCGWindowLayer": 3},
        {"kCGWindowOwnerPID": 900, "kCGWindowOwnerName": "NEO", "kCGWindowLayer": 0},
        {"kCGWindowOwnerPID": 4242, "kCGWindowOwnerName": "Chess", "kCGWindowLayer": 0},
    ]
    monkeypatch.setattr(Quartz, "CGWindowListCopyWindowInfo", lambda opts, wid: wins)
    assert ax.working_app() == ("Chess", 4242)


def test_an_app_running_a_shell_is_told_by_its_processes(monkeypatch):
    kids = {100: [101], 101: [102], 200: [201], 1: [100, 200]}
    procs = {101: ("login", True), 102: ("zsh", True), 201: ("zsh", False)}  # 201: a script, on no terminal
    monkeypatch.setattr(focus, "_children", lambda pid: kids.get(pid, []))
    monkeypatch.setattr(focus, "_proc", lambda pid: procs.get(pid, ("", False)))
    monkeypatch.setattr(focus, "_shells", lambda: frozenset({"zsh", "bash"}))
    monkeypatch.setattr(focus, "_holds_pty", lambda pid: False)
    real = _REAL_HOSTS_SHELL
    assert real(100) and not real(200) and not real(1)  # (never launchd, the parent of everything)
    monkeypatch.setattr(focus, "_holds_pty", lambda pid: pid in (100, 1))  # a terminal draws its shells' output
    assert real(100) == "terminal" and real(1) == ""


def test_a_chromium_or_electron_apps_web_tree_is_switched_on_while_its_ids_are_used(monkeypatch):
    from neo.decide import axkit

    monkeypatch.setattr(ax, "_web_held", {})
    app = types.SimpleNamespace(bundleURL=lambda: types.SimpleNamespace(path=lambda: "/Applications/Slack.app"))
    monkeypatch.setattr(ax, "_running", lambda pid: app)
    monkeypatch.setattr(axkit, "runtime", lambda path: "electron")
    switched, restored = [], []
    monkeypatch.setattr(axkit, "enable_web_reading", lambda root, rt: switched.append(rt) or [("AXManualAccessibility", False)])
    monkeypatch.setattr(axkit, "restore_web_tree", lambda root, changed: restored.append(changed))
    monkeypatch.setattr(ax, "_WEB_HOLD_S", 60.0)
    root = object()
    assert ax._expose_web(7, root) is True and switched == ["electron"]
    assert ax._expose_web(7, root) is False and switched == ["electron"]  # held: not switched again
    ax._release_web(7)
    assert restored == [[("AXManualAccessibility", False)]] and 7 not in ax._web_held


def test_a_native_app_is_left_alone(monkeypatch):
    from neo.decide import axkit

    monkeypatch.setattr(ax, "_web_held", {})
    app = types.SimpleNamespace(bundleURL=lambda: types.SimpleNamespace(path=lambda: "/System/Applications/Chess.app"))
    monkeypatch.setattr(ax, "_running", lambda pid: app)
    monkeypatch.setattr(axkit, "runtime", lambda path: "native")
    monkeypatch.setattr(axkit, "enable_web_reading", lambda root, rt: pytest.fail("switched a native app"))
    assert ax._expose_web(8, object()) is False


def test_a_dialog_ok_is_read_off_the_event_loop(monkeypatch):
    import threading

    loop_thread = threading.get_ident()
    seen = []

    def window_elements(app=None):
        seen.append(threading.get_ident())
        return "Mail", (0, 0, 10, 10), [ax.Element("w1", "AXStaticText", "", "Are you sure you want to delete 3 messages?", 0, 0, 1, 1)]

    monkeypatch.setattr(ax, "window_elements", window_elements)
    gate = asyncio.run(comp._ask_first("click", {"x": 1, "y": 2}, "OK", "Mail"))
    assert seen and seen[0] != loop_thread and gate and gate.startswith(confirm.NEEDS_CONFIRM)
    confirm.cancel("t")


def test_sheet_words_and_a_point_outside_the_sheet(fake, monkeypatch):
    sheet = El("AXSheet", frame=(500, 60, 400, 200), children=[
        El("AXStaticText", value="Start a New Game?", frame=(520, 80, 200, 20)),
        button("Start", frame=(800, 220, 60, 24)), button("Cancel", frame=(720, 220, 60, 24)),
    ])  # fmt: skip
    board = El("AXGroup", frame=(0, 60, 1512, 900))
    win = window("Chess", board, sheet)
    fake.add_app("Chess", 42, El("AXApplication", children=[win]))
    board.attrs["AXWindow"] = win
    fake.AXUIElementCopyElementAtPosition = lambda sys, x, y, _: (OK, board)
    chess = types.SimpleNamespace(localizedName=lambda: "Chess", activationPolicy=lambda: 0)
    monkeypatch.setattr(ax, "_running", lambda pid: chess if pid == 42 else None)
    h = _REAL_HIT(300.0, 500.0)
    assert h.pid == 42 and "Start a New Game?" in h.sheet and "Start, Cancel" in h.sheet
    fake.AXUIElementCopyElementAtPosition = lambda sys, x, y, _: (OK, board)
    assert h.app == "Chess" and h.regular
    assert _REAL_HIT(600.0, 100.0).sheet == ""  # a point on the sheet itself is fine


def test_an_app_modal_alert_blocks_clicks_on_its_other_windows(fake, monkeypatch):
    board = El("AXGroup", frame=(0, 60, 1512, 900))
    main = window("Chess", board)
    alert = El("AXWindow", "", frame=(500, 300, 400, 150), AXModal=True, children=[
        El("AXStaticText", value="Resign this game?", frame=(520, 320, 200, 20)), button("Resign", frame=(800, 400, 60, 24)),
    ])  # fmt: skip
    fake.add_app("Chess", 42, El("AXApplication", children=[main, alert], AXFocusedWindow=alert))
    board.attrs["AXWindow"] = main
    fake.AXUIElementCopyElementAtPosition = lambda sys, x, y, _: (OK, board)
    monkeypatch.setattr(ax, "_running", lambda pid: None)
    h = _REAL_HIT(100.0, 700.0)
    assert "Resign this game?" in h.sheet and "Resign" in h.sheet


def test_the_real_pointer_pipeline_never_brings_neo_forward(monkeypatch):
    monkeypatch.setattr(ax, "hit", lambda x, y: ax.Hit(pid=900, app="NEO", regular=True))
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": pid == 900)
    monkeypatch.setattr(apps, "bring_pid_front", lambda pid: pytest.fail("brought NEO forward"))
    with pytest.raises(pointer.Refused):
        asyncio.run(pointer.probe([(10.0, 10.0)]))


# ---- where a Return runs a command: the app's own processes, and where the keyboard is ------------------------


def _procs(monkeypatch, kids, procs, pty_holders):
    monkeypatch.setattr(focus, "_children", lambda p: kids.get(p, []))
    monkeypatch.setattr(focus, "_proc", lambda p: procs.get(p, ("", False)))
    monkeypatch.setattr(focus, "_shells", lambda: frozenset({"zsh", "bash"}))
    monkeypatch.setattr(focus, "_holds_pty", lambda p: p in pty_holders)
    monkeypatch.setattr(focus, "hosts_shell", _REAL_HOSTS_SHELL)
    monkeypatch.setattr(comp, "_doc_like", lambda n, p: False)
    monkeypatch.setattr(ax, "focused_role", lambda: ("AXTextArea", ""))
    monkeypatch.setattr(ax, "focused_element_text", lambda: "me@mac ~ % rm -rf old")


def test_a_terminal_whose_shells_run_under_a_server_is_still_a_terminal(monkeypatch):
    # iTerm2 with session restoration (its default) runs every shell under a long-lived server whose
    # parent is launchd, not iTerm2: no shell is iTerm2's descendant. iTerm2 itself holds each
    # pseudo-terminal's master side (the server hands it over), and that is the evidence.
    ITERM, SERVER, LOGIN, ZSH = 100, 200, 201, 202
    _procs(monkeypatch, {ITERM: [], SERVER: [LOGIN], LOGIN: [ZSH]},
           {SERVER: ("iTermServer-3.5", False), LOGIN: ("login", True), ZSH: ("-zsh", True)}, {ITERM})  # fmt: skip
    assert _REAL_HOSTS_SHELL(ITERM) == "terminal"
    assert asyncio.run(comp._line_break("iTerm2", ITERM, "AXTextArea")) == ("", comp._SHELL_JOINED)
    gate = asyncio.run(comp._ask_before_keys({"keys": "return"}, "return", "iTerm2", ToolContext(), ITERM))
    assert gate and gate.startswith(confirm.NEEDS_CONFIRM) and "rm -rf old" in confirm.peek().detail
    confirm.cancel("t")


def test_an_apps_terminal_pane_counts_only_while_the_keyboard_is_in_it(monkeypatch):
    # A chat or editor app with a terminal pane: the shell runs under its helper, which holds the
    # pseudo-terminal; the app itself doesn't. Its chat box is a chat box.
    APP, HELPER, ZSH = 629, 1137, 1138
    _procs(monkeypatch, {APP: [HELPER], HELPER: [ZSH]}, {HELPER: ("App Helper", False), ZSH: ("zsh", True)}, {HELPER})
    assert _REAL_HOSTS_SHELL(APP) == "pane"
    monkeypatch.setattr(ax, "focused_terminal", lambda: False)  # the keyboard is in its chat box
    assert asyncio.run(comp._line_break("App", APP, "AXTextArea")) == ("shift+return", "")
    assert asyncio.run(comp._ask_before_keys({"keys": "return"}, "return", "App", ToolContext(), APP)) is None
    monkeypatch.setattr(ax, "focused_terminal", lambda: True)  # …and now in its terminal pane
    assert asyncio.run(comp._line_break("App", APP, "AXTextArea")) == ("", comp._SHELL_JOINED)
    gate = asyncio.run(comp._ask_before_keys({"keys": "return"}, "return", "App", ToolContext(), APP))
    assert gate and "runs the command" in gate
    confirm.cancel("t")


def test_the_focused_element_says_whether_it_is_a_terminal(fake):
    fake.system.attrs["AXFocusedUIElement"] = El("AXTextArea", AXDescription="Terminal input")
    assert ax.focused_terminal()
    fake.system.attrs["AXFocusedUIElement"] = El("AXTextArea", AXDescription="Write your prompt to Claude")
    assert not ax.focused_terminal()
    del fake.system.attrs["AXFocusedUIElement"]
    assert not ax.focused_terminal()


def test_a_pseudo_terminal_the_app_holds_is_read_from_its_open_files(monkeypatch):
    import ctypes

    files = {100: [(3, 1, "/dev/ptmx"), (4, 2, ""), (5, 1, "/Users/me/notes.txt")], 200: [(3, 1, "/dev/ttys003"), (4, 1, "/dev/null")]}

    class Lib:
        def proc_pidinfo(self, pid, flavor, arg, buf, size):
            got = files.get(pid, [])
            for i, (fd, kind, _path) in enumerate(got):
                buf[i].proc_fd, buf[i].proc_fdtype = fd, kind
            return len(got) * ctypes.sizeof(focus._FDInfo)

        def proc_pidfdinfo(self, pid, fd, flavor, buf, size):
            path = next(p for f, _k, p in files[pid] if f == fd).encode()
            ctypes.memmove(ctypes.addressof(buf) + focus._VNODE_PATH_AT, path, len(path))
            return size

    monkeypatch.setattr(focus, "_proclib", lambda: Lib())
    assert focus._holds_pty(100) and not focus._holds_pty(200) and not focus._holds_pty(300)


def test_system_wide_lookups_answer_within_the_short_timeout(fake):
    fake.system.attrs["AXFocusedUIElement"] = El("AXTextArea", value="hi")
    fake.at_point = El("AXButton", "OK")
    for call in (lambda: ax._element_at(10, 10), ax.focused_role, ax.focused_editable, _REAL_FOCUSED_TEXT, ax.focused_terminal):
        fake.timeouts.clear()
        call()
        assert (fake.system, ax._AX_TIMEOUT_S) in fake.timeouts  # a hung app holds NEO 2 s, not ~6 s


def test_the_words_around_a_drop_spot_are_what_its_part_of_the_window_says(fake):
    # An AirDrop window: its recipients are just names; the text beside them says what a drop does.
    # (Its sidebar comes first in the window and says nothing about it.)
    person = El("AXImage", "Priya’s iPhone")
    intro = El("AXStaticText", value="AirDrop lets you share instantly with people nearby.")
    view = El("AXGroup", children=[intro, El("AXGroup", children=[person])])
    sidebar = El("AXOutline", children=[El("AXRow", children=[El("AXStaticText", value=n)]) for n in ("Recents", "Desktop", "Documents", "Downloads")])
    win = window("AirDrop", sidebar, view)
    fake.add_app("Finder", 3, El("AXApplication", children=[win]))
    said = ax.window_words(ax.Hit(pid=3, app="Finder", label="Priya’s iPhone", el=person, window=win))
    assert "share instantly" in said and "AirDrop" in said
    assert policy.drop_commits("Priya’s iPhone", said)
    body = El("AXTextArea", value="See the attached notes.")
    draft = window("New Message", El("AXStaticText", value="To:"), El("AXStaticText", value="Subject:"), El("AXScrollArea", children=[body]))
    fake.add_app("Mail", 4, El("AXApplication", children=[draft]))
    around = ax.window_words(ax.Hit(pid=4, app="Mail", label="See the attached notes.", el=body, window=draft))
    assert "New Message" in around and not policy.drop_commits("See the attached notes.", around)
    assert policy.drop_commits("Trash")  # onto the Trash: its own name says what a drop there does
