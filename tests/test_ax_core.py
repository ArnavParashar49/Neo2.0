"""ax-core: fast, bounded, truthful Accessibility and input primitives.

Hermetic: ApplicationServices is a fake tree of elements that records every round trip (one per
call, as the real IPC is), time is a fake clock (a hung element costs the timeout that applies to
it), the window server and LaunchServices are fakes, and no event is ever posted — input events are
built and caught before posting."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from neo.tools.computer import apps, ax, axwatch, focus, wait
from neo.tools.computer import input as inp

OK, NO_VALUE, UNSUPPORTED, CANNOT_COMPLETE, INVALID, FAILURE = 0, -25212, -25205, -25204, -25202, -25200
DEFAULT_TIMEOUT_S = 6.0
_REAL_HIT = ax.hit  # (the suite's conftest swaps these for "nothing there" in every test)
_REAL_SCENE = ax.scene
_REAL_RAISE = ax.raise_window
_REAL_FRONTMOST = ax.frontmost_app  # (the fake below swaps it)


# ---- a fake Accessibility layer ---------------------------------------------------------------------------


class Err:
    """What a batched read puts where a value failed (an AXValue of the error type)."""

    __ax_error__ = True

    def __init__(self, code):
        self.code = code


class El:
    def __init__(self, role, title="", value=None, frame=(0, 0, 0, 0), actions=(), children=(), pid=0, **extra):
        self.attrs = {"AXRole": role, "AXEnabled": True, "AXFocused": False, **extra}
        if title:
            self.attrs["AXTitle"] = title
        if value is not None:
            self.attrs["AXValue"] = value
        self.frame(*frame)
        self.children, self.actions, self.pid, self.parent = list(children), list(actions), pid, None
        self.hung = False  # every call waits out the timeout that applies, then -25204
        self.fails = False  # every call answers -25204 at once (an odd element, not a timeout)
        self.no_batch = False  # the batched read answers errors only (some Java/Qt builds)
        self.gone = False  # the element no longer exists
        self.ignore_set = False
        self.act_err = OK
        self.lazy_kids = 0  # AXChildren answers nothing this many times first (a menu built when asked)
        self.settable = {"AXValue", "AXFocused", "AXMain"}
        for c in self.children:
            c.parent = self

    def frame(self, x, y, w, h):
        self.attrs["AXPosition"] = SimpleNamespace(x=x, y=y)
        self.attrs["AXSize"] = SimpleNamespace(width=w, height=h)

    def set_pid(self, pid):
        self.pid = pid
        for c in self.children:
            c.set_pid(pid)
        return self

    def add(self, *kids):
        for c in kids:
            c.parent = self
            c.set_pid(self.pid)
        self.children += kids
        return self

    def __repr__(self):
        return f"<{self.attrs.get('AXRole')} {self.attrs.get('AXTitle') or self.attrs.get('AXValue') or ''}>"


def _window_of(el):
    win = top = None
    p = el
    while p is not None:
        r = p.attrs.get("AXRole")
        if r in ("AXSheet", "AXDrawer") and top is None:
            top = p
        if r == "AXWindow":
            win, top = p, top or p
            break
        p = p.parent
    return win, top


class FakeAS:
    kAXRoleAttribute, kAXSubroleAttribute, kAXTitleAttribute = "AXRole", "AXSubrole", "AXTitle"
    kAXDescriptionAttribute, kAXValueAttribute = "AXDescription", "AXValue"
    kAXPositionAttribute, kAXSizeAttribute, kAXEnabledAttribute = "AXPosition", "AXSize", "AXEnabled"
    kAXFocusedAttribute, kAXChildrenAttribute = "AXFocused", "AXChildren"
    kAXFocusedWindowAttribute, kAXMainWindowAttribute = "AXFocusedWindow", "AXMainWindow"
    kAXFocusedUIElementAttribute = "AXFocusedUIElement"
    kAXValueCGPointType, kAXValueCGSizeType, kAXValueAXErrorType = 1, 2, 5

    def __init__(self, clock=None, latency=0.0, batch=True, sleep_s=0.0):
        self.apps: dict[int, El] = {}
        self.calls: list[tuple] = []
        self.performed: list = []
        self.sets: list = []
        self.timeouts: list = []
        self.system = El("AXSystemWide")
        self.clock = clock
        self.latency = latency
        self.sleep_s = sleep_s  # real seconds per call (for a walk that must take real time)
        self.global_timeout = None
        self.at_point = None
        self.values: dict = {}  # (id(el), name) → callable giving the value now
        self._lock = threading.Lock()
        if not batch:
            self.AXUIElementCopyMultipleAttributeValues = None

    def add(self, pid, root):
        self.apps[pid] = root.set_pid(pid)
        return root

    def _cost(self, el, kind, *what):
        with self._lock:
            self.calls.append((kind, el, *what))
        if self.sleep_s:
            time.sleep(self.sleep_s)
        if self.clock is not None and self.latency:
            self.clock.advance(self.latency)
        if getattr(el, "gone", False):
            return INVALID
        if getattr(el, "hung", False):
            t = getattr(el, "timeout", None) or self.global_timeout or DEFAULT_TIMEOUT_S
            if self.clock is not None:
                self.clock.advance(t)
            return CANNOT_COMPLETE
        if getattr(el, "fails", False):
            return CANNOT_COMPLETE
        return OK

    def count(self, kinds=("get", "multi", "actions", "count", "range", "hit")):
        return sum(1 for c in self.calls if c[0] in kinds)

    def _value(self, el, name):
        dyn = self.values.get((id(el), name))
        if dyn is not None:
            v = dyn()
            return (OK, v) if v is not None else (NO_VALUE, None)
        if name == "AXChildren":
            if el.lazy_kids > 0:
                el.lazy_kids -= 1
                return NO_VALUE, None
            return (OK, list(el.children)) if el.children else (NO_VALUE, None)
        if name == "AXParent":
            return (OK, el.parent) if el.parent is not None else (NO_VALUE, None)
        if name in el.attrs:
            v = el.attrs[name]
            return (OK, v) if v is not None else (NO_VALUE, None)
        if name in ("AXWindow", "AXTopLevelUIElement") and el.attrs.get("AXRole") not in ("AXApplication", "AXSystemWide"):
            win, top = _window_of(el)
            v = win if name == "AXWindow" else top
            return (OK, v) if v is not None else (NO_VALUE, None)
        if name == "AXFocusedWindow" and el.attrs.get("AXRole") == "AXApplication":
            win = next((c for c in el.children if c.attrs.get("AXRole") == "AXWindow"), None)
            return (OK, win) if win is not None else (NO_VALUE, None)
        return UNSUPPORTED, None

    # ---- the API ----
    def AXIsProcessTrusted(self):
        return True

    def AXUIElementCreateApplication(self, pid):
        return self.apps.get(pid) or El("AXApplication", pid=pid)

    def AXUIElementCreateSystemWide(self):
        return self.system

    def AXUIElementSetMessagingTimeout(self, el, t):
        self.timeouts.append((el, t))
        if el is self.system:
            self.global_timeout = t
        else:
            el.timeout = t
        return OK

    def AXUIElementCopyAttributeValue(self, el, name, _):
        err = self._cost(el, "get", name)
        return (err, None) if err else self._value(el, name)

    def AXUIElementCopyMultipleAttributeValues(self, el, names, options, _):
        err = self._cost(el, "multi", tuple(names))
        if err:
            return err, None
        out = []
        for n in names:
            if el.no_batch:
                out.append(Err(FAILURE))
                continue
            e, v = self._value(el, n)
            out.append(v if e == OK else Err(e))
        return OK, out

    def AXUIElementGetAttributeValueCount(self, el, name, _):
        err = self._cost(el, "count", name)
        if err:
            return err, 0
        e, v = self._value(el, name)
        return (OK, len(v)) if e == OK and isinstance(v, list) else (OK, 0)

    def AXValueGetValue(self, v, kind, _):
        if isinstance(v, str):
            raise TypeError("not an AXValue")
        return True, v

    def AXValueGetType(self, v):
        return 5 if isinstance(v, Err) else 1

    def AXUIElementCopyActionNames(self, el, _):
        err = self._cost(el, "actions")
        return (err, None) if err else (OK, list(el.actions))

    def AXUIElementPerformAction(self, el, action):
        self.performed.append((el, action))
        err = self._cost(el, "perform", action)
        return err or el.act_err

    def AXUIElementSetAttributeValue(self, el, name, value):
        self.sets.append((el, name, value))
        err = self._cost(el, "set", name)
        if err:
            return err
        if name not in el.settable:
            return FAILURE
        if not el.ignore_set:
            el.attrs[name] = value
        return OK

    def AXUIElementIsAttributeSettable(self, el, name, _):
        return OK, name in el.settable

    def AXUIElementGetPid(self, el, _):
        return OK, el.pid

    def AXUIElementCopyElementAtPosition(self, el, x, y, _):
        self._cost(el, "hit", x, y)
        at = self.at_point(x, y) if callable(self.at_point) else self.at_point
        return (OK, at) if at is not None else (NO_VALUE, None)


@pytest.fixture
def clock():
    c = wait.FakeClock()
    with wait.use_clock(c):
        yield c


@pytest.fixture
def fake(monkeypatch):
    """A fake AX layer with the apps a test adds (`fake.add_app(name, pid, root)`); the first one
    is in front."""
    f = FakeAS()
    monkeypatch.setattr(ax, "AS", f)
    monkeypatch.setattr(ax, "_timeout_for", None)
    names: dict[int, str] = {}

    def add(name, pid, root):
        names[pid] = name
        return f.add(pid, root)

    f.add_app = add
    f.names = names
    monkeypatch.setattr(ax, "pid_for_app", lambda n: next((p for p, x in names.items() if x.lower() == n.lower()), None))
    monkeypatch.setattr(ax, "working_app", lambda: next(((x, p) for p, x in names.items()), ("", 0)))
    monkeypatch.setattr(ax, "frontmost_app", lambda: next(((x, p) for p, x in names.items()), ("", 0)))
    monkeypatch.setattr(ax, "app_of", lambda el: names.get(getattr(el, "pid", 0), ""))
    monkeypatch.setattr(ax, "_running", lambda pid: SimpleNamespace(localizedName=lambda: names.get(pid, ""), activationPolicy=lambda: 0) if pid in names else None)
    monkeypatch.setattr(ax, "_expose_web", lambda pid, root: False)
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": False)
    for name in ("_last_tree", "_issued", "_cache", "_win_cache"):
        monkeypatch.setattr(ax, name, {})
    monkeypatch.setattr(ax, "_last_tree_app", "")
    monkeypatch.setattr(ax, "_last_tree_pid", 0)
    monkeypatch.setattr(ax, "_last_tree_ts", 0.0)
    monkeypatch.setattr(focus, "front_app", lambda: next(((x, p) for p, x in names.items()), ("", 0)))
    return f


def window(title, *kids, frame=(0, 25, 1512, 957), **extra):
    return El("AXWindow", title, frame=frame, children=kids, **extra)


def button(title, frame=(10, 60, 80, 24), actions=("AXPress",), **extra):
    return El("AXButton", title, frame=frame, actions=actions, **extra)


def app_root(*kids, **extra):
    return El("AXApplication", children=kids, **extra)


def _ids(tree: str) -> dict[str, str]:
    """title → id in a printed tree (element lines and the one-line lists alike)."""
    out = {}
    for line in tree.splitlines():
        for part in line.replace("·", "\n").splitlines():
            bits = part.split()
            for b in bits:
                if b.startswith("e") and b[1:].isdigit() and '"' in part[part.index(b):]:
                    out[part[part.index(b):].split('"')[1]] = b
                    break
    return out


# ---- the synthetic app of the speed research: 2,522 elements --------------------------------------------


def synthetic(rows=40, doc_items=600, menus=10, menu_items=20, bg_items=200):
    """A Finder/Safari-like app: a menu bar (closed menus), a focused window with a toolbar, a
    sidebar list and a long web document (mostly scrolled out of view), plus a background window."""
    mb = El("AXMenuBar", frame=(0, 0, 1512, 24), children=[
        El("AXMenuBarItem", f"Menu{m}", frame=(40 + 60 * m, 0, 50, 24), actions=("AXPress", "AXCancel"), children=[
            El("AXMenu", frame=(0, 0, 0, 0), children=[
                El("AXMenuItem", f"Item {m}.{i}", frame=(0, 0, 0, 0), actions=("AXPress", "AXCancel"))
                for i in range(menu_items)])])
        for m in range(menus)])  # fmt: skip
    toolbar = El("AXToolbar", frame=(120, 70, 1100, 52), children=[
        El("AXButton", f"Tool {i}", frame=(130 + 40 * i, 80, 32, 28), actions=("AXPress",)) for i in range(12)])  # fmt: skip
    side_rows = [El("AXRow", frame=(120, 130 + 22 * i, 220, 22), children=[
        El("AXCell", frame=(120, 130 + 22 * i, 220, 22), children=[
            El("AXImage", frame=(126, 132 + 22 * i, 16, 16), AXDescription="folder"),
            El("AXStaticText", value=f"Folder {i}", frame=(146, 132 + 22 * i, 180, 18))])])
        for i in range(rows)]  # fmt: skip
    sidebar = El("AXScrollArea", frame=(120, 122, 220, 748), children=[
        El("AXOutline", frame=(120, 122, 220, 22 * rows), children=side_rows)])  # fmt: skip
    items = []
    for i in range(doc_items):
        y = 130 + 30 * i
        kids = [El("AXStaticText", value=f"Paragraph {i} text about things", frame=(360, y, 700, 16)),
                El("AXLink", f"link {i}", frame=(360, y + 16, 80, 14), actions=("AXPress",))]  # fmt: skip
        if i % 5 == 0:
            kids.append(El("AXButton", f"Action {i}", frame=(460, y + 14, 60, 16), actions=("AXPress",)))
        items.append(El("AXGroup", frame=(350, y, 800, 30), children=kids))
    web = El("AXWebArea", "Page", frame=(350, 122, 870, 30 * doc_items + 20), children=items)
    content = El("AXScrollArea", frame=(350, 122, 870, 748), children=[web])
    win = El("AXWindow", "Front", frame=(120, 70, 1100, 800), children=[
        toolbar, El("AXSplitGroup", frame=(120, 122, 1100, 748), children=[sidebar, content])],
        AXFocused=True, AXSubrole="AXStandardWindow")  # fmt: skip
    bg = El("AXWindow", "Back", frame=(200, 200, 800, 600), AXSubrole="AXStandardWindow", children=[
        El("AXStaticText", value=f"bg {i}", frame=(210, 210 + 2 * i, 300, 16)) for i in range(bg_items)])  # fmt: skip
    root = El("AXApplication", "Demo", children=[mb, win, bg])
    root.attrs["AXFocusedWindow"] = win
    btn = toolbar.children[3]
    root.attrs["AXFocusedUIElement"] = btn
    return root, win, btn


def _count(el):
    return 1 + sum(_count(c) for c in el.children)


def calls(fake, fn):
    fake.calls.clear()
    out = fn()
    return fake.count(), out


# ---- batched reads ---------------------------------------------------------------------------------------


def test_an_elements_look_is_one_round_trip_and_error_placeholders_are_nothing(fake):
    el = button("Save", AXDescription="Save the file")
    fake.add_app("Doc", 1, app_root(window("Doc", el)))
    n, got = calls(fake, lambda: ax._attrs(el, ("AXRole", "AXTitle", "AXHelp", "AXValue", "AXDescription")))
    assert n == 1 and got == {"AXRole": "AXButton", "AXTitle": "Save", "AXHelp": None, "AXValue": None, "AXDescription": "Save the file"}


def test_a_batch_of_errors_only_is_read_again_one_attribute_at_a_time(fake):
    el = button("Old toolkit")
    el.no_batch = True  # (a Java/Qt build that answers single reads only)
    fake.add_app("Java", 2, app_root(window("J", el)))
    n, got = calls(fake, lambda: ax._attrs(el, ("AXRole", "AXTitle", "AXHelp")))
    assert got == {"AXRole": "AXButton", "AXTitle": "Old toolkit", "AXHelp": None}
    assert n == 1 + 3  # the batch, then each attribute


def test_without_the_batched_call_single_reads_still_work(monkeypatch):
    f = FakeAS(batch=False)
    monkeypatch.setattr(ax, "AS", f)
    el = button("Plain")
    assert ax._attrs(el, ("AXRole", "AXTitle")) == {"AXRole": "AXButton", "AXTitle": "Plain"}


def test_a_gone_element_is_told_apart_from_a_busy_one(fake):
    el = button("Close")
    el.gone = True
    assert ax._attrs_err(el, ("AXRole",))[1] == INVALID
    el.gone, el.fails = False, True
    assert ax._attrs_err(el, ("AXRole",))[1] == CANNOT_COMPLETE


# ---- round trips on the synthetic app ---------------------------------------------------------------------


@pytest.fixture
def demo(fake):
    root, win, btn = synthetic()
    fake.add_app("Demo", 4242, root)
    fake.at_point = btn
    return root, win, btn


def test_ax_tree_on_the_big_app_takes_a_few_hundred_round_trips_not_eight_thousand(fake, demo):
    root, win, _btn = demo
    assert _count(root) == 2522
    n, tree = calls(fake, ax.snapshot)
    assert n <= 1000, n  # 7,879 before (the speed research's count on this app)
    # …and finds the same things: the toolbar, the sidebar rows in view, the paragraphs in view
    said = {e.title or e.value for e in ax._last_tree.values()}
    assert {"Tool 11", "Folder 0", "Folder 30", "Paragraph 0 text about things", "Paragraph 20 text about things"} <= said
    assert "Paragraph 30 text about things" not in said and "Folder 39" not in said  # scrolled out of view
    assert "Tool 11" in tree and "Action 20" in tree  # (what can be acted on is shown first)
    assert tree.count('Window "Back"') == 1 and "bg 5" not in tree  # a background window is one line
    menu = next(line for line in tree.splitlines() if "Menu bar:" in line)
    assert 'MenuBarItem "Menu1"' in menu and "Menu0" not in menu  # (the first item: the system menu)
    assert "Item 1.3" not in tree  # closed menus are never walked


def test_the_other_primitives_take_a_handful_of_round_trips(fake, demo):
    _root, _win, btn = demo
    x, y = btn.attrs["AXPosition"].x + 5, btn.attrs["AXPosition"].y + 5
    n, (name, frame, els) = calls(fake, ax.window_elements)
    assert n <= 600 and any(e.title == "Tool 3" for e in els), n  # 4,072 before
    assert calls(fake, lambda: _REAL_HIT(x, y))[0] <= 7  # 13 before
    assert calls(fake, lambda: _REAL_SCENE(4242))[0] <= 5  # 9 before
    assert calls(fake, lambda: _REAL_SCENE(4242, front=("Demo", 4242)))[0] <= 5
    assert calls(fake, lambda: ax.label_at(x, y))[0] <= 3  # 7 before


def test_action_names_are_asked_only_of_what_is_pressed(fake, demo):
    fake.calls.clear()
    tree = ax.snapshot()
    assert fake.count(("actions",)) == 0
    e = ax.get(_ids(tree)["Tool 2"])
    assert ax.press_el(e).startswith("Pressed") and fake.count(("actions",)) == 1
    assert fake.performed[-1] == (e.ref, "AXPress")


def test_a_long_list_is_entered_where_it_is_on_screen(fake):
    rows = [El("AXGroup", frame=(0, 100 + 20 * i, 400, 20), children=[El("AXStaticText", value=f"line {i}", frame=(0, 100 + 20 * i, 400, 20))]) for i in range(500)]
    area = El("AXScrollArea", frame=(0, 100, 400, 200), children=[El("AXGroup", frame=(0, 100, 400, 10000), children=rows)])
    fake.add_app("Log", 3, app_root(window("Log", area)))
    n, tree = calls(fake, ax.snapshot)
    shown = [i for i in range(500) if f"'line {i}'" in tree]
    assert shown and min(shown) == 0 and 9 <= max(shown) <= 11  # exactly what's in view…
    assert n < 120  # …found by halving: not 1,000 reads


def test_a_list_not_laid_out_top_to_bottom_is_walked_whole(fake):
    rows = [El("AXStaticText", value=f"cell {i}", frame=(0, 100 + (i * 37 % 50) * 5, 100, 5)) for i in range(60)]
    area = El("AXScrollArea", frame=(0, 100, 400, 400), children=[El("AXGroup", frame=(0, 100, 400, 400), children=rows)])
    fake.add_app("Grid", 3, app_root(window("Grid", area)))
    tree = ax.snapshot()
    assert all(f"'cell {i}'" in tree for i in range(60))  # nothing lost to a wrong guess


def test_a_table_says_which_rows_are_visible(fake):
    rows = [El("AXRow", f"Row {i}", frame=(0, 100 + 20 * i, 300, 20)) for i in range(200)]
    table = El("AXTable", frame=(0, 100, 300, 4000), children=rows, AXVisibleRows=rows[3:9])
    fake.add_app("Mail", 4, app_root(window("Inbox", El("AXScrollArea", frame=(0, 100, 300, 400), children=[table]))))
    tree = ax.snapshot()
    assert '"Row 3"' in tree and '"Row 8"' in tree and '"Row 2"' not in tree and '"Row 9"' not in tree


# ---- timeouts, deadlines, cancel ------------------------------------------------------------------------------


def test_one_timeout_for_the_whole_process_set_once_and_none_per_element(fake, demo):
    for _ in range(3):
        ax.snapshot()
    ax.window_elements()
    assert fake.timeouts == [(fake.system, ax._AX_TIMEOUT_S)]  # once, on the system-wide element
    assert ax._AX_TIMEOUT_S == 1.5


def test_in_a_host_process_the_timeout_is_never_process_wide(fake, demo, monkeypatch):
    from neo import runtime

    monkeypatch.setattr(runtime, "_library", True)
    ax.snapshot("Demo")
    assert fake.global_timeout is None and all(el is not fake.system for el, _t in fake.timeouts)
    assert any(el is fake.apps[4242] for el, _t in fake.timeouts)  # the app's element has its own


def test_a_pressed_element_gets_a_longer_timeout_of_its_own(fake):
    b = button("Export")
    ids = _ids((fake.add_app("Pages", 5, app_root(window("Doc", b))), ax.snapshot())[1])
    ax.press(ids["Export"])
    assert (b, ax._PRESS_TIMEOUT_S) in fake.timeouts


def _hung_app(fake, clock, hung_at=(5,), n=40):
    kids = [button(f"B{i}", frame=(10 + 20 * i, 60, 18, 18)) for i in range(n)]
    for i in hung_at:
        kids[i].hung = True
    fake.clock, fake.latency = clock, 0.0002
    fake.add_app("Stuck", 7, app_root(window("Stuck", *kids)))


def test_a_child_that_times_out_ends_the_walk_with_what_was_read(fake, clock):
    _hung_app(fake, clock, hung_at=(5, 6, 7))
    t0 = clock.now()
    tree = ax.snapshot("Stuck")
    assert clock.now() - t0 <= 1.6  # one timeout (1.5 s), not 3 elements × several reads × 6 s
    assert '"B4"' in tree and '"B5"' not in tree and '"B30"' not in tree
    assert "isn't answering" in tree


def test_an_odd_element_that_refuses_at_once_is_not_a_hung_app(fake, clock):
    kids = [button(f"B{i}", frame=(10 + 20 * i, 60, 18, 18)) for i in range(20)]
    kids[3].fails = True  # answers -25204 immediately: an odd element, not a timeout
    fake.add_app("Odd", 8, app_root(window("Odd", *kids)))
    tree = ax.snapshot("Odd")
    assert '"B19"' in tree and "isn't answering" not in tree
    for i in (5, 6, 7):
        kids[i].fails = True  # …but several in a row: the app has stopped answering
    tree = ax.snapshot("Odd")
    assert "isn't answering" in tree and '"B19"' not in tree


def test_a_walk_past_its_deadline_is_partial_and_says_so(fake, clock, demo):
    fake.clock, fake.latency = clock, 0.01  # a slow app: 10 ms a question
    t0 = clock.now()
    tree = ax.snapshot(deadline_s=1.5)
    assert clock.now() - t0 <= 1.6
    assert "stopped after 1.5 s" in tree and tree.startswith("App: Demo")


def test_a_cancelled_look_stops_its_worker_thread(fake):
    root, _win, _btn = synthetic(doc_items=600)
    fake.add_app("Demo", 4242, root)
    fake.sleep_s = 0.005  # 5 ms a question in real time: the whole walk would take ~1.5 s

    async def look():
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(ax.tree(), 0.05)

    t0 = time.monotonic()
    asyncio.run(look())  # (which waits for its worker threads to end)
    assert time.monotonic() - t0 < 0.4  # nobody waits for it: it stopped, not a zombie walk
    n = len(fake.calls)
    time.sleep(0.05)
    assert len(fake.calls) == n and n < 60


def test_the_tool_call_runs_the_look_off_the_event_loop(fake, demo):
    seen = []
    real = ax.snapshot

    def spy(*a, **k):
        seen.append(threading.get_ident())
        return real(*a, **k)

    ax_snapshot = ax.snapshot
    try:
        ax.snapshot = spy
        asyncio.run(ax.tree())
    finally:
        ax.snapshot = ax_snapshot
    assert seen and seen[0] != threading.get_ident()


# ---- the front app ----------------------------------------------------------------------------------------------


@pytest.fixture
def front_world(monkeypatch, fake):
    """The system-wide focused app, the window server's front window, and lsappinfo — all fakes."""
    world = SimpleNamespace(ax_pid=10, cg=("Pages", 10), ls=("Pages", 10), runs=[])
    fake.values[(id(fake.system), "AXFocusedApplication")] = lambda: El("AXApplication", pid=world.ax_pid) if world.ax_pid else None
    monkeypatch.setattr(focus, "_cg_front", lambda: world.cg)
    monkeypatch.setattr(focus, "_app_name", lambda pid: {10: "Pages", 11: "Keynote"}.get(pid, ""))

    def run(cmd, **kw):
        world.runs.append(cmd)
        if cmd[:2] == ["lsappinfo", "front"]:
            return SimpleNamespace(stdout="ASN:0x0-0x1:\n")
        return SimpleNamespace(stdout=f'"pid"={world.ls[1]}\n"LSDisplayName"="{world.ls[0]}"\n')

    monkeypatch.setattr(focus.subprocess, "run", run)
    monkeypatch.setattr(focus, "_brought_at", -1e9)
    return world


_REAL_FRONT_APP = focus.front_app


def test_the_front_app_is_one_accessibility_question_when_the_window_server_agrees(fake, front_world):
    fake.calls.clear()
    assert _REAL_FRONT_APP() == ("Pages", 10)
    assert fake.count() == 1 and front_world.runs == []  # no lsappinfo


def test_lsappinfo_decides_when_they_disagree_and_right_after_a_bring(fake, front_world):
    front_world.cg, front_world.ls = ("Keynote", 11), ("Keynote", 11)
    assert _REAL_FRONT_APP() == ("Keynote", 11) and front_world.runs  # they disagreed: lsappinfo
    front_world.runs.clear()
    front_world.cg = ("Pages", 10)
    front_world.ls = ("Pages", 10)
    focus.note_bring()
    assert _REAL_FRONT_APP() == ("Pages", 10) and front_world.runs  # just brought: lsappinfo
    front_world.ax_pid = 0  # a locked screen: Accessibility can't say
    focus._brought_at = -1e9
    front_world.runs.clear()
    assert _REAL_FRONT_APP() == ("Pages", 10) and front_world.runs


def test_a_known_front_app_saves_the_question(fake, demo, monkeypatch):
    monkeypatch.setattr(focus, "front_app", lambda: pytest.fail("asked again"))
    s = _REAL_SCENE(4242, front=("Demo", 4242))
    assert s.known and s.front_pid == 4242 and s.title == "Front"


def test_pid_lookups_are_kept_a_moment_while_the_app_still_runs(monkeypatch):
    asked = []
    monkeypatch.setattr(ax, "_pid_cache", {})
    monkeypatch.setattr(focus, "pid_running", lambda n: asked.append(n) or [("Notes", 77)])

    class Running:
        def __init__(self, pid):
            self.pid = pid

        def localizedName(self):
            return "Notes"

        def activationPolicy(self):
            return 0

        def processIdentifier(self):
            return self.pid

        def isTerminated(self):
            return self.pid in dead

    dead: set[int] = set()
    import AppKit

    monkeypatch.setattr(AppKit, "NSRunningApplication", SimpleNamespace(runningApplicationWithProcessIdentifier_=Running))
    assert ax.pid_for_app("Notes") == 77 and ax.pid_for_app("notes") == 77 and asked == ["notes"]
    dead.add(77)  # it quit: asked afresh
    ax.pid_for_app("Notes")
    assert len(asked) == 2


# ---- polls instead of fixed sleeps --------------------------------------------------------------------------


def test_until_answers_at_the_first_change_and_honestly_at_the_deadline(clock):
    t0 = clock.now()
    got = asyncio.run(wait.until(lambda: clock.now() - t0 >= 0.1 and "yes", 1.0, every=0.02))
    assert got == "yes" and 0.1 <= clock.now() - t0 < 0.13
    t0 = clock.now()
    asked = []
    assert asyncio.run(wait.until(lambda: asked.append(clock.now()) or False, 0.1, every=0.03)) is False
    assert abs(clock.now() - t0 - 0.1) < 1e-9 and asked[-1] - t0 == pytest.approx(0.1)  # asked at the deadline too
    assert wait.until_sync(lambda: 0, 0.0) == 0 and wait.until_sync(lambda: 5, 3.0) == 5


def test_bringing_an_app_forward_polls_from_the_start_and_skips_one_already_in_front(clock, monkeypatch):
    state = {"at": None, "forced": []}

    def front():
        return ("Target", 99) if state["at"] is not None and clock.now() >= state["at"] else ("Other", 1)

    def force(pid):
        state["forced"].append(clock.now())
        state["at"] = clock.now() + 0.03  # it comes forward 30 ms later
        return True

    monkeypatch.setattr(focus, "front_app", front)
    monkeypatch.setattr(focus, "force_front", force)
    target = SimpleNamespace(localizedName=lambda: "Target", processIdentifier=lambda: 99)
    t0 = clock.now()
    assert asyncio.run(apps._raise(target)) == (True, "Target")
    assert state["forced"] == [t0]  # no wait before the switch…
    assert clock.now() - t0 <= 0.03 + 0.013  # …and done within one poll of it taking (was ≥100 ms)
    state["forced"].clear()
    assert asyncio.run(apps._raise(target)) == (True, "Target") and state["forced"] == []


def test_a_file_opened_is_done_when_its_app_is_in_front_with_a_window(clock, monkeypatch):
    state = {"at": None}

    async def run(cmd, timeout=30):
        state["at"] = clock.now() + 0.12  # the handler comes forward 120 ms after `open`
        return 0, "", ""

    monkeypatch.setattr(apps, "_run", run)
    monkeypatch.setattr(apps, "front_name", lambda: "Viewer" if state["at"] is not None and clock.now() >= state["at"] else "Editor")
    monkeypatch.setattr(apps, "_handler_for", lambda target: ("Viewer", 0))
    monkeypatch.setattr(apps, "_has_window", lambda name: True)
    t0 = clock.now()
    assert asyncio.run(apps.open_app("/tmp/report.pdf")) == "Opened /tmp/report.pdf"
    assert apps.target() == "Viewer" and 0.12 <= clock.now() - t0 <= 0.16  # not 0.6 s every time


def test_launch_waits_for_a_pid_by_polling(clock, monkeypatch):
    from neo.decide import axkit
    from neo.decide.kinds import launch

    t0 = clock.now()
    monkeypatch.setattr(axkit, "pid_of", lambda bid: 55 if bid == "test.app" and clock.now() - t0 >= 0.12 else None)
    assert asyncio.run(launch.wait_pid("test.app")) == 55 and clock.now() - t0 <= 0.17  # (was a 0.2 s step)
    t1 = clock.now()
    assert asyncio.run(launch.wait_pid("test.none", 0.3)) is None and clock.now() - t1 == pytest.approx(0.3)


def test_a_web_tree_switched_on_is_waited_for_only_until_it_fills_in(fake, clock, monkeypatch):
    area = El("AXWebArea", "Page", frame=(0, 60, 1000, 700))
    fake.add_app("Browser", 11, app_root(window("Browser", El("AXGroup", frame=(0, 60, 1000, 700), children=[area]))))
    t0 = clock.now()
    fake.values[(id(area), "AXChildren")] = lambda: [button("Go", frame=(10, 100, 40, 20)).set_pid(11)] if clock.now() - t0 >= 0.1 else None
    monkeypatch.setattr(ax, "_expose_web", lambda pid, root: True)
    tree = ax.snapshot("Browser")
    assert '"Go"' in tree and 0.1 <= clock.now() - t0 <= 0.2  # (a fixed 0.4 s before)


# ---- read-back ----------------------------------------------------------------------------------------------------


@pytest.fixture
def field(monkeypatch, clock):
    from neo.decide import axkit, media

    f = SimpleNamespace(value="", set_at=None, want="", delay=0.06, takes=True, typed=[])

    def attr(el, name, default=None):
        if name == "AXValue":
            if f.takes and f.set_at is not None and clock.now() - f.set_at >= f.delay:
                return f.want
            return f.value
        return {"AXRole": "AXTextField"}.get(name, default)

    def set_(el, name, value):
        if name == "AXValue":
            f.set_at, f.want = clock.now(), value
        return True

    def type_(pid, text, newline):
        f.typed.append(text)
        if f.takes:
            f.set_at, f.want = clock.now(), text

    monkeypatch.setattr(media, "_attr", attr)
    monkeypatch.setattr(media, "_set", set_)
    monkeypatch.setattr(axkit, "_pid_of", lambda el: 5)
    monkeypatch.setattr(axkit, "_type", type_)
    return f


def test_a_field_that_shows_its_text_a_moment_later_is_not_typed_into_again(field, clock):
    from neo.decide import axkit

    t0 = clock.now()
    assert axkit._put(object(), "hello", "return") and field.typed == []  # before: "didn't take", then typed twice
    assert 0.06 <= clock.now() - t0 <= 0.09


def test_a_field_that_never_takes_the_text_fails_honestly_after_the_read_back(field, clock):
    from neo.decide import axkit

    field.takes = False
    t0 = clock.now()
    assert not axkit._put(object(), "hello", "return")
    assert field.typed == ["hello"]  # typed once, never twice
    assert clock.now() - t0 == pytest.approx(2 * axkit._READBACK_S)


# ---- blockers --------------------------------------------------------------------------------------------------------


def _sheet_app(fake, pid=42):
    sheet = El("AXSheet", frame=(500, 60, 400, 200), children=[
        El("AXStaticText", value="Start a New Game?", frame=(520, 80, 200, 20)),
        El("AXPopUpButton", "Players", frame=(520, 120, 150, 24), actions=("AXPress",)),
        button("Start", frame=(800, 220, 60, 24)), button("Cancel", frame=(720, 220, 60, 24)),
    ])  # fmt: skip
    board = El("AXGroup", frame=(0, 60, 1512, 900))
    win = window("Game", board, sheet)
    fake.add_app("Game", pid, app_root(win))
    return win, sheet, board


def test_ax_tree_names_an_open_sheet_first_with_its_controls(fake):
    _sheet_app(fake)
    tree = ax.snapshot("Game")
    line = tree.splitlines()[1]
    assert line.startswith("A sheet “Start a New Game?” is open:") and line.endswith("answer it first.")
    ids = _ids(tree)
    assert f"{ids['Start']} Start" in line and f"{ids['Cancel']} Cancel" in line and f"{ids['Players']} Players" in line


def test_blockers_sheet_modal_dialog_and_another_apps_window(fake, monkeypatch):
    _win, sheet, _board = _sheet_app(fake)
    got = ax.blockers(42)
    assert [b.kind for b in got] == ["sheet"] and got[0].ref is sheet
    assert [e.title for e in got[0].buttons] == ["Players", "Start", "Cancel"] and "New Game" in got[0].text
    assert ax.get(got[0].buttons[1].id).title == "Start"  # pressable by id
    alert = El("AXWindow", "", frame=(500, 300, 400, 150), AXModal=True, children=[
        El("AXStaticText", value="Delete this photo?", frame=(520, 320, 200, 20)), button("Delete", frame=(800, 400, 60, 24))])  # fmt: skip
    fake.add_app("Photos", 50, app_root(window("Library"), alert, AXFocusedWindow=alert))
    got = ax.blockers(50)
    assert [b.kind for b in got] == ["modal"] and "Delete this photo?" in got[0].text
    prefs = window("Settings", button("Done"), AXSubrole="AXDialog")  # a dialog window, not modal
    fake.add_app("Prefs", 51, app_root(prefs, AXFocusedWindow=prefs))
    assert [b.kind for b in ax.blockers(51)] == ["dialog"]
    sysdlg = window("", button("OK"), AXSubrole="AXSystemDialog")
    fake.add_app("Sys", 52, app_root(sysdlg, AXFocusedWindow=sysdlg))
    assert [b.kind for b in ax.blockers(52)] == ["modal"]
    # another process's window over the middle of this one (the window server's evidence)
    fake.add_app("Plain", 53, app_root(window("Plain", button("Go"))))
    monkeypatch.setattr(ax, "window_owner_at", lambda x, y: ("Overlay", 900) if 500 < x < 1000 and 300 < y < 700 else ("Plain", 53))
    monkeypatch.setattr(ax, "_layer_at", lambda pid, x, y: 3)
    got = ax.blockers(53)
    assert [(b.kind, b.owner_pid) for b in got] == [("covering", 900)]
    assert "Another app's window covers this one" in ax.snapshot("Plain")


def test_a_banner_in_a_corner_or_an_ordinary_window_in_front_is_not_in_the_way(fake, monkeypatch):
    fake.add_app("Plain", 53, app_root(window("Plain", button("Go"))))
    monkeypatch.setattr(ax, "window_owner_at", lambda x, y: ("Banner", 901) if x > 1100 and y < 300 else ("Plain", 53))
    assert ax.blockers(53) == []
    monkeypatch.setattr(ax, "window_owner_at", lambda x, y: ("Editor", 902))  # another app's window, in front…
    monkeypatch.setattr(ax, "_layer_at", lambda pid, x, y: 0)  # …an ordinary one: bringing this app forward uncovers it
    assert ax.blockers(53) == []
    monkeypatch.setattr(ax, "_layer_at", lambda pid, x, y: 8)  # a floating panel
    assert [b.owner_pid for b in ax.blockers(53)] == [902]


def test_a_modal_dialog_blocks_a_point_accessibility_cannot_see(fake, monkeypatch):
    board = window("Board", frame=(0, 25, 1512, 957))  # drawn by the app: no element at the point
    alert = El("AXWindow", "", frame=(500, 300, 400, 150), AXModal=True, children=[
        El("AXStaticText", value="Resign this game?", frame=(520, 320, 200, 20)), button("Resign", frame=(800, 400, 60, 24))])  # fmt: skip
    fake.add_app("Game", 42, app_root(board, alert, AXFocusedWindow=alert))
    fake.at_point = None
    monkeypatch.setattr(ax, "_window_owner_at", lambda x, y: (42, "Game"))
    h = _REAL_HIT(100.0, 700.0)
    assert h.pid == 42 and "Resign this game?" in h.sheet
    assert _REAL_HIT(600.0, 350.0).sheet == ""  # on the dialog itself


# ---- menus ----------------------------------------------------------------------------------------------------------


def _menu_app(fake, pid=60):
    apple = El("AXMenuBarItem", "Apple", frame=(10, 0, 30, 24), actions=("AXPress",), children=[
        El("AXMenu", frame=(0, 0, 0, 0), children=[El("AXMenuItem", "Restart…", actions=("AXPress",))])])  # fmt: skip
    new_game = El("AXMenuItem", "New Game", actions=("AXPress",), AXMenuItemCmdChar="N", AXMenuItemCmdModifiers=0)
    export = El("AXMenu", frame=(0, 0, 0, 0), children=[
        El("AXMenuItem", "PDF…", actions=("AXPress",), AXMenuItemCmdChar="P", AXMenuItemCmdModifiers=3)])  # fmt: skip
    lazy = El("AXMenu", frame=(0, 0, 0, 0), children=[El("AXMenuItem", "Hint", actions=("AXPress",), AXMenuItemCmdChar="H", AXMenuItemCmdModifiers=1)])
    lazy.lazy_kids = 1  # (filled in when first asked)
    game = El("AXMenuBarItem", "Game", frame=(60, 0, 50, 24), actions=("AXPress",), children=[
        El("AXMenu", frame=(0, 0, 0, 0), children=[
            new_game,
            El("AXMenuItem", ""),  # a separator
            El("AXMenuItem", "Export", actions=("AXPress",), children=[export]),
            El("AXMenuItem", "Undo Move", actions=("AXPress",), AXEnabled=False, AXMenuItemCmdChar="Z"),
        ])])  # fmt: skip
    moves = El("AXMenuBarItem", "Moves", frame=(120, 0, 60, 24), actions=("AXPress",), children=[lazy])
    bar = El("AXMenuBar", frame=(0, 0, 1512, 24), children=[apple, game, moves])
    root = fake.add_app("Game", pid, app_root(window("Game"), bar, AXMenuBar=bar))
    return root, apple, new_game


def test_menu_items_are_read_without_opening_a_menu_and_never_the_system_menu(fake):
    _root, _apple, new_game = _menu_app(fake)
    items = ax.menu_items(60)
    paths = [i.where() for i in items]
    assert paths == ["Game › New Game", "Game › Export", "Game › Export › PDF…", "Game › Undo Move", "Moves › Hint"]
    by = {i.title: i for i in items}
    assert by["New Game"].chord == "cmd+n" and by["New Game"].ref is new_game and by["New Game"].enabled
    assert by["PDF…"].chord == "cmd+shift+alt+p" and by["Hint"].chord == "cmd+shift+h"  # (a lazily built menu)
    assert not by["Undo Move"].enabled and "Restart…" not in paths
    assert fake.performed == []  # nothing opened


def test_a_menu_command_is_pressed_through_the_one_door_and_never_a_system_one(fake, monkeypatch):
    _root, apple, _new_game = _menu_app(fake)
    seen = []
    monkeypatch.setattr(ax, "_ax_watch", lambda kind, el, what, value: seen.append((kind, what)))
    items = {i.title: i for i in ax.menu_items(60)}
    assert ax.press_menu_item(items["New Game"]) == "Chose Game › New Game" and seen == [("perform", "AXPress")]
    assert "greyed out" in ax.press_menu_item(items["Undo Move"])
    restart = ax.MenuItem("Restart…", ("Apple", "Restart…"), True, ref=apple.children[0].children[0], pid=60)
    assert ax.press_menu_item(restart) == ax._SYSTEM_MENU
    assert len(fake.performed) == 1


def test_the_system_menu_is_never_pressed_by_id_either(fake):
    _root, apple, _ng = _menu_app(fake)
    restart = apple.children[0].children[0]
    e = ax.Element("e9", "AXMenuItem", "Restart…", "", 0, 30, 100, 20, actions=["AXPress"], ref=restart)
    assert ax.press_el(e) == ax._SYSTEM_MENU and ax.press_element(e) == ax._SYSTEM_MENU and fake.performed == []


# ---- what's on screen, for marks ----------------------------------------------------------------------------------


def test_visible_elements_put_the_sheet_first_then_controls_then_size(fake):
    _win, _sheet, _board = _sheet_app(fake)
    els = ax.visible_elements(42)
    assert [e.title for e in els[:3]] == ["Start", "Cancel", "Players"] and all(e.visible for e in els)
    assert all(ax.get(e.id) is e for e in els)


def test_visible_frames_are_cut_to_the_window(fake):
    half = button("Half", frame=(1400, 60, 200, 24))  # half past the window's right edge
    fake.add_app("W", 70, app_root(window("W", half, frame=(0, 25, 1512, 957))))
    e = next(e for e in ax.visible_elements(70) if e.title == "Half")
    assert e.visible == (1400, 60, 112, 24)


# ---- the tree's own words --------------------------------------------------------------------------------------------


def test_the_tool_text_names_no_app_and_a_board_is_told_by_evidence(fake):
    assert "Chess" not in ax.AX_TREE_DESCRIPTION and "board" in ax.AX_TREE_DESCRIPTION
    board = El("AXGroup", frame=(100, 100, 900, 800))  # childless, not pressable, most of the window
    fake.add_app("Game", 42, app_root(window("Game", board, button("Hint", frame=(1100, 100, 60, 24)))))
    tree = ax.snapshot("Game")
    assert "no accessible parts (a board or canvas)" in tree and "@(100,100 900x800)" in tree
    board.actions = ["AXPress"]  # a big button is a button
    assert "no accessible parts" not in ax.snapshot("Game")


def test_a_background_window_can_be_looked_inside_by_title(fake):
    fake.add_app("Mail", 9, app_root(window("Inbox", button("Reply")), window("Draft", button("Send"), frame=(300, 300, 600, 400))))
    tree = ax.snapshot("Mail")
    assert '"Send"' not in tree and 'ax_tree window="Draft"' in tree
    inside = ax.snapshot("Mail", window="draft")
    assert '"Send"' in inside and '"Reply"' not in inside
    assert ax.snapshot("Mail", window="nope").startswith("Error")


def test_compact_is_controls_only(fake):
    fake.add_app("Doc", 9, app_root(window("Doc", El("AXStaticText", value="A long paragraph", frame=(10, 60, 300, 20)), button("Share"))))
    compact = ax.snapshot("Doc", compact=True)
    assert '"Share"' in compact and "long paragraph" not in compact


# ---- the one door for AX actions ------------------------------------------------------------------------------------


def test_every_ax_action_and_write_is_shown_to_the_watcher_first(fake, monkeypatch):
    b, f = button("OK"), El("AXTextField", "Name", value="", frame=(10, 100, 200, 24))
    ids = _ids((fake.add_app("Form", 3, app_root(window("Form", b, f))), ax.snapshot())[1])
    seen = []
    monkeypatch.setattr(ax, "_ax_watch", lambda kind, el, what, value: seen.append((kind, what)))
    ax.press(ids["OK"])
    ax.focus_el(f)
    other = window("Other", frame=(0, 0, 10, 10))
    fake.apps[3].add(other)
    fake.apps[3].attrs["AXFocusedWindow"] = fake.apps[3].children[0]
    assert _REAL_RAISE(ax.Hit(pid=3, app="Form", window=other))
    assert seen == [("perform", "AXPress"), ("set", "AXFocused"), ("perform", "AXRaise"), ("set", "AXMain")]

    def refuse(kind, el, what, value):
        raise RuntimeError("outside the test window")

    monkeypatch.setattr(ax, "_ax_watch", refuse)
    n = len(fake.performed)
    with pytest.raises(RuntimeError):
        ax.press(ids["OK"])
    assert len(fake.performed) == n  # stopped before it was made


# ---- what a press changed ----------------------------------------------------------------------------------------------


def test_a_press_reports_a_toggle_a_closed_dialog_and_no_change(fake, clock, monkeypatch):
    box = El("AXCheckBox", "Wrap lines", value=0, frame=(10, 60, 120, 20), actions=("AXPress",))
    ok = button("OK", frame=(10, 100, 60, 24))
    root = fake.add_app("Editor", 12, app_root(window("Editor", box, ok)))
    monkeypatch.setattr(ax, "scene", lambda pid, front=None: _REAL_SCENE(pid, front=("Editor", 12)))
    ids = _ids(ax.snapshot("Editor"))
    e = ax.get(ids["Wrap lines"])
    w = ax.press_watch(e, 12)
    box.attrs["AXValue"] = 1  # the press ticked it
    ch = ax.press_change(w, e)
    assert ch.changed is True and "the element changed" in ch.said
    e2 = ax.get(ids["OK"])
    w = ax.press_watch(e2, 12)
    ok.gone = True  # its dialog closed with it
    assert ax.press_change(w, e2).changed is True
    w = ax.press_watch(e, 12)
    t0 = clock.now()
    ch = ax.press_change(w, e, timeout=0.5)
    assert ch.changed is False and clock.now() - t0 == pytest.approx(0.5, abs=0.01)
    assert root.pid == 12


# ---- the repeat look: Accessibility notifications ----------------------------------------------------------------------


class FakeBackend(axwatch.Backend):
    def __init__(self):
        self.started, self.stopped, self.notes = [], [], {}

    def start(self, pid, on_note):
        self.started.append(pid)
        self.notes[pid] = on_note
        return {"pid": pid}

    def stop(self, handle):
        self.stopped.append(handle["pid"])

    def fire(self, pid, name="AXValueChanged"):
        self.notes[pid](pid, name)


@pytest.fixture
def notes(monkeypatch):
    b = FakeBackend()
    axwatch.use_backend(b)
    yield b
    axwatch.use_backend(None)


def test_generations_move_with_notifications_and_waits_end_at_the_first(notes):
    assert axwatch.generation(5) is None and axwatch.wait_change(5, 0, 0.01) is None  # not watched: can't tell
    assert axwatch.watch(5) and notes.started == [5] and axwatch.generation(5) == 0
    notes.fire(5)
    assert axwatch.generation(5) == 1 and axwatch.wait_change(5, 0, 0.0) is True
    assert axwatch.wait_change(5, 1, 0.01) is False
    threading.Timer(0.02, notes.fire, (5,)).start()
    t0 = time.monotonic()
    assert axwatch.wait_change(5, 1, 2.0) is True and time.monotonic() - t0 < 1.0
    assert axwatch.watch(6) and notes.stopped == [5] and axwatch.generation(5) is None  # one app at a time


def test_no_watching_in_a_host_process_or_when_switched_off(notes, monkeypatch):
    from neo import runtime

    monkeypatch.setattr(runtime, "_library", True)
    assert not axwatch.watch(5) and notes.started == []
    monkeypatch.setattr(runtime, "_library", False)
    monkeypatch.setenv("NEO_AX_WATCH", "0")
    assert not axwatch.watch(5)


def test_the_real_observer_never_starts_while_effects_are_off():
    axwatch.use_backend(None)
    assert not axwatch.enabled() and not axwatch.watch(123)


def test_a_repeat_look_with_nothing_changed_costs_no_walk(fake, notes, demo, monkeypatch):
    from neo.decide import axkit

    monkeypatch.setattr(ax, "_running", lambda pid: SimpleNamespace(localizedName=lambda: "Demo", activationPolicy=lambda: 0,
                                                                    bundleURL=lambda: SimpleNamespace(path=lambda: "/Applications/Demo.app")))  # fmt: skip
    monkeypatch.setattr(axkit, "runtime", lambda path: "native")
    ax.snapshot()  # starts watching
    second = ax.snapshot()  # walks (the watch began during the first) and keeps it
    n, third = calls(fake, ax.snapshot)
    assert third == second and n <= 3  # the same look, the same ids: no walk
    assert ax.get(_ids(third)["Tool 3"]).title == "Tool 3"
    notes.fire(4242)  # the app changed something
    n, fourth = calls(fake, ax.snapshot)
    assert n > 100 and fourth != third  # walked again: new ids
    ax.snapshot()
    monkeypatch.setattr(inp, "_sent", inp._sent + 1)  # NEO sent an input event since
    assert calls(fake, ax.snapshot)[0] > 100
    ax.window_elements()
    assert calls(fake, ax.window_elements)[0] <= 3  # the gate's walk of a window just walked: free


def test_chromium_and_electron_apps_are_not_watched(fake, notes, demo, monkeypatch):
    from neo.decide import axkit

    monkeypatch.setattr(ax, "_running", lambda pid: SimpleNamespace(localizedName=lambda: "Demo", activationPolicy=lambda: 0,
                                                                    bundleURL=lambda: SimpleNamespace(path=lambda: "/Applications/Chat.app")))  # fmt: skip
    monkeypatch.setattr(axkit, "runtime", lambda path: "electron")
    ax.snapshot()
    ax.snapshot()
    assert notes.started == [] and calls(fake, ax.snapshot)[0] > 100


# ---- input: the leftover audit items -----------------------------------------------------------------------------------


@pytest.fixture
def posted(monkeypatch):
    got: list = []
    monkeypatch.setattr(inp, "_post", got.append)  # built, never posted
    monkeypatch.setattr(inp, "time", SimpleNamespace(sleep=lambda s: got.append(("sleep", s))))
    return got


def test_the_scroll_wheel_event_carries_its_point(posted):
    import Quartz as Q

    inp.scroll(321, 654, -3)
    wheel = [e for e in posted if not isinstance(e, tuple) and Q.CGEventGetType(e) == Q.kCGEventScrollWheel]
    loc = Q.CGEventGetLocation(wheel[0])
    assert (round(loc.x), round(loc.y)) == (321, 654)
    assert ("sleep", inp._SCROLL_SETTLE_S) in posted  # the pointer rests there first


def test_a_click_ends_with_its_last_event(posted):
    inp.click(10, 20)
    assert not isinstance(posted[-1], tuple)  # no pause after the button comes up (was 50 ms)
    posted.clear()
    inp.click(10, 20, count=2)
    assert [p for p in posted if isinstance(p, tuple)] == [("sleep", inp._MOVE_SETTLE_S), ("sleep", inp._CLICK_GAP_S)]


def test_text_goes_in_whole_characters_of_at_most_20_utf16_units(posted):
    import Quartz as Q

    text = "नमस्ते दुनिया 👨‍👩‍👧‍👦 héllo wörld 🇮🇳🇮🇳 किताब क्षत्रिय " * 2
    inp.type_text(text)
    downs = [e for e in posted if not isinstance(e, tuple) and Q.CGEventGetType(e) == Q.kCGEventKeyDown]
    pieces = [Q.CGEventKeyboardGetUnicodeString(e, 64, None, None) for e in downs]
    assert all(n <= 20 for n, _s in pieces) and "".join(s for _n, s in pieces) == text
    assert all(not inp._extends(s[0]) and s[0] != "‍" for _n, s in pieces)  # no piece starts mid-character
    assert inp.clusters("क्षत्रिय") == ["क्ष", "त्रि", "य"] and inp.clusters("👍🏽🇮🇳") == ["👍🏽", "🇮🇳"]


@pytest.mark.parametrize(
    "said, flags, key",
    [
        ("pagedown", 0, "pagedown"), ("page_down", 0, "pagedown"), ("page-down", 0, "pagedown"),
        ("arrowdown", 0, "down"), ("spacebar", 0, "space"), ("cmd+comma", "cmd", ","), ("f13", 0, "f13"),
        ("ctrl-shift-tab", "ctrl+shift", "tab"), ("cmd+?", "cmd+shift", "/"),
    ],
)  # fmt: skip
def test_key_names_models_use(said, flags, key):
    want = inp.mods_flags(flags) if flags else 0
    assert inp.parse_chord(said) == (want, key)


def test_letters_follow_the_keyboard_layout(monkeypatch, posted):
    import Quartz as Q

    azerty = {"a": 12, "z": 13, "q": 0, "w": 6, "m": 41}
    monkeypatch.setattr(inp, "_layout", ("com.apple.keylayout.French", azerty))
    monkeypatch.setattr(inp, "_layout_id", lambda: "com.apple.keylayout.French")
    inp.hotkey("cmd+z")
    down = next(e for e in posted if not isinstance(e, tuple))
    assert Q.CGEventGetIntegerValueField(down, Q.kCGKeyboardEventKeycode) == 13  # the key that types z here
    assert inp.keycode("return") == 36 and inp.keycode("left") == 123  # named keys never move
    monkeypatch.setattr(inp, "_layout_id", lambda: "com.apple.keylayout.ABC")  # switched since: not trusted
    assert inp.keycode("z") == 6


def test_the_layout_is_read_on_the_main_thread_only(monkeypatch):
    monkeypatch.setattr(inp, "_tis_letters", lambda: pytest.fail("read off the main thread"))
    got = []
    t = threading.Thread(target=lambda: got.append(inp.refresh_layout()))
    t.start()
    t.join()
    assert got == [False]


# ---- media keeps working under the process-wide timeout -------------------------------------------------------------


def test_media_walks_under_the_new_timeout_are_faster_and_unchanged(fake, clock):
    from neo.decide import media

    fake.clock = clock
    ax._ensure_timeout()  # NEO.app: the process-wide 1.5 s
    stuck = El("AXTextField", "Search", frame=(10, 60, 200, 24))
    stuck.hung = True
    good = El("AXSearchField", "Search", frame=(10, 100, 200, 24))
    root = fake.add_app("Music", 30, app_root(window("Music", stuck, good)))
    fake.AXUIElementSetMessagingTimeout(root, 2.0)  # (media's own root keeps its 2 s)
    t0 = clock.now()
    hits = media.find_search(root)
    assert [h.el for h in hits] == [good]  # found as before
    took = clock.now() - t0
    asks = sum(1 for c in fake.calls if c[1] is stuck)
    assert took == pytest.approx(asks * 1.5)  # each question to the hung field: 1.5 s (was 6 s)
    assert root.timeout == 2.0 and fake.global_timeout == 1.5


def test_media_in_a_host_process_is_as_before(fake, clock, monkeypatch):
    from neo import runtime
    from neo.decide import media

    monkeypatch.setattr(runtime, "_library", True)
    fake.clock = clock
    stuck = El("AXTextField", "Search", frame=(10, 60, 200, 24))
    stuck.hung = True
    root = fake.add_app("Music", 30, app_root(window("Music", stuck)))
    t0 = clock.now()
    media.find_search(root)
    asks = sum(1 for c in fake.calls if c[1] is stuck)
    assert fake.global_timeout is None and clock.now() - t0 == pytest.approx(asks * 6.0)


# ---- nothing here ever acts for real ---------------------------------------------------------------------------------------


def test_the_choke_points_are_refused_while_effects_are_off(monkeypatch):
    import ApplicationServices

    from neo import effects

    monkeypatch.setattr(ax, "AS", ApplicationServices)
    with pytest.raises(effects.EffectRefused):
        ax._perform(object(), "AXPress", "press something")
    with pytest.raises(effects.EffectRefused):
        ax._set(object(), "AXValue", "x", "write something")



def test_a_labelled_look_ends_at_the_first_exact_match(fake, demo):
    n_all, (_n, _f, every) = calls(fake, ax.window_elements)
    n, (_n, _f, els) = calls(fake, lambda: ax.window_elements(early_exit="tool 2"))
    assert els[-1].title == "Tool 2" and n < n_all / 4 and len(els) < len(every)
    assert els[-1].actions == ["AXPress"]  # (read while walking: the gate and click_text use them)


def test_something_frontmost_for_a_moment_is_told_by_how_it_runs_not_its_name(monkeypatch, fake):
    import AppKit

    panel = El("AXApplication", pid=31)
    fake.add(31, panel)  # a notification: no windows, not a regular app
    monkeypatch.setattr(focus, "front_app", lambda: ("Banners", 31))
    monkeypatch.setattr(ax, "regular", lambda pid: pid != 31)
    bar = SimpleNamespace(localizedName=lambda: "Editor", processIdentifier=lambda: 8)
    monkeypatch.setattr(AppKit, "NSWorkspace", SimpleNamespace(sharedWorkspace=lambda: SimpleNamespace(menuBarOwningApplication=lambda: bar)))
    monkeypatch.setattr(ax, "NSWorkspace", AppKit.NSWorkspace)
    assert _REAL_FRONTMOST() == ("Editor", 8)
    panel.add(window("Search"))  # …unless the user is in a window of it
    assert _REAL_FRONTMOST() == ("Banners", 31)
