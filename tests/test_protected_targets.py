"""NEO's input tools never click, press, drag, scroll or type into NEO's own overlay (its card's
"Yes, do it" is the user's to press) or into an app listed in `protected_apps`."""

from __future__ import annotations

import asyncio

import pytest

from neo2.core.agent import confirm
from neo2.core.agent.registry import ToolContext, registry
from neo2.core.events import EventBus

# Screen layout of the fake desktop: x < 100 is NEO's overlay, 100 <= x < 200 the host's approval
# dialog, anything else Finder.
_OWNERS = {"ref-neo": "NEO", "ref-ui": "neo-ui", "ref-approver": "MyApprover", "ref-finder": "Finder"}


def _owner_at(x: float, y: float) -> str:
    return "NEO" if x < 100 else "MyApprover" if x < 200 else "Finder"


_PIDS = {"NEO": 100, "MyApprover": 101, "Finder": 102}  # (as the window server numbers them below)
_FRONTED: list[int] = []  # the apps brought to the front (pids) in a test


@pytest.fixture
def desk(monkeypatch):
    from neo2.core import events
    from neo2.core.tools import computer as comp
    from neo2.core.tools import load_all
    from neo2.core.tools.computer import ax

    monkeypatch.setattr(events, "_bus", EventBus())
    load_all()
    confirm.cancel("test setup")
    did: list[tuple] = []
    elements = {
        "e1": ax.Element("e1", "AXButton", "Yes, do it", "", 10, 10, 40, 20, actions=["AXPress"], ref="ref-neo"),
        "e2": ax.Element("e2", "AXButton", "Allow", "", 10, 10, 40, 20, actions=["AXPress"], ref="ref-ui"),
        "e3": ax.Element("e3", "AXButton", "Approve", "", 150, 10, 40, 20, actions=["AXPress"], ref="ref-approver"),
        "e4": ax.Element("e4", "AXButton", "Open", "", 300, 10, 40, 20, actions=["AXPress"], ref="ref-finder"),
        "e5": ax.Element("e5", "AXTextField", "", "", 300, 50, 80, 20, ref="ref-finder"),
        "e6": ax.Element("e6", "AXTextField", "", "", 10, 50, 80, 20, ref="ref-neo"),
        # on NEO's overlay, but AX can't name their owner
        "e7": ax.Element("e7", "AXButton", "Yes, do it", "", 10, 10, 40, 20, actions=["AXPress"], ref="ref-?"),
        "e8": ax.Element("e8", "AXTextField", "", "", 10, 50, 80, 20, ref="ref-?"),
    }
    monkeypatch.setattr(ax, "get", lambda eid: elements.get(eid))

    def hit(x, y):
        """What accessibility says is under a point: its owner (as app_at names it) and label."""
        owner = ax.app_at(x, y)
        if not owner:
            return ax.Hit()  # AX couldn't tell: the window server is asked
        return ax.Hit(pid=_PIDS[owner], app=owner, label=ax.label_at(x, y), role="AXButton", regular=True)

    frames = {e.ref: (e.x, e.y, e.w, e.h) for e in reversed(elements.values())}  # (the first element per ref)
    monkeypatch.setattr(ax, "hit", hit)
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": name in ("NEO", "neo-ui"))
    monkeypatch.setattr(ax, "pid_of", lambda ref: _PIDS.get(_OWNERS.get(ref, ""), 0))
    monkeypatch.setattr(ax, "visible_frame", lambda ref: frames.get(ref))
    monkeypatch.setattr(ax, "regular", lambda pid: True)
    monkeypatch.setattr(comp.inp, "move", lambda x, y: None)
    _FRONTED.clear()

    async def bring_pid_front(pid):
        _FRONTED.append(pid)
        return True, next((n for n, p in _PIDS.items() if p == pid), "")

    monkeypatch.setattr(comp.apps, "bring_pid_front", bring_pid_front)
    monkeypatch.setattr(comp.pointer, "SETTLE_S", 0.0)
    from neo2.core.tools.computer import focus

    monkeypatch.setattr(focus, "front_app", lambda: ("Finder", 102))
    monkeypatch.setattr(ax, "app_of", lambda ref: _OWNERS.get(ref, ""))
    monkeypatch.setattr(ax, "app_at", _owner_at)
    # the window server's view of the same desktop, and every app's other names
    monkeypatch.setattr(ax, "window_owner_at", lambda x, y: (_owner_at(x, y), 100 + int(x >= 100) + int(x >= 200)))
    # every app's other names, by its pid or by its name (what names_of_app says of that name)
    monkeypatch.setattr(ax, "app_names", lambda pid: ax.names_of_app(next((n for n, p in _PIDS.items() if p == pid), "")))
    monkeypatch.setattr(ax, "names_of_app", lambda name: set())
    monkeypatch.setattr(ax, "label_at", lambda x, y: "Yes, do it" if x < 100 else "Open")
    monkeypatch.setattr(ax, "press", lambda eid: did.append(("press", eid)) or f"Pressed {eid}")
    monkeypatch.setattr(ax, "set_value", lambda eid, v: did.append(("set", eid, v)) or f"Set {eid}")
    monkeypatch.setattr(ax, "press_element", lambda e: did.append(("press", e.id)) or "Pressed")
    monkeypatch.setattr(ax, "press_el", lambda e: did.append(("press", e.id)) or f"Pressed {e.id}")
    monkeypatch.setattr(ax, "set_el_value", lambda e, v: did.append(("set", e.id, v)) or f"Set {e.id}")
    monkeypatch.setattr(comp.inp, "click", lambda x, y, **kw: did.append(("click", x, y)) or "Clicked")
    monkeypatch.setattr(comp.inp, "drag", lambda *a: did.append(("drag", *a)) or "Dragged")
    monkeypatch.setattr(comp.inp, "scroll", lambda x, y, dy, dx: did.append(("scroll", x, y)) or "Scrolled")
    monkeypatch.setattr(comp.inp, "hotkey", lambda keys: did.append(("hotkey", keys)) or f"Pressed {keys}")
    monkeypatch.setattr(comp.inp, "mouse_position", lambda: (50.0, 50.0))
    monkeypatch.setattr(comp.apps, "_target", None)
    monkeypatch.setattr(comp.apps, "front_name", lambda: "Finder")
    monkeypatch.setattr(ax, "frontmost_app", lambda: ("Finder", 1))
    return did, monkeypatch


def _run(tool: str, args: dict):
    return asyncio.run(registry().invoke(tool, args, ToolContext(user_text="")))


def _refused(out) -> bool:
    return out.halt and out.text.startswith(confirm.NEEDS_USER) and "NEO's own window" in out.text


@pytest.mark.parametrize(
    "tool, args",
    [
        ("ax_press", {"id": "e1"}),
        ("ax_press", {"id": "e2"}),
        ("click", {"id": "e1"}),
        ("click", {"x": 20, "y": 20}),
        ("drag", {"x1": 300, "y1": 20, "x2": 20, "y2": 20}),  # dropped onto the overlay
        ("drag", {"x1": 20, "y1": 20, "x2": 300, "y2": 20}),  # picked up from the overlay
        ("scroll", {"x": 30, "y": 30, "dy": 5}),
        ("ax_set_value", {"id": "e6", "value": "hi"}),
    ],
)
def test_nothing_touches_neos_overlay(desk, tool, args):
    did, _ = desk
    out = _run(tool, args)
    assert _refused(out), out.text
    assert did == [] and confirm.peek() is None  # refused before any gate: nothing to approve
    assert _FRONTED == []  # and nothing was brought forward first


def test_scroll_at_a_pointer_over_neos_overlay_never_scrolls_neo(desk):
    """The pointer rests on NEO's overlay (the user just clicked the orb): "scroll down" means the
    app they work in — scrolled in its window's middle — never NEO; a protected app there is refused."""
    did, mp = desk
    from neo2.core import config
    from neo2.core.tools.computer import ax

    mp.setattr(ax, "work_window_frame", lambda: None)  # no other window in front
    out = _run("scroll", {"at_pointer": True, "dy": -5})
    assert not out.ok and "NEO's own window" in out.text and did == []
    mp.setattr(ax, "work_window_frame", lambda: (300.0, 0.0, 200.0, 100.0))  # Finder's window behind it
    assert _run("scroll", {"at_pointer": True, "dy": -5}).ok and did == [("scroll", 400.0, 50.0)]
    did.clear()
    mp.setenv("NEO_PROTECTED_APPS", '["MyApprover"]')
    mp.setattr(config, "_settings", None)
    mp.setattr(ax, "work_window_frame", lambda: (100.0, 0.0, 100.0, 100.0))  # the host's approval dialog
    assert _refused(_run("scroll", {"at_pointer": True, "dy": -5})) and did == [] and _FRONTED == []


def test_click_text_refuses_neos_window(desk):
    did, mp = desk
    from neo2.core.tools.computer import ax

    els = [ax.Element("e1", "AXButton", "Yes, do it", "", 10, 10, 40, 20, actions=["AXPress"])]
    mp.setattr(ax, "window_elements", lambda app=None: ("NEO", (0, 0, 400, 300), els))
    out = _run("click_text", {"label": "yes, do it"})
    assert _refused(out) and did == [] and confirm.peek() is None


@pytest.mark.parametrize(
    "tool, args",
    [
        ("ax_press", {"id": "e3"}),
        ("click", {"id": "e3"}),
        ("click", {"x": 150, "y": 20}),
        ("drag", {"x1": 300, "y1": 20, "x2": 150, "y2": 20}),
        ("scroll", {"x": 150, "y": 30, "dy": -3}),
    ],
)
def test_protected_apps_from_the_environment(desk, tool, args):
    did, mp = desk
    from neo2.core import config

    mp.setenv("NEO_PROTECTED_APPS", '["myapprover"]')  # case doesn't matter
    mp.setattr(config, "_settings", None)
    assert config.settings().protected_apps == ["myapprover"]
    out = _run(tool, args)
    assert _refused(out) and did == [] and confirm.peek() is None
    assert _FRONTED == []  # the host's dialog is in the background: it was never brought forward


def test_protected_apps_also_block_keystrokes_and_click_text(desk):
    did, mp = desk
    from neo2.core import config
    from neo2.core.tools import computer as comp
    from neo2.core.tools.computer import ax

    mp.setenv("NEO_PROTECTED_APPS", '["MyApprover"]')
    mp.setattr(config, "_settings", None)
    mp.setattr(comp.apps, "front_name", lambda: "MyApprover")
    for tool, args in (("hotkey", {"keys": "return"}), ("type_text", {"text": "yes"})):
        out = _run(tool, args)
        assert out.halt and out.text.startswith(confirm.NEEDS_USER), (tool, out.text)
    mp.setattr(comp.apps, "front_name", lambda: "Finder")
    els = [ax.Element("e1", "AXButton", "Approve", "", 10, 10, 40, 20, actions=["AXPress"])]
    mp.setattr(ax, "window_elements", lambda app=None: ("MyApprover", (0, 0, 400, 300), els))
    assert _refused(_run("click_text", {"label": "approve"}))
    assert did == [] and confirm.peek() is None


def test_without_the_setting_the_host_app_is_an_ordinary_app(desk):
    """Unlisted, a host's "Approve" button is just a committing button: the gate asks (as today)."""
    did, _ = desk
    out = _run("ax_press", {"id": "e3"})
    assert out.halt and out.text.startswith(confirm.NEEDS_CONFIRM) and did == []
    confirm.cancel("test")
    assert _run("click", {"x": 150, "y": 20}).ok and did == [("click", 150, 20)]
    assert _FRONTED == [101]  # an ordinary app in the background comes forward for the click


def test_finder_targets_behave_as_before(desk):
    did, mp = desk
    from neo2.core.tools.computer import ax

    assert _run("ax_press", {"id": "e4"}).ok
    assert _run("click", {"id": "e4"}).ok
    assert _run("click", {"x": 320, "y": 20}).ok
    assert _run("drag", {"x1": 300, "y1": 20, "x2": 400, "y2": 20}).ok
    assert _run("scroll", {"x": 320, "y": 20, "dy": -3}).ok
    assert _run("ax_set_value", {"id": "e5", "value": "notes.txt"}).ok
    els = [ax.Element("e1", "AXButton", "Open", "", 10, 10, 40, 20, actions=["AXPress"])]
    mp.setattr(ax, "window_elements", lambda app=None: ("Finder", (0, 0, 400, 300), els))
    assert _run("click_text", {"label": "open"}).ok
    assert [d[0] for d in did] == ["press", "click", "click", "drag", "scroll", "set", "press"]
    assert confirm.peek() is None


def test_a_committing_button_in_an_ordinary_app_still_asks(desk):
    """The protection sits in front of the gate: an ordinary app's "Delete" still goes to the user."""
    did, mp = desk
    from neo2.core.tools.computer import ax

    mp.setattr(ax, "get", lambda eid: ax.Element(eid, "AXButton", "Delete", "", 300, 10, 40, 20, ref="ref-finder"))
    out = _run("ax_press", {"id": "e9"})
    assert out.halt and out.text.startswith(confirm.NEEDS_CONFIRM) and did == []
    confirm.cancel("test")


# ---- an owner AX can't name --------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool, args",
    [
        ("click", {"x": 20, "y": 20}),
        ("ax_press", {"id": "e7"}),
        ("click", {"id": "e7"}),
        ("ax_set_value", {"id": "e8", "value": "hi"}),
        ("drag", {"x1": 300, "y1": 20, "x2": 20, "y2": 20}),
        ("scroll", {"x": 30, "y": 30, "dy": 5}),
    ],
)
def test_an_owner_ax_cant_name_is_asked_of_the_window_server(desk, tool, args):
    did, mp = desk
    from neo2.core.tools.computer import ax

    mp.setattr(ax, "app_at", lambda x, y: "")  # AX couldn't resolve who owns the point
    out = _run(tool, args)
    assert _refused(out), out.text
    assert did == [] and confirm.peek() is None


def test_when_nobody_can_say_whose_it_is_nothing_is_touched(desk):
    did, mp = desk
    from neo2.core.tools.computer import ax

    mp.setattr(ax, "app_at", lambda x, y: "")
    mp.setattr(ax, "window_owner_at", lambda x, y: None)  # the window server couldn't say either
    assert _refused(_run("click", {"x": 320, "y": 20})) and did == []


def test_an_unnamed_owner_elsewhere_is_clicked_as_before(desk):
    did, mp = desk
    from neo2.core.tools.computer import ax

    mp.setattr(ax, "app_at", lambda x, y: "")
    assert _run("click", {"x": 320, "y": 20}).ok  # the window server says Finder
    mp.setattr(ax, "window_owner_at", lambda x, y: ("", 0))  # no window there at all
    assert _run("scroll", {"x": 320, "y": 20, "dy": -3}).ok
    assert [d[0] for d in did] == ["click", "scroll"]


def test_protected_apps_match_a_bundle_id_or_executable_too(desk):
    did, mp = desk
    from neo2.core import config
    from neo2.core.tools.computer import ax

    mp.setenv("NEO_PROTECTED_APPS", '["com.example.approver", "neo-approverd"]')
    mp.setattr(config, "_settings", None)
    others = {"MyApprover": {"com.example.approver", "Approver"}, "Finder": {"com.apple.finder", "Finder"}}
    mp.setattr(ax, "names_of_app", lambda name: others.get(name, set()))
    assert _refused(_run("click", {"x": 150, "y": 20}))  # "MyApprover" is com.example.approver
    assert _refused(_run("ax_press", {"id": "e3"}))
    assert _run("click", {"x": 320, "y": 20}).ok
    # named only by the window server: its pid's names
    mp.setattr(ax, "app_at", lambda x, y: "")
    mp.setattr(ax, "window_owner_at", lambda x, y: ("", 777))
    mp.setattr(ax, "app_names", lambda pid: {"neo-approverd"} if pid == 777 else set())
    assert _refused(_run("click", {"x": 320, "y": 20}))
    mp.setattr(ax, "app_names", lambda pid: {"com.arnav.neo"})  # NEO's overlay, by its bundle id
    mp.setenv("NEO_PROTECTED_APPS", "[]")
    mp.setattr(config, "_settings", None)
    assert _refused(_run("click", {"x": 320, "y": 20}))
    assert did == [("click", 320, 20)]


def test_a_protected_app_is_never_brought_forward_first(desk):
    """hotkey(app=...) used to activate the app it names and only then refuse: NEO's own overlay
    or the host's approval dialog took focus, though no key was pressed."""
    did, mp = desk
    from neo2.core import config
    from neo2.core.tools import computer as comp

    fronted: list[str] = []

    async def bring_front(name, *, wait=1.5):
        fronted.append(name)
        return True, name

    mp.setattr(comp.apps, "bring_front", bring_front)
    mp.setenv("NEO_PROTECTED_APPS", '["MyApprover"]')
    mp.setattr(config, "_settings", None)
    for app in ("NEO", "neo-ui", "MyApprover", "myapprover"):
        out = _run("hotkey", {"keys": "return", "app": app})
        assert _refused(out), (app, out.text)
    mp.setattr(comp.apps, "_target", ("NEO", 9e18))  # the app NEO last worked in
    assert _refused(_run("type_text", {"text": "yes"}))
    assert fronted == [] and did == []
