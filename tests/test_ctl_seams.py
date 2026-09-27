"""Step 0's seams for the computer-controls work: fields and hooks the workstreams build on, each
with no behaviour of its own yet — a tool's checked effect and its hand-off to a model, the
by-utterance lane (the user's own words only), an image's detail and the strictest image limit of a
chain, polls on a fake clock, the settings, one door for Accessibility actions with a watcher, the
capture effect, every acting tool in neo2's INPUT_TOOLS, and nothing process-wide in library mode."""

from __future__ import annotations

import ast
import asyncio
import dataclasses
from pathlib import Path

import pytest

from neo import effects, runtime
from neo.agent.registry import (
    Choice,
    Effect,
    Handoff,
    RegisteredTool,
    ToolContext,
    ToolOutput,
    utterance_lane,
)
from neo.providers.base import ImagePart

# ---- ToolOutput.effect and .handoff ------------------------------------------------------------------


def test_new_tool_output_fields_come_last_with_defaults():
    names = [f.name for f in dataclasses.fields(ToolOutput)]
    assert names[-2:] == ["effect", "handoff"]
    out = ToolOutput("Pressed", [], True)  # positional, as before
    assert out.effect is None and out.handoff is None
    assert Effect() == Effect(changed=None, scene_changed=False, sheet="", looked=False)
    h = Handoff("two buttons are called Start", [Choice("e3", "Start (sheet)"), Choice("e9", "Start")])
    assert [c.id for c in h.choices] == ["e3", "e9"]


def test_an_effect_and_a_handoff_reach_the_caller_through_the_registry(monkeypatch):
    from neo.agent import registry as reg

    r = reg.Registry()
    got = ToolOutput("Clicked.", effect=Effect(changed=False), handoff=Handoff("which one?", [Choice("e1", "OK")]))
    r.register(RegisteredTool("t_effect", "", {"type": "object", "properties": {}}, lambda a, c: got))
    out = asyncio.run(r.invoke("t_effect", {}, ToolContext()))
    assert out.effect == Effect(changed=False) and out.handoff.choices == [Choice("e1", "OK")]


# ---- @tool(by_utterance): the user's own words only --------------------------------------------------


def test_the_utterance_lane_is_for_the_users_own_words_only():
    t = RegisteredTool("t_utt", "", {}, lambda a, c: "", by_utterance=True)
    plain = RegisteredTool("t_plain", "", {}, lambda a, c: "")
    assert utterance_lane(t, "user")
    for origin in ("live", "agent", "approval"):  # Live relaying, a model (neo2's too), a replay
        assert not utterance_lane(t, origin), origin
    assert not utterance_lane(plain, "user") and not utterance_lane(None, "user")


def test_the_utterance_lane_is_closed_to_an_embedded_agent(monkeypatch):
    """neo2 says "user" for its default source: an embedded agent's words are a host's, never the
    user's own to NEO.app (tests/sdk/test_sdk_restriction.py goes through ask() with both sources)."""
    t = RegisteredTool("t_utt", "", {}, lambda a, c: "", by_utterance=True)
    monkeypatch.setattr(runtime, "library_bound", lambda: True)  # an embedded agent's request runs
    assert not utterance_lane(t, "user")
    monkeypatch.setattr(runtime, "library_bound", lambda: False)
    monkeypatch.setattr(runtime, "_library", True)  # an embedded agent exists in this process
    assert not utterance_lane(t, "user")


def test_the_tool_decorator_takes_by_utterance():
    import inspect

    from neo.agent import registry as reg

    assert inspect.signature(reg.tool).parameters["by_utterance"].default is False
    assert RegisteredTool("x", "", {}, lambda a, c: "").by_utterance is False


# ---- ImagePart.detail and image_limit() ---------------------------------------------------------------


def test_an_image_has_a_detail_level_last_with_a_default():
    assert [f.name for f in dataclasses.fields(ImagePart)] == ["data", "mime", "detail"]
    assert ImagePart(b"x", "image/jpeg").detail == "auto"


class _Brain:
    def __init__(self, name, vision=True, limit=None):
        self.name, self.supports_vision, self.supports_tools = name, vision, True
        if limit is not None:
            self.image_limit = limit


def test_image_limit_is_the_strictest_of_the_seeing_brains():
    from neo.providers import Chain

    assert Chain([_Brain("a"), _Brain("b")]).image_limit is None  # none declares one
    chain = Chain([_Brain("a", limit=(2576, 3_750_000)), _Brain("b", limit=(1568, 1_150_000)), _Brain("t", vision=False, limit=(10, 10))])
    assert chain.image_limit == (1568, 1_150_000)  # a text-only brain's limit doesn't count
    nested = Chain([_Brain("a", limit=(2000, 4_000_000)), Chain([_Brain("b", limit=(3000, 2_000_000))])])
    assert nested.image_limit == (2000, 2_000_000)


def test_image_limit_covers_a_hosts_own_brains(monkeypatch):
    from neo import providers

    assert providers.image_limit("vision") is None  # NEO.app with no keys: nothing declared
    monkeypatch.setattr(providers, "_hosts", lambda purpose: [_Brain("host", limit=(1024, 800_000))])
    assert providers.image_limit("vision") == (1024, 800_000)


# ---- wait.until on a fake clock ------------------------------------------------------------------


def test_until_returns_at_the_first_poll_that_sees_it(fake_clock):
    from neo.tools.computer import wait

    ready_at = fake_clock.t + 0.06
    got = asyncio.run(wait.until(lambda: fake_clock.t >= ready_at and "in front", timeout=2.0, every=0.015))
    assert got == "in front" and len(fake_clock.sleeps) == 4 and sum(fake_clock.sleeps) == pytest.approx(0.06)


def test_until_gives_up_at_its_timeout_and_takes_async_checks(fake_clock):
    from neo.tools.computer import wait

    async def never():
        return None

    start = fake_clock.t
    assert asyncio.run(wait.until(never, timeout=0.4, every=0.1)) is None
    assert fake_clock.t - start == pytest.approx(0.4) and sum(fake_clock.sleeps) == pytest.approx(0.4)
    fake_clock.sleeps.clear()
    assert asyncio.run(wait.until(lambda: 7, timeout=0.0)) == 7 and fake_clock.sleeps == []  # checked at once


# ---- settings ---------------------------------------------------------------------------------------


def test_the_controls_settings_and_their_defaults(monkeypatch):
    from neo import config

    s = config.settings()
    assert (s.screen_effort, s.prefetch_look, s.shot_format, s.resolver) == ("high", True, "jpeg", "shadow")
    assert (s.resolver_laya_p, s.local_grounder, s.ground_port, s.sck_capture, s.sck_stream) == (0.7, "", 8081, False, False)
    for key, typo, default in (("resolver", "livee", "shadow"), ("shot_format", "gif", "jpeg"), ("screen_effort", "max", "high")):
        monkeypatch.setenv(f"NEO_{key.upper()}", typo)
        monkeypatch.setattr(config, "_settings", None)
        assert getattr(config.settings(), key) == default, key  # a typo turns nothing on
    monkeypatch.setenv("NEO_RESOLVER", " Live ")
    monkeypatch.setattr(config, "_settings", None)
    assert config.settings().resolver == "live"


# ---- one door for Accessibility actions ----------------------------------------------------------------


def test_every_ax_action_goes_through_the_one_door_and_its_watcher_first(monkeypatch):
    import ApplicationServices as AS

    from neo.tools.computer import ax

    done: list[tuple] = []
    seen: list[tuple] = []

    def fake_perform(el, action):
        done.append(("perform", el, action))
        return 0

    def fake_set(el, name, value):
        done.append(("set", el, name, value))
        return 0

    monkeypatch.setattr(AS, "AXUIElementPerformAction", fake_perform)
    monkeypatch.setattr(AS, "AXUIElementSetAttributeValue", fake_set)
    monkeypatch.setattr(ax, "_watch", lambda kind, el, name, value: seen.append((kind, el, name)))
    assert ax.perform_action("el", "AXPress") == 0 and ax.set_attribute("el", "AXFocused", True) == 0
    assert seen == [("perform", "el", "AXPress"), ("set", "el", "AXFocused")] and len(done) == 2

    def refuse(kind, el, name, value):
        raise RuntimeError("not in the test window")

    monkeypatch.setattr(ax, "_watch", refuse)
    done.clear()
    with pytest.raises(RuntimeError):
        ax.perform_action("el", "AXPress")
    with pytest.raises(RuntimeError):
        ax.set_attribute("el", "AXValue", "x")
    assert done == []  # the watcher said no: nothing was done


def test_the_door_is_refused_with_effects_off():
    from neo.tools.computer import ax

    with pytest.raises(effects.EffectRefused):
        ax.perform_action(object(), "AXPress")
    with pytest.raises(effects.EffectRefused):
        ax.set_attribute(object(), "AXValue", "x")


def test_presses_focus_raises_and_the_task_kinds_all_use_the_door(monkeypatch):
    from neo.decide import media
    from neo.tools.computer import ax

    calls: list[tuple] = []
    monkeypatch.setattr(ax, "perform_action", lambda el, action, **k: calls.append(("perform", action, k.get("kind"))) or 0)
    monkeypatch.setattr(ax, "set_attribute", lambda el, name, v, **k: calls.append(("set", name, k.get("kind"))) or 0)
    monkeypatch.setattr(ax, "_attr", lambda el, name, default=None: True if name == "AXFocused" else default)
    e = ax.Element("e1", "AXButton", "OK", "", 0, 0, 10, 10, actions=["AXPress"], ref="ref")
    assert ax.press_el(e).startswith("Pressed") and ax.press_element(e) == "Pressed"
    assert ax.focus_el("ref") is True
    assert media._perform("el", "AXPress") and media._set("el", "AXValue", "x")
    assert [c[:2] for c in calls] == [("perform", "AXPress"), ("perform", "AXPress"), ("set", "AXFocused"), ("perform", "AXPress"), ("set", "AXValue")]


def test_no_ax_action_bypasses_the_door():
    """AXUIElementPerformAction / SetAttributeValue are called in one place: the door."""
    root = Path(runtime.__file__).resolve().parent
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("AXUIElementPerformAction", "AXUIElementSetAttributeValue")
            ):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert sorted(offenders) == ["tools/computer/ax.py:" + str(n) for n in _door_lines(root / "tools/computer/ax.py")]


def _door_lines(path: Path) -> list[int]:
    tree = ast.parse(path.read_text())
    doors = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("perform_action", "set_attribute")]
    return sorted(
        c.lineno
        for d in doors
        for c in ast.walk(d)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr.startswith("AXUIElement")
    )


# ---- capturing the screen in NEO's own process ---------------------------------------------------------


def test_an_in_process_capture_is_an_effect_of_its_own():
    assert effects.CAPTURE == "capture"
    with pytest.raises(effects.EffectRefused) as e:
        effects.guard(effects.CAPTURE, "capture the screen in NEO's process")
    assert "capture" in str(e.value)
    with effects.allow(effects.CAPTURE):
        effects.guard(effects.CAPTURE, "capture the screen in NEO's process")  # opted in: through


# ---- every tool that acts is one of neo2's INPUT_TOOLS ---------------------------------------------------

_COMPUTER_DIR = Path(runtime.__file__).resolve().parent / "tools" / "computer"
_SCANNED = ("gates", "tools_ax", "tools_pointer", "tools_keys", "tools_label", "tools_apps", "pointer", "ax", "input",
            "apps", "focus", "screen")  # fmt: skip
# What makes a tool one that acts: an input event, or an Accessibility press, value, focus or text.
# (Bringing an app forward and switching a browser's web tree on for reading are not input.)
_ACTUATORS = {
    ("input", n) for n in ("click", "drag", "scroll", "type_text", "hotkey", "_post")
} | {("ax", n) for n in ("press", "press_el", "press_element", "set_value", "set_el_value", "focus_el")} | {
    ("axkit", n) for n in ("perform", "set_text", "set_search", "clear_text", "post_return")
}


def _call_graph() -> dict[tuple[str, str], set[tuple[str, str]]]:
    graph: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for mod in _SCANNED:
        tree = ast.parse((_COMPUTER_DIR / f"{mod}.py").read_text())
        alias: dict[str, str] = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module in ("neo.tools.computer", "neo.decide"):
                for a in n.names:
                    alias[a.asname or a.name] = a.name
        for fn in tree.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            refs: set[tuple[str, str]] = set()
            for n in ast.walk(fn):
                if isinstance(n, ast.Name):
                    refs.add((mod, n.id))
                elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in alias:
                    refs.add((alias[n.value.id], n.attr))
            graph[(mod, fn.name)] = refs
    return graph


def _acts(start: tuple[str, str], graph) -> bool:
    todo, seen = [start], set()
    while todo:
        node = todo.pop()
        if node in _ACTUATORS:
            return True
        if node in seen or node not in graph:
            continue
        seen.add(node)
        todo.extend(graph[node])
    return False


def test_every_computer_tool_that_acts_is_an_input_tool():
    from neo.agent.registry import catalog
    from neo.tools import load_all
    from neo2.mcp.server import INPUT_TOOLS

    load_all()
    graph = _call_graph()
    mine = [t for t in catalog().all() if t.handler.__module__.startswith("neo.tools.computer.")]
    acting = {t.name for t in mine if _acts((t.handler.__module__.rsplit(".", 1)[1], t.handler.__name__), graph)}
    assert acting >= {"click", "drag", "scroll", "type_text", "dictate", "hotkey", "click_text", "ax_press", "ax_set_value"}
    assert acting <= INPUT_TOOLS, acting - INPUT_TOOLS
    # and by what a tool says of itself: a computer tool that isn't read-only acts
    declared = {t.name for t in mine if t.category == "computer" and not t.read_only}
    assert declared <= INPUT_TOOLS, declared - INPUT_TOOLS


def test_input_tools_and_the_planned_tools_are_known_names():
    from neo.agent.registry import catalog
    from neo.tools import MODULES_FOR, PLANNED, PRESETS, load_all
    from neo2.mcp.server import INPUT_TOOLS

    load_all()
    assert set(PRESETS["all"]) | set(PLANNED) >= INPUT_TOOLS
    assert "menu_command" in INPUT_TOOLS and PLANNED == {"menu_command": "mac", "look": "mac_read"}
    for name, preset in PLANNED.items():
        assert MODULES_FOR[name] == "neo.tools.computer"
        # once it is registered it must be in its preset (else no embedded agent can have it)
        assert catalog().get(name) is None or name in PRESETS[preset], name
    from neo.tools.computer import ORDER

    assert "menu_command" in ORDER and "look" in ORDER  # their places in the catalog are kept


# ---- nothing process-wide in library mode -----------------------------------------------------------------


def test_process_wide_work_only_as_neo_app(monkeypatch):
    assert runtime.process_wide_ok()  # NEO.app (the suite's default: not embedded)
    monkeypatch.setattr(runtime, "_library", True)  # an embedded agent exists in this process
    assert not runtime.process_wide_ok()
    monkeypatch.setattr(runtime, "_library", False)
    monkeypatch.setattr(runtime, "library_bound", lambda: True)  # an embedded agent's request runs
    assert not runtime.process_wide_ok()


def test_no_web_tree_switch_or_timer_thread_in_library_mode(monkeypatch):
    """ax_tree on a Chromium app switches its web tree on and holds it with a timer thread: as
    NEO.app only — never inside a host's process."""
    from neo.tools.computer import ax

    asked: list[int] = []
    monkeypatch.setattr(ax, "_running", lambda pid: asked.append(pid))
    monkeypatch.setattr(runtime, "_library", True)
    assert ax._expose_web(4242, object()) is False and asked == [] and ax._web_held == {}
    monkeypatch.setattr(runtime, "_library", False)
    assert ax._expose_web(4242, object()) is False and asked == [4242]  # NEO.app: it looks (no bundle here)


try:  # (the suite fakes it: nothing has the keyboard focus)
    from neo.tools.computer.ax import focused_element_text as _REAL_FOCUSED_TEXT
except ImportError:  # no macOS frameworks
    _REAL_FOCUSED_TEXT = None


def _ax_timeouts(monkeypatch):
    """ApplicationServices faked for the system-wide lookups: every messaging timeout NEO sets, by
    element, and the focused element and the element at any point."""
    import ApplicationServices as AS

    system, focused, at_point = object(), object(), object()
    set_on: list[tuple[object, float]] = []

    def copy(el, name, _):
        if el is system and name == "AXFocusedUIElement":
            return 0, focused
        return -25212, None

    monkeypatch.setattr(AS, "AXUIElementCreateSystemWide", lambda: system)
    monkeypatch.setattr(AS, "AXUIElementSetMessagingTimeout", lambda el, t: set_on.append((el, t)) or 0)
    monkeypatch.setattr(AS, "AXUIElementCopyAttributeValue", copy)
    monkeypatch.setattr(AS, "AXUIElementCopyElementAtPosition", lambda el, x, y, _: (0, at_point))
    return system, focused, at_point, set_on


def test_no_process_wide_ax_timeout_in_library_mode(monkeypatch):
    """A timeout on the system-wide element is the whole process's default: as NEO.app only. In a
    host's process the elements NEO reads through it get their own timeout instead."""
    from neo.tools.computer import ax

    system, focused, at_point, set_on = _ax_timeouts(monkeypatch)
    monkeypatch.setattr(runtime, "_library", True)
    for call in (ax.focused_role, ax.focused_editable, _REAL_FOCUSED_TEXT, ax.focused_terminal,
                 lambda: ax._element_at(10, 10), lambda: ax.label_at(10, 10)):  # fmt: skip
        set_on.clear()
        call()
        assert all(el is not system for el, _ in set_on), call  # the host's default is left as it is
        assert set_on and all(t == ax._AX_TIMEOUT_S for _, t in set_on)  # what was read answers briefly
    assert (focused, ax._AX_TIMEOUT_S) in set_on or (at_point, ax._AX_TIMEOUT_S) in set_on
    set_on.clear()
    ax.focused_role()
    assert (focused, ax._AX_TIMEOUT_S) in set_on
    monkeypatch.setattr(runtime, "_library", False)  # NEO.app: its own process's default, 2 s
    set_on.clear()
    ax.focused_role()
    assert (system, ax._AX_TIMEOUT_S) in set_on


def test_the_self_tests_hit_test_never_leaves_its_own_process_wide_timeout(monkeypatch):
    """The controls self-test's watcher asks Accessibility what is under each point: through NEO's
    one system-wide helper, so NEO.app's default stays 2 s after a test (never 0.5 s)."""
    from neo.selftest import mac as smac
    from neo.tools.computer import ax

    system, _focused, _at, set_on = _ax_timeouts(monkeypatch)
    m = smac.Mac.__new__(smac.Mac)
    with effects.allow("selftest"):
        m.hit_pid(1, 1)
    assert [t for el, t in set_on if el is system] == [ax._AX_TIMEOUT_S]


# ---- the doors: every event and AX action, the typed fallbacks too; a refusal is never swallowed --------


class _Stop(Exception):  # like neo.selftest.watch.Stop: a watcher's refusal is a plain Exception
    pass


def _text_field(monkeypatch, pid: int = 4242):
    """A text field of app `pid` that ignores AXValue and reads back empty: NEO types into it."""
    import ApplicationServices as AS

    from neo.decide import media
    from neo.tools.computer import ax

    e = ax.Element("e3", "AXTextField", "", "", 10, 10, 100, 20, ref=object())
    monkeypatch.setattr(ax, "_attr", lambda el, name, default=None: default)  # not secure, no ancestor
    monkeypatch.setattr(media, "_attr", lambda el, name, default=None: "")  # the field ignores AXValue
    monkeypatch.setattr(AS, "AXUIElementGetPid", lambda el, _: (0, pid))
    monkeypatch.setattr(AS, "AXUIElementSetAttributeValue", lambda el, n, v: -25200)  # "didn't take it"
    monkeypatch.setattr(ax, "app_names", lambda p: {"textedit"})
    return e


def test_typing_into_one_app_goes_through_the_input_door_and_its_watcher(monkeypatch):
    import Quartz

    from neo.tools.computer import ax
    from neo.tools.computer import input as inp

    e = _text_field(monkeypatch)
    posted, seen = [], []
    monkeypatch.setattr(Quartz, "CGEventPostToPid", lambda pid, ev: posted.append(pid))
    monkeypatch.setattr(inp, "_watch", lambda ev: seen.append(Quartz.CGEventGetIntegerValueField(ev, Quartz.kCGEventTargetUnixProcessID)))
    ax.set_el_value(e, "hello")
    assert posted and set(posted) == {4242}
    assert len(seen) == len(posted) and set(seen) == {4242}  # the watcher saw each, with where it goes

    def refuse(ev):
        raise _Stop("keys would have left the test window")

    monkeypatch.setattr(inp, "_watch", refuse)
    posted.clear()
    with pytest.raises(_Stop):
        ax.set_el_value(e, "hello")
    assert posted == []


def test_a_refusing_ax_watcher_reaches_the_caller_and_nothing_is_typed(monkeypatch):
    import ApplicationServices as AS
    import Quartz

    from neo.decide import media
    from neo.tools.computer import ax

    e = _text_field(monkeypatch)
    posted = []

    def watcher(kind, el, name, value):
        raise _Stop("not in the test window")

    monkeypatch.setattr(ax, "_watch", watcher)
    monkeypatch.setattr(Quartz, "CGEventPostToPid", lambda pid, ev: posted.append(pid))
    with pytest.raises(_Stop):
        ax.set_el_value(e, "hello")  # the focus set is refused: no typing takes its place
    assert posted == []
    for call in (lambda: media._set("el", "AXValue", "x"), lambda: media._perform("el", "AXPress"),
                 lambda: ax.focus_el("el")):  # fmt: skip
        with pytest.raises(_Stop):
            call()
    win = object()
    monkeypatch.setattr(ax, "_attr", lambda el, name, default=None: "AXWindow" if el is win else object())
    monkeypatch.setattr(ax, "_same", lambda a, b: False)
    # (the suite fakes raise_window: the real one here)
    real_raise = next(old for obj, name, old in monkeypatch._setattr if obj is ax and name == "raise_window")
    with pytest.raises(_Stop):
        real_raise(ax.Hit(pid=4242, app="TextEdit", window=win))
    # an app that fails (or an element that can't be asked) is a failure, not a refusal
    monkeypatch.setattr(ax, "_watch", None)
    monkeypatch.setattr(AS, "AXUIElementPerformAction", lambda el, a: (_ for _ in ()).throw(ValueError("bad ref")))
    assert ax.perform_action(object(), "AXPress") != 0 and media._perform(object(), "AXPress") is False


def test_no_event_is_posted_past_the_input_door():
    """CGEventPost / CGEventPostToPid are called in input's two doors only (and by the self-test's
    watch to let go of NEO's own gesture, which it posts through its Mac)."""
    root = Path(runtime.__file__).resolve().parent
    posts: list[tuple[str, str]] = []  # (file, the top-level function or class it is in)
    for path in root.rglob("*.py"):
        for top in ast.parse(path.read_text()).body:
            for node in ast.walk(top):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("CGEventPost", "CGEventPostToPid"):
                    posts.append((str(path.relative_to(root)), getattr(top, "name", "")))
    assert sorted(posts) == [
        ("selftest/mac.py", "Mac"),  # Mac.post: the watch letting go of NEO's own button, in the test window
        ("tools/computer/input.py", "_post"),
        ("tools/computer/input.py", "_post_to_pid"),
    ]


def test_the_doors_never_act_in_neos_own_process_or_a_protected_app(monkeypatch):
    import ApplicationServices as AS
    import Quartz

    from neo import config
    from neo.decide import media
    from neo.tools.computer import ax
    from neo.tools.computer import input as inp

    done, posted = [], []
    monkeypatch.setattr(AS, "AXUIElementPerformAction", lambda el, a: done.append(a) or 0)
    monkeypatch.setattr(AS, "AXUIElementSetAttributeValue", lambda el, n, v: done.append(n) or 0)
    monkeypatch.setattr(Quartz, "CGEventPostToPid", lambda pid, ev: posted.append(pid))
    pids = {"own": 0, "guarded": 0}
    monkeypatch.setattr(AS, "AXUIElementGetPid", lambda el, _: (0, pids[el]))
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": pid == 101)
    monkeypatch.setattr(ax, "app_names", lambda pid: {"Approver", "com.example.approver"} if pid == 202 else {"Other"})
    monkeypatch.setenv("NEO_PROTECTED_APPS", '["Approver"]')
    monkeypatch.setattr(config, "_settings", None)
    pids.update(own=101, guarded=202)
    for el in ("own", "guarded"):
        with pytest.raises(ax.OffLimits):
            ax.perform_action(el, "AXPress")
        with pytest.raises(ax.OffLimits):
            ax.set_attribute(el, "AXValue", "x")
        with pytest.raises(ax.OffLimits):
            media._perform(el, "AXPress")  # a task kind's press: the same door, the same no
        assert ax.set_attribute(el, "AXManualAccessibility", True) == 0  # making it readable is no action
    for pid in (101, 202):
        with pytest.raises(ax.OffLimits):
            inp._post_to_pid(pid, Quartz.CGEventCreateKeyboardEvent(None, 36, True))
    assert done == ["AXManualAccessibility", "AXManualAccessibility"] and posted == []
    inp._post_to_pid(303, Quartz.CGEventCreateKeyboardEvent(None, 0, True))  # any other app: through
    assert posted == [303]


def test_the_self_tests_watch_judges_keys_for_one_app_by_where_they_go():
    """Keys posted to one app (input._post_to_pid) go there whatever is in front: the watch lets
    them through only when that app is the test window's, and refuses them for any other."""
    import Quartz as Q

    from neo.selftest.watch import Stop, Watch

    class Mac:
        front_app = ("Safari", 77)

        def front(self):
            return self.front_app

        def buttons_down(self):
            return 0

        def post(self, ev):
            pass

    def key(target: int):
        ev = Q.CGEventCreateKeyboardEvent(None, 0, True)
        Q.CGEventSetIntegerValueField(ev, Q.kCGEventTargetUnixProcessID, target)
        return ev

    mac = Mac()
    Watch(mac, 4242, {})(key(4242))  # into the test window's app, with the user's Safari in front
    mac.front_app = ("Test", 4242)
    w = Watch(mac, 4242, {})
    with pytest.raises(Stop):
        w(key(77))  # into another app, though the test window is in front
    assert "pid 77" in w.refused


# ---- ax_set_value never types a line break (a Return would run or send it) --------------------------


def _set_value_world(monkeypatch, *, hosts_shell, terminal: bool):
    from neo.decide import axkit, media
    from neo.tools.computer import ax, focus, tools_ax

    typed, values = [], []
    monkeypatch.setattr(axkit, "_type", lambda pid, text, newline: typed.append((pid, text, newline)))
    monkeypatch.setattr(axkit, "_pid_of", lambda el: 4242)
    monkeypatch.setattr(media, "_set", lambda el, name, v: values.append((name, v)) or True)
    # the value is "taken", but the field reads back empty (a web terminal's input textarea)
    monkeypatch.setattr(media, "_attr", lambda el, name, default=None: "" if name == "AXValue" else default)
    e = ax.Element("e7", "AXTextArea", "", "", 10, 10, 300, 20, ref="REF")
    monkeypatch.setattr(ax, "get", lambda eid: e)
    monkeypatch.setattr(ax, "_attr", lambda el, name, default=None: default)
    monkeypatch.setattr(tools_ax, "_element_pid", lambda el: 4242)
    monkeypatch.setattr(ax, "app_of", lambda ref: "Code")
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": False)
    monkeypatch.setattr(ax, "app_names", lambda pid: {"code"})
    monkeypatch.setattr(focus, "hosts_shell", lambda pid, depth=3: hosts_shell)
    monkeypatch.setattr(ax, "focused_terminal", lambda pid=0: terminal)
    monkeypatch.setattr(ax, "is_terminal", lambda el: terminal)
    return typed, values


@pytest.mark.parametrize("value", ["cd /tmp\nrm -rf build", "cd /tmp\rrm -rf build", "cd /tmp\r\nrm -rf build"])
def test_ax_set_value_joins_the_lines_on_a_command_line(monkeypatch, value):
    from neo.agent import confirm
    from neo.agent.registry import ToolContext
    from neo.tools.computer import tools_ax

    typed, values = _set_value_world(monkeypatch, hosts_shell="pane", terminal=True)
    monkeypatch.setattr(confirm, "gated", lambda *a, **k: pytest.fail("nothing here needs asking: nothing runs"))
    asyncio.run(tools_ax.ax_set_value({"id": "e7", "value": value}, ToolContext(origin="agent")))
    assert values and all("\n" not in v and "\r" not in v for n, v in values if n == "AXValue")
    assert typed and all("\n" not in t and "\r" not in t for _, t, _ in typed)  # no Return, ever
    from neo.decide import media  # a field that takes the value: the result says it went in as one line

    monkeypatch.setattr(media, "_attr", lambda el, name, default=None: values[-1][1] if name == "AXValue" else default)
    out = asyncio.run(tools_ax.ax_set_value({"id": "e7", "value": value}, ToolContext(origin="agent")))
    assert out.startswith("Set e7") and "one line" in out and values[-1] == ("AXValue", "cd /tmp rm -rf build")


def test_ax_set_value_types_no_line_break_where_the_value_did_not_take(monkeypatch):
    from neo.agent.registry import ToolContext
    from neo.tools.computer import tools_ax

    typed, values = _set_value_world(monkeypatch, hosts_shell=False, terminal=False)
    out = asyncio.run(tools_ax.ax_set_value({"id": "e7", "value": "Dear team,\nthanks!"}, ToolContext(origin="agent")))
    assert ("AXValue", "Dear team,\nthanks!") in values  # a value set is no keystroke: lines kept
    assert typed == [] and out.startswith("Error") and "type_text" in out  # typing them: type_text's call
    typed.clear()
    asyncio.run(tools_ax.ax_set_value({"id": "e7", "value": "no breaks here"}, ToolContext(origin="agent")))
    assert typed == [(4242, "no breaks here", "")]
