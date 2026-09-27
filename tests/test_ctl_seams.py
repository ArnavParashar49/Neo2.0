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

from neo2.core import effects, runtime
from neo2.core.agent.registry import (
    Choice,
    Effect,
    Handoff,
    RegisteredTool,
    ToolContext,
    ToolOutput,
    utterance_lane,
)
from neo2.core.providers.base import ImagePart

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
    from neo2.core.agent import registry as reg

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

    from neo2.core.agent import registry as reg

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
    from neo2.core.providers import Chain

    assert Chain([_Brain("a"), _Brain("b")]).image_limit is None  # none declares one
    chain = Chain([_Brain("a", limit=(2576, 3_750_000)), _Brain("b", limit=(1568, 1_150_000)), _Brain("t", vision=False, limit=(10, 10))])
    assert chain.image_limit == (1568, 1_150_000)  # a text-only brain's limit doesn't count
    nested = Chain([_Brain("a", limit=(2000, 4_000_000)), Chain([_Brain("b", limit=(3000, 2_000_000))])])
    assert nested.image_limit == (2000, 2_000_000)


def test_image_limit_covers_a_hosts_own_brains(monkeypatch):
    from neo2.core import providers

    assert providers.image_limit("vision") is None  # NEO.app with no keys: nothing declared
    monkeypatch.setattr(providers, "_hosts", lambda purpose: [_Brain("host", limit=(1024, 800_000))])
    # (brain-effort: the limit is an ImageLimit — edge, pixels and bytes; a declared pair still counts)
    assert providers.image_limit("vision") == providers.ImageLimit(1024, 800_000)


# ---- wait.until on a fake clock ------------------------------------------------------------------


def test_until_returns_at_the_first_poll_that_sees_it(fake_clock):
    from neo2.core.tools.computer import wait

    ready_at = fake_clock.t + 0.06
    got = asyncio.run(wait.until(lambda: fake_clock.t >= ready_at and "in front", timeout=2.0, every=0.015))
    assert got == "in front" and len(fake_clock.sleeps) == 4 and sum(fake_clock.sleeps) == pytest.approx(0.06)


def test_until_gives_up_at_its_timeout_and_takes_async_checks(fake_clock):
    from neo2.core.tools.computer import wait

    async def never():
        return None

    start = fake_clock.t
    assert asyncio.run(wait.until(never, timeout=0.4, every=0.1)) is None
    assert fake_clock.t - start == pytest.approx(0.4) and sum(fake_clock.sleeps) == pytest.approx(0.4)
    fake_clock.sleeps.clear()
    assert asyncio.run(wait.until(lambda: 7, timeout=0.0)) == 7 and fake_clock.sleeps == []  # checked at once


# ---- settings ---------------------------------------------------------------------------------------


def test_the_controls_settings_and_their_defaults(monkeypatch):
    from neo2.core import config

    monkeypatch.delenv("NEO_PREFETCH_LOOK", raising=False)  # (the suite turns the first look off: tests/conftest.py)
    monkeypatch.setattr(config, "_settings", None)
    s = config.settings()
    assert (s.screen_effort, s.prefetch_look, s.shot_format, s.resolver) == ("high", True, "jpeg", "shadow")
    assert (s.prefetch_look_s, s.prewarm_resolver) == (0.5, True)
    assert (s.resolver_laya_p, s.local_grounder, s.ground_port, s.sck_capture, s.sck_stream) == (0.9, "", 8081, False, False)
    assert s.resolver_promoted == ""  # (no shadow stage acts until promoted)
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

    from neo2.core.tools.computer import ax

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
    from neo2.core.tools.computer import ax

    with pytest.raises(effects.EffectRefused):
        ax.perform_action(object(), "AXPress")
    with pytest.raises(effects.EffectRefused):
        ax.set_attribute(object(), "AXValue", "x")


def test_presses_focus_raises_and_the_task_kinds_all_use_the_door(monkeypatch):
    from neo2.core.decide import media
    from neo2.core.tools.computer import ax

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

_NEO = Path(runtime.__file__).resolve().parent
_PKG = runtime.__name__.rpartition(".")[0]  # NEO's package (neo2.core), as its modules are named
# Where a tool's handler can lead: the tools, the computer package, and the task kinds with the
# Accessibility kit they share (neo2.core.decide: axkit, media, the kinds).
_SCANNED_DIRS = ("tools", "tools/computer", "decide", "decide/kinds")
# What makes a tool one that acts: an input event (to the front app or to one app), or an
# Accessibility press, value, focus or text. (Bringing an app forward — its window raised through
# the AX door — and switching a browser's web tree on for reading are not input.)
_ACTUATORS = (
    {("neo2.core.tools.computer.input", n) for n in ("click", "drag", "scroll", "type_text", "hotkey", "_post", "_post_to_pid")}
    | {("neo2.core.tools.computer.ax", n) for n in ("press", "press_el", "press_element", "set_value", "set_el_value", "focus_el")}
    | {("neo2.core.decide.axkit", n) for n in ("perform", "set_text", "set_search", "clear_text", "post_return", "_put", "_type")}
    | {("neo2.core.decide.media", n) for n in ("_perform", "_post_return", "_set_query")}
)  # fmt: skip
# A task kind's driver is its own module (run.run_best(sys.modules[__name__], …)): these are called.
_DRIVER_HOOKS = ("attempt", "fallback", "accepts")


def _modules() -> dict[str, Path]:
    out = {}
    for d in _SCANNED_DIRS:
        for path in sorted((_NEO / d).glob("*.py")):
            rel = path.relative_to(_NEO).with_suffix("")
            parts = [_PKG, *rel.parts] if rel.name != "__init__" else [_PKG, *rel.parts[:-1]]
            out[".".join(parts)] = path
    return out


def _call_graph() -> dict[tuple[str, str], set[tuple[str, str]]]:
    """(module, top-level function) -> what it names: its module's functions, and the functions
    of the scanned modules it imports (``from neo2.core.decide.kinds import site as sk`` … ``sk.search``,
    ``from neo2.core.decide.media import play_best``), called or handed on (a worker, a driver)."""
    mods = _modules()
    graph: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for mod, path in mods.items():
        tree = ast.parse(path.read_text())
        alias: dict[str, str] = {}  # name -> a scanned module
        func: dict[str, tuple[str, str]] = {}  # name -> (a scanned module, its function)
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module:
                for a in n.names:
                    if f"{n.module}.{a.name}" in mods:
                        alias[a.asname or a.name] = f"{n.module}.{a.name}"
                    elif n.module in mods:
                        func[a.asname or a.name] = (n.module, a.name)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.asname and a.name in mods:
                        alias[a.asname] = a.name
        for top in tree.body:
            if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = top.name
            elif isinstance(top, ast.Assign) and len(top.targets) == 1 and isinstance(top.targets[0], ast.Name):
                name = top.targets[0].id  # a table of workers or methods
            else:
                continue
            refs: set[tuple[str, str]] = set()
            for n in ast.walk(top):
                if isinstance(n, ast.Name):
                    refs.add(func.get(n.id, (mod, n.id)))
                elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in alias:
                    refs.add((alias[n.value.id], n.attr))
                elif isinstance(n, ast.Subscript) and ast.unparse(n) == "sys.modules[__name__]":
                    refs |= {(mod, h) for h in _DRIVER_HOOKS}
            graph[(mod, name)] = refs
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


def test_every_tool_that_acts_is_an_input_tool():
    """Every registered tool that posts an event or takes an Accessibility action — the computer
    tools, and the task tools whose kinds act in an app's own UI (a site's search box, a new note)
    — is one of neo2's INPUT_TOOLS, so an MCP server asking its client for approvals (--approve
    elicit) can't hand them out without the client's app protected."""
    from neo2.core.agent.registry import catalog
    from neo2.core.tools import PRESETS, load_all
    from neo2.mcp.server import INPUT_TOOLS

    load_all()
    graph = _call_graph()
    tools = catalog().all()
    acting = {t.name for t in tools if _acts((t.handler.__module__, t.handler.__name__), graph)}
    assert acting >= {"click", "drag", "scroll", "type_text", "dictate", "hotkey", "click_text", "ax_press", "ax_set_value"}
    assert acting >= {"search_site", "notes_create"}  # a site's search box and Return; a new note's New and text
    # NEO.app's own (never an embedded agent's: in no preset) need no entry — nor could they have one
    embeddable = acting & set(PRESETS["all"])
    assert embeddable <= INPUT_TOOLS, embeddable - INPUT_TOOLS
    # and by what a tool says of itself: a computer tool that isn't read-only acts
    mine = [t for t in tools if t.handler.__module__.startswith("neo2.core.tools.computer.")]
    declared = {t.name for t in mine if t.category == "computer" and not t.read_only}
    assert declared <= INPUT_TOOLS, declared - INPUT_TOOLS
    assert not acting & {t.name for t in tools if t.read_only}  # (a look never acts)


def test_input_tools_and_the_planned_tools_are_known_names():
    from neo2.core.agent.registry import catalog
    from neo2.core.tools import MODULES_FOR, PLANNED, PRESETS, load_all
    from neo2.mcp.server import INPUT_TOOLS

    load_all()
    assert set(PRESETS["all"]) | set(PLANNED) >= INPUT_TOOLS
    assert "menu_command" in INPUT_TOOLS and PLANNED == {"menu_command": "mac", "look": "mac_read"}
    for name, preset in PLANNED.items():
        assert MODULES_FOR[name] == "neo2.core.tools.computer"
        # once it is registered it must be in its preset (else no embedded agent can have it)
        assert catalog().get(name) is None or name in PRESETS[preset], name
    from neo2.core.tools.computer import ORDER

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
    from neo2.core.tools.computer import ax

    asked: list[int] = []
    monkeypatch.setattr(ax, "_running", lambda pid: asked.append(pid))
    monkeypatch.setattr(runtime, "_library", True)
    assert ax._expose_web(4242, object()) is False and asked == [] and ax._web_held == {}
    monkeypatch.setattr(runtime, "_library", False)
    assert ax._expose_web(4242, object()) is False and asked == [4242]  # NEO.app: it looks (no bundle here)


try:  # (the suite fakes it: nothing has the keyboard focus)
    from neo2.core.tools.computer.ax import focused_element_text as _REAL_FOCUSED_TEXT
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
    from neo2.core.tools.computer import ax

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
    from neo2.core.selftest import mac as smac
    from neo2.core.tools.computer import ax

    system, _focused, _at, set_on = _ax_timeouts(monkeypatch)
    m = smac.Mac.__new__(smac.Mac)
    with effects.allow("selftest"):
        m.hit_pid(1, 1)
    assert [t for el, t in set_on if el is system] == [ax._AX_TIMEOUT_S]


# ---- the doors: every event and AX action, the typed fallbacks too; a refusal is never swallowed --------


class _Stop(Exception):  # like neo2.core.selftest.watch.Stop: a watcher's refusal is a plain Exception
    pass


def _text_field(monkeypatch, pid: int = 4242):
    """A text field of app `pid` that ignores AXValue and reads back empty: NEO types into it."""
    import ApplicationServices as AS

    from neo2.core.decide import media
    from neo2.core.tools.computer import ax

    e = ax.Element("e3", "AXTextField", "", "", 10, 10, 100, 20, ref=object())
    monkeypatch.setattr(ax, "_attr", lambda el, name, default=None: default)  # not secure, no ancestor
    monkeypatch.setattr(media, "_attr", lambda el, name, default=None: "")  # the field ignores AXValue
    monkeypatch.setattr(AS, "AXUIElementGetPid", lambda el, _: (0, pid))
    monkeypatch.setattr(AS, "AXUIElementSetAttributeValue", lambda el, n, v: -25200)  # "didn't take it"
    monkeypatch.setattr(ax, "app_names", lambda p: {"textedit"})
    return e


def test_typing_into_one_app_goes_through_the_input_door_and_its_watcher(monkeypatch):
    import Quartz

    from neo2.core.tools.computer import ax
    from neo2.core.tools.computer import input as inp

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

    from neo2.core.decide import media
    from neo2.core.tools.computer import ax

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

    from neo2.core import config
    from neo2.core.decide import media
    from neo2.core.tools.computer import ax
    from neo2.core.tools.computer import input as inp

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

    from neo2.core.selftest.watch import Stop, Watch

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
    from neo2.core.decide import axkit, media
    from neo2.core.tools.computer import ax, focus, tools_ax

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
    from neo2.core.agent import confirm
    from neo2.core.agent.registry import ToolContext
    from neo2.core.tools.computer import tools_ax

    typed, values = _set_value_world(monkeypatch, hosts_shell="pane", terminal=True)
    monkeypatch.setattr(confirm, "gated", lambda *a, **k: pytest.fail("nothing here needs asking: nothing runs"))
    asyncio.run(tools_ax.ax_set_value({"id": "e7", "value": value}, ToolContext(origin="agent")))
    assert values and all("\n" not in v and "\r" not in v for n, v in values if n == "AXValue")
    assert typed and all("\n" not in t and "\r" not in t for _, t, _ in typed)  # no Return, ever
    from neo2.core.decide import media  # a field that takes the value: the result says it went in as one line

    monkeypatch.setattr(media, "_attr", lambda el, name, default=None: values[-1][1] if name == "AXValue" else default)
    out = asyncio.run(tools_ax.ax_set_value({"id": "e7", "value": value}, ToolContext(origin="agent")))
    assert out.startswith("Set e7") and "one line" in out and values[-1] == ("AXValue", "cd /tmp rm -rf build")


def test_ax_set_value_types_no_line_break_where_the_value_did_not_take(monkeypatch):
    from neo2.core.agent.registry import ToolContext
    from neo2.core.tools.computer import tools_ax

    typed, values = _set_value_world(monkeypatch, hosts_shell=False, terminal=False)
    out = asyncio.run(tools_ax.ax_set_value({"id": "e7", "value": "Dear team,\nthanks!"}, ToolContext(origin="agent")))
    assert ("AXValue", "Dear team,\nthanks!") in values  # a value set is no keystroke: lines kept
    assert typed == [] and out.startswith("Error") and "type_text" in out  # typing them: type_text's call
    typed.clear()
    asyncio.run(tools_ax.ax_set_value({"id": "e7", "value": "no breaks here"}, ToolContext(origin="agent")))
    assert typed == [(4242, "no breaks here", "")]


# ---- the system (Apple) menu: never pressed, clicked or answered -------------------------------------


class _MenuEl:
    def __init__(self, role, title="", parent=None):
        self.role, self.title, self.parent, self.children = role, title, parent, []
        if parent is not None:
            parent.children.append(self)


def _menu_bar(monkeypatch):
    """A menu bar as every app's AX tree has it: the system's menu first, then the app's own."""
    from neo2.core.tools.computer import ax

    bar = _MenuEl("AXMenuBar")
    apple, app_menu = _MenuEl("AXMenuBarItem", "Apple", bar), _MenuEl("AXMenuBarItem", "TextEdit", bar)
    system = _MenuEl("AXMenu", "", apple)
    items = {t: _MenuEl("AXMenuItem", t, system) for t in ("Restart…", "Shut Down…", "Force Quit…", "Log Out")}
    new = _MenuEl("AXMenuItem", "New", _MenuEl("AXMenu", "", app_menu))
    window = _MenuEl("AXWindow", "Untitled")
    context = _MenuEl("AXMenuItem", "Restart", _MenuEl("AXMenu", "", window))  # a context menu's own item

    def attr(el, name, default=None):
        if not isinstance(el, _MenuEl):
            return default
        got = {"AXRole": el.role, "AXTitle": el.title, "AXParent": el.parent, "AXChildren": el.children}.get(name)
        return got if got not in (None, [], "") else default

    monkeypatch.setattr(ax, "_attr", attr)
    return apple, items, new, context


def test_the_system_menu_is_told_by_its_place_in_the_menu_bar(monkeypatch):
    from neo2.core.tools.computer import gates

    apple, items, new, context = _menu_bar(monkeypatch)
    assert gates._in_system_menu(apple) and all(gates._in_system_menu(el) for el in items.values())
    assert not gates._in_system_menu(new) and not gates._in_system_menu(context)
    assert not gates._in_system_menu(context.parent.parent)  # (a window is no menu)


def _menu_world(monkeypatch):
    from neo2.core.agent import confirm
    from neo2.core.tools.computer import ax, tools_ax

    apple, items, new, context = _menu_bar(monkeypatch)
    els = {f"e{n}": ax.Element(f"e{n}", "AXMenuItem", el.title, "", 20, 40 + 22 * n, 220, 22, actions=["AXPress", "AXCancel"], ref=el)
           for n, el in enumerate([*items.values(), new], start=1)}  # fmt: skip
    monkeypatch.setattr(ax, "get", lambda eid: els.get(eid))
    monkeypatch.setattr(tools_ax, "_element_pid", lambda el: 555)  # the front app's pid (its menu bar)
    monkeypatch.setattr(ax, "app_of", lambda ref: "TextEdit")
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": False)
    monkeypatch.setattr(ax, "app_names", lambda pid: {"textedit"})
    monkeypatch.setattr(ax, "regular", lambda pid: True)
    pressed = []
    monkeypatch.setattr(ax, "press_el", lambda e: pressed.append(e.title) or f"Pressed {e.id}")
    monkeypatch.setattr(ax, "press_element", lambda e: pressed.append(e.title) or "Pressed")
    monkeypatch.setattr(confirm, "gated", lambda *a, **k: "NEEDS_CONFIRM: x")
    return els, pressed


@pytest.mark.parametrize("eid", ["e1", "e2", "e3", "e4"])
def test_ax_press_and_click_by_id_never_touch_the_system_menu(monkeypatch, eid):
    from neo2.core.agent.registry import ToolContext
    from neo2.core.tools.computer import input as inp
    from neo2.core.tools.computer import tools_ax, tools_pointer

    els, pressed = _menu_world(monkeypatch)
    clicked = []
    monkeypatch.setattr(inp, "click", lambda *a, **k: clicked.append(a))
    out = asyncio.run(tools_ax.ax_press({"id": eid}, ToolContext(origin="agent")))
    assert pressed == [] and not out.ok and "system menu" in out.text, out
    out = asyncio.run(tools_pointer.click({"id": eid}, ToolContext(origin="agent")))
    assert clicked == [] and not out.ok and "system menu" in out.text, out
    asyncio.run(tools_ax.ax_press({"id": "e5"}, ToolContext(origin="agent")))  # the app's own menu: pressed
    assert pressed == ["New"]


def test_a_click_at_a_system_menu_item_is_refused(monkeypatch):
    from neo2.core.agent.registry import ToolContext
    from neo2.core.tools.computer import ax, tools_pointer
    from neo2.core.tools.computer import input as inp

    _apple, items, _new, _context = _menu_bar(monkeypatch)
    monkeypatch.setattr(ax, "hit", lambda x, y: ax.Hit(pid=555, app="TextEdit", label="Restart…", role="AXMenuItem", regular=True, el=items["Restart…"]))
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": False)
    monkeypatch.setattr(ax, "app_names", lambda pid: {"textedit"})
    clicked = []
    monkeypatch.setattr(inp, "click", lambda *a, **k: clicked.append(a))
    out = asyncio.run(tools_pointer.click({"x": 60, "y": 90}, ToolContext(user_text="restart", origin="agent")))
    assert clicked == [] and not out.ok and "system menu" in out.text, out


def test_click_text_never_picks_a_system_menu_item(monkeypatch):
    from neo2.core.agent.registry import ToolContext
    from neo2.core.tools.computer import ax, gates, tools_label

    els, pressed = _menu_world(monkeypatch)

    async def none(*a, **k):
        return None

    monkeypatch.setattr(gates, "_guard_front", none)
    shown = [els["e1"]]  # an open Apple menu, its "Restart…" the only match
    monkeypatch.setattr(ax, "window_elements", lambda app=None: ("TextEdit", (0, 0, 800, 600), shown))
    out = asyncio.run(tools_label.click_text({"label": "restart"}, ToolContext(origin="agent")))
    assert pressed == [] and str(out).startswith("Error"), out
    game = ax.Element("e9", "AXButton", "Restart", "", 300, 300, 80, 24, actions=["AXPress"], ref=None)
    shown.append(game)  # a game's own Restart button beside it: that one, never the system's
    from neo2.core.agent import confirm

    asked: list[str] = []  # ("Restart" commits by meaning: the resolver asks first — here, a yes)
    monkeypatch.setattr(confirm, "gated", lambda tool, a, what, **k: asked.append(what))
    asyncio.run(tools_label.click_text({"label": "restart"}, ToolContext(origin="agent")))
    assert pressed == ["Restart"] and all("Restart…" not in q for q in asked)


# ---- click_text's mouse fallback: what is really under the spot is gated too -------------------------


def _fallback_world(monkeypatch, hit):
    from neo2.core.agent import confirm
    from neo2.core.tools.computer import ax, gates, pointer
    from neo2.core.tools.computer import input as inp

    async def none(*a, **k):
        return None

    monkeypatch.setattr(gates, "_guard_front", none)
    e = ax.Element("e3", "AXStaticText", "Archive", "", 100, 100, 60, 20, actions=[], ref=None)  # no AXPress
    monkeypatch.setattr(ax, "window_elements", lambda app=None: ("Notes", (0, 0, 800, 600), [e]))
    monkeypatch.setattr(pointer, "spot", lambda frame: (130.0, 110.0))
    monkeypatch.setattr(ax, "hit", lambda x, y: hit)
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": False)
    monkeypatch.setattr(ax, "app_names", lambda pid: {hit.app.lower()})
    staged, clicked = [], []
    monkeypatch.setattr(confirm, "gated", lambda *a, **k: staged.append(a[2]) or "NEEDS_CONFIRM: x")
    monkeypatch.setattr(inp, "click", lambda *a, **k: clicked.append(a))
    return staged, clicked


def test_click_texts_fallback_never_clicks_another_apps_window_over_the_element(monkeypatch):
    from neo2.core.agent.registry import ToolContext
    from neo2.core.tools.computer import ax, tools_label

    # another app's floating panel covers the spot: its "Send" button is what the click would hit
    staged, clicked = _fallback_world(monkeypatch, ax.Hit(pid=77, app="Mail", label="Send", role="AXButton", regular=True))
    out = asyncio.run(tools_label.click_text({"label": "archive"}, ToolContext(origin="agent")))
    assert clicked == [] and not out.ok and out.text.startswith("Error") and "Mail" in out.text, out


def test_click_texts_fallback_asks_first_when_the_spot_is_a_committing_control(monkeypatch):
    from neo2.core.agent.registry import ToolContext
    from neo2.core.tools.computer import ax, tools_label

    # the same app, but what lies under the spot is its "Send" button, not the text meant
    staged, clicked = _fallback_world(monkeypatch, ax.Hit(pid=0, app="Notes", label="Send", role="AXButton", regular=True))
    out = asyncio.run(tools_label.click_text({"label": "archive"}, ToolContext(origin="agent")))
    assert clicked == [] and staged and "Send" in staged[0] and str(out).startswith("NEEDS_CONFIRM"), out
    staged, clicked = _fallback_world(monkeypatch, ax.Hit(pid=0, app="Notes", label="Archive", role="AXStaticText"))
    asyncio.run(tools_label.click_text({"label": "archive"}, ToolContext(origin="agent")))
    assert staged == [] and clicked == [(130.0, 110.0)]  # the text meant: clicked


# ---- one window-server lookup ------------------------------------------------------------------------


def test_one_window_server_lookup_names_a_point_accessibility_cannot(monkeypatch):
    """ax.hit and the protected check ask the window server the same way (window_owner_at): a board
    drawn with Metal is named by one lookup; when the window server can't say, the point is unnamed
    and nothing there is touched (fail closed)."""
    import ApplicationServices as AS
    import Quartz

    from neo2.core.tools.computer import ax, pointer

    real = {name: old for obj, name, old in monkeypatch._setattr if obj is ax and name in ("hit", "window_owner_at")}
    for name, fn in real.items():  # (the suite fakes both: the real ones here)
        monkeypatch.setattr(ax, name, fn)
    assert not hasattr(ax, "_window_owner_at")  # one lookup, one set of rules
    lists = []
    board = {"kCGWindowBounds": {"X": 0, "Y": 0, "Width": 800, "Height": 600}, "kCGWindowAlpha": 1.0,
             "kCGWindowOwnerPID": 4242, "kCGWindowOwnerName": "Board"}  # fmt: skip
    monkeypatch.setattr(AS, "AXUIElementCreateSystemWide", lambda: object())
    monkeypatch.setattr(AS, "AXUIElementSetMessagingTimeout", lambda el, t: 0)
    monkeypatch.setattr(AS, "AXUIElementCopyElementAtPosition", lambda el, x, y, _: (-25212, None))
    monkeypatch.setattr(Quartz, "CGWindowListCopyWindowInfo", lambda o, w: lists.append(1) or [board])
    monkeypatch.setattr(ax, "regular", lambda pid: True)
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": False)
    monkeypatch.setattr(ax, "app_names", lambda pid: {"board"})
    t = asyncio.run(pointer.probe([(10.0, 10.0)]))
    assert (t.hits[0].pid, t.hits[0].app, t.hits[0].regular) == (4242, "Board", True) and len(lists) == 1
    monkeypatch.setattr(Quartz, "CGWindowListCopyWindowInfo", lambda o, w: None)  # it can't say
    assert ax.hit(10, 10) == ax.Hit()
    with pytest.raises(pointer.Refused):
        asyncio.run(pointer.probe([(10.0, 10.0)]))


def test_a_menu_item_whose_place_cant_be_told_is_not_pressed(monkeypatch):
    """A menu item whose way up to its menu bar can't be read might be the system's: refused (fail
    closed), like a target whose owner can't be told."""
    from neo2.core.agent import confirm
    from neo2.core.agent.registry import ToolContext
    from neo2.core.tools.computer import ax, tools_ax

    item = ax.Element("e88", "AXMenuItem", "Restart…", "", 20, 120, 220, 22, actions=["AXPress", "AXCancel"], ref="REF")
    monkeypatch.setattr(ax, "_attr", lambda el, name, default=None: default)  # nothing can be read
    monkeypatch.setattr(ax, "get", lambda eid: item)
    monkeypatch.setattr(tools_ax, "_element_pid", lambda el: 555)
    monkeypatch.setattr(ax, "app_of", lambda ref: "TextEdit")
    monkeypatch.setattr(ax, "is_neo", lambda pid, name="": False)
    monkeypatch.setattr(ax, "app_names", lambda pid: {"textedit"})
    pressed = []
    monkeypatch.setattr(confirm, "gated", lambda *a, **k: "NEEDS_CONFIRM: x")
    monkeypatch.setattr(ax, "press_el", lambda e: pressed.append(e.title) or f"Pressed {e.id}")
    out = asyncio.run(tools_ax.ax_press({"id": "e88"}, ToolContext(origin="agent")))
    assert pressed == [] and not out.ok
