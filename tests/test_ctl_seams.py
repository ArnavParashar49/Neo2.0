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
