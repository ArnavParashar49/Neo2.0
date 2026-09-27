"""neo2/core/tools/computer split into files (gates, tools_ax, tools_pointer, tools_keys, tools_label,
tools_apps, phrases) with no change in behaviour: the catalog keeps its order, every gate lives once
in the frozen gates module and is looked up there at call time, and a patch through the package
reaches what the tools really call."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from neo2.core import effects
from neo2.core.tools import computer
from neo2.core.tools.computer import gates

ROOT = Path(__file__).resolve().parents[1]
COMPUTER = Path(computer.__file__).resolve().parent
TOOL_FILES = ("tools_ax.py", "tools_pointer.py", "tools_keys.py", "tools_label.py", "tools_apps.py", "phrases.py")

# NEO.app's catalog as it was before the split (a fresh process, every module loaded).
CATALOG_BEFORE = [
    "ax_tree", "ax_find", "screenshot", "apps_running", "ax_press", "ax_set_value", "click", "drag", "scroll",
    "type_text", "dictate", "hotkey", "click_text", "open_app", "activate_app", "quit_app", "close", "applescript",
    "open_url", "page_read", "safari_read", "mail_send", "mail_unread", "calendar_today", "calendar_add",
    "notes_create", "reminder_add", "finder_reveal", "shell", "read_file", "write_file", "edit_file", "trash",
    "selftest_controls", "volume", "brightness", "clock", "sleep_display", "timer", "web_search", "web_fetch",
    "search_site", "weather", "memory", "browser_open", "browser_read", "browser_click", "browser_type",
    "browser_press", "browser_scroll", "browser_back", "browser_tabs", "browser_screenshot", "browser_handoff",
    "browser_close", "browse", "play_music",
]  # fmt: skip

# Every gate and protection helper: defined in gates.py, and nowhere else.
GATES = (
    "AX_OFF", "_AX_STILL_OFF", "_AX_DENIED", "_ax_trusted", "_host_app", "_host_who", "_off_for_host",
    "_ax_off_texts", "_ax_off_reply", "_needs_ax", "INPUT_DOUBT", "_needs_input", "_input_doubt",
    "_dialog_text", "_ask_before_click", "_ask_first", "_gate_click", "_gate_drop", "_NEVER_TYPE_INTO",
    "_protected", "_protected_at", "_refuse_protected", "_protected_out", "_protected_hit",
    "_off_limits_el", "allow_test_window", "_test_window_front", "_guard_front", "_taint",
    "_read_from_screen", "_refused", "_watch", "_changes", "_DOC_APPS", "_doc_like", "_command_line",
    "_chord", "_return_runs_or_sends", "_ask_before_keys",
)  # fmt: skip


def test_neo_apps_catalog_order_is_the_one_before_the_split(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ, "HOME": str(home), "NEO_DATA_DIR": str(tmp_path / "data"), "NEO_REFLEX": "off"}
    env.pop("PYTEST_CURRENT_TEST", None)
    code = (
        "import json\nfrom neo2.core.agent.registry import catalog\nfrom neo2.core.tools import load_all\nload_all()\n"
        "print(json.dumps([t.name for t in catalog().all()]))"
    )
    with effects.allow("process"):
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    names = json.loads(out.stdout.strip().splitlines()[-1])
    assert [n for n in names if n in CATALOG_BEFORE] == CATALOG_BEFORE


def test_every_computer_tool_has_its_place_in_the_order():
    from neo2.core.agent.registry import catalog
    from neo2.core.tools.computer import _catalog

    assert _catalog.pending() == []  # all registered
    mine = [t.name for t in catalog().all() if t.handler.__module__.startswith("neo2.core.tools.computer.")]
    assert mine and all(n in computer.ORDER for n in mine), [n for n in mine if n not in computer.ORDER]
    assert mine == [n for n in computer.ORDER if n in mine]


def test_the_gates_live_in_gates_py_only():
    for name in GATES:
        assert hasattr(gates, name), name
        assert computer._HOME[name] is gates, name
    for f in TOOL_FILES:
        tree = ast.parse((COMPUTER / f).read_text())
        defined = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        defined |= {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
        assert not defined & set(GATES), (f, defined & set(GATES))
        # a gate is always called through the module, never bound by name (a patch would miss it)
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module and n.module.endswith("computer.gates"):
                pytest.fail(f"{f} binds {[a.name for a in n.names]} from gates")


def test_each_name_is_defined_in_one_file():
    seen: dict[str, str] = {}
    for f in ("gates.py", *TOOL_FILES):
        tree = ast.parse((COMPUTER / f).read_text())
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert n.name not in seen, (n.name, seen.get(n.name), f)
                seen[n.name] = f


def test_a_patch_through_the_package_reaches_the_file_that_defines_it(monkeypatch):
    for name, home in computer._HOME.items():
        if name.startswith("__") or not callable(getattr(home, name)):
            continue
        marker = object()
        with monkeypatch.context() as m:
            m.setattr(computer, name, marker)
            assert getattr(home, name) is marker and getattr(computer, name) is marker, name
        assert getattr(home, name) is not marker and getattr(computer, name) is getattr(home, name)
    assert computer._test_window_pid == 0
    with computer.allow_test_window(77):
        assert computer._test_window_pid == 77
    assert isinstance(computer.ax, types.ModuleType) and computer.inp.__name__ == "neo2.core.tools.computer.input"


def test_the_tools_call_the_gates_at_call_time(monkeypatch):
    """A gate patched through the package is the one ax_press, click_text, type_text and hotkey
    call — no tool kept a copy of it."""
    from neo2.core.agent.registry import ToolContext, registry
    from neo2.core.tools import load_all
    from neo2.core.tools.computer import ax

    load_all()
    asked: list[str] = []

    async def ask_first(tool_name, a, label, app="", els=None):
        asked.append(tool_name)
        return "Error: asked"

    async def guard(want=""):
        asked.append("guard")
        return "Error: guarded"

    async def keys(a, keys, app, c, pid=0):
        asked.append("keys")
        return "Error: asked"

    e = ax.Element("e1", "AXButton", "Go", "", 10, 10, 20, 20, actions=["AXPress"], ref=None)
    monkeypatch.setattr(ax, "get", lambda eid: e)
    monkeypatch.setattr(ax, "app_of", lambda ref: "Finder")
    monkeypatch.setattr(ax, "window_owner_at", lambda x, y: ("Finder", 5))
    monkeypatch.setattr(computer, "_ask_first", ask_first)
    monkeypatch.setattr(computer, "_guard_front", guard)
    monkeypatch.setattr(computer, "_ask_before_keys", keys)
    monkeypatch.setattr(computer.inp, "hotkey", lambda k: pytest.fail("a key was pressed"))
    monkeypatch.setattr(ax, "frontmost_app", lambda: ("Finder", 5))

    def run(name, args):
        return asyncio.run(registry().invoke(name, args, ToolContext(user_text="")))

    assert run("ax_press", {"id": "e1"}).text == "Error: asked"
    assert run("click_text", {"label": "go"}).text == "Error: guarded"
    assert run("type_text", {"text": "hi"}).text == "Error: guarded"

    async def let_through(want=""):
        asked.append("guard")

    monkeypatch.setattr(computer, "_guard_front", let_through)
    assert run("hotkey", {"keys": "return"}).text == "Error: asked"
    assert asked == ["ax_press", "guard", "guard", "guard", "keys"]
