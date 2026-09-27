"""Tools load by preset without macOS frameworks; presets resolve per platform; tags are right."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from neo import runtime
from neo.tools import MACOS_ONLY, MODULES_FOR, PRESETS, REFUSED, load, modules_for, resolve

ROOT = Path(__file__).resolve().parents[1]

# NEO.app's catalog order on main at 7e11113 (load_all): fast paths break ties by it, Live declares tools in it.
APP_ORDER = ["ax_tree", "ax_find", "screenshot", "apps_running", "ax_press", "ax_set_value", "click", "drag", "scroll", "type_text", "dictate", "hotkey", "click_text", "open_app", "activate_app", "quit_app", "close", "applescript", "open_url", "page_read", "safari_read", "mail_send", "mail_unread", "calendar_today", "calendar_add", "notes_create", "reminder_add", "finder_reveal", "shell", "read_file", "write_file", "edit_file", "trash", "volume", "brightness", "clock", "sleep_display", "timer", "web_search", "web_fetch", "search_site", "weather", "memory", "browser_open", "browser_read", "browser_click", "browser_type", "browser_press", "browser_scroll", "browser_back", "browser_tabs", "browser_screenshot", "browser_handoff", "browser_close", "browse", "play_music"]

_NO_MACOS = textwrap.dedent(
    """
    import sys, importlib.abc
    BLOCKED = ("AppKit", "Quartz", "ApplicationServices", "objc", "Foundation", "CoreFoundation",
               "EventKit", "Contacts", "PyObjCTools", "ScriptingBridge")
    class NoMacOS(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ImportError(f"no macOS here: {name}")
            return None
    sys.meta_path.insert(0, NoMacOS())
    """
)


def _run(code: str, tmp_path: Path, *, macos: bool = False) -> dict:
    """`code` in a fresh interpreter (its own HOME and data dir), without macOS frameworks unless
    `macos`."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {**os.environ, "HOME": str(home), "NEO_DATA_DIR": str(tmp_path / "data"), "NEO_REFLEX": "off"}
    env.pop("PYTEST_CURRENT_TEST", None)
    src = textwrap.dedent(code) if macos else _NO_MACOS + textwrap.dedent(code)
    out = subprocess.run([sys.executable, "-c", src], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_portable_tools_work_without_any_macos_framework(tmp_path):
    got = _run(
        """
        import asyncio, json, sys
        import neo.tools.web, neo.tools.weather, neo.tools.memory_tool, neo.tools.system
        from neo.agent.registry import ToolContext, catalog
        from neo.agent import fastpath
        reg = catalog()
        fetch = asyncio.run(reg.invoke("web_fetch", {"url": "http://127.0.0.1:9/admin"}, ToolContext()))
        clock = asyncio.run(reg.invoke("clock", {}, ToolContext()))
        match = fastpath.match("search for cats in notes")  # the web_search check that consults neo.decide
        print(json.dumps({
            "tools": [t.name for t in reg.all()],
            "fetch": fetch.text, "clock": clock.text, "match": match,
            "loaded": sorted(m for m in ("neo.tools.computer", "neo.decide", "AppKit", "objc") if m in sys.modules),
        }))
        """,
        tmp_path,
    )
    assert got["loaded"] == []  # no Mac tools, no task-kind drivers, no pyobjc
    assert {"web_search", "web_fetch", "weather", "memory", "clock"} <= set(got["tools"])
    assert got["fetch"].startswith("Error: not fetching http://127.0.0.1:9/admin")  # the policy, not ImportError
    assert got["clock"].startswith("It's ")
    assert got["match"] is None


def test_load_imports_only_what_a_preset_needs(tmp_path):
    got = _run(
        """
        import json, sys
        from neo.tools import load
        from neo.agent.registry import catalog
        web = load("web")
        after_web = sorted(t.name for t in catalog().all())
        mac = load("mac", platform="darwin")  # the Mac tools can't load here: reported, not raised
        print(json.dumps({"web": web, "after_web": after_web, "mac": mac,
                          "computer": "neo.tools.computer" in sys.modules and hasattr(sys.modules["neo.tools.computer"], "shell_tool")}))
        """,
        tmp_path,
    )
    assert got["web"] == {}
    assert got["after_web"] == sorted(
        ["web_search", "web_fetch", "search_site", "weather", "clock", "volume", "brightness", "sleep_display", "timer"]
    )
    assert list(got["mac"]) == ["neo.tools.computer"] and "no macOS here" in got["mac"]["neo.tools.computer"]
    assert not got["computer"]


def test_importing_the_web_tools_stays_small(tmp_path):
    got = _run(
        """
        import json, sys
        import neo.tools.web
        from neo.agent.registry import catalog
        print(json.dumps({"n": len(catalog().all()), "mods": [m for m in sys.modules if m.startswith(("neo.tools.computer", "neo.decide"))]}))
        """,
        tmp_path,
    )
    assert got == {"n": 3, "mods": []}  # at 4764b73: 35 tools, AppKit and Quartz


def test_the_old_policy_path_is_the_same_module():
    import neo.security.policy as new
    import neo.tools.computer.policy as old
    from neo.tools.computer import policy as via_package

    assert old is new is via_package
    assert old._GENERIC_OK is new._GENERIC_OK and old.is_secret is new.is_secret


def test_a_monkeypatch_through_either_name_is_seen_by_both(monkeypatch):
    import neo.security.policy as new
    from neo.tools.computer import policy as old

    monkeypatch.setattr(old, "commits", lambda label, context="": True)
    assert new.commits("anything")


# ---- presets --------------------------------------------------------------------------------------


def test_the_web_preset():
    assert resolve("web") == ["web_search", "web_fetch", "weather", "clock"]
    assert not set(resolve("web")) & MACOS_ONLY


def test_presets_globs_names_and_excludes_mix():
    got = resolve(("web", "browser_*", "memory"), exclude=("browser_close", "web_f*"), platform="darwin")
    assert set(got) == {"web_search", "weather", "clock", "memory"} | set(PRESETS["browser"]) - {"browser_close"}
    assert got == [n for n in PRESETS["all"] if n in got]  # one order, whatever the selection's
    assert resolve("none") == [] and resolve([]) == []
    assert resolve("shell", platform="darwin") == ["shell"]


def test_off_macos(monkeypatch):
    for sel in ("mac", "mac_read", "shell", ("web", "open_url")):
        with pytest.raises(runtime.UnsupportedPlatform):
            resolve(sel, platform="linux")
    portable = resolve("all", platform="linux")
    assert set(portable) == {"web_search", "web_fetch", "weather", "clock", "memory"} | set(PRESETS["browser"])
    assert resolve("*_read", platform="linux") == ["browser_read"]  # a glob just skips the Mac ones


def test_unknown_names_and_play_music_are_refused():
    with pytest.raises(ValueError, match="unknown tool or preset"):
        resolve("shel")
    with pytest.raises(ValueError, match="play_music isn't available to embedded agents"):
        resolve(("web", "play_music"))
    assert "play_music" not in resolve("*", platform="darwin") and "play_music" in REFUSED


def test_every_catalog_tool_is_in_a_preset_with_its_module():
    from neo.agent.registry import catalog
    from neo.tools import load_all

    load_all()
    names = {t.name for t in catalog().all()} - {"zz_declared_while_bound"}
    assert names - set(PRESETS["all"]) == {"play_music", "browse"} == set(REFUSED)
    assert set(PRESETS["all"]) <= set(MODULES_FOR)
    for n in PRESETS["all"]:
        assert catalog().get(n) is not None, n


def test_requires_tags_match_the_presets():
    from neo.agent.registry import catalog
    from neo.tools import load_all

    load_all()
    mac = {t.name for t in catalog().all() if "macos" in t.requires}
    assert mac == MACOS_ONLY | {"play_music"}
    assert {t.name for t in catalog().all() if "playwright" in t.requires} == set(PRESETS["browser"])
    for n in ("web_search", "web_fetch", "weather", "clock", "memory"):
        assert not catalog().get(n).requires
    assert catalog().get("search_site").requires == {"macos"}  # it opens the Mac's own browser


def test_neo_apps_catalog_order_is_unchanged(tmp_path):
    """As NEO.app loads it, in a fresh process: in this one, whichever test first imported a tool
    module has already decided the catalog's order."""
    names = _run(
        """
        import json
        from neo.agent.registry import catalog
        from neo.tools import load_all
        load_all()
        print(json.dumps([t.name for t in catalog().all()]))
        """,
        tmp_path,
        macos=True,
    )
    assert [n for n in names if n in APP_ORDER] == APP_ORDER


def test_modules_for_load_the_task_tools_with_the_computer_tools():
    assert modules_for(["mail_send", "clock", "web_search", "memory"]) == [
        "neo.tools.computer", "neo.tools.system", "neo.tools.web", "neo.tools.memory_tool",
    ]


def test_load_never_switches_task_modes(monkeypatch):
    from neo.tools import tasks

    calls = []
    monkeypatch.setattr(tasks, "apply_modes", lambda: calls.append(1))
    assert load(("mac", "memory"), platform="darwin") == {}
    assert calls == []


def test_the_clock_module_loads_mac_scripting_only_when_used(monkeypatch):
    import asyncio

    from neo.tools import system
    from neo.tools.computer import apps

    ran = []

    async def fake(script, timeout=30):
        ran.append(script)
        return "40"

    monkeypatch.setattr(apps, "osascript", fake)
    assert asyncio.run(system.osascript("output volume of (get volume settings)")) == "40"
    assert system._q('say "hi"') == apps._q('say "hi"')
    assert ran == ["output volume of (get volume settings)"]
