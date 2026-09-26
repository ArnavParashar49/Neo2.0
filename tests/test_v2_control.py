"""Zero-LLM computer control: click by label, keys, scroll, and command fast paths."""

from __future__ import annotations

import asyncio

import pytest

from neo.agent.registry import ToolContext, registry
from neo.agent.session import _match_fast_path
from neo.events import EventBus


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    from neo import config, events
    from neo.tools import load_all

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(events, "_bus", EventBus())
    load_all()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("click play", ("click_text", {"label": "play"})),
        ("press the send button", ("click_text", {"label": "send"})),
        ("click on the sign in link", ("click_text", {"label": "sign in"})),
        ("press enter", ("hotkey", {"keys": "enter"})),  # key names win over labels
        ("select all", ("hotkey", {"keys": "cmd+a"})),
        ("close this tab", ("hotkey", {"keys": "cmd+w"})),
        ("next tab", ("hotkey", {"keys": "ctrl+tab"})),
        ("go back", ("hotkey", {"keys": "cmd+["})),
        ("zoom in", ("hotkey", {"keys": "cmd+="})),
        ("take a screenshot", ("hotkey", {"keys": "cmd+shift+3"})),
        ("scroll down", ("scroll", {"x": -1, "y": -1, "dy": -8})),
        ("scroll to the top", ("hotkey", {"keys": "cmd+up"})),
        ("quit spotify", ("quit_app", {"name": "spotify"})),
        ("close this window", ("hotkey", {"keys": "cmd+w"})),
        ("switch to safari", ("open_app", {"target": "safari"})),
        ("what apps are open", ("apps_running", {})),
        ("check my emails", ("mail_unread", {})),
        ("any new mail?", ("mail_unread", {})),
        ("check my emails and summarize the important ones", None),  # a task, not a command
        ("what's on my calendar today", ("calendar_today", {})),
        ("make a note called groceries", ("notes_create", {"title": "groceries", "body": None})),
        ("make a note called groceries saying milk and eggs", ("notes_create", {"title": "groceries", "body": "milk and eggs"})),
        ("create a note saying buy milk", ("notes_create", {"title": "", "body": "buy milk"})),
        ("search for the best laptops under 1000", ("web_search", {"query": "the best laptops under 1000"})),
        ("search for cat videos in spotify", None),  # app-specific search: not the web
        ("remember that i park on level 3", ("memory", {"action": "remember", "text": "i park on level 3", "kind": "note"})),
        ("remember to call mom", None),  # a reminder, not a memory
        ("what did i say about parking", ("memory", {"action": "recall", "query": "parking"})),
        ("lock the screen", ("sleep_display", {})),
        ("click", None),
    ],
)
def test_control_fast_paths(text, expected):
    assert _match_fast_path(text) == expected


def test_click_text_prefers_exact_titles_and_buttons(monkeypatch):
    from neo.tools.computer import ax
    from neo.tools import computer as comp

    els = {
        "e1": ax.Element("e1", "AXStaticText", "Play next episode", "", 0, 0, 200, 20, actions=[]),
        "e2": ax.Element("e2", "AXButton", "Play", "", 10, 10, 40, 40, actions=["AXPress"]),
        "e3": ax.Element("e3", "AXButton", "Play", "", 0, 0, 400, 400, actions=["AXPress"]),
    }
    monkeypatch.setattr(ax, "snapshot", lambda app=None, **kw: None)
    monkeypatch.setattr(ax, "_last_tree", els)
    pressed = []
    monkeypatch.setattr(ax, "press", lambda eid: pressed.append(eid) or "Pressed")
    clicked = []
    monkeypatch.setattr(comp.inp, "click", lambda x, y, **kw: clicked.append((x, y)) or "clicked")
    out = asyncio.run(registry().invoke("click_text", {"label": "play"}, ToolContext(user_text="")))
    assert out.ok and pressed == ["e2"] and clicked == []  # exact title, button role, smallest
    out = asyncio.run(registry().invoke("click_text", {"label": "settings"}, ToolContext(user_text="")))
    assert not out.ok and "nothing on screen" in out.text


def test_scroll_fast_path_uses_the_pointer(monkeypatch):
    from neo.tools import computer as comp

    monkeypatch.setattr(comp.inp, "mouse_position", lambda: (640.0, 400.0))
    seen = {}
    monkeypatch.setattr(comp.inp, "scroll", lambda x, y, dy, dx: seen.update(x=x, y=y, dy=dy) or "ok")
    tool, args = _match_fast_path("scroll down")
    asyncio.run(registry().invoke(tool, args, ToolContext(user_text="")))
    assert seen == {"x": 640.0, "y": 400.0, "dy": -8}


def test_control_tools_are_quiet_and_live_direct():
    from neo.voice import live

    for n in ("click_text", "hotkey", "type_text", "dictate", "open_app"):
        assert n in live._DIRECT_TOOLS and registry().get(n).quiet
    for n in ("clock", "mail_unread", "apps_running", "calendar_today", "web_search"):
        assert not registry().get(n).quiet  # answers are spoken


def test_dictate_appends_on_a_new_line_when_the_note_has_text(monkeypatch):
    from neo.tools import computer as comp
    from neo.tools.computer import ax
    import neo.providers as prov
    from neo.providers.base import Turn

    typed = []

    async def no_focus():
        return None

    monkeypatch.setattr(comp, "_ensure_text_focus", no_focus)
    monkeypatch.setattr(comp.inp, "type_text", lambda s, **kw: typed.append(s) or "ok")
    monkeypatch.setattr(comp.inp, "hotkey", lambda k: typed.append(f"<{k}>") or "ok")
    monkeypatch.setattr(ax, "main_text_area", lambda app=None: ax.Element("e1", "AXTextArea", "", "Laptops", 0, 0, 10, 10))

    class B:
        name = "b"
        supports_vision = False
        supports_tools = True

        async def complete(self, messages, **kw):
            return Turn(text="- light\n- fast")

    monkeypatch.setattr(prov, "make", lambda n: B())
    out = asyncio.run(registry().invoke("dictate", {"brief": "points about laptops"}, ToolContext(user_text="")))
    assert out.ok and typed == ["<return>", "- light", "<return>", "- fast"]  # new line first


def test_scroll_is_callable_without_coordinates():
    t = registry().get("scroll")
    assert t.parameters.get("required", []) == [] and t.quiet
    from neo.voice import live

    assert "scroll" in live._DIRECT_TOOLS


def test_confirmation_expires(monkeypatch):
    from neo.agent import confirm

    monkeypatch.setattr(confirm, "_TTL_S", 0.01)
    confirm.stage("a1", "trash", {"path": "/x"}, "move /x to the Trash")
    assert confirm.peek() is not None
    import time

    time.sleep(0.02)
    assert confirm.peek() is None  # lapsed: the next request is not treated as an answer
