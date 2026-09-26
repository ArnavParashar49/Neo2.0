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
        ("go back", ("hotkey", {"keys": "cmd+[", "app": "browser|finder"})),
        ("zoom in", ("hotkey", {"keys": "cmd+="})),
        ("take a screenshot", ("hotkey", {"keys": "cmd+shift+3"})),
        ("scroll down", ("scroll", {"at_pointer": True, "dy": -8})),
        ("scroll to the top", ("hotkey", {"keys": "cmd+up", "app": "browser"})),
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
        ("search for cat videos in spotify", ("search_site", {"site": "spotify", "query": "cat videos"})),
        ("search 2tb ssd on amazon", ("search_site", {"site": "amazon", "query": "2tb ssd"})),
        ("search amazon for crucial x9 pro", ("search_site", {"site": "amazon", "query": "crucial x9 pro"})),
        ("open youtube and search for cat videos", ("search_site", {"site": "youtube", "query": "cat videos"})),
        ("open it on amazon", None),  # "it" comes from the conversation: the model resolves it
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
    from neo.tools import computer as comp
    from neo.tools.computer import ax

    els = [
        ax.Element("e1", "AXStaticText", "Play next episode", "", 0, 0, 200, 20, actions=[]),
        ax.Element("e2", "AXButton", "Play", "", 10, 10, 40, 40, actions=["AXPress"]),
        ax.Element("e3", "AXButton", "Play", "", 0, 0, 400, 400, actions=["AXPress"]),
        ax.Element("e4", "AXButton", "Display", "", 0, 0, 40, 40, actions=["AXPress"]),
        ax.Element("e5", "AXButton", "Play", "", 5000, 5000, 40, 40, actions=["AXPress"]),  # off-window
    ]
    monkeypatch.setattr(ax, "window_elements", lambda app=None: ("Music", (0, 0, 1000, 800), els))
    monkeypatch.setattr(comp.apps, "_target", None)
    monkeypatch.setattr(comp.apps, "front_name", lambda: "Music")
    pressed = []
    monkeypatch.setattr(ax, "press_element", lambda e: pressed.append(e.id) or "Pressed")
    clicked = []
    monkeypatch.setattr(comp.inp, "click", lambda x, y, **kw: clicked.append((x, y)) or "clicked")
    out = asyncio.run(registry().invoke("click_text", {"label": "play"}, ToolContext(user_text="")))
    assert out.ok and pressed == ["e2"] and clicked == []  # exact title, button, smallest, on screen
    pressed.clear()
    out = asyncio.run(registry().invoke("click_text", {"label": "ok"}, ToolContext(user_text="")))
    assert not out.ok  # "ok" is not a substring hit on "Bookmarks"-like labels
    out = asyncio.run(registry().invoke("click_text", {"label": "it"}, ToolContext(user_text="")))
    assert not out.ok and pressed == []  # vague labels are never clicked


def test_click_text_asks_when_ambiguous(monkeypatch):
    from neo.tools.computer import ax

    els = [
        ax.Element("e1", "AXButton", "Save draft", "", 0, 0, 40, 40, actions=["AXPress"]),
        ax.Element("e2", "AXButton", "Save as", "", 50, 0, 40, 40, actions=["AXPress"]),
    ]
    monkeypatch.setattr(ax, "window_elements", lambda app=None: ("Mail", (0, 0, 1000, 800), els))
    from neo.tools import computer as comp

    monkeypatch.setattr(comp.apps, "_target", None)
    monkeypatch.setattr(comp.apps, "front_name", lambda: "Mail")
    pressed = []
    monkeypatch.setattr(ax, "press_element", lambda e: pressed.append(e.id) or "Pressed")
    out = asyncio.run(registry().invoke("click_text", {"label": "save"}, ToolContext(user_text="")))
    assert not out.ok and "which one" in out.text and pressed == []


def test_scroll_fast_path_uses_the_pointer(monkeypatch):
    from neo.tools import computer as comp

    monkeypatch.setattr(comp.inp, "mouse_position", lambda: (640.0, 400.0))
    seen = {}
    monkeypatch.setattr(comp.inp, "scroll", lambda x, y, dy, dx: seen.update(x=x, y=y, dy=dy) or "ok")
    tool, args = _match_fast_path("scroll down")
    asyncio.run(registry().invoke(tool, args, ToolContext(user_text="")))
    assert seen == {"x": 640.0, "y": 400.0, "dy": -8}
    # an explicit coordinate left of the main display is a real coordinate, not "the pointer"
    asyncio.run(registry().invoke("scroll", {"x": -300, "y": 200, "dy": -3}, ToolContext(user_text="")))
    assert seen == {"x": -300, "y": 200, "dy": -3}


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
        return None, False

    monkeypatch.setattr(comp, "_ensure_text_focus", no_focus)
    monkeypatch.setattr(comp.inp, "type_text", lambda s, **kw: typed.append(s) or "ok")
    monkeypatch.setattr(comp.inp, "hotkey", lambda k: typed.append(f"<{k}>") or "ok")
    monkeypatch.setattr(ax, "frontmost_app", lambda: ("Notes", 1))
    monkeypatch.setattr(comp.apps, "front_name", lambda: "Notes")
    monkeypatch.setattr(comp.apps, "_target", None)
    monkeypatch.setattr(ax, "focused_role", lambda: ("AXTextArea", ""))
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


def test_search_site_builds_real_search_urls(monkeypatch):
    from neo.tools import web

    monkeypatch.setattr(web, "_country", lambda: "IN")
    assert web.site_search_url("amazon", "crucial x9 pro 2tb") == "https://www.amazon.in/s?k=crucial+x9+pro+2tb"
    assert web.site_search_url("Google", "ssd") == "https://www.google.com/search?q=ssd"
    assert web.site_search_url("bestbuy.com", "ssd").endswith("site%3Abestbuy.com+ssd")
    opened = []

    async def fake_open(url):
        opened.append(url)
        return f"Opened {url} in Chrome"

    monkeypatch.setattr(web, "_country", lambda: "")
    from neo.tools.computer import apps

    monkeypatch.setattr(apps, "open_app", fake_open)
    out = asyncio.run(registry().invoke("search_site", {"site": "youtube", "query": "lofi"}, ToolContext(user_text="")))
    assert out.ok and opened == ["https://www.youtube.com/results?search_query=lofi"]
    out = asyncio.run(registry().invoke("search_site", {"site": "amazon", "query": "it"}, ToolContext(user_text="")))
    assert not out.ok and len(opened) == 1  # never searches for the word "it"


def test_agent_gets_the_spoken_conversation(monkeypatch):
    import neo.agent.session as sess_mod
    from neo.agent.session import Session
    from neo.providers.base import Turn
    from neo.reflex.schema import Decision

    seen = {}

    class B:
        name = "fake"
        supports_vision = False
        supports_tools = True

        async def complete(self, messages, *, system="", **kw):
            seen["system"] = system
            return Turn(text="ok")

        async def stream(self, messages, **kw):
            yield ""

    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": B())

    class R:
        async def decide(self, t):
            return Decision("agent_task", 0.9, False, 0.0, False, 0.0, "fake")

    conv = "user: what's a cheaper SSD?\nNEO: The Crucial X9 Pro 2TB is about half the price."
    asyncio.run(Session(reflex=R()).handle("open it on amazon", conversation=conv))
    assert "Crucial X9 Pro 2TB" in seen["system"] and "RECENT CONVERSATION" in seen["system"]


@pytest.mark.parametrize(
    "text, place, when",
    [
        ("what's the weather", None, None),
        ("what's the weather like in new york tomorrow", "new york", "tomorrow"),
        ("is it going to rain tomorrow", None, "tomorrow"),
        ("weather forecast for the week", None, "week"),
        ("do i need an umbrella", None, None),
    ],
)
def test_weather_fast_paths(text, place, when):
    tool, args = _match_fast_path(text)
    assert tool == "weather" and args["place"] == place and (args["when"] or args["when2"]) == when


def test_weather_formats_open_meteo(monkeypatch):
    from neo.tools import weather as w

    class Resp:
        def __init__(self, j):
            self._j = j

        def json(self):
            return self._j

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None, timeout=None):
            if "geocoding" in url:
                return Resp({"results": [{"name": "Dubai", "country": "UAE", "latitude": 25.2, "longitude": 55.3}]})
            return Resp(
                {
                    "current": {"temperature_2m": 36.2, "apparent_temperature": 41, "weather_code": 0, "wind_speed_10m": 12, "relative_humidity_2m": 50},
                    "daily": {
                        "time": ["2026-09-26", "2026-09-27"],
                        "weather_code": [0, 61],
                        "temperature_2m_max": [39, 37],
                        "temperature_2m_min": [28, 27],
                        "precipitation_probability_max": [0, 60],
                    },
                }
            )

    monkeypatch.setattr(w.httpx, "Client", Client)
    today = w.forecast("Dubai", "today")
    assert today.startswith("Dubai, UAE: 36°C now") and "Tomorrow: light rain" in today
    assert "60% chance of rain" in w.forecast("Dubai", "tomorrow")


def test_live_caps_searching_per_question():
    from neo.voice import live

    assert live._LOOKUP_BUDGET == {"web_search": 2, "web_fetch": 1} and {"weather", "web_fetch"} <= live._DIRECT_TOOLS


def test_fourth_lookup_in_one_question_is_refused(monkeypatch):
    import neo.voice.live as live_mod

    calls = []

    async def fake_search(a, c):
        calls.append(a["query"])
        return "1. result"

    monkeypatch.setattr(registry().get("web_search"), "handler", fake_search)
    v = live_mod.LiveVoice.__new__(live_mod.LiveVoice)
    v._agent_session, v._tool_tasks, v._goals, v._server_cancelled = None, {}, {}, set()
    v._live, v._ui_lock, v._dialog = None, asyncio.Lock(), []
    v._new_question()

    class FC:
        def __init__(self, i):
            self.id, self.name, self.args = f"c{i}", "web_search", {"query": f"q{i}"}

    async def run():
        for i in range(5):
            await v._run_tool(None, FC(i), "what's new with the iphone")

    asyncio.run(run())
    assert calls == ["q0", "q1"]  # two searches per question; the rest are told to answer


def test_identical_read_only_calls_run_once_per_question(monkeypatch):
    import neo.voice.live as live_mod

    calls = []

    async def fake_search(a, c):
        calls.append(a["query"])
        return "1. result"

    monkeypatch.setattr(registry().get("web_search"), "handler", fake_search)
    v = live_mod.LiveVoice.__new__(live_mod.LiveVoice)
    v._agent_session, v._tool_tasks, v._goals, v._server_cancelled = None, {}, {}, set()
    v._live, v._ui_lock, v._dialog = None, asyncio.Lock(), []
    v._new_question()

    class FC:
        def __init__(self, i, q):
            self.id, self.name, self.args = f"c{i}", "web_search", {"query": q}

    async def run():
        await v._run_tool(None, FC(0, "t9 alternative"), "q")
        await v._run_tool(None, FC(1, "T9 alternative "), "q")  # same call, spelled slightly differently
        v._new_question()
        await v._run_tool(None, FC(2, "t9 alternative"), "q")  # a new question may ask again

    asyncio.run(run())
    assert calls == ["t9 alternative", "t9 alternative"]
