"""Every voice command that works today keeps working (plan §7.4), with the task kinds in shadow
(the release default) and switched on. Each command goes through the session: a fast path, or the
model's call (a fake brain) when no fast path covers it. Every app, script, store, page and
driver is a fake that records who answered: today's code ("legacy") or the kind's driver."""

from __future__ import annotations

import asyncio

import pytest

import neo.agent.session as sess_mod
from neo.agent import confirm
from neo.agent.session import Session
from neo.decide import run
from neo.decide.inventory import App
from neo.events import EventBus
from neo.providers.base import ToolCall, Turn
from neo.reflex.schema import Decision

KINDS = ("write_note", "add_reminder", "add_event", "read_calendar", "read_mail", "open_app", "web_search")
NOTES = App(bundle_id="test.notes", name="Notes", path="/Applications/Notes.app")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    from neo import events
    from neo.decide import inventory, sites
    from neo.tools import load_all

    monkeypatch.setattr(events, "_bus", EventBus())
    # what the fast-path checks read: the apps on this "Mac", its country, which words are everyday
    monkeypatch.setattr(inventory, "snapshot", lambda: [NOTES])
    monkeypatch.setattr(sites, "_country", lambda: "in")
    monkeypatch.setattr(sites, "everyday", lambda words: False)
    confirm.cancel("test")
    load_all()
    yield
    confirm.cancel("test")


class Brain:
    """The model: makes the given call, then says it's done."""

    name = "fake"
    supports_vision = False
    supports_tools = True

    def __init__(self, call: ToolCall | None):
        self.turns = [Turn(tool_calls=[call])] if call else []
        self.calls = 0

    async def complete(self, messages, **kw):
        self.calls += 1
        return self.turns.pop(0) if self.turns else Turn(text="Done.")

    async def stream(self, messages, **kw):
        yield ""


class Reflex:
    async def decide(self, text):
        return Decision("agent_task", 0.95, False, 0.0, False, 0.0, "laya+head", reply=False, reply_p=0.1)


@pytest.fixture
def world(monkeypatch, real_handlers):
    """Who answered each tool call: ("legacy", what) for today's code, ("driver", kind) for the
    kind's driver, ("web", query) for the web search both share."""
    from neo.decide import axkit
    from neo.decide.kinds import launch, mail, note, schedule, site
    from neo.tools import computer, web
    from neo.tools.computer import apps, ax, legacy
    from neo.tools.computer import input as inp

    log: list = []

    def legacy_fake(name):
        async def fake(*a, **k):
            log.append(("legacy", name))
            return f"{name} done"

        return fake

    for name in ("notes_create", "reminder_add", "calendar_add", "calendar_today", "mail_unread"):
        monkeypatch.setattr(legacy, name, legacy_fake(name))

    async def open_app(target):
        log.append(("legacy", f"open {target}"))
        return f"Opened {target}"

    monkeypatch.setattr(apps, "open_app", open_app)
    monkeypatch.setattr(
        web, "_race", lambda q, n: log.append(("web", q)) or [{"title": "t", "href": "h", "body": "b"}]
    )
    monkeypatch.setattr(web, "_country", lambda: "")  # legacy's store for "amazon": the .com one

    def driver(kind):
        async def fake(*a, **k):
            log.append(("driver", kind))
            return run.Result(f"{kind} done", True)

        return fake

    async def carry_out(kind, a, text="", *, ask_access=False):
        return await driver(kind)()

    monkeypatch.setattr(schedule, "carry_out", carry_out)
    monkeypatch.setattr(note, "write", driver("write_note"))
    monkeypatch.setattr(mail, "unread", driver("read_mail"))
    monkeypatch.setattr(launch, "open_target", driver("open_app"))
    monkeypatch.setattr(site, "search", driver("web_search"))
    # keys: the front app, NEO's own guard, the notes app's New command
    monkeypatch.setattr(computer, "_guard_front", lambda want="": asyncio.sleep(0))
    monkeypatch.setattr(ax, "frontmost_app", lambda: ("Notes", 42))
    monkeypatch.setattr(
        inp, "hotkey", lambda keys: log.append(("legacy", f"keys {keys}")) or f"Pressed {keys}"
    )

    async def top_app(said=""):
        return NOTES

    async def window_pid(app, ready_s):
        return 42

    monkeypatch.setattr(note, "top_app", top_app)
    monkeypatch.setattr(axkit, "window_pid", window_pid)
    monkeypatch.setattr(axkit, "menu_item", lambda pid, ch, mods=0: ("item", "New Note"))
    monkeypatch.setattr(axkit, "perform", lambda el, action: log.append(("driver", "new note")) or True)
    return log


def _say(text: str, call: ToolCall | None, monkeypatch) -> tuple[str, int]:
    brain = Brain(call)
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": brain)
    r = asyncio.run(Session(reflex=Reflex()).handle(text, from_user=True))
    return r.route, brain.calls


def _call(tool: str, **args) -> ToolCall:
    return ToolCall("1", tool, args)


ISO = "2026-09-27T17:00"
# (words, what the model calls when no fast path covers them, who answers in shadow, and when on)
MATRIX = [
    ("open safari", None, ("legacy", "open safari"), ("driver", "open_app")),
    ("launch vs code", None, ("legacy", "open vs code"), ("driver", "open_app")),
    ("open github", None, ("legacy", "open github"), ("driver", "open_app")),
    ("open github.com", None, ("legacy", "open github.com"), ("driver", "open_app")),
    (
        "search 2tb ssd on amazon",
        _call("search_site", site="amazon", query="2tb ssd"),
        ("legacy", "open https://www.amazon.com/s?k=2tb+ssd"),
        ("driver", "web_search"),
    ),
    (
        "search youtube for cats",
        _call("search_site", site="youtube", query="cats"),
        ("legacy", "open https://www.youtube.com/results?search_query=cats"),
        ("driver", "web_search"),
    ),
    (
        "open youtube and search for cats",
        _call("search_site", site="youtube", query="cats"),
        ("legacy", "open https://www.youtube.com/results?search_query=cats"),
        ("driver", "web_search"),
    ),
    (
        "search news on apple",
        _call("web_search", query="news on apple"),
        ("web", "news on apple"),
        ("web", "news on apple"),
    ),
    (
        "search articles on climate change",
        _call("web_search", query="articles on climate change"),
        ("web", "articles on climate change"),
        ("web", "articles on climate change"),
    ),
    ("search for the best laptops", None, ("web", "the best laptops"), ("web", "the best laptops")),
    ("google cheap flights", None, ("web", "cheap flights"), ("web", "cheap flights")),
    (
        "look up the weather on monday",
        _call("web_search", query="the weather on monday"),
        ("web", "the weather on monday"),
        ("web", "the weather on monday"),
    ),
    ("make a note saying buy milk", None, ("legacy", "notes_create"), ("driver", "write_note")),
    (
        "create a note called groceries saying milk",
        None,
        ("legacy", "notes_create"),
        ("driver", "write_note"),
    ),
    ("what's on my calendar", None, ("legacy", "calendar_today"), ("driver", "read_calendar")),
    ("what's on my calendar today", None, ("legacy", "calendar_today"), ("driver", "read_calendar")),
    ("check my email", None, ("legacy", "mail_unread"), ("driver", "read_mail")),
    ("any new emails", None, ("legacy", "mail_unread"), ("driver", "read_mail")),
    (
        "remind me to call mom at 5",
        _call("reminder_add", title="call mom", due=ISO),
        ("legacy", "reminder_add"),
        ("driver", "add_reminder"),
    ),
    (
        "put lunch with sam on my calendar tomorrow at 1",
        _call("calendar_add", title="lunch with sam", start=ISO, end="2026-09-27T18:00"),
        ("legacy", "calendar_add"),
        ("driver", "add_event"),
    ),
    ("new note", None, ("legacy", "keys cmd+n"), ("driver", "new note")),
]


@pytest.mark.parametrize("mode", ["shadow", "on"])
@pytest.mark.parametrize("text, call, in_shadow, when_on", MATRIX, ids=[m[0] for m in MATRIX])
def test_every_command_still_works(world, decide_modes, monkeypatch, mode, text, call, in_shadow, when_on):
    decide_modes(**(dict.fromkeys(KINDS, "on") if mode == "on" else {}))
    route, model_calls = _say(text, call, monkeypatch)
    assert world == [in_shadow if mode == "shadow" else when_on]
    if route == "agent":  # no fast path covered it: the model made the call
        assert call is not None and model_calls >= 1
    else:
        assert route == "quick" and model_calls == 0


# Where each command goes: a fast path (tool) or a model (None), today and with every kind on.
ROUTES = [
    ("search 2tb ssd on amazon", "search_site", None),  # the site word is learned after one landing
    ("search youtube for cats", "search_site", None),
    ("search news on apple", "web_search", None),  # no website opened by mistake
    ("search articles on climate change", "web_search", None),
    ("search for restaurants in Delhi", "web_search", "web_search"),
    ("search milk in notes", None, None),  # "in notes": the agent, as today
    ("remind me to call mom at 5", None, "reminder_add"),
    ("put lunch with sam on my calendar tomorrow at 1", None, "calendar_add"),
    ("what's on my calendar tomorrow", None, "calendar_today"),
    ("press enter", "hotkey", "hotkey"),
    ("new document", "hotkey", "hotkey"),
    ("new tab", "hotkey", "hotkey"),
    ("type hello in slack", "type_text", "type_text"),
    ("play despacito", "play_music", "play_music"),
]


def _fast_tool(text: str) -> str | None:
    from neo.agent.fastpath import match

    hit = match(text)
    return hit[0] if hit else None


@pytest.mark.parametrize("text, today, when_on", ROUTES, ids=[r[0] for r in ROUTES])
def test_each_command_takes_the_route_it_should(decide_modes, text, today, when_on):
    assert _fast_tool(text) == today
    decide_modes(**dict.fromkeys(KINDS, "on"))
    assert _fast_tool(text) == when_on
