"""Commands are done quietly; questions get answers."""

from __future__ import annotations

import asyncio

import pytest

from neo.agent.loop import AgentResult, Step
from neo.agent.registry import registry
from neo.agent.session import Session, _quietly
from neo.events import EventBus
from neo.providers.base import ToolCall, Turn
from neo.reflex import _with_reply
from neo.reflex.schema import Decision, wants_reply


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    from neo import config, events
    from neo.tools import load_all

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(events, "_bus", EventBus())
    load_all()


@pytest.mark.parametrize(
    "text, wants",
    [
        ("open notes", False),
        ("type Laptops in the heading", False),
        ("move the report to archive", False),
        ("set a timer for 10 minutes", False),
        ("what time is it", True),
        ("check my unread emails", True),
        ("how many emails do I have from Sam?", True),
        ("summarize the article that's open", True),
        ("is it going to rain today", True),
        ("can you tell me the weather", True),
        ("please read the report and summarise it", True),
        ("do I have any meetings today", True),
        ("delete the duplicates in Downloads", False),
        ("Neo, show me my calendar", True),
    ],
)
def test_wants_reply_regex(text, wants):
    assert wants_reply(text) is wants


def test_with_reply_rules():
    chat = _with_reply(Decision("chat", 0.9, False, 0, False, 0, "laya+head", reply=False, reply_p=0.1), "hi")
    assert chat.reply  # conversation always answers
    d = _with_reply(Decision("quick_action", 0.9, False, 0, False, 0, "laya+head"), "open notes")
    assert d.reply is False and d.reply_p == 0.5  # no reply head → regex fallback
    d = _with_reply(Decision("quick_action", 0.9, False, 0, False, 0, "laya+head", reply=False, reply_p=0.2), "what time is it")
    assert d.reply is False and d.reply_p == 0.2  # the head's verdict is kept as is


def _reflex(intent, reply, tool="", tool_p=0.0):
    class R:
        async def decide(self, t):
            return Decision(intent, 0.95, False, 0.0, False, 0.0, "laya+head", tool=tool, tool_p=tool_p, reply=reply, reply_p=0.9 if reply else 0.1)

    return R()


def _events():
    from neo.events import bus

    seen = []
    bus().subscribe(lambda e: seen.append((e.type, e.data.get("text", ""), e.data.get("role", ""))))
    return seen


def test_quiet_action_says_nothing_but_notes_for_the_ui(monkeypatch):
    calls = []

    async def fake_open(a, c):
        calls.append(a["target"])
        return "Opened notes"

    monkeypatch.setattr(registry().get("open_app"), "handler", fake_open)
    seen = _events()
    r = asyncio.run(Session(reflex=_reflex("quick_action", False)).handle("open notes"))
    assert r.route == "quick" and r.silent and calls == ["notes"]
    assert not [e for e in seen if e[0] == "transcript" and e[2] == "assistant"]  # no bubble
    assert any(e[0] == "note" and "Opened" in e[1] for e in seen)


def test_information_tool_is_spoken_even_when_phrased_as_a_command(monkeypatch):
    async def fake_clock(a, c):
        return "It's noon."

    monkeypatch.setattr(registry().get("clock"), "handler", fake_clock)
    seen = _events()
    r = asyncio.run(Session(reflex=_reflex("quick_action", False, "clock", 0.95)).handle("tell the time"))
    assert r.route == "quick" and not r.silent
    assert any(e[0] == "transcript" and e[2] == "assistant" and "noon" in e[1] for e in seen)


def test_failed_quiet_action_is_reported(monkeypatch):
    async def fake_open(a, c):
        return "Error: no such app"

    monkeypatch.setattr(registry().get("open_app"), "handler", fake_open)
    from neo.agent.registry import ToolOutput  # noqa: F401 — ok flag comes from the registry

    r = asyncio.run(Session(reflex=_reflex("quick_action", False)).handle("open blorp"))
    assert not r.silent


def _agent_session(monkeypatch, reply, text_out="Done."):
    import neo.agent.session as sess_mod

    class B:
        name = "fake"
        supports_vision = False
        supports_tools = True
        # tool call → answer → the verification turn's confirmation (open_app is a side effect)
        turns = [Turn(tool_calls=[ToolCall("1", "open_app", {"target": "notes"})]), Turn(text=text_out), Turn(text=text_out)]

        async def complete(self, messages, **kw):
            return self.turns.pop(0)

        async def stream(self, messages, **kw):
            yield ""

    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": B())

    async def fake_open(a, c):
        return "Opened notes"

    monkeypatch.setattr(registry().get("open_app"), "handler", fake_open)
    return Session(reflex=_reflex("agent_task", reply))


def test_agent_command_finishes_quietly(monkeypatch):
    seen = _events()
    r = asyncio.run(_agent_session(monkeypatch, reply=False).handle("open notes and get ready"))
    assert r.route == "agent" and r.silent
    assert not [e for e in seen if e[0] == "transcript" and e[2] == "assistant"]


def test_agent_question_is_answered(monkeypatch):
    seen = _events()
    r = asyncio.run(_agent_session(monkeypatch, reply=True, text_out="Notes is open with 3 notes.").handle("what's in notes?"))
    assert not r.silent and any(e[0] == "transcript" and e[2] == "assistant" for e in seen)


def test_quietly_rules():
    d = Decision("agent_task", 0.9, False, 0, False, 0, "laya+head", reply=False)
    ok = AgentResult("Done.", "done", [Step("open_app", {}, "ok", True)])
    assert _quietly(d, ok)
    assert not _quietly(d, AgentResult("I couldn't find it.", "done", [Step("open_app", {}, "ok", True)]))
    assert not _quietly(d, AgentResult("Stopped.", "max_steps", [Step("x", {}, "", True)]))
    assert not _quietly(d, AgentResult("Which one?", "done", [Step("x", {}, "", True)], question="Which one?"))
    assert not _quietly(d, AgentResult("Done.", "done", [Step("x", {}, "boom", False)]))
    assert not _quietly(None, ok)


def test_app_does_not_speak_silent_replies(monkeypatch):
    from neo.agent.session import Reply
    from neo.app import App

    spoken = []

    class V:
        async def speak(self, t):
            spoken.append(t)

    app = App.__new__(App)
    app.voice = V()
    app._tasks = set()

    class S:
        memory_context = ""

        async def handle(self, text):
            return Reply("Opened notes", "quick", silent=text.startswith("open"))

    app.session = S()
    import neo.app as app_mod

    class Store:
        def prompt_context(self, t):
            return ""

    monkeypatch.setattr(app_mod, "store", lambda: Store())
    asyncio.run(app.handle_text("open notes"))
    asyncio.run(app.handle_text("what time is it"))
    assert spoken == ["Opened notes"]  # only the answer, not the command's result
