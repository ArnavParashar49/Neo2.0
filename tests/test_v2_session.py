"""Session routing tests with a fake reflex and fake brains (no network, no models)."""

from __future__ import annotations

import asyncio

import pytest

from neo.agent import session as sess_mod
from neo.agent.registry import registry
from neo.agent.session import Session, _match_fast_path
from neo.events import EventBus
from neo.reflex.schema import Decision


class FakeReflex:
    def __init__(self, intent="chat", **kw):
        self.d = Decision(
            intent, 0.9, kw.get("destructive", False), 0.0, kw.get("needs_screen", False), 0.0, "fake"
        )

    async def decide(self, text):
        return self.d


class FakeChain:
    name = "fake"
    supports_vision = False
    supports_tools = True

    def __init__(self):
        self.streamed = 0
        self.completed = 0

    async def stream(self, messages, *, system="", max_tokens=2048):
        self.streamed += 1
        yield "hello "
        yield "there"

    async def complete(self, messages, *, system="", tools=None, effort="medium", max_tokens=4096):
        from neo.providers.base import Turn

        self.completed += 1
        return Turn(text="agent did it")


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    from neo import config, events

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(events, "_bus", EventBus())
    from neo.tools import load_all

    load_all()


def test_fast_path_matches_volume_and_open():
    assert _match_fast_path("turn the volume up") == ("volume", {"action": "up"})
    assert _match_fast_path("open Safari") == ("open_app", {"target": "Safari"})
    assert _match_fast_path("please write me a poem") is None


def test_chat_route_streams(monkeypatch):
    chain = FakeChain()
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": chain)
    s = Session(reflex=FakeReflex("chat"))
    r = asyncio.run(s.handle("how are you"))
    assert r.route == "chat" and r.text == "hello there"
    assert chain.streamed == 1 and chain.completed == 0
    assert [m.role for m in s.history] == ["user", "assistant"]


def test_agent_route_uses_loop(monkeypatch):
    chain = FakeChain()
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": chain)
    s = Session(reflex=FakeReflex("agent_task"))
    r = asyncio.run(s.handle("organize my desktop"))
    assert r.route == "agent" and r.text == "agent did it"
    assert chain.completed == 1


def test_quick_route_runs_tool_without_llm(monkeypatch):
    chain = FakeChain()
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": chain)

    async def fake_clock(a, c):
        return "It's noon."

    registry().get("clock").handler = fake_clock  # type: ignore[union-attr]
    s = Session(reflex=FakeReflex("quick_action"))
    r = asyncio.run(s.handle("what time is it"))
    assert r.route == "quick" and r.text == "It's noon."
    assert chain.completed == 0 and chain.streamed == 0


def test_stop_route():
    s = Session(reflex=FakeReflex("stop"))
    r = asyncio.run(s.handle("stop"))
    assert r.route == "stop"


def test_fast_path_used_when_reflex_unavailable(monkeypatch):
    chain = FakeChain()
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": chain)

    async def fake_clock(a, c):
        return "It's noon."

    registry().get("clock").handler = fake_clock  # type: ignore[union-attr]
    # source="rules" is what Reflex returns when neither Laya nor Flash-Lite is available
    r = FakeReflex("agent_task")
    r.d.source = "rules"
    s = Session(reflex=r)
    out = asyncio.run(s.handle("what time is it"))
    assert out.route == "quick" and chain.completed == 0
