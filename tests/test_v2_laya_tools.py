"""Laya v2: embedding-only decisions with a tool head, and the session lane that acts on them."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from neo.agent.session import Session
from neo.events import EventBus
from neo.providers.base import ToolCall, Turn
from neo.reflex.schema import Decision


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    from neo import config, events
    from neo.tools import load_all

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(events, "_bus", EventBus())
    load_all()


def test_v2_head_decides_from_the_embedding_alone(tmp_path):
    from neo.reflex import laya_reflex as lr

    d = 8
    intents = ["chat", "quick_action", "agent_task", "stop"]
    W_int = np.zeros((4, d + 1), np.float32)
    W_int[1, 0] = 5.0  # feature 0 high → quick_action
    W_int[0, 1] = 5.0  # feature 1 high → chat
    tool_classes = ["clock", "none", "web_search"]
    W_tool = np.zeros((3, d + 1), np.float32)
    W_tool[0, 2] = 6.0  # feature 2 → clock
    head = {
        "version": 2,
        "mu": [0.0] * (d + 1),
        "sd": [1.0] * (d + 1),
        "intent": {"W": W_int.tolist(), "classes": intents},
        "destructive": {"W": np.zeros((2, d + 1)).tolist()},
        "needs_screen": {"W": np.zeros((2, d + 1)).tolist()},
        "tool": {"W": W_tool.tolist(), "classes": tool_classes},
        "zs_below": 0.6,
    }
    path = tmp_path / "reflex_head.json"
    path.write_text(json.dumps(head))
    loaded = lr._load_head(path)
    assert loaded["version"] == 2 and loaded["tool_classes"] == tool_classes

    vecs = {"what time is it": [1, 0, 1, 0, 0, 0, 0, 0], "tell me a joke": [0, 1, 0, 0, 0, 0, 0, 0], "hmm": [0.1, 0.1, 0, 0, 0, 0, 0, 0]}
    calls = []

    class Agent:
        def system_one(self, item, questions):
            calls.append(item["text"])
            return {"answers": {"intent": {"probabilities": {"chat": 0.1, "quick_action": 0.1, "agent_task": 0.7, "stop": 0.1}}}}

    r = lr.LayaReflex.__new__(lr.LayaReflex)
    r._agent, r._head = Agent(), loaded
    monkeypatch_embed = lambda agent, texts: np.array([vecs[t] for t in texts], np.float32)  # noqa: E731
    import neo.reflex.features as feats

    orig = feats.embed
    feats.embed = monkeypatch_embed
    try:
        a = r.decide("what time is it")
        assert a.intent == "quick_action" and a.tool == "clock" and a.tool_p > 0.9 and a.source == "laya+head"
        b = r.decide("tell me a joke")
        assert b.intent == "chat" and b.tool == "" and calls == []  # confident: zero-shot never ran
        c = r.decide("hmm")  # unsure → zero-shot consulted and blended
        assert c.source == "laya+zs" and calls == ["hmm"] and c.intent == "agent_task"
    finally:
        feats.embed = orig


def _reflex(intent, tool="", tool_p=0.0):
    class R:
        async def decide(self, t):
            return Decision(intent, 0.95, False, 0.0, False, 0.0, "laya+head", tool=tool, tool_p=tool_p)

    return R()


def test_laya_tool_without_arguments_runs_with_no_model(monkeypatch):
    import neo.agent.session as sess_mod
    from neo.agent.registry import registry

    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": pytest.fail("no model needed"))

    async def fake_clock(a, c):
        return "It's noon."

    monkeypatch.setattr(registry().get("clock"), "handler", fake_clock)
    r = asyncio.run(Session(reflex=_reflex("quick_action", "clock", 0.97)).handle("do you know the hour"))
    assert r.route == "quick" and r.text == "It's noon."


def test_laya_tool_with_arguments_gets_a_single_tool_light_call(monkeypatch):
    import neo.agent.session as sess_mod

    seen = {}

    class B:
        name = "fake"
        supports_vision = False
        supports_tools = True
        turns = [Turn(tool_calls=[ToolCall("1", "web_search", {"query": "weather mumbai"})]), Turn(text="Sunny.")]

        async def complete(self, messages, *, tools=None, **kw):
            seen.setdefault("tools", [t.name for t in tools or []])
            return self.turns.pop(0)

        async def stream(self, messages, **kw):
            yield ""

    purposes = []

    def fake_brain(purpose="agent"):
        purposes.append(purpose)
        return B()

    monkeypatch.setattr(sess_mod, "brain", fake_brain)
    from neo.agent.registry import registry

    async def fake_search(a, c):
        return "28°C and sunny"

    monkeypatch.setattr(registry().get("web_search"), "handler", fake_search)
    r = asyncio.run(Session(reflex=_reflex("quick_action", "web_search", 0.93)).handle("any idea what the skies look like over mumbai"))
    assert r.route == "agent" and r.text == "Sunny."
    assert seen["tools"] == ["web_search", "more_tools"] and purposes == ["light"]


def test_low_tool_confidence_falls_back_to_the_normal_route(monkeypatch):
    import neo.agent.session as sess_mod

    class B:
        name = "fake"
        supports_vision = False
        supports_tools = True

        async def complete(self, messages, *, tools=None, **kw):
            assert len(tools) > 3  # the usual per-request selection, not a single forced tool
            return Turn(text="ok")

        async def stream(self, messages, **kw):
            yield ""

    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": B())
    r = asyncio.run(Session(reflex=_reflex("quick_action", "web_search", 0.5)).handle("something vague please"))
    assert r.route == "agent"


def test_dataset_tool_labels():
    from neo.reflex.dataset import tool_for

    assert tool_for("open Safari") == "open_app"
    assert tool_for("is it going to rain today") == "weather"
    assert tool_for("type hello world") == "type_text"
    assert tool_for("remember that I park on level 3") == "memory"
    assert tool_for("play some jazz") == ""
