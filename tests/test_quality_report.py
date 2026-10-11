"""The reliability report (neo2.core.tools.quality): an explicit ask reads back the local counters; a
greeting ("how are you doing?") is conversation and never gets statistics. No model, no network."""

from __future__ import annotations

import asyncio

import pytest

from neo2.core.agent import fastpath


@pytest.fixture(autouse=True)
def _tools():
    from neo2.core.tools import load_all

    load_all()


ASKS = [
    "how's your reliability?",
    "how is your reliability",
    "how's neo doing?",
    "show me your quality report",
    "show neo's recent reliability",
]

GREETINGS = [
    "how are you doing?",
    "How are you doing, Neo?",
    "how have you been doing?",
    "how is neo",
    "how's neo?",
    "how are you",
    "how have you been",
]


@pytest.mark.parametrize("text", ASKS)
def test_an_explicit_ask_is_the_report(text):
    assert fastpath.plan(text) == [("quality_report", {"days": 7})]


@pytest.mark.parametrize("text", GREETINGS)
def test_a_greeting_is_not(text):
    steps = fastpath.plan(text) or []
    assert all(t != "quality_report" for t, _ in steps)


class _Reflex:
    def __init__(self, intent="chat"):
        from neo2.core.reflex.schema import Decision

        self.d = Decision(intent, 0.9, False, 0.0, False, 0.0, "fake")

    async def decide(self, text):
        return self.d


def test_the_session_chats_back_to_a_greeting(monkeypatch):
    from neo2.core import config, events
    from neo2.core.agent import session as sess_mod

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(events, "_bus", events.EventBus())
    chats: list[str] = []

    async def chat(self, text, d, job):
        chats.append(text)
        await events.bus().end_job(job)
        return sess_mod.Reply("I'm good, thanks!", "chat", d, job=job)

    monkeypatch.setattr(sess_mod.Session, "_chat", chat)
    s = sess_mod.Session(reflex=_Reflex("chat"))
    r = asyncio.run(s.handle("Neo, how are you doing?", from_user=True))
    assert r.route == "chat" and chats == ["how are you doing?"]
    r = asyncio.run(s.handle("how's your reliability?", from_user=True))
    assert r.route == "quick" and r.tools == ("quality_report",) and chats == ["how are you doing?"]
