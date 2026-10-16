"""A request that needs a model when NEO has none: NoBrain, the words for the screen and the voice, and
Brains & Keys opened — only for the user's own request, at most once every 10 minutes, and never
without NEO.app (then the terminal command is shown). And the composer's key guard: typed text that
starts with an API key goes to no model. Fakes only; no brain is asked."""

from __future__ import annotations

import asyncio

import pytest

from neo2.core import config, events, onboarding
from neo2.core import providers as prov
from neo2.core.agent import session as sess_mod
from neo2.core.agent.loop import run_agent
from neo2.core.agent.session import Session
from neo2.core.providers.base import Message, NoBrain, Unavailable
from neo2.core.reflex.schema import Decision

_VARS = ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "CEREBRAS_API_KEY", "OPENAI_API_KEY",
         "OPENROUTER_API_KEY", "NEO_BRAIN", "NEO_FAST_BRAIN")  # fmt: skip


class FakeReflex:
    def __init__(self, intent="chat"):
        self.d = Decision(intent, 0.9, False, 0.0, False, 0.0, "fake")

    async def decide(self, text):
        return self.d


@pytest.fixture
def no_keys(monkeypatch):
    for k in _VARS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(prov, "_cache", {})
    monkeypatch.setattr(prov, "_demoted", {})

    def make(n):
        raise Unavailable(f"no {n}")  # (no key, nothing running on this Mac)

    monkeypatch.setattr(prov, "make", make)
    bus = events.EventBus()
    monkeypatch.setattr(events, "_bus", bus)
    seen: list[tuple[str, dict]] = []
    bus.subscribe(lambda ev: seen.append((ev.type, dict(ev.data))))
    monkeypatch.setattr(onboarding, "_opener", None)
    from neo2.core.tools import load_all

    load_all()
    return seen


def test_no_key_at_all_is_no_brain(no_keys):
    with pytest.raises(NoBrain) as e:
        asyncio.run(prov.brain("agent").complete([Message.user("hi")]))
    assert e.value.kind == "no_brain" and "Brains & Keys" in str(e.value)


def test_an_embedded_agent_keeps_the_old_words(no_keys, monkeypatch):
    from neo2.core import runtime

    monkeypatch.setattr(runtime, "_library", True)
    monkeypatch.setattr(runtime, "check_unbound", lambda what: None)
    with pytest.raises(NoBrain, match="no API key set"):
        asyncio.run(prov.brain("agent").complete([Message.user("hi")]))


def test_the_loop_ends_on_how_to_connect_one(no_keys):
    res = asyncio.run(run_agent("what's on my calendar", provider=prov.brain("agent"), system="", tools=[]))
    assert res.stopped == "error" and res.failure == "no_brain" and res.text == onboarding.OPEN_IT


def _ask(text: str, *, origin_user: bool = True) -> sess_mod.Reply:
    s = Session(reflex=FakeReflex("chat"))
    return asyncio.run(s.handle(text, from_user=origin_user))


def test_the_users_request_opens_brains_and_keys(no_keys):
    asked: list[bool] = []

    def opener(open_now: bool) -> str:
        asked.append(open_now)
        return "opened" if open_now else "busy"

    onboarding.set_opener(opener)
    r = _ask("tell me a joke")
    assert r.text == onboarding.OPENED and r.spoken == ""  # (the voice says the same words)
    assert asked == [True]
    assert ("setup_needed", {"shell": True, "opened": True, "command": onboarding.COMMAND}) in no_keys
    said = [d["text"] for t, d in no_keys if t == "transcript" and d.get("role") == "assistant" and d.get("final")]
    assert said[-1] == onboarding.OPENED


def test_a_request_the_user_didnt_make_never_opens_it(no_keys):
    asked: list[bool] = []
    onboarding.set_opener(lambda open_now: (asked.append(open_now), "busy")[1])
    r = asyncio.run(Session(reflex=FakeReflex("chat")).handle("summarise the page", origin="live"))
    assert asked == [False] and r.text == onboarding.OPEN_IT


def test_without_neo_app_the_screen_shows_the_terminal_command(no_keys):
    r = _ask("tell me a joke")  # (no opener: a core started by hand, the text REPL)
    assert r.text == onboarding.TERMINAL and onboarding.COMMAND in r.text
    assert r.spoken == onboarding.TERMINAL_SPOKEN
    assert ("setup_needed", {"shell": False, "opened": False, "command": onboarding.COMMAND}) in no_keys


class _UI:
    def __init__(self, shell=True):
        self.shell, self.opened = shell, 0

    def has_shell(self):
        return self.shell

    def open_keys(self):
        self.opened += 1
        return self.shell


def test_the_app_opens_it_at_most_every_ten_minutes(monkeypatch):
    from neo2.core import app as app_mod

    clock = [1000.0]
    monkeypatch.setattr(app_mod.time, "monotonic", lambda: clock[0])
    a = app_mod.App.__new__(app_mod.App)
    a.ui, a._keys_opened_at = _UI(), None
    assert a.open_keys(False) == "busy" and a.ui.opened == 0  # (not the user's request: never)
    assert a.open_keys(True) == "opened" and a.ui.opened == 1
    clock[0] += 599
    assert a.open_keys(True) == "busy" and a.ui.opened == 1
    clock[0] += 2
    assert a.open_keys(True) == "opened" and a.ui.opened == 2
    a.ui = _UI(shell=False)
    assert a.open_keys(True) == "no_app"
    a.ui = None
    assert a.open_keys(True) == "no_app"


def test_the_voice_says_the_spoken_words(no_keys):
    from neo2.core import app as app_mod

    class Voice:
        said: list[str] = []

        async def speak(self, text):
            self.said.append(text)

    a = app_mod.App.__new__(app_mod.App)
    a.session, a.voice, a._tasks = Session(reflex=FakeReflex("chat")), Voice(), set()
    asyncio.run(a.handle_text("tell me a joke"))
    assert Voice.said == [onboarding.TERMINAL_SPOKEN]


# ---- the composer's key guard ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "typed",
    [
        "sk-ant-TEST-0000-abcdefghijklmnop",
        "  AIzaTEST0000abcdefghijklmnopqrstu  ",
        "gsk_TEST0000abcdefghijklmnop please",
        "sk-proj-TEST0000abcdefghijklmnop",
        "sk-or-v1-TEST0000abcdefghijklmnop",
        "xai-TEST0000abcdefghijklmnop",
        "sk-TEST0000abcdefghijklmnopqrstuvwxyz0123456789",
    ],
)
def test_a_typed_key_goes_to_no_model(no_keys, typed, monkeypatch):
    from neo2.core import app as app_mod

    handled: list[str] = []

    class Live:
        handles_text = True

        async def send_text(self, text):
            handled.append(text)

    class S:
        async def handle(self, text, **kw):
            handled.append(text)

    a = app_mod.App.__new__(app_mod.App)
    a.session, a.voice, a._tasks = S(), Live(), set()
    asyncio.run(a.on_command({"cmd": "text", "text": typed}))
    a.voice = None
    asyncio.run(a.on_command({"cmd": "text", "text": typed}))
    asyncio.run(a.handle_text(typed))
    assert handled == []  # (not Gemini Live, not the session: no model, no journal)
    said = [d for t, d in no_keys if t == "transcript"]
    assert said and all(d["text"] == onboarding.KEY_TYPED and d["role"] == "assistant" for d in said)


@pytest.mark.parametrize("typed", ["sk-learn is a python library", "what is 3f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a", "AIza is a prefix"])
def test_the_guard_is_narrow(no_keys, typed):
    from neo2.core import app as app_mod

    handled: list[str] = []

    class S:
        async def handle(self, text, **kw):
            handled.append(text)
            return sess_mod.Reply("ok", "chat")

    a = app_mod.App.__new__(app_mod.App)
    a.session, a.voice, a._tasks = S(), None, set()
    asyncio.run(a.on_command({"cmd": "text", "text": typed}))
    assert handled == [typed]
