"""The dictation lane: typing, keys and short composition without the agent loop."""

from __future__ import annotations

import asyncio

import pytest

from neo.agent import early as early_mod
from neo.agent.early import EarlyActor
from neo.agent.registry import registry
from neo.agent.session import Session, _match_fast_path
from neo.events import EventBus
from neo.providers.base import Turn
from neo.reflex.schema import Decision


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    from neo import config, events
    from neo.tools import load_all

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(events, "_bus", EventBus())
    monkeypatch.setattr(early_mod, "_actor", None)
    load_all()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("type Laptops in the heading", ("type_text", {"text": "Laptops"})),
        ("Type 'Laptops' in the heading of the current note.", ("type_text", {"text": "Laptops"})),
        ("type hello world", ("type_text", {"text": "hello world"})),
        ("write see you in the morning", ("type_text", {"text": "see you in the morning"})),
        ("type in the search bar hello", ("type_text", {"text": "hello"})),
        ("write down milk eggs bread", ("type_text", {"text": "milk eggs bread"})),
        ('type "Meeting notes" as the title', ("type_text", {"text": "Meeting notes"})),
        ("and then type done", ("type_text", {"text": "done"})),
        ("now type some points related to laptops", ("dictate", {"brief": "some points related to laptops"})),
        ("write a short paragraph about cats in the note", ("dictate", {"brief": "a short paragraph about cats"})),
        ("type 3 reasons to buy a mac", ("dictate", {"brief": "3 reasons to buy a mac"})),
        ("type something about the weather", ("dictate", {"brief": "something about the weather"})),
        ("press enter", ("hotkey", {"keys": "enter"})),
        ("hit the return key", ("hotkey", {"keys": "return"})),
        ("new line", ("hotkey", {"keys": "return"})),
        ("select all", ("hotkey", {"keys": "cmd+a"})),
        ("scratch that", ("hotkey", {"keys": "cmd+z"})),
        ("make a new note", ("hotkey", {"keys": "cmd+n"})),
        ("open new tab", ("hotkey", {"keys": "cmd+t"})),
        ("open notes", ("open_app", {"target": "notes"})),  # untouched
        # not dictation: tasks, questions, conversation
        ("write an email to john saying hi", None),
        ("write a message to mom saying I'm late", None),
        ("please write me a poem", None),
        ("what type of laptop should I buy", None),
        ("add a reminder for tomorrow", None),
        ("type", None),
        ("copy", None),
    ],
)
def test_dictation_fast_paths(text, expected):
    assert _match_fast_path(text) == expected


def _stub_typing(monkeypatch):
    from neo.tools import computer as comp

    typed: list[str] = []

    async def no_focus_needed():
        return None

    monkeypatch.setattr(comp, "_ensure_text_focus", no_focus_needed)
    monkeypatch.setattr(comp.inp, "type_text", lambda s, **kw: typed.append(s) or f"Typed {len(s)}")
    monkeypatch.setattr(comp.inp, "hotkey", lambda k: typed.append(f"<{k}>") or f"Pressed {k}")
    return typed


def test_type_text_presses_return_between_lines(monkeypatch):
    typed = _stub_typing(monkeypatch)
    from neo.agent.registry import ToolContext

    out = asyncio.run(registry().invoke("type_text", {"text": "a\nb\n"}, ToolContext(user_text="")))
    assert out.ok and typed == ["a", "<return>", "b", "<return>"]


def test_dictate_composes_with_the_fast_brain_then_types(monkeypatch):
    typed = _stub_typing(monkeypatch)
    from neo.agent.registry import ToolContext
    import neo.providers as prov

    class B:
        name = "fake"
        supports_vision = False
        supports_tools = True
        seen = {}

        async def complete(self, messages, *, system="", **kw):
            B.seen = {"brief": messages[-1].text, "system": system}
            return Turn(text='"- light\n- fast"')

    monkeypatch.setattr(prov, "make", lambda n: B())
    out = asyncio.run(
        registry().invoke("dictate", {"brief": "some points about laptops"}, ToolContext(user_text=""))
    )
    assert out.ok and "2 line(s)" in out.text
    assert typed == ["- light", "<return>", "- fast"]
    assert B.seen["brief"] == "some points about laptops" and "ONLY the text" in B.seen["system"]


def test_early_actor_waits_for_the_sentence_end_before_typing(monkeypatch):
    typed = _stub_typing(monkeypatch)
    ea = EarlyActor()

    async def run():
        ea.feed("type Lap")
        ea._changed -= 1.0  # stable for a second: still not typed — the words are the payload
        assert await ea.tick() == [] and typed == []
        ea.feed("type Laptops in the heading")
        assert await ea.tick(final=True) and typed == ["Laptops"]
        assert ea.remaining() == ""
        assert ea.recent(5.0) == [("type_text", {"text": "Laptops"})]

    asyncio.run(run())


def test_open_app_still_fires_early_but_dictation_after(monkeypatch):
    typed = _stub_typing(monkeypatch)
    calls = []

    async def fake_open(a, c):
        calls.append(a["target"])
        return "Opened"

    monkeypatch.setattr(registry().get("open_app"), "handler", fake_open)
    ea = EarlyActor()

    async def run():
        ea.feed("open notes and type hello")
        ea._changed -= 1.0
        await ea.tick()
        assert calls == ["notes"] and typed == []  # the app opens mid-sentence; the typing waits
        await ea.tick(final=True)
        assert typed == ["hello"]

    asyncio.run(run())


def test_session_takes_the_fast_path_even_when_the_reflex_says_agent_task(monkeypatch):
    typed = _stub_typing(monkeypatch)
    import neo.agent.session as sess_mod

    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": pytest.fail("no model needed"))

    class R:
        async def decide(self, t):
            return Decision("agent_task", 0.98, True, 0.0, False, 0.0, "laya")

    r = asyncio.run(Session(reflex=R()).handle("type Laptops in the heading"))
    assert r.route == "quick" and typed == ["Laptops"] and "Laptops" in r.text


def test_agent_prompt_lists_what_already_ran(monkeypatch):
    typed = _stub_typing(monkeypatch)
    from neo.agent.session import _already_done_note

    assert _already_done_note() == ""
    ea = early_mod.early()

    async def run():
        ea.feed("type hi")
        await ea.tick(final=True)

    asyncio.run(run())
    note = _already_done_note()
    assert "do NOT repeat" in note and "type_text" in note and "hi" in note
