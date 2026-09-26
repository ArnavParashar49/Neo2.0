"""The dictation lane: typing, keys and short composition without the agent loop."""

from __future__ import annotations

import asyncio

import pytest

from neo.agent import early as early_mod
from neo.agent.early import EarlyActor
from neo.agent.registry import ToolContext, registry
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
        ("scratch that", ("hotkey", {"keys": "cmd+z", "after": "input"})),
        ("make a new note", ("hotkey", {"keys": "cmd+n", "app": "Notes"})),
        ("open new tab", ("hotkey", {"keys": "cmd+t", "app": "tabs"})),
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
        return None, False

    monkeypatch.setattr(comp, "_ensure_text_focus", no_focus_needed)
    monkeypatch.setattr(comp.inp, "type_text", lambda s, **kw: typed.append(s) or f"Typed {len(s)}")
    monkeypatch.setattr(comp.inp, "hotkey", lambda k: typed.append(f"<{k}>") or f"Pressed {k}")
    monkeypatch.setattr(comp.ax, "frontmost_app", lambda: (front[0], 1))
    monkeypatch.setattr(comp.apps, "front_name", lambda: front[0])
    monkeypatch.setattr(comp.apps, "_target", None)
    monkeypatch.setattr(comp.ax, "focused_role", lambda: (role[0], ""))
    monkeypatch.setattr(comp.ax, "main_text_area", lambda app=None: None)
    return typed


front, role = ["Notes"], ["AXTextArea"]  # what the stubbed Mac reports; tests may change these


def test_type_text_presses_return_between_lines(monkeypatch):
    typed = _stub_typing(monkeypatch)
    from neo.agent.registry import ToolContext

    out = asyncio.run(registry().invoke("type_text", {"text": "a\nb\n"}, ToolContext(user_text="")))
    assert out.ok and typed == ["a", "<return>", "b"]  # a trailing newline is never a Return


@pytest.mark.parametrize(
    "app, focus, expected",
    [
        ("Safari", "AXTextField", ["a b"]),  # single-line field: joined, never submitted
        ("Slack", "AXTextArea", ["a", "<shift+return>", "b"]),  # chat: a line break, not "send"
        ("Terminal", "AXTextArea", ["a b"]),  # a shell never gets a multi-line enter
        ("Notes", "AXTextArea", ["a", "<return>", "b"]),
    ],
)
def test_line_breaks_never_send(monkeypatch, app, focus, expected):
    typed = _stub_typing(monkeypatch)
    front[0] = app
    role[0] = focus
    try:
        out = asyncio.run(registry().invoke("type_text", {"text": "a\nb"}, ToolContext(user_text="")))
        assert out.ok and typed == expected
    finally:
        front[0], role[0] = "Notes", "AXTextArea"


def test_never_types_into_a_password_field(monkeypatch):
    from neo.tools import computer as comp

    typed = []
    monkeypatch.setattr(comp.ax, "focused_is_secure", lambda: True)
    monkeypatch.setattr(comp.inp, "type_text", lambda s, **kw: typed.append(s) or "ok")
    out = asyncio.run(registry().invoke("type_text", {"text": "hunter2"}, ToolContext(user_text="")))
    assert not out.ok and "password" in out.text and typed == []


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

    asyncio.run(run())


def test_open_app_fires_early_but_dictation_waits_and_keeps_its_and(monkeypatch):
    typed = _stub_typing(monkeypatch)
    calls = []

    async def fake_open(a, c):
        calls.append(a["target"])
        return "Opened"

    monkeypatch.setattr(registry().get("open_app"), "handler", fake_open)
    ea = EarlyActor()

    async def run():
        ea.feed("open notes and type salt and pepper")
        ea._changed -= 1.0
        await ea.tick()
        assert calls == ["notes"] and typed == []  # the app opens mid-sentence; the typing waits
        await ea.tick(final=True)
        assert typed == ["salt and pepper"]  # never split at "and"
        assert calls == ["notes"]  # and the app isn't opened twice

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


def test_agent_prompt_lists_what_already_ran_once(monkeypatch):
    calls = []

    async def fake_open(a, c):
        calls.append(a["target"])
        return "Opened"

    monkeypatch.setattr(registry().get("open_app"), "handler", fake_open)
    from neo.agent.session import _already_done_note

    assert _already_done_note() == ""
    ea = early_mod.early()

    async def run():
        ea.feed("open notes and summarize my inbox into it")
        ea._changed -= 1.0
        await ea.tick()

    asyncio.run(run())
    note = _already_done_note()
    assert "do NOT repeat" in note and "open_app" in note and "notes" in note
    assert _already_done_note() == ""  # consumed: the next request doesn't inherit it


# ---- keystrokes only go to the app NEO is working in ---------------------------------------------


def _guard_env(monkeypatch, *, front_app, target, can_raise):
    from neo.tools import computer as comp

    typed: list[str] = []
    state = {"front": front_app}

    async def bring_front(name, wait=1.5):
        if can_raise:
            state["front"] = name
        return can_raise, name

    monkeypatch.setattr(comp.apps, "front_name", lambda: state["front"])
    monkeypatch.setattr(comp.apps, "bring_front", bring_front)
    monkeypatch.setattr(comp.apps, "_target", (target, __import__("time").time()) if target else None)
    monkeypatch.setattr(comp.ax, "frontmost_app", lambda: (state["front"], 1))
    monkeypatch.setattr(comp.ax, "focused_is_secure", lambda: False)
    monkeypatch.setattr(comp.ax, "focused_editable", lambda: True)
    monkeypatch.setattr(comp.ax, "focused_role", lambda: ("AXTextArea", ""))
    monkeypatch.setattr(comp.ax, "main_text_area", lambda app=None: None)
    monkeypatch.setattr(comp.inp, "type_text", lambda t, **kw: typed.append(t) or "ok")
    monkeypatch.setattr(comp.inp, "hotkey", lambda k: typed.append(f"<{k}>") or "ok")
    return typed, state


def test_never_types_into_another_app_when_the_target_wont_come_forward(monkeypatch):
    typed, _ = _guard_env(monkeypatch, front_app="Claude", target="Notes", can_raise=False)
    out = asyncio.run(registry().invoke("type_text", {"text": "salt and pepper"}, ToolContext(user_text="")))
    assert out.text.startswith("NEEDS_USER") and "Notes" in out.text and typed == []
    out = asyncio.run(registry().invoke("hotkey", {"keys": "cmd+w"}, ToolContext(user_text="")))
    assert out.text.startswith("NEEDS_USER") and typed == []  # no ⌘W into the user's app either


def test_brings_the_target_forward_then_types(monkeypatch):
    typed, state = _guard_env(monkeypatch, front_app="Claude", target="Notes", can_raise=True)
    out = asyncio.run(registry().invoke("type_text", {"text": "Laptops"}, ToolContext(user_text="")))
    assert out.ok and typed == ["Laptops"] and state["front"] == "Notes"


def test_without_a_recent_target_types_where_the_cursor_is(monkeypatch):
    typed, _ = _guard_env(monkeypatch, front_app="TextEdit", target=None, can_raise=False)
    out = asyncio.run(registry().invoke("type_text", {"text": "hi"}, ToolContext(user_text="")))
    assert out.ok and typed == ["hi"]


def test_new_note_presses_nothing_unless_notes_is_in_front(monkeypatch):
    typed, _ = _guard_env(monkeypatch, front_app="Claude", target=None, can_raise=False)
    out = asyncio.run(registry().invoke("hotkey", {"keys": "cmd+n", "app": "Notes"}, ToolContext(user_text="")))
    assert out.text.startswith("NEEDS_USER") and typed == []


def test_never_types_into_neos_own_overlay(monkeypatch):
    typed, _ = _guard_env(monkeypatch, front_app="neo-ui", target=None, can_raise=False)
    out = asyncio.run(registry().invoke("type_text", {"text": "hi"}, ToolContext(user_text="")))
    assert out.text.startswith("NEEDS_USER") and typed == []


def test_needs_user_is_spoken_not_handed_to_the_agent(monkeypatch):
    import neo.agent.session as sess_mod

    typed, _ = _guard_env(monkeypatch, front_app="Claude", target="Notes", can_raise=False)
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": pytest.fail("no model: tell the user"))

    class R:
        async def decide(self, t):
            return Decision("quick_action", 0.95, False, 0.0, False, 0.0, "laya+head")

    r = asyncio.run(Session(reflex=R()).handle("type hello"))
    assert r.route == "quick" and not r.silent and "didn't type" in r.text and typed == []


def test_live_multi_step_commands_are_not_silent():
    from neo.voice.live import _multi_step

    assert _multi_step("type github.com and press enter")
    assert not _multi_step("type salt and pepper")
    assert not _multi_step("open notes")


def test_live_agent_task_uses_the_users_words_when_fast_paths_cover_them(monkeypatch):
    import neo.voice.live as live_mod

    seen = []

    class S:
        async def handle(self, text, **kw):
            from neo.agent.session import Reply

            seen.append(text)
            return Reply("ok", "quick", silent=True)

    v = live_mod.LiveVoice.__new__(live_mod.LiveVoice)
    v._agent_session, v._tool_tasks, v._goals, v._server_cancelled = S(), {}, {}, set()
    v._live, v._ui_lock, v._dialog = None, asyncio.Lock(), []

    class FC:
        id, name, args = "c1", "agent_task", {"goal": "Create a new note in the Notes app"}

    async def run():
        await v._run_tool(None, FC(), "new note")
        await v._run_tool(None, FC(), "put the highlights of my emails in a note")

    asyncio.run(run())
    assert seen == ["new note", "Create a new note in the Notes app"]
