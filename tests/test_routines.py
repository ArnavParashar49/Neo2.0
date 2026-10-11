"""Spoken routines (neo2.core.tools.routines): which sentences teach one, how the plain "when I say …,
open …" form is read locally, what a routine may open, the confirmation a page-read request and a
delete need, and how the session routes a definition. Every test has its own data dir (conftest):
nothing here reads or writes the real ~/.neo, and no app or page really opens."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import ModuleType

import pytest

from neo2.core.agent.registry import RegisteredTool, ToolContext, ToolOutput, catalog

routines: ModuleType  # neo2.core.tools.routines, imported by _routines_after_the_catalog below


@pytest.fixture(autouse=True)
def _routines_after_the_catalog():
    """Import routines inside a test, after load_all: imported while the tests are collected, its tools
    would be registered ahead of every other one, and a test elsewhere that fits "every tool" into a
    small budget (test_v2_brain_fallback) reads the catalog in its order."""
    global routines
    from neo2.core.tools import load_all

    load_all()
    from neo2.core.tools import routines as module

    routines = module


def _app(name: str) -> dict:
    return {"kind": "open_app", "target": name}


def _url(url: str) -> dict:
    return {"kind": "open_url", "target": url}


# ---- which sentences teach a routine ---------------------------------------------------------------

SETUP = [
    "When I say I want to study, open Notion",
    "when i say study time open Notion",
    "Neo, when I say good night, open Safari",
    "hey neo whenever I say focus mode, open Notion and Safari",
    "Please, when I say deep work, open Notion",
    "can you make it so when I say study time, open Notion",
    "make it so that whenever I say study time you open Notion",
    "Create a spoken routine",
    "make a voice shortcut for studying",
    "set up a voice routine",
    "save my spoken phrase",
    # A spoken transcript often starts with a filler: still the user's own definition.
    "Okay, when I say focus mode, open Notion.",
    "So when I say focus mode, open Notion",
    "Um, when I say focus mode, open Notion",
    "Alright, when I say focus mode, open Notion",
    "okay so uh when I say focus mode open Notion",
    "So Neo, when I say focus mode, open Notion",
    "And can you make it so when I say study time, open Notion",
]

NOT_SETUP = [
    "Make a shortcut to Downloads",  # a Finder alias, not a spoken phrase
    "make a shortcut",
    "create a keyboard shortcut for screenshots",
    "Why do you open Safari when I say Neo?",
    "what happens when I say stop",
    "I want to know what you do when I say hello",
    "remind me what I said when I say goodbye to Sam",
    "create a morning routine in my calendar",
    "save a shortcut to the desktop",
    "Okay, why do you open Safari when I say Neo?",  # a filler doesn't make a question a definition
    "So make a shortcut to Downloads",
    "um, what happens when I say stop",
    "",
]


@pytest.mark.parametrize("text", SETUP)
def test_an_explicit_definition_is_a_setup_request(text):
    assert routines.is_setup_request(text)


@pytest.mark.parametrize("text", NOT_SETUP)
def test_the_phrase_somewhere_in_a_sentence_is_not(text):
    assert not routines.is_setup_request(text)


# ---- the local parser ---------------------------------------------------------------------------------

PARSED = [
    # The README's own example (curly single quotes, the Oxford comma, a full stop after the URL).
    ("When I say ‘I want to study’, open Notion, Safari, and https://example.com.",
     "I want to study", [_app("Notion"), _app("Safari"), _url("https://example.com")]),
    # routines.parse_local_setup's docstring examples.
    ("When I say “I want to study”, open Notion, Safari, and youtube.com",
     "I want to study", [_app("Notion"), _app("Safari"), _url("https://youtube.com")]),
    ("When I say I want to study, then open Notion and Safari",
     "I want to study", [_app("Notion"), _app("Safari")]),
    # The trigger ends at the comma, not at a verb inside it ("time to start working").
    ("When I say time to start working, open VS Code", "time to start working", [_app("VS Code")]),
    # No comma (a spoken transcript): "open" wins over the "start" inside the phrase.
    ("When I say time to start working open VS Code", "time to start working", [_app("VS Code")]),
    ("When I say study time open Notion, Safari", "study time", [_app("Notion"), _app("Safari")]),
    ('When I say "deep work", open Notion', "deep work", [_app("Notion")]),
    ("When I say 'let's study', open Notion", "let's study", [_app("Notion")]),
    ("When I say “study time,” open Notion", "study time", [_app("Notion")]),
    ("Neo, when I say focus mode; open Notion; Safari", "focus mode", [_app("Notion"), _app("Safari")]),
    ("When I say let's go, open Notion then launch Safari", "let's go", [_app("Notion"), _app("Safari")]),
    ("When I say study time, open Notion, and then open Safari", "study time", [_app("Notion"), _app("Safari")]),
    ("can you make it so when I say study time, open Notion", "study time", [_app("Notion")]),
    ("When I say work mode, open the app Slack and the website github.com/me please.",
     "work mode", [_app("Slack"), _url("https://github.com/me")]),
    ("Okay, when I say focus mode, open Notion.", "focus mode", [_app("Notion")]),
    ("So um, when I say deep work open Notion and Safari", "deep work", [_app("Notion"), _app("Safari")]),
    ("When I say work mode, open the Notion app and start Slack", "work mode", [_app("Notion"), _app("Slack")]),
]


@pytest.mark.parametrize(("text", "trigger", "actions"), PARSED)
def test_the_plain_form_is_parsed_locally(text, trigger, actions):
    assert routines.parse_local_setup(text) == {"action": "create", "trigger": trigger, "actions": actions}


UNPARSED = [
    "When I say good night, turn off the lights",  # not an "open" routine at all
    "When I say good night, open Safari and turn off the lights",  # one part is a command
    "When I say good night, open Safari, then lock the screen",
    "When I say good night, turn off the lights and open Safari",  # (the phrase is never "good night, turn…")
    "When I say music time, open Spotify and play my liked songs",
    "When I say study time, open the notes app I use for my classes",  # a description, not a name
    "When I say study, open Notion",  # a one-word trigger is too easy to say by accident
    "When I say study time, open",
    "When I say study time, open localhost:3000",
    "When I say study time, open 192.168.1.10",
    "Make a shortcut to Downloads",
    "When I say good night",
    # Not an app or a site: "start" followed by what to start, a question or a thank-you tacked on,
    # a choice between two.
    "When I say chill out, open Spotify and start playing",
    "When I say chill out, start playing",
    "When I say focus mode, open Notion and start the timer",
    "When I say focus mode, open Safari, does that work?",
    "When I say focus mode, open Safari, right?",
    "When I say focus mode, open Notion please, thanks",
    "When I say focus mode, open Safari or Chrome",
    "Okay, why do you open Safari when I say Neo?",
]


@pytest.mark.parametrize("text", UNPARSED)
def test_anything_else_is_left_to_the_constrained_manager(text):
    assert routines.parse_local_setup(text) is None


# ---- what a routine may open ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "action",
    [
        _url("localhost:8080"),
        _url("http://127.0.0.1/admin"),
        _url("https://192.168.1.1"),
        _url("http://10.0.0.5"),
        _url("http://[::1]/"),
        _url("printer.local"),
        _url("https://user:pass@example.com"),
        _url("file:///etc/passwd"),
        _url("javascript://alert(1)"),
        _url("https://example.com/?token=ghp_abcdefghijklmnopqrstuvwxyz0123456789"),
        _app("Safari/../../bin/sh"),
        _app("https://example.com"),
        {"kind": "shell", "target": "rm -rf ~"},
        {"kind": "open_app", "target": ""},
        {"kind": "open_app", "target": "Notes\x07rm"},
    ],
)
def test_unsafe_destinations_are_refused(action):
    with pytest.raises(ValueError):
        routines._safe_action(action)


def test_a_url_keeps_its_path_and_query_but_not_its_fragment():
    assert routines._safe_action(_url("Example.com/watch?v=1#t=3")) == _url("https://Example.com/watch?v=1")


# ---- create, match, run, delete ----------------------------------------------------------------------


def _ctx(words: str, *, origin: str = "user", taint: list[str] | None = None) -> ToolContext:
    return ToolContext(user_text=words, origin=origin, taint=list(taint or []))  # type: ignore[arg-type]


def _create(trigger: str, actions: list[dict], *, words: str | None = None, **kw):
    words = words or f"When I say {trigger}, open " + " and ".join(a["target"] for a in actions)
    args = {"action": "create", "trigger": trigger, "actions": actions}
    return asyncio.run(routines.routine_manage(args, _ctx(words, **kw)))


def test_create_saves_locally_and_the_exact_phrase_matches():
    out = _create("I want to study", [_app("Notion"), _url("youtube.com")])
    assert out.startswith("Saved locally") and "Notion and https://youtube.com" in out
    row = routines.match("Neo, I want to study!")
    assert row is not None and row["actions"] == [_app("Notion"), _url("https://youtube.com")]
    assert routines.match("I want to study maths") is None  # an exact phrase, never a prefix
    assert routines.match("study") is None
    path = routines._path()
    assert path.exists() and oct(path.stat().st_mode & 0o777) == "0o600"


def test_the_models_call_for_a_definition_said_after_a_filler_is_saved():
    """Live often hears "Okay, …" first: the user still said to define it, so the model may save it."""
    out = _create("focus mode", [_app("Notion")], words="Okay, when I say focus mode, open Notion.")
    assert str(out).startswith("Saved locally") and routines.match("focus mode") is not None


def test_a_model_cannot_create_one_the_user_never_asked_for():
    out = _create("open sesame", [_app("Terminal")], words="what's the weather like?", origin="agent")
    assert str(out).startswith("Error:") and routines.match("open sesame") is None


def test_after_reading_a_page_a_new_routine_waits_for_the_users_yes():
    out = _create("study time", [_app("Notion")], taint=["page_read"])
    assert isinstance(out, ToolOutput) and out.halt and out.text.startswith("NEEDS_CONFIRM")
    assert routines.match("study time") is None  # nothing saved until they approve it


def test_list_names_every_routine():
    assert "don't have any spoken routines" in asyncio.run(routines.routine_manage({"action": "list"}, _ctx("")))
    _create("I want to study", [_app("Notion")])
    listed = asyncio.run(routines.routine_manage({"action": "list"}, _ctx("list my routines")))
    assert "I want to study" in listed and "open Notion" in listed


def _delete(name: str):
    return asyncio.run(routines.routine_manage({"action": "delete", "name": name}, _ctx(f"delete the {name} routine")))


def test_delete_asks_first_and_matches_whole_words(monkeypatch):
    _create("I want to study", [_app("Notion")])
    out = _delete("study")  # "delete the study routine" (the README's example)
    assert isinstance(out, ToolOutput) and out.halt and out.text.startswith("NEEDS_CONFIRM")
    assert routines.match("I want to study") is not None  # still there until they say yes
    assert not str(_delete("stud")).startswith("NEEDS_CONFIRM")  # part of a word is not the routine

    from neo2.core import config

    monkeypatch.setenv("NEO_CONFIRM_DESTRUCTIVE", "false")
    monkeypatch.setattr(config, "_settings", None)
    assert _delete("Study") == "Deleted the “I want to study” routine."
    assert routines.match("I want to study") is None


def test_delete_with_several_matches_asks_which_one():
    _create("I want to study", [_app("Notion")])
    _create("study music", [_app("Music")])
    out = _delete("study")
    assert isinstance(out, str) and not out.startswith("Error") and out.startswith("Which one")
    assert "I want to study" in out and "study music" in out
    assert routines.match("I want to study") and routines.match("study music")


def test_delete_with_no_match_lists_them_without_an_error():
    """Not an "Error": the quick lane would hand that to the cloud agent with every tool."""
    _create("I want to study", [_app("Notion")])
    out = _delete("gym")
    assert isinstance(out, str) and not out.lower().startswith(("error", "failed"))
    assert "couldn't find" in out and "I want to study" in out


@contextmanager
def _fake_launchers():
    calls: list[tuple[str, dict]] = []
    saved = {}
    for name in ("open_app", "open_url"):

        async def handler(args, ctx, _name=name):
            calls.append((_name, dict(args)))
            return ToolOutput("Done.")

        saved[name] = catalog().get(name)
        catalog().register(RegisteredTool(name, name, {"type": "object", "properties": {}}, handler))
    try:
        yield calls
    finally:
        for name, prev in saved.items():
            if prev is not None:
                catalog().register(prev)  # replaced where it stands: never moved to the end
            else:
                catalog().remove(name)


@pytest.fixture
def launchers():
    """Fake open_app / open_url in NEO.app's catalog: what a routine would launch, recorded."""
    with _fake_launchers() as calls:
        yield calls


def test_the_fake_launchers_leave_the_catalog_as_it_was():
    """Another test fits "every tool" into a small budget in the catalog's order: faking the launchers
    for a while must leave both the tools and their order as they were."""
    before = list(catalog()._tools.items())
    with _fake_launchers():
        assert catalog().get("open_app") is not dict(before).get("open_app")
    assert list(catalog()._tools.items()) == before


def test_run_launches_in_order_under_the_running_job(launchers, monkeypatch):
    from neo2.core import events

    monkeypatch.setattr(events, "_bus", events.EventBus())
    seen: list[tuple[str, str]] = []
    events.bus().subscribe(lambda e: seen.append((e.type, e.data.get("job", ""))) if e.type.startswith("tool_") else None)
    _create("I want to study", [_app("Notion"), _url("youtube.com")])
    rid = routines.match("I want to study")["id"]

    async def run():
        events.current_job.set("j42")  # what the session's tool_start set for routine_run's own step
        return await routines.routine_run({"routine_id": rid}, ToolContext(origin="user"))

    out = asyncio.run(run())
    assert out.ok and "Started your" in out.text
    assert launchers == [("open_app", {"target": "Notion"}), ("open_url", {"url": "https://youtube.com"})]
    assert seen and all(job == "j42" for _, job in seen)
    assert routines.match("I want to study")["uses"] == 1


def test_run_never_reads_the_job_from_the_hosts_extras(launchers, monkeypatch):
    from neo2.core import events

    monkeypatch.setattr(events, "_bus", events.EventBus())
    seen: list[str] = []
    events.bus().subscribe(lambda e: seen.append(e.data.get("job", "")) if e.type == "tool_start" else None)
    _create("I want to study", [_app("Notion")])
    rid = routines.match("I want to study")["id"]
    asyncio.run(routines.routine_run({"routine_id": rid}, ToolContext(origin="user", extras={"neo_job": "spoofed"})))
    assert seen == [""]


# ---- the session routes a definition -------------------------------------------------------------


@pytest.fixture
def session_env(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from neo2.core import config, events
    from neo2.core.agent import session as sess_mod
    from neo2.core.tools import load_all

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(events, "_bus", events.EventBus())
    load_all()
    agent_calls: list[dict] = []

    async def fake_agent(self, text, d, images, job, *, only=None, **kw):
        from neo2.core.agent.session import Reply

        agent_calls.append({"text": text, "only": only})
        await events.bus().end_job(job)
        return Reply("agent did it", "agent", d, job=job)

    async def fake_chat(self, text, d, job):
        from neo2.core.agent.session import Reply

        agent_calls.append({"text": text, "chat": True})
        await events.bus().end_job(job)
        return Reply("a chat reply", "chat", d, job=job)

    monkeypatch.setattr(sess_mod.Session, "_agent", fake_agent)
    monkeypatch.setattr(sess_mod.Session, "_chat", fake_chat)
    return SimpleNamespace(calls=agent_calls)


class _Reflex:
    def __init__(self, intent="chat"):
        from neo2.core.reflex.schema import Decision

        self.d = Decision(intent, 0.9, False, 0.0, False, 0.0, "fake")

    async def decide(self, text):
        return self.d


@pytest.mark.parametrize("intent", ["chat", "agent_task"])
def test_the_plain_form_is_saved_without_a_model(session_env, intent):
    from neo2.core.agent.session import Session

    r = asyncio.run(Session(reflex=_Reflex(intent)).handle(
        "When I say I want to study, open Notion, Safari, and youtube.com", from_user=True))
    assert r.route == "quick" and r.tools == ("routine_manage",) and "Saved locally" in r.text
    assert session_env.calls == []
    assert routines.match("i want to study")["actions"][-1] == _url("https://youtube.com")


@pytest.mark.parametrize(
    "text", ["Make a shortcut to Downloads", "Why do you open Safari when I say Neo?", "what happens when I say stop"]
)
def test_a_sentence_that_only_mentions_it_goes_where_laya_says(session_env, text):
    from neo2.core.agent.session import Session

    asyncio.run(Session(reflex=_Reflex("chat")).handle(text, from_user=True))
    assert session_env.calls == [{"text": text, "chat": True}]
    asyncio.run(Session(reflex=_Reflex("agent_task")).handle(text, from_user=True))
    assert session_env.calls[-1].get("only") is None  # every tool, not just the routine manager


def test_an_open_ended_definition_gets_only_the_routine_manager(session_env):
    from neo2.core.agent.session import Session

    text = "When I say good night, turn off the lights"
    asyncio.run(Session(reflex=_Reflex("agent_task")).handle(text, from_user=True))
    assert session_env.calls == [{"text": text, "only": ["routine_manage"]}]


def test_an_open_ended_definition_laya_calls_chat_stays_chat(session_env):
    from neo2.core.agent.session import Session

    text = "When I say good night, turn off the lights"
    asyncio.run(Session(reflex=_Reflex("chat")).handle(text, from_user=True))
    assert session_env.calls == [{"text": text, "chat": True}]


def test_a_live_relay_never_defines_one(session_env):
    from neo2.core.agent.session import Session

    asyncio.run(Session(reflex=_Reflex("agent_task")).handle(
        "When I say I want to study, open Notion", origin="live"))
    assert routines.match("I want to study") is None
    assert session_env.calls and session_env.calls[-1].get("only") != ["routine_manage"]
