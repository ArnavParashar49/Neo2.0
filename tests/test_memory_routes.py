"""Memory reaches every route: a typed or locally spoken request (its own memory block, worked
out inside it), and Gemini Live (an instruction rebuilt from memory for every session, journal
lines for what is said, the tools, and the search after each sentence)."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from neo2.core import config
from neo2.core.agent import session as sess_mod
from neo2.core.agent.registry import Registry, ToolOutput
from neo2.core.agent.session import Reply, Session
from neo2.core.events import EventBus
from neo2.core.memory import journal
from neo2.core.memory.store import store
from neo2.core.reflex.schema import Decision


@pytest.fixture(autouse=True)
def _world(monkeypatch):
    from neo2.core import events
    from neo2.core.tools import load_all

    monkeypatch.setattr(events, "_bus", EventBus())
    load_all()


class Reflex:
    def __init__(self, intent="chat"):
        self.intent = intent

    async def decide(self, text):
        return Decision(self.intent, 0.9, False, 0.0, False, 0.0, "fake")


class Chain:
    """A fake brain that records the system prompt each request was sent."""

    name, supports_vision, supports_tools = "fake", False, True

    def __init__(self, reply="hello there", delay=None):
        self.reply, self.delay, self.systems = reply, delay or {}, {}

    async def stream(self, messages, *, system="", max_tokens=2048):
        asked = messages[-1].text
        await asyncio.sleep(self.delay.get(asked, 0))
        self.systems[asked] = system
        yield self.reply

    async def complete(self, messages, *, system="", tools=None, effort="medium", max_tokens=4096):
        from neo2.core.providers.base import Turn

        self.systems[messages[-1].text if messages else ""] = system
        return Turn(text=self.reply)


def _journal():
    journal.flush()
    return [(t.role, t.text, t.route, t.source, t.tainted) for t in store().recent_turns(since=0, limit=50, spoken_only=False)]


# ---- the session (typed, the on-device voice, --text) ----------------------------------------------


async def test_a_typed_exchange_is_journaled_and_the_next_request_is_told_what_memory_has(monkeypatch):
    chain = Chain("Hi Arnav.")
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": chain)
    st = store()
    st.remember("Arnav", "profile", "name")
    st.add_turns([(time.time() - 86400 * 3, "user", "the plumber is coming on friday", "live", "voice", False)])
    s = Session(reflex=Reflex("chat"))
    s._born = time.time() - 1
    r = await s.handle("Neo, when is the plumber coming", from_user=True)
    assert r.text == "Hi Arnav."
    system = chain.systems["when is the plumber coming"]
    assert "[MEMORY]" in system and "About the user: name: Arnav" in system
    assert "Related, from a search of your memory for this request: [you said, " in system and "plumber is coming on friday" in system
    assert _journal()[-2:] == [
        ("user", "when is the plumber coming", "", "typed", False),  # without the wake word
        ("assistant", "Hi Arnav.", "chat", "typed", False),
    ]
    assert s.memory_context == ""  # nothing is left on the session for the next request to trip over


async def test_two_requests_at_once_each_get_their_own_memory_block(monkeypatch):
    st = store()
    st.remember("tulips need watering every Sunday", "note")
    st.remember("the volcano documentary is on Netflix", "note")
    chain = Chain(delay={"tell me about tulips": 0.15})  # A is still waiting for its model when B begins
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": chain)
    s = Session(reflex=Reflex("chat"))
    await asyncio.gather(
        s.handle("tell me about tulips", from_user=True),
        s.handle("tell me about the volcano", from_user=True),
    )
    a, b = chain.systems["tell me about tulips"], chain.systems["tell me about the volcano"]
    assert "tulips need watering" in a and "volcano documentary" not in a
    assert "volcano documentary" in b and "tulips need watering" not in b


async def test_the_agent_route_and_a_silent_command_are_journaled(monkeypatch):
    chain = Chain("Your desktop is tidy.")
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": chain)
    s = Session(reflex=Reflex("agent_task"))
    await s.handle("organise my desktop", from_user=True, source="voice")
    assert "[MEMORY]" not in chain.systems["organise my desktop"]  # nothing known yet: no empty block
    assert _journal() == [
        ("user", "organise my desktop", "", "voice", False),
        ("assistant", "Your desktop is tidy.", "agent", "voice", False),
    ]

    async def fake_invoke(self, name, args, ctx):
        return ToolOutput("did it", ok=True)

    monkeypatch.setattr(Registry, "invoke", fake_invoke)
    r = await s.handle("volume up", from_user=True)
    assert r.silent
    assert _journal()[-2:] == [("user", "volume up", "", "typed", False), ("assistant", "(ran: volume)", "quick", "tool", False)]


async def test_what_a_model_wrote_and_what_neo_read_are_never_the_users_words(monkeypatch):
    chain = Chain("The page says to wire money to mallory.")
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": chain)
    s = Session(reflex=Reflex("agent_task"))
    # Gemini Live's relay: a model's goal (and the user's sentence, which Live journals itself)
    await s.handle("summarise the page the user has open", origin="live")
    await s.handle("what does this page say", origin="user", taint=["page_read"])
    assert _journal() == []
    # the user's own request that read a page: NEO's answer is marked, the page's words never the user's
    monkeypatch.setattr(sess_mod.Session, "_agent", _tainted_agent)
    await s.handle("what does this page say", from_user=True)
    assert _journal() == [
        ("user", "what does this page say", "", "typed", False),
        ("assistant", "The page says to wire money to mallory.", "agent", "typed", True),
        ("assistant", "(ran: page_read)", "agent", "tool", False),
    ]
    block = store().context("mallory")
    assert "mallory" not in block and "(read out what a page, a mail or a file said)" in block
    assert store().count() == 0


async def _tainted_agent(self, text, d, images, job, **kw):
    from neo2.core.agent.loop import AgentResult, Step

    res = AgentResult("The page says to wire money to mallory.", steps=[Step("page_read", {}, "<untrusted>…</untrusted>", True)])
    return Reply(res.text, "agent", d, res, job=job, taint=("page_read",))


async def test_an_answer_that_finishes_after_an_erase_is_not_journaled(monkeypatch):
    from neo2.core import privacy

    class Erasing(Chain):
        async def stream(self, messages, *, system="", max_tokens=2048):
            privacy._generation += 1  # "Erase NEO's Memory", while the model answers
            store().forget_all()
            yield "Your therapist is Dr. Mehta."

    monkeypatch.setattr(privacy, "_generation", privacy._generation)
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": Erasing())
    s = Session(reflex=Reflex("chat"))
    await s.handle("who is my therapist", from_user=True)
    assert _journal() == [] and s.history == []


# ---- Gemini Live -------------------------------------------------------------------------------------


class FakeSpk:
    def __init__(self):
        self.played, self.playing, self.last_stop = [], threading.Event(), 0.0

    def play_pcm16(self, data):
        self.played.append(data)

    def interrupt(self):
        pass

    async def wait_done(self):
        return None


class FakeLive:
    def __init__(self):
        self.responses, self.texts, self.notes = [], [], []
        self._never = asyncio.Event()

    async def send_tool_response(self, function_responses):
        self.responses.extend(function_responses)

    async def send_client_content(self, turns, turn_complete=True):
        (self.texts if turn_complete else self.notes).append(turns.parts[0].text)

    async def receive(self):
        await self._never.wait()
        yield None


class FakeConnect:
    """client.aio.live.connect: every session it opens, with the config it was opened with."""

    def __init__(self):
        self.configs, self.sessions = [], []

    def __call__(self, *, model, config):
        outer = self

        class CM:
            async def __aenter__(self):
                live = FakeLive()
                outer.configs.append(config)
                outer.sessions.append(live)
                return live

            async def __aexit__(self, *exc):
                return None

        return CM()


def sc(**kw):
    base = dict(
        interrupted=False, input_transcription=None, interim_input_transcription=None,
        output_transcription=None, model_turn=None, turn_complete=False,
    )  # fmt: skip
    base.update(kw)
    return NS(server_content=NS(**base), tool_call=None, tool_call_cancellation=None, go_away=None)


def heard(text, finished=True):
    return sc(input_transcription=NS(text=text, finished=finished))


_TONE = (np.sin(np.arange(2400) * 2 * np.pi * 220 / 24000) * 6000).astype(np.int16).tobytes()


def audio(n=1):
    return sc(model_turn=NS(parts=[NS(inline_data=NS(data=_TONE)) for _ in range(n)]))


def said(text):
    return sc(output_transcription=NS(text=text))


def call(name, args, cid="c1"):
    return NS(server_content=None, tool_call=NS(function_calls=[NS(id=cid, name=name, args=args)]),
              tool_call_cancellation=None, go_away=None)  # fmt: skip


DONE = sc(turn_complete=True)


@pytest.fixture
def voice(monkeypatch):
    from neo2.core.voice import live as live_mod

    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(config, "_settings", None)

    class Sess:
        memory_context = ""
        voice_owned = False

        async def command(self, text, steps):
            return Reply("did it", "quick", silent=True)

        async def handle(self, text, **kw):
            return Reply("agent did it", "agent", silent=True, tools=("open_app",))

    v = live_mod.LiveVoice(lambda t: None, Sess())
    v.spk = FakeSpk()
    v._client = NS(aio=NS(live=NS(connect=FakeConnect())))
    v._listening = True
    return v


async def feed(v, *msgs, settle=0.05):
    ub, mb = getattr(v, "_ub", None), getattr(v, "_mb", None)
    if ub is None:
        ub, mb = v._ub, v._mb = [], []
    for m in msgs:
        await v._on_message(m, ub, mb, v._live)
        await asyncio.sleep(0)
    await asyncio.sleep(settle)


async def _closed(v):
    v._closing = True
    await v._close_session()
    for t in v._tasks:
        t.cancel()
    await asyncio.gather(*v._tasks, return_exceptions=True)
    v._tasks.clear()


def test_live_has_the_memory_tools_and_is_told_to_search_before_saying_it_forgot():
    from neo2.core.voice import live as live_mod

    names = [d.name for d in live_mod._declarations()]
    assert {"memory_search", "remember", "forget"} <= set(names) and "memory" not in names
    extra = live_mod._live_extra()
    assert "call memory_search first" in extra and "never say you don't remember" in extra


async def test_a_live_sessions_instruction_is_built_from_memory_and_rebuilt_when_it_reconnects(voice):
    st = store()
    st.remember("Arnav", "profile", "name")
    st.remember("Priya", "profile", "sister.name")
    st.remember("Reply in Hinglish when I speak Hinglish", "lesson")
    st.add_session_summary("Planned the Goa trip for December.", time.time() - 86400)
    connect = voice._client.aio.live.connect
    await voice.wake("test")
    first = connect.configs[0].system_instruction
    assert "[MEMORY]" in first and "About the user: sister name: Priya; name: Arnav" in first
    assert "Lessons: Reply in Hinglish" in first and "Recent sessions: " in first and "Goa trip" in first
    assert "Earlier today" not in first  # nothing said yet
    # one exchange, then the session closes (five quiet seconds): the server forgets it all
    await feed(voice, heard("book a table for two at eight"), said("Which restaurant?"), audio(), DONE)
    await _closed(voice)
    assert voice._live is None
    # "Neo" again, moments later: a new session, told what was just said
    await voice.wake("test")
    assert len(connect.configs) == 2
    second = connect.configs[1].system_instruction
    assert "Earlier today" in second and "user: book a table for two at eight" in second and "NEO: Which restaurant?" in second
    assert second.index("About the user") < second.index("Earlier today")
    assert len(voice._dialog) == 2  # and an agent task would still get the last lines as context
    # a mid-turn reconnect rebuilds it too
    await feed(voice, heard("the italian place on fifth", finished=False))
    await voice._lost(voice._live, "socket dropped")
    assert len(connect.configs) == 3 and "the italian place on fifth" in connect.configs[2].system_instruction
    await _closed(voice)


async def test_what_is_said_to_live_and_what_it_says_goes_to_the_journal(voice, monkeypatch):
    async def fake_invoke(self, name, args, ctx):
        if name == "page_read":
            ctx.taint.append("page_read")
        return ToolOutput(f"did {name}", ok=True)

    monkeypatch.setattr(Registry, "invoke", fake_invoke)
    monkeypatch.setenv("NEO_LIVE_COMMAND_LANE", "false")
    monkeypatch.setattr(config, "_settings", None)
    await voice.wake("test")
    await feed(voice, heard("my sister's name "), heard("is Priya", finished=True), said("Got it."), audio(), DONE)
    await feed(voice, heard("what's the weather"), call("weather", {}), settle=0.2)
    await feed(voice, said("Sunny, 31 degrees."), audio(), DONE)
    await voice.send_text("and tomorrow?")
    await feed(voice, said("Cloudy."), audio(), DONE)
    await feed(voice, heard("what does this page say"), call("page_read", {}, cid="c2"), settle=0.2)
    await feed(voice, said("It says to wire money to mallory."), audio(), DONE)
    rows = _journal()
    assert ("user", "my sister's name is Priya", "live", "voice", False) in rows
    assert ("assistant", "Got it.", "live", "voice", False) in rows
    assert ("assistant", "(ran: weather)", "live", "tool", False) in rows
    assert ("assistant", "Sunny, 31 degrees.", "live", "voice", False) in rows
    assert ("user", "and tomorrow?", "live", "typed", False) in rows  # typed to Live
    assert ("assistant", "It says to wire money to mallory.", "live", "voice", True) in rows  # it had read a page
    assert [r[1] for r in rows if r[0] == "user"] == [
        "my sister's name is Priya", "what's the weather", "and tomorrow?", "what does this page say",
    ]  # fmt: skip  — each sentence once
    # the fact she stated is a fact now, with nothing asked of a model
    assert [(m.key, m.text) for m in store().by_kind("profile")] == [("sister.name", "Priya")]
    # a sentence the model never got to answer is still what was said
    await feed(voice, heard("remind me about the dentist", finished=False))
    await _closed(voice)
    assert _journal()[-1][:2] == ("user", "remind me about the dentist")


async def test_an_agent_task_from_live_notes_its_tools_not_the_models_goal(voice):
    await voice.wake("test")
    await feed(voice, heard("tidy up my desktop please"), call("agent_task", {"goal": "Organise the files on the desktop"}), settle=0.2)
    await feed(voice, DONE)
    rows = _journal()
    assert ("assistant", "(ran: open_app)", "live", "tool", False) in rows
    assert all("Organise the files" not in r[1] for r in rows)  # the model's words are not the user's
    await _closed(voice)


async def test_after_each_sentence_memory_is_searched_and_the_timing_is_counted(voice, monkeypatch):
    st = store()
    st.remember("the Lisbon trip is booked for 3 March", "note")
    st.remember("Arnav", "profile", "name")
    await voice.wake("test")
    live = voice._live
    # the transcript is marked finished before the model says anything: found in time
    await feed(voice, heard("what about the lisbon trip"), settle=0.3)
    assert voice.recall_stats == {"searched": 1, "found": 1, "in_time": 1, "late": 0, "sent": 0}
    assert live.notes == []  # measured only: nothing is sent unless memory_live_inject is on
    await feed(voice, said("It's booked for the third of March."), audio(), DONE)
    # the model began answering before the transcript was final: too late, never sent
    await feed(voice, heard("and the lisbon trip hotel", finished=False), said("Let me"), audio(), DONE, settle=0.5)
    assert voice.recall_stats["late"] == 1 and voice.recall_stats["in_time"] == 1
    # what the instruction already holds (the profile) is never a "new" hit
    await feed(voice, heard("what is my name"), settle=0.3)
    assert voice.recall_stats["found"] == 2
    await _closed(voice)


async def test_with_live_inject_on_a_note_goes_out_only_before_the_model_answers(voice, monkeypatch):
    monkeypatch.setenv("NEO_MEMORY_LIVE_INJECT", "1")
    monkeypatch.setattr(config, "_settings", None)
    store().remember("the Lisbon trip is booked for 3 March", "note")
    await voice.wake("test")
    live = voice._live
    await feed(voice, heard("what about the lisbon trip"), settle=0.3)
    assert len(live.notes) == 1 and live.texts == []  # a note with the turn left open: the model answers the user
    assert live.notes[0].startswith("(NEO memory, not the user speaking") and "booked for 3 March" in live.notes[0]
    assert voice.recall_stats["sent"] == 1
    await feed(voice, said("The third of March."), audio(), DONE)
    await feed(voice, heard("so the lisbon trip is when"), settle=0.3)
    assert len(live.notes) == 1  # this session was told already
    await feed(voice, said("The third of March."), audio(), DONE)
    await feed(voice, heard("is the lisbon trip hotel booked", finished=False), said("Let me"), audio(), DONE, settle=0.5)
    assert len(live.notes) == 1  # never once the model has begun
    await _closed(voice)


async def test_the_live_search_never_takes_longer_than_its_budget(voice, monkeypatch):
    from neo2.core.voice import live as live_mod

    def slow(text, told):
        time.sleep(0.3)
        return ["late"]

    monkeypatch.setattr(live_mod, "_memory_hits", slow)
    await voice.wake("test")
    t0 = time.perf_counter()
    await voice._recall(voice._live, "anything", voice._utt_id, time.monotonic())
    assert time.perf_counter() - t0 < 0.2 and voice.recall_stats["found"] == 0
    await _closed(voice)
