"""Routing to song_identify (neo2.core.agent.song_words, fastpath.song): the user's own words "what song
is this" / "ye kaunsa gaana hai" / "यह कौन सा गाना है" make NEO listen; "play this song", a song they
name, a model's words, Live's own idea and library mode never do; "stop" stops it. Fakes only: nothing
listens, no model, no network."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from neo2.core.agent import fastpath, song_words
from neo2.core.agent.registry import USERS_WORDS, RegisteredTool, ToolOutput, catalog

SONG_ASKS = [
    "what song is this",
    "What song is this?",
    "Neo, what song is this?",
    "hey neo, which song is this?",
    "what's this song",
    "whats this song",
    "what is this song called",
    "which song is playing",
    "which song is playing right now",
    "what song is playing",
    "what song's playing",
    "what's the song that's playing",
    "what's the name of this song",
    "name this song",
    "name that tune",
    "shazam this",
    "Shazam this song",
    "identify this song",
    "recognise this song",
    "recognize the song playing",
    "can you identify this song",
    "do you know this song",
    "can you tell me what song this is",
    "who sings this",
    "who is singing this song",
    "whose song is this",
    "what am I listening to",
    "what's playing right now",
    "listen and tell me what song this is",
    "what track is this",
    "what's this tune",
    # Hinglish
    "ye kaunsa gaana hai",
    "yeh konsa gana hai",
    "ye kaun sa song hai",
    "kaunsa gaana chal raha hai",
    "ye gaana kaunsa hai",
    "is gaane ka naam kya hai",
    "ye kiska gaana hai",
    "kaunsa gaana baj raha hai",
    "ye kaunsa gaana hai batao",
    "shazam karo",
    # Hindi
    "यह कौन सा गाना है",
    "ये कौनसा गाना है?",
    "कौन सा गाना बज रहा है",
    "इस गाने का नाम क्या है",
    "यह किसका गाना है",
]

NOT_SONG = [
    "play this song",
    "play a song",
    "next song",
    "what songs do you know",
    "sing a song",
    "who sings Blinding Lights",  # a song they name: a web question
    "who sang bohemian rhapsody",
    "who is the singer of despacito",
    "play despacito",
    "what song should I listen to",
    "what's the best song ever",
    "what is the song of the year",
    "add this song to my playlist",
    "like this song",
    "pause the song",
    "skip this song",
    "what's this song about",
    "search for this song on youtube",
    "gaana bajao",
    "koi gaana sunao",
    "गाना बजाओ",
    "turn the music on",
    "play some music",
    "what is music",
    "remind me to ask what song this is",
    "what am I holding",  # the camera's
    "what is this",
    "what is on my screen",
]


@pytest.mark.parametrize("text", SONG_ASKS)
def test_song_asks_are_recognised(text):
    assert song_words.wants_song(text) and song_words.asks_about_song(text)


@pytest.mark.parametrize("text", NOT_SONG)
def test_other_requests_are_not(text):
    assert not song_words.wants_song(text)


@pytest.mark.parametrize("text", SONG_ASKS)
def test_no_song_ask_turns_the_camera_on(text):
    from neo2.core.agent import camera_words

    assert not camera_words.wants_camera(text)


def test_a_models_call_needs_a_sentence_about_the_song_playing():
    for said in (
        "what's the song playing in this video",
        "hey what's that music",
        "what song is in the background",
    ):
        assert song_words.asks_about_song(said), said
    for said in (
        "who sings Blinding Lights",
        "what songs do you know",
        "play the song despacito",
        "what song should I listen to",
    ):
        assert not song_words.asks_about_song(said), said


def test_stop_listening_words():
    for said in ("stop listening", "Neo, stop listening", "sunna band karo", "सुनना बंद करो", "don't listen"):
        assert song_words.wants_stop(said), said
    for said in ("stop the music", "listen to me", "what song is this"):
        assert not song_words.wants_stop(said), said


# ---- fastpath.song: who may make NEO listen ----------------------------------------------------------


class _Calls:
    def __init__(self, text="That's Blinding Lights by The Weeknd, from After Hours, 2020.", ok=True):
        self.calls: list[tuple[dict, object]] = []
        self.text, self.ok = text, ok

    async def __call__(self, args, ctx):
        self.calls.append((dict(args), ctx))
        return ToolOutput(self.text, ok=self.ok)


@pytest.fixture
def song_tool():
    """A fake song_identify in NEO.app's catalog (the real one is neo2.core.tools.music)."""
    handler = _Calls()
    prev = catalog().get(fastpath.SONG_TOOL)
    catalog().register(
        RegisteredTool(
            fastpath.SONG_TOOL,
            "Listen and name the song playing.",
            {"type": "object", "properties": {}},
            handler,
            category="music",
            slow=True,
            untrusted=True,
        )
    )
    yield handler
    catalog().remove(fastpath.SONG_TOOL)
    if prev is not None:
        catalog().register(prev)


def test_song_step_for_the_users_own_words(song_tool):
    assert fastpath.song("what song is this?", origin="user") == ("song_identify", {})
    assert fastpath.song("यह कौन सा गाना है", origin="user") == ("song_identify", {})
    assert fastpath.song("play this song", origin="user") is None


@pytest.mark.parametrize("origin", ["live", "agent", "approval"])
def test_never_for_a_models_words(song_tool, origin):
    assert fastpath.song("what song is this", origin=origin) is None


def test_no_song_tool_no_route():
    prev = catalog().remove(fastpath.SONG_TOOL)
    try:
        assert fastpath.song("what song is this", origin="user") is None
    finally:
        if prev is not None:
            catalog().register(prev)


def test_library_mode_never_routes_to_listening(song_tool, monkeypatch):
    from neo2.core import runtime

    monkeypatch.setattr(runtime, "library_mode", lambda: True)
    monkeypatch.setattr(runtime, "library_bound", lambda: False)
    assert fastpath.song("what song is this", origin="user") is None


def test_the_regex_fast_paths_never_take_a_song_ask(song_tool):
    from neo2.core.tools import load_all

    load_all()
    for text in ("what song is this", "ye kaunsa gaana hai", "shazam this", "which song is playing"):
        steps = fastpath.plan(text) or []
        assert all(t != "song_identify" for t, _ in steps)


# ---- the session routes it -------------------------------------------------------------------------


@pytest.fixture
def session_env(tmp_path, monkeypatch):
    from neo2.core import config, events
    from neo2.core.agent import session as sess_mod
    from neo2.core.tools import load_all

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(events, "_bus", events.EventBus())
    load_all()

    class Chain:
        name = "fake"
        supports_vision = False
        supports_tools = True
        streamed = completed = 0

        async def stream(self, messages, *, system="", max_tokens=2048):
            Chain.streamed += 1
            yield "a chat reply"

        async def complete(self, messages, *, system="", tools=None, effort="medium", max_tokens=4096):
            from neo2.core.providers.base import Turn

            Chain.completed += 1
            return Turn(text="agent did it")

    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": Chain())
    return SimpleNamespace(chain=Chain)


class _Reflex:
    def __init__(self, intent="chat"):
        from neo2.core.reflex.schema import Decision

        self.d = Decision(intent, 0.9, False, 0.0, False, 0.0, "fake")

    async def decide(self, text):
        return self.d


@pytest.mark.parametrize("intent", ["chat", "quick_action", "agent_task"])
def test_session_listens_for_the_users_question_whatever_laya_says(song_tool, session_env, intent):
    from neo2.core.agent.session import Session

    s = Session(reflex=_Reflex(intent))
    r = asyncio.run(s.handle("Neo, what song is this?", from_user=True))
    assert r.route == "quick" and r.tools == ("song_identify",)
    assert r.text.startswith("That's Blinding Lights")
    ((args, ctx),) = song_tool.calls
    assert args == {} and ctx.origin == "user" and ctx.get(USERS_WORDS)
    assert session_env.chain.streamed == 0 and session_env.chain.completed == 0


def test_session_hinglish_and_hindi(song_tool, session_env):
    from neo2.core.agent.session import Session

    s = Session(reflex=_Reflex("chat"))
    for said in ("ye kaunsa gaana hai", "यह कौन सा गाना है?"):
        r = asyncio.run(s.handle(said, from_user=True))
        assert r.tools == ("song_identify",), said
    assert len(song_tool.calls) == 2


def test_session_named_song_and_play_never_listen(song_tool, session_env):
    from neo2.core.agent.session import Session

    s = Session(reflex=_Reflex("chat"))
    for said in ("who sings Blinding Lights", "what songs do you know"):
        asyncio.run(s.handle(said, from_user=True))
    assert song_tool.calls == []


def test_session_live_goal_never_listens(song_tool, session_env):
    from neo2.core.agent.session import Session

    s = Session(reflex=_Reflex("chat"))
    asyncio.run(s.handle("what song is this", origin="live"))  # a model's goal, not the user's words
    assert song_tool.calls == []


def test_session_failure_is_said_not_handed_to_a_model(song_tool, session_env):
    from neo2.core.agent.session import Session

    song_tool.text, song_tool.ok = "I couldn't reach Shazam to recognise it.", False
    s = Session(reflex=_Reflex("agent_task"))
    r = asyncio.run(s.handle("what song is this", from_user=True))
    assert r.route == "quick" and "Shazam" in r.text
    assert session_env.chain.completed == 0  # no brain guessing a song it never heard


class _Listening:
    """The song card, listening (neo2.core.tools.music.live)."""

    def __init__(self, monkeypatch):
        from neo2.core.tools.music import live

        live.reset()
        self.live = live.song_live
        self.stops: list[str] = []
        monkeypatch.setattr(self.live, "sid", "s1")
        monkeypatch.setattr(self.live, "status", "listening")
        monkeypatch.setattr(self.live, "channel", None)

        def stop(reason="stopped"):
            self.stops.append(reason)
            self.live.sid, self.live.status = "", ""
            return True

        monkeypatch.setattr(self.live, "stop", stop)


def test_session_stop_listening_and_cancel_all_stop_it(song_tool, session_env, monkeypatch):
    from neo2.core.agent.session import Session

    card = _Listening(monkeypatch)
    s = Session(reflex=_Reflex("chat"))
    r = asyncio.run(s.handle("stop listening", from_user=True))
    assert r.route == "stop" and card.stops == ["stopped"] and "stopped listening" in r.text
    card.live.sid, card.live.status = "s2", "listening"
    r = asyncio.run(s.handle("stop", from_user=True))  # a bare stop: everything, listening too
    assert r.route == "stop" and card.stops == ["stopped", "stopped"]
    card.live.sid, card.live.status = "s3", "listening"
    asyncio.run(s.cancel_all())
    assert card.stops == ["stopped"] * 3


def test_stop_listening_when_nothing_listens_is_no_song_stop(song_tool, session_env, monkeypatch):
    from neo2.core.agent.session import Session

    card = _Listening(monkeypatch)
    card.live.sid, card.live.status = "", ""
    s = Session(reflex=_Reflex("chat"))
    r = asyncio.run(s.handle("stop listening", from_user=True))
    assert card.stops == [] and "stopped listening" not in r.text  # (whatever it means then, not this)


def test_the_stop_button_stops_listening(monkeypatch):
    from neo2.core.server import ws

    card = _Listening(monkeypatch)
    srv = ws.UIServer(on_command=lambda cmd: asyncio.sleep(0))

    class Conn:
        def __init__(self, msgs):
            self.msgs = msgs

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            import json

            for m in self.msgs:
                yield json.dumps(m)

        async def send(self, data):
            pass

    async def run():
        await srv._handle(Conn([{"cmd": "stop"}]))
        await asyncio.sleep(0)

    asyncio.run(run())
    assert card.stops == ["stopped"]


# ---- prompts, and Gemini Live ---------------------------------------------------------------------------


def test_agent_prompt_knows_the_tool_only_when_it_is_there():
    from neo2.core.agent import prompt

    assert "song_identify" in prompt.persona(["song_identify", "screenshot"])
    assert "song_identify" not in prompt.persona(["screenshot"])
    assert "song_identify" not in prompt.PERSONA  # a library's persona never has it


def test_neo_apps_prompt_has_the_song_line_only_while_the_tool_is_there(song_tool):
    from neo2.core.agent import prompt

    assert "song_identify" in prompt.persona()
    catalog().remove(fastpath.SONG_TOOL)
    assert "song_identify" not in prompt.persona()


def test_live_declares_and_explains_song_identify(song_tool):
    from neo2.core.voice import live as live_mod

    assert "song_identify" in live_mod._direct_tools()
    assert "song_identify" in [d.name for d in live_mod._declarations()]
    extra = live_mod._live_extra()
    assert "song_identify" in extra and "Say nothing until its answer comes" in extra


def test_live_extra_without_the_tool_never_mentions_it():
    from neo2.core.voice import live as live_mod

    prev = catalog().remove(fastpath.SONG_TOOL)
    try:
        assert "song_identify" not in live_mod._live_extra()
        assert "song_identify" not in [d.name for d in live_mod._declarations()]
    finally:
        if prev is not None:
            catalog().register(prev)


def _voice():
    from neo2.core.voice import live as live_mod

    v = live_mod.LiveVoice.__new__(live_mod.LiveVoice)
    v._taint, v._seen_calls, v._recent_idem = [], {}, {}
    v._own_utt = -1
    return v


def test_live_call_runs_only_when_the_users_words_asked(song_tool, monkeypatch):
    from neo2.core import events

    monkeypatch.setattr(events, "_bus", events.EventBus())
    v = _voice()
    fc = SimpleNamespace(name="song_identify", args={}, id="c1")
    # the model's own idea: the user was talking about something else
    for asked in ("summarize this page", "", "who sings Blinding Lights", "play this song"):
        out = asyncio.run(v._song_call(fc, {}, asked, "c1", 1))
        assert out.startswith("NEEDS_USER") and song_tool.calls == [], asked
    # the user asked: it runs as theirs, its answer fenced (Shazam's words are data)
    fc2 = SimpleNamespace(name="song_identify", args={}, id="c2")
    out = asyncio.run(v._song_call(fc2, {}, "ye kaunsa gaana hai", "c2", 2))
    assert "Blinding Lights" in out and out.startswith("<untrusted")
    ((args, ctx),) = song_tool.calls
    assert args == {} and ctx.origin == "user" and ctx.get(USERS_WORDS) == "ye kaunsa gaana hai"
    assert v._own_utt == 2
    # the same sentence again (a retry after the user's stop), or once a page was read: never
    fc3 = SimpleNamespace(name="song_identify", args={}, id="c3")
    assert asyncio.run(v._song_call(fc3, {}, "ye kaunsa gaana hai", "c3", 2)).startswith("NEEDS_USER")
    v._taint = ["web_fetch"]
    assert asyncio.run(v._song_call(fc3, {}, "what song is this", "c3", 3)).startswith("NEEDS_USER")
    assert len(song_tool.calls) == 1


def test_live_sends_no_audio_up_while_neo_listens_and_stop_stops_it(monkeypatch):
    from neo2.core.voice import live as live_mod

    card = _Listening(monkeypatch)
    monkeypatch.setattr(live_mod, "_song_listening", lambda: bool(card.live.sid))
    frames = [b"\0" * 640, b"\0" * 640, b""]
    hits = iter([None, "stop_final"])
    sent: list[bytes] = []

    class Mic:
        async def read(self):
            return frames.pop(0)

    class Wake:
        def feed(self, frame):
            return next(hits, None)

    class Session:
        async def send_realtime_input(self, audio):
            sent.append(audio)

    v = live_mod.LiveVoice.__new__(live_mod.LiveVoice)
    v.mic, v._wake, v._live = Mic(), Wake(), Session()
    asyncio.run(v._mic_loop())
    assert sent == []  # the room's music never went to the model as the user's words
    assert card.stops == ["stopped"]  # a committed "stop" stopped the listening


def test_live_typed_stop_stops_listening(monkeypatch):
    from neo2.core.voice import live as live_mod

    card = _Listening(monkeypatch)
    assert live_mod._song_stop_if_asked("stop listening") and card.stops == ["stopped"]
    card.live.sid, card.live.status = "s2", "listening"
    assert not live_mod._song_stop_if_asked("what song is this") and card.stops == ["stopped"]
    assert live_mod._song_stop_if_asked("stop")
