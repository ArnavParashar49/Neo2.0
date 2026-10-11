"""Retry words ("try again", "phir se try karo") and where they go (neo2.core.agent.retry_words): to the
last retryable thing NEO did, a camera look or a song it couldn't name, and only while that lane's
window is open. A retry never starts the camera or the mic on its own: an unrelated request in
between, or the words that belong to the other lane, leave it to Laya. Fakes only: nothing looks or
listens, no model, no network."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from neo2.core.agent import camera_words, fastpath, retry_words, song_words
from neo2.core.agent.registry import RegisteredTool, ToolOutput, catalog
from neo2.core.agent.retry_words import is_retry_request


@pytest.fixture(autouse=True)
def _no_open_windows():
    """The lanes' windows are process-wide: none carries from one test into the next."""
    camera_words.forget()
    song_words.clear_retry()
    retry_words.forget()
    yield
    camera_words.forget()
    song_words.clear_retry()
    retry_words.forget()


BOTH = [
    "try again",
    "Try it again.",
    "neo, try that again please",
    "can you try again",
    "please try once more",
    "retry",
    "retry that",
    "one more time",
    "once more",
    "once again",
    "Once again, Neo!",
    "give it another try",
    "phir se",
    "phir se try karo",
    "fir se try kar do",
    "dobara",
    "dobara karo",
    "arre neo dobara",
    "arey neo, phir se try karo",
    "ek baar aur try karo",
    "ek baar phir se karke dekho",
    "फिर से",
    "दोबारा कोशिश करो",
    "एक बार और ट्राई करो",
]

CAMERA_ONLY = [
    "try",
    "again",
    "do it again",
    "redo it",
    "repeat that",
    "look again",
    "look at it again",
    "take another look",
    "phir se dekho",
    "dobara dekh lo",
    "फिर से देखो",
]

SONG_ONLY = [
    "listen again",
    "listen to it again",
    "listen once more",
    "phir se suno",
    "ek baar aur sun lo",
    "दोबारा सुनो",
]

NEITHER = [
    "repeat",
    "try harder",
    "try the other one",
    "again and again",
    "what did you say again",
    "say that again",
    "do it",
    "play it again",
    "open it again",
    "try again later",
    "phir milte hain",
    "dobara mat karna",
    "",
    "neo",
]


@pytest.mark.parametrize("text", BOTH)
def test_retries_for_either_lane(text):
    assert is_retry_request(text, "camera") and is_retry_request(text, "song")


@pytest.mark.parametrize("text", CAMERA_ONLY)
def test_looking_again_is_the_cameras_only(text):
    assert is_retry_request(text, "camera") and not is_retry_request(text, "song")


@pytest.mark.parametrize("text", SONG_ONLY)
def test_listening_again_is_the_songs_only(text):
    assert is_retry_request(text, "song") and not is_retry_request(text, "camera")


@pytest.mark.parametrize("text", NEITHER)
def test_other_words_are_no_retry(text):
    assert not is_retry_request(text, "camera") and not is_retry_request(text, "song")


def test_one_address_for_every_lane():
    for module in (camera_words, song_words):
        assert module._ADDRESS is retry_words.ADDRESS
    assert retry_words.ADDRESS.match("arre neo ") and retry_words.ADDRESS.match("arey neo, ")


# ---- the windows ---------------------------------------------------------------------------------


def test_a_song_retry_needs_a_fresh_miss():
    assert not song_words.retry_after_miss("try again", now=100.0)
    song_words.mark_no_match(now=100.0)
    assert song_words.retry_after_miss("try again", now=100.0 + song_words.RETRY_WINDOW_S - 1)
    assert not song_words.retry_after_miss("try again", now=100.0 + song_words.RETRY_WINDOW_S + 1)
    assert song_words.RETRY_WINDOW_S <= 20.0
    assert song_words.consume_retry("try again", now=101.0)
    assert not song_words.retry_after_miss("try again", now=101.0)  # one retry per miss


def test_a_camera_retry_needs_the_cameras_window(monkeypatch):
    monkeypatch.setattr(camera_words.time, "monotonic", lambda: 12.0)
    assert not camera_words.follow_up("try again")
    camera_words.mark_look(10.0, question="what am I holding")
    assert camera_words.follow_up("try again")
    assert camera_words.follow_up_question("phir se dekho") == "what am I holding"
    assert not camera_words.follow_up("listen again")  # (the song's words, not a look)
    assert not camera_words.follow_up("try again", now=10.0 + camera_words._window() + 1)


def test_the_last_lane_takes_a_neutral_retry(monkeypatch):
    now = 1000.0
    monkeypatch.setattr(camera_words.time, "monotonic", lambda: now)
    monkeypatch.setattr(song_words.time, "monotonic", lambda: now)
    camera_words.mark_look(question="what am I holding")
    song_words.mark_no_match()  # the song missed after the look: it is the last thing NEO did
    assert song_words.retry_after_miss("try again") and not camera_words.follow_up("try again")
    camera_words.mark_look(question="what colour is this")  # …and now a look after the miss
    assert camera_words.follow_up("try again") and not song_words.retry_after_miss("try again")
    assert camera_words.follow_up("and now?")  # (a camera follow-up that isn't a retry: still the camera's)


def test_another_request_ends_both_retries(monkeypatch):
    monkeypatch.setattr(camera_words.time, "monotonic", lambda: 1000.0)
    camera_words.mark_look(question="what am I holding")
    song_words.mark_no_match(now=1000.0)
    retry_words.mark("other")
    assert not camera_words.follow_up("try again") and not song_words.retry_after_miss("try again", now=1000.0)


def test_a_new_listen_is_the_last_lane():
    camera_words.mark_look(question="what am I holding")
    song_words.begin_listen()
    assert retry_words.last_lane() == "song"
    assert not camera_words.follow_up("try again")


# ---- the session routes them -----------------------------------------------------------------------


class _Calls:
    def __init__(self, text):
        self.calls: list[tuple[dict, object]] = []
        self.text = text

    async def __call__(self, args, ctx):
        self.calls.append((dict(args), ctx))
        return ToolOutput(self.text)


@pytest.fixture
def lanes(monkeypatch):
    """Fake camera_look and song_identify in NEO.app's catalog, and a session that never reaches a
    model (its chat and agent are recorded)."""
    from neo2.core import config, events
    from neo2.core.agent import session as sess_mod

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(events, "_bus", events.EventBus())
    camera, song = _Calls("You're holding a mug."), _Calls("I couldn't recognise it.")
    saved = {}
    for name, handler in ((fastpath.CAMERA_TOOL, camera), (fastpath.SONG_TOOL, song)):
        saved[name] = catalog().get(name)
        catalog().register(RegisteredTool(name, name, {"type": "object", "properties": {}}, handler, slow=True))
    others: list[str] = []

    async def elsewhere(self, text, d, *a, **k):
        others.append(text)
        await events.bus().end_job(a[-1] if a and isinstance(a[-1], str) else k.get("job", ""))
        return sess_mod.Reply("a chat reply", "chat", d)

    monkeypatch.setattr(sess_mod.Session, "_chat", elsewhere)
    monkeypatch.setattr(sess_mod.Session, "_agent", elsewhere)
    monkeypatch.setattr(sess_mod, "_camera_on", lambda: True)
    yield SimpleNamespace(camera=camera.calls, song=song.calls, others=others)
    for name, prev in saved.items():
        catalog().remove(name)
        if prev is not None:
            catalog().register(prev)


class _Reflex:
    def __init__(self, intent="chat"):
        from neo2.core.reflex.schema import Decision

        self.d = Decision(intent, 0.9, False, 0.0, False, 0.0, "fake")

    async def decide(self, text):
        return self.d


def _say(s, text, **kw):
    return asyncio.run(s.handle(text, from_user=True, **kw))


def test_try_again_after_a_look_looks_again_with_the_same_question(lanes):
    from neo2.core.agent.session import Session

    s = Session(reflex=_Reflex())
    _say(s, "what am I holding?")
    r = _say(s, "phir se try karo")
    assert r.tools == (fastpath.CAMERA_TOOL,)
    assert [a["question"] for a, _ in lanes.camera] == ["what am I holding?"] * 2


def test_try_again_after_a_song_miss_listens_again(lanes):
    from neo2.core.agent.session import Session

    s = Session(reflex=_Reflex())
    song_words.mark_no_match()  # (what song_identify does when nothing matched)
    r = _say(s, "try again")
    assert r.tools == (fastpath.SONG_TOOL,)
    ((args, ctx),) = lanes.song
    assert ctx.get("song_retry") is True and "_retry" not in args


def test_an_unrelated_request_ends_the_song_retry(lanes):
    from neo2.core.agent.session import Session

    s = Session(reflex=_Reflex())
    song_words.mark_no_match()
    _say(s, "tell me a joke")
    _say(s, "one more time")
    assert lanes.song == [] and lanes.others == ["tell me a joke", "one more time"]


def test_an_unrelated_request_ends_the_camera_retry_but_not_its_follow_ups(lanes):
    from neo2.core.agent.session import Session

    s = Session(reflex=_Reflex())
    _say(s, "what am I holding?")
    _say(s, "tell me a joke")
    _say(s, "try again")
    assert len(lanes.camera) == 1 and lanes.others == ["tell me a joke", "try again"]
    _say(s, "and now?")  # still the camera's own follow-up while it is on
    assert len(lanes.camera) == 2


def test_with_both_windows_open_the_most_recent_lane_wins(lanes):
    from neo2.core.agent.session import Session

    s = Session(reflex=_Reflex())
    _say(s, "what am I holding?")
    song_words.mark_no_match()  # then a song NEO couldn't name
    _say(s, "try again")
    assert len(lanes.song) == 1 and len(lanes.camera) == 1


def test_bare_repeat_after_a_song_miss_never_opens_the_mic(lanes):
    from neo2.core.agent.session import Session

    s = Session(reflex=_Reflex())
    for said in ("repeat that", "again", "do it again", "redo", "look again"):
        song_words.mark_no_match()
        _say(s, said)
    assert lanes.song == []


# ---- Gemini Live, the default voice, routes them the same way --------------------------------------


def _msg(**kw):
    content = dict(
        interrupted=False, input_transcription=None, interim_input_transcription=None,
        output_transcription=None, model_turn=None, turn_complete=False,
    )
    content.update(kw)
    return SimpleNamespace(
        server_content=SimpleNamespace(**content), tool_call=None, tool_call_cancellation=None, go_away=None
    )


def _heard(text):
    """The user's sentence as Live transcribes it."""
    return _msg(input_transcription=SimpleNamespace(text=text, finished=True))


def _answer(text):
    """The model's spoken answer, as Live transcribes it."""
    return _msg(output_transcription=SimpleNamespace(text=text))


def _model_calls(name, cid):
    fc = SimpleNamespace(id=cid, name=name, args={})
    return SimpleNamespace(
        server_content=None, tool_call=SimpleNamespace(function_calls=[fc]), tool_call_cancellation=None, go_away=None
    )


_DONE = _msg(turn_complete=True)


class _LiveSpeaker:
    def __init__(self):
        import threading

        self.playing = threading.Event()

    def play_pcm16(self, data):
        pass

    def interrupt(self):
        pass

    async def wait_done(self):
        return None


class _LiveConnection:
    def __init__(self):
        self.responses = []

    async def send_tool_response(self, function_responses):
        self.responses.extend(function_responses)

    async def send_client_content(self, turns):
        pass


@pytest.fixture
def live(lanes, monkeypatch):
    """Gemini Live with a fake connection and speaker; the model's calls reach lanes' fake camera_look
    and song_identify, and a sentence NEO does itself goes to a session that only records it."""
    from neo2.core import config
    from neo2.core.voice import live as live_mod

    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(config, "_settings", None)

    class _Session:
        memory_context = ""

        async def command(self, text, steps):
            from neo2.core.agent.session import Reply

            return Reply("did it", "quick", silent=True)

        async def handle(self, text, **kw):
            from neo2.core.agent.session import Reply

            return Reply("a chat reply", "chat", silent=True)

    v = live_mod.LiveVoice(lambda t: None, _Session())
    v.spk, v._live, v._listening = _LiveSpeaker(), _LiveConnection(), True
    return v


def _turn(v, *msgs):
    async def run():
        user_buf, model_buf = [], []
        for m in msgs:
            await v._on_message(m, user_buf, model_buf, v._live)
            await asyncio.sleep(0)
        await asyncio.sleep(0.05)

    asyncio.run(run())


def test_live_another_request_ends_a_missed_songs_retry(live, lanes):
    """As in a typed chat (Session._handle): a question the model answers in between is the user's
    last request, so their "try again" after it never opens the mic, whatever the model calls."""
    song_words.mark_no_match()
    _turn(live, _heard("what's the capital of France?"), _answer("Paris."), _DONE)
    assert retry_words.last_lane() == "other" and not song_words.retry_after_miss("try again")
    _turn(live, _heard("try again"), _model_calls(fastpath.SONG_TOOL, "s1"), _DONE)
    assert lanes.song == []


def test_live_try_again_right_after_a_miss_still_listens(live, lanes):
    """The sentence is weighed before the model's call for it runs: the retry it asks for is still
    there to pass on (the call spends it)."""
    song_words.mark_no_match()
    _turn(live, _heard("try again"), _model_calls(fastpath.SONG_TOOL, "s1"), _DONE)
    ((_, ctx),) = lanes.song
    assert ctx.get("song_retry") is True


def test_live_another_request_ends_the_camera_retry_but_not_its_follow_ups(live):
    camera_words.mark_look(question="what am I holding?")
    _turn(live, _heard("tell me a joke"), _answer("Why did the scarecrow win an award?"), _DONE)
    assert retry_words.last_lane() == "other"
    assert not camera_words.follow_up("try again") and camera_words.follow_up("and now?")
