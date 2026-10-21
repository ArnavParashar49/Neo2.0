"""Routing to Search Screen: the user's own "search my screen" runs search_screen — in the Session
(ahead of the camera and the web, no model, no Laya), in Gemini Live's command lane and in the
on-device voice (the early actor leaves it to the session) — never web_search, never for a model's
words, Live's own idea after it read a page, or an embedded agent. The tool is hidden while Search
Screen is off or no NEO.app runs the core, and says why when asked anyway. While the overlay holds
an area, what the user says is about that area. Fakes only: no overlay, no models."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from neo2.core import config, events
from neo2.core.agent import fastpath
from neo2.core.agent.registry import ToolContext, catalog
from neo2.core.lens import live
from neo2.core.tools import lens as lens_tool
from neo2.core.tools import load_all
from tests.test_v2_speech_gate import DONE, call, feed, heard, voice  # noqa: F401 — (the Live fixture)


@pytest.fixture
def opened(monkeypatch):
    """Search Screen's overlay, as NEO.app's shell would open it: what was asked."""
    load_all()
    got: list[str] = []
    monkeypatch.setattr(live.lens_live, "open_overlay", lambda question="": got.append(question) or "opened")
    from neo2.core.tools.computer import screen

    monkeypatch.setattr(screen, "can_capture", lambda: True)
    return got


def test_the_users_own_words_are_the_lens_step(opened):
    assert fastpath.lens("search my screen", origin="user") == ("search_screen", {})
    assert fastpath.lens("Neo, search my screen for the price of this?", origin="user") == (
        "search_screen", {"question": "the price of this"},
    )  # fmt: skip
    assert fastpath.lens("screen pe search karo", origin="user") == ("search_screen", {})
    for origin in ("live", "agent", "approval"):
        assert fastpath.lens("search my screen", origin=origin) is None
    assert fastpath.lens("search the web for screen protectors", origin="user") is None


def test_library_mode_never_routes_to_it(opened, monkeypatch):
    from neo2.core import runtime

    monkeypatch.setattr(runtime, "library_mode", lambda: True)
    assert fastpath.lens("search my screen", origin="user") is None


def test_web_search_never_takes_search_screens_words(opened):
    for text in ("search my screen", "search the screen", "search my screen for shoes", "search what's on my screen",
                 "search on my screen"):  # fmt: skip
        steps = fastpath.plan(text) or []
        assert all(t != "web_search" for t, _ in steps), text
    assert fastpath.plan("search the screen protector deals") == [("web_search", {"query": "the screen protector deals"})]
    assert fastpath.plan("search cats") == [("web_search", {"query": "cats"})]


# ---- the session routes it --------------------------------------------------------------------------


@pytest.fixture
def session_env(tmp_path, monkeypatch, opened):
    from neo2.core.agent import session as sess_mod

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(events, "_bus", events.EventBus())

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

    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent", **kw: Chain())
    return SimpleNamespace(chain=Chain, opened=opened)


class _Reflex:
    def __init__(self, intent="agent_task"):
        from neo2.core.reflex.schema import Decision

        self.d = Decision(intent, 0.9, False, 0.0, False, 0.0, "fake")
        self.asked = 0

    async def decide(self, text):
        self.asked += 1
        return self.d


def test_session_opens_search_screen_for_the_users_words_without_a_model(session_env):
    from neo2.core.agent.session import Session

    reflex = _Reflex("chat")
    s = Session(reflex=reflex)
    r = asyncio.run(s.handle("Neo, search my screen for what these shoes cost", from_user=True))
    assert r.route == "quick" and r.tools == ("search_screen",) and r.text == lens_tool.OPENED
    assert session_env.opened == ["what these shoes cost"]
    assert reflex.asked == 0 and session_env.chain.streamed == session_env.chain.completed == 0
    r = asyncio.run(s.handle("स्क्रीन पर सर्च करो", from_user=True))
    assert r.tools == ("search_screen",) and session_env.opened[-1] == ""


def test_session_a_models_goal_never_opens_it(session_env):
    from neo2.core.agent.session import Session

    asyncio.run(Session(reflex=_Reflex()).handle("search my screen", origin="live"))
    assert session_env.opened == []


def test_session_off_or_no_app_is_said_not_handed_to_a_model(session_env, monkeypatch):
    from neo2.core.agent.session import Session

    monkeypatch.setattr(live.lens_live, "open_overlay", lambda question="": "no_app")
    r = asyncio.run(Session(reflex=_Reflex()).handle("search my screen", from_user=True))
    assert r.text == lens_tool.NO_APP and session_env.chain.completed == 0
    monkeypatch.setenv("NEO_LENS", "off")
    monkeypatch.setattr(config, "_settings", None)
    r = asyncio.run(Session(reflex=_Reflex()).handle("circle to search", from_user=True))
    assert r.text == lens_tool.OFF and session_env.chain.completed == 0


def test_the_tool_without_screen_recording_still_opens_and_says_so(session_env, monkeypatch):
    from neo2.core.tools.computer import screen

    monkeypatch.setattr(screen, "can_capture", lambda: False)
    out = asyncio.run(lens_tool.search_screen({}, ToolContext(user_text="search my screen", origin="user")))
    assert out.ok and out.text.startswith(lens_tool.OPENED) and "Screen Recording" in out.text


def test_a_model_after_reading_a_page_cant_open_it(session_env):
    ctx = ToolContext(user_text="", taint=["web_fetch"], origin="agent")
    out = asyncio.run(lens_tool.search_screen({"question": "x"}, ctx))
    assert not out.ok and out.text == lens_tool.NOT_FOR_MODELS and session_env.opened == []
    ok = asyncio.run(lens_tool.search_screen({"question": "these shoes"}, ToolContext(user_text="", origin="agent")))
    assert ok.ok and session_env.opened == ["these shoes"]  # (the user asked the agent about something on screen)


def test_the_tool_is_hidden_while_off_or_without_neo_app(opened, monkeypatch):
    t = catalog().get("search_screen")
    try:
        lens_tool.start(has_shell=False)
        assert t.hidden
        lens_tool.start(has_shell=True)
        assert not t.hidden
        monkeypatch.setenv("NEO_LENS", "off")
        monkeypatch.setattr(config, "_settings", None)
        lens_tool.start(has_shell=True)
        assert t.hidden
    finally:
        t.hidden = False


def test_an_embedded_agent_never_gets_it():
    from neo2.core.tools import REFUSED, resolve

    assert "search_screen" in REFUSED
    with pytest.raises(ValueError):
        resolve(["search_screen"])


# ---- the voices ----------------------------------------------------------------------------------------


def test_lives_command_lane_opens_it_before_the_model_or_the_web(voice):  # noqa: F811
    async def run():
        await feed(voice, heard("search my screen"), call("web_search", {"query": "my screen"}, "m1"), DONE)

    asyncio.run(run())
    assert voice.lane == [("search my screen", ["search_screen"])]
    assert voice.ran == []  # the model's web search never ran


def test_lives_prompt_and_declarations_name_it_only_while_it_is_there(opened):
    from neo2.core.voice import live as live_mod

    assert "search_screen" in [d.name for d in live_mod._declarations()]
    assert "call search_screen" in live_mod._live_extra()
    t = catalog().get("search_screen")
    t.hidden = True
    try:
        assert "search_screen" not in [d.name for d in live_mod._declarations()]
        assert "search_screen" not in live_mod._live_extra()
    finally:
        t.hidden = False


def test_the_early_actor_leaves_it_to_the_session(opened, monkeypatch):
    from neo2.core.agent.early import EarlyActor

    ran = []

    async def invoke(self, name, args, ctx):
        ran.append(name)
        raise AssertionError("nothing runs early for search my screen")

    from neo2.core.agent.registry import Registry

    monkeypatch.setattr(Registry, "invoke", invoke)
    e = EarlyActor()
    e.feed("search my screen")
    assert asyncio.run(e.tick(final=True)) == [] and e.remaining() == "search my screen" and ran == []


# ---- while it holds an area -------------------------------------------------------------------------------


def test_while_it_holds_an_area_the_users_words_are_about_it(monkeypatch, voice):  # noqa: F811
    heard_: list[str] = []

    async def hear(text):
        heard_.append(text)
        return "That's a Brand X Runner 4. It comes in three colours."

    monkeypatch.setattr(live.lens_live, "takes_voice", lambda: True)
    monkeypatch.setattr(live.lens_live, "hear", hear)
    from neo2.core.agent.early import EarlyActor

    e = EarlyActor()
    e.feed("open safari")
    assert asyncio.run(e.tick(final=True)) == []  # (no command runs early while it holds the voice)

    async def run():
        await feed(voice, heard("what brand is this"), call("web_search", {"query": "brand"}, "m1"), DONE, settle=0.1)

    asyncio.run(run())
    assert heard_ == ["what brand is this"] and voice.lane == [] and voice.ran == []
    note = voice._live.texts[-1].parts[0].text
    assert "That's a Brand X Runner 4." in note and "three colours" not in note  # (its first sentence, said)


def test_the_on_device_voice_asks_the_area_and_says_the_first_sentence(monkeypatch):
    from neo2.core.app import App

    said: list[str] = []

    async def hear(text):
        return "Close the bracket on line 3. Then run it again."

    monkeypatch.setattr(live.lens_live, "takes_voice", lambda: True)
    monkeypatch.setattr(live.lens_live, "hear", hear)
    app = App.__new__(App)
    app._wellbeing_reply_context = None

    async def speak(text):
        said.append(text)

    app.voice = SimpleNamespace(speak=speak, handles_text=False)

    async def never(*a, **k):
        raise AssertionError("never the session: the area's question stays out of the chat")

    app.handle_text = never
    asyncio.run(app.handle_speech("how do I fix this error"))
    assert said == ["Close the bracket on line 3."]


def test_close_and_continue_while_it_holds_an_area(monkeypatch):
    from neo2.core.lens import capture

    closed, handed = [], []
    lv = live.LensLive()
    lv.ticket, lv.client = "ab" * 32, object()
    lv.spot = capture.Spot(1, 0, 0, 100, 100, 1)
    lv._area = live._Area(1, (0, 0, 10, 10), "drag", SimpleNamespace(image=None))
    monkeypatch.setattr(lv, "close", lambda why="done": closed.append(why) or True)

    async def cont(a, text):
        handed.append(text)

    monkeypatch.setattr(lv, "_continue", cont)
    assert lv.takes_voice()
    assert asyncio.run(lv.hear("never mind")) == "" and closed == ["done"]
    assert asyncio.run(lv.hear("continue in chat")) == "" and handed == [""]


def test_what_is_said_aloud_is_the_first_sentence():
    assert live.first_sentence("That's a Brand X Runner 4. It comes in three colours.") == "That's a Brand X Runner 4."
    assert live.first_sentence("No. 5 is the answer, see step 2.") == "No. 5 is the answer, see step 2."  # (too short to cut)
    assert len(live.first_sentence("word " * 200)) == 240


def test_the_same_pixels_are_the_same_area():
    from PIL import Image

    a = Image.new("RGB", (3000, 2000), (10, 20, 30))
    b = Image.new("RGB", (3000, 2000), (10, 20, 31))
    assert live._digest(a) == live._digest(a.copy()) and live._digest(a) != live._digest(b)
    assert live._digest(a) != live._digest(a.resize((300, 200)))
