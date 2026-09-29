"""Routing to the camera (neo2.core.agent.camera_words, fastpath.camera): the user's own words
"what am I holding" / "ye kya hai" / "मेरे हाथ में क्या है" run camera_look; screen questions, a
model's words, Live's own idea and library mode never do. Fakes only: no camera, no models."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from neo2.core.agent import camera_words, fastpath
from neo2.core.agent.registry import USERS_WORDS, RegisteredTool, ToolOutput, catalog

CAMERA_ASKS = [
    "what am I holding",
    "What am I holding?",
    "Neo, what am I holding right now?",
    "Neo, can you please tell me what I'm holding?",
    "what's in my hand",
    "whats in my hand",
    "what do I have in my hand",
    "what is this",
    "What's this?",
    "what is this thing",
    "what is this called",
    "can you tell me what this is",
    "identify this",
    "look at this",
    "take a look at this",
    "can you see me",
    "Can you see me?",
    "do you see this",
    "what do you see",
    "read this label",
    "please read this label",
    "what does this label say",
    "how many fingers",
    "how many fingers am I holding up",
    "what colour is this",
    "what color is my shirt",
    "what kind of plant is this",
    "what am I wearing",
    "turn on the camera",
    "turn on the camera and tell me what I'm holding",
    # Hinglish
    "ye kya hai",
    "yeh kya hai?",
    "ye kya hai batao",
    "dekho ye kya hai",
    "mere haath mein kya hai",
    "mere hath me kya hai",
    "maine kya pakda hai",
    "kitni ungliyan",
    "ye kis rang ka hai",
    "kya tum mujhe dekh sakte ho",
    "camera on karo",
    # Hindi
    "मेरे हाथ में क्या है",
    "मेरे हाथ में क्या है?",
    "यह क्या है",
    "ये क्या है?",
    "कितनी उंगलियां",
    "क्या तुम मुझे देख सकते हो",
]

NOT_CAMERA = [
    # the screen: the screen tools' job
    "what is on my screen",
    "what's on my screen",
    "what is this on my screen",
    "what do you see on my screen",
    "can you see my screen",
    "look at this page",
    "look at this tab",
    "read this email",
    "what is this error",
    "what is this website",
    "what's this page about",
    "summarize this page",
    "what colour is this button",
    "what is this in the photo",
    "yeh screen pe kya hai",
    "स्क्रीन पर क्या है",
    # not a look at all
    "open camera",
    "open the camera app",
    "remind me to ask what this is",
    "what is that",
    "what is this about",
    "read this",
    "read this note",
    "what time is it",
    "what is python",
    "what am I looking at",
    "type what is this",
    "",
]


@pytest.mark.parametrize("text", CAMERA_ASKS)
def test_camera_asks_are_recognised(text):
    assert camera_words.wants_camera(text), text


@pytest.mark.parametrize("text", NOT_CAMERA)
def test_screen_and_other_requests_are_not(text):
    assert not camera_words.wants_camera(text), text


def test_follow_up_only_inside_the_window():
    camera_words.forget()
    assert not camera_words.follow_up("and now?")
    t0 = time.monotonic()
    camera_words.mark_look(t0)
    for said in ("and now?", "what about now", "and this one?", "what about this one", "aur ye?", "ab?", "और अब?"):
        assert camera_words.follow_up(said, now=t0 + 5), said
    for said in ("what is on my screen", "open safari", "and close it", "now open notes"):
        assert not camera_words.follow_up(said, now=t0 + 5), said
    assert not camera_words.follow_up("and now?", now=t0 + camera_words.FOLLOW_UP_S + 1)
    camera_words.forget()
    assert not camera_words.follow_up("and now?", now=t0 + 1)


# ---- fastpath.camera: who may turn the camera on --------------------------------------------------


class _Calls:
    def __init__(self, text="You're holding a coffee mug.", ok=True):
        self.calls: list[tuple[dict, object]] = []
        self.text, self.ok = text, ok

    async def __call__(self, args, ctx):
        self.calls.append((dict(args), ctx))
        return ToolOutput(self.text, ok=self.ok)


@pytest.fixture
def cam_tool():
    """A fake camera_look in NEO.app's catalog (the real one is neo2.core.tools.camera)."""
    handler = _Calls()
    prev = catalog().get(fastpath.CAMERA_TOOL)
    catalog().register(
        RegisteredTool(
            fastpath.CAMERA_TOOL,
            "Look through the camera and answer the user's question.",
            {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
            handler,
            category="camera",
            slow=True,
        )
    )
    camera_words.forget()
    yield handler
    catalog().remove(fastpath.CAMERA_TOOL)
    if prev is not None:
        catalog().register(prev)
    camera_words.forget()


def test_camera_step_for_the_users_own_words(cam_tool):
    assert fastpath.camera("what am I holding?", origin="user") == ("camera_look", {"question": "what am I holding?"})
    assert fastpath.camera("मेरे हाथ में क्या है", origin="user") == ("camera_look", {"question": "मेरे हाथ में क्या है"})


@pytest.mark.parametrize("origin", ["live", "agent", "approval"])
def test_camera_never_for_a_models_words(cam_tool, origin):
    assert fastpath.camera("what am I holding", origin=origin) is None


def test_camera_never_for_the_screen(cam_tool):
    assert fastpath.camera("what is on my screen", origin="user") is None
    assert fastpath.camera("what is this on my screen", origin="user") is None


def test_no_camera_tool_no_route():
    camera_words.forget()
    prev = catalog().remove(fastpath.CAMERA_TOOL)
    try:
        assert fastpath.camera("what am I holding", origin="user") is None
    finally:
        if prev is not None:
            catalog().register(prev)


def test_library_mode_never_routes_to_the_camera(cam_tool, monkeypatch):
    from neo2.core import runtime

    monkeypatch.setattr(runtime, "library_mode", lambda: True)
    monkeypatch.setattr(runtime, "library_bound", lambda: False)
    assert fastpath.camera("what am I holding", origin="user") is None


def test_the_regex_fast_paths_never_take_a_camera_ask(cam_tool):
    from neo2.core.tools import load_all

    load_all()
    for text in ("what am I holding", "ye kya hai", "मेरे हाथ में क्या है", "can you see me"):
        steps = fastpath.plan(text) or []
        assert all(t != "camera_look" for t, _ in steps)


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
    warmed: list[int] = []
    monkeypatch.setattr(camera_words, "prewarm", lambda: warmed.append(1) or True)

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
    return SimpleNamespace(chain=Chain, warmed=warmed)


class _Reflex:
    def __init__(self, intent="chat"):
        from neo2.core.reflex.schema import Decision

        self.d = Decision(intent, 0.9, False, 0.0, False, 0.0, "fake")

    async def decide(self, text):
        return self.d


def test_session_runs_camera_for_the_users_question_even_if_laya_says_chat(cam_tool, session_env):
    from neo2.core.agent.session import Session

    s = Session(reflex=_Reflex("chat"))
    r = asyncio.run(s.handle("Neo, what am I holding?", from_user=True))
    assert r.route == "quick" and r.tools == ("camera_look",)
    assert r.text == "You're holding a coffee mug."
    (args, ctx), = cam_tool.calls
    assert args == {"question": "what am I holding?"} or args["question"].lower().startswith("what am i holding")
    assert ctx.origin == "user" and ctx.get(USERS_WORDS)
    assert session_env.warmed  # the answer's brains warmed while it was routed
    assert session_env.chain.streamed == 0 and session_env.chain.completed == 0


def test_session_hinglish_and_hindi(cam_tool, session_env):
    from neo2.core.agent.session import Session

    s = Session(reflex=_Reflex("chat"))
    for said in ("mere haath mein kya hai", "यह क्या है?"):
        r = asyncio.run(s.handle(said, from_user=True))
        assert r.tools == ("camera_look",), said
    assert len(cam_tool.calls) == 2


def test_session_screen_question_never_turns_the_camera_on(cam_tool, session_env):
    from neo2.core.agent.session import Session

    s = Session(reflex=_Reflex("chat"))
    r = asyncio.run(s.handle("what is on my screen", from_user=True))
    assert cam_tool.calls == []
    assert "camera_look" not in (r.tools or ())


def test_session_live_goal_never_turns_the_camera_on(cam_tool, session_env):
    from neo2.core.agent.session import Session

    s = Session(reflex=_Reflex("chat"))
    asyncio.run(s.handle("what am I holding", origin="live"))  # a model's goal, not the user's words
    assert cam_tool.calls == []


def test_session_follow_up_reuses_the_camera_then_stop_ends_it(cam_tool, session_env):
    from neo2.core.agent.session import Session

    s = Session(reflex=_Reflex("chat"))
    asyncio.run(s.handle("what is this", from_user=True))
    r = asyncio.run(s.handle("and now?", from_user=True))
    assert r.tools == ("camera_look",) and len(cam_tool.calls) == 2
    asyncio.run(s.cancel_all())
    asyncio.run(s.handle("and now?", from_user=True))
    assert len(cam_tool.calls) == 2


def test_session_camera_failure_is_said_not_handed_to_a_model(cam_tool, session_env):
    from neo2.core.agent.session import Session

    cam_tool.text, cam_tool.ok = "Error: the camera isn't allowed yet — turn it on in System Settings.", False
    s = Session(reflex=_Reflex("agent_task"))
    r = asyncio.run(s.handle("what am I holding", from_user=True))
    assert r.route == "quick" and "camera" in r.text
    assert session_env.chain.completed == 0  # no blind brain guessing what's in the user's hand


# ---- early: the answer's brains warm while the user speaks ------------------------------------------


def test_early_prewarms_on_a_camera_ask_once(monkeypatch):
    from neo2.core.agent import early as early_mod
    from neo2.core.agent import prefetch
    from neo2.core.providers import cerebras_hooks

    monkeypatch.setattr(prefetch, "prewarm", lambda reason="": False)
    monkeypatch.setattr(cerebras_hooks, "prewarm_soon", lambda: None)
    warmed: list[int] = []
    monkeypatch.setattr(camera_words, "prewarm", lambda: warmed.append(1) or True)
    a = early_mod.EarlyActor()
    a.feed("open")
    a.feed("open safari")
    assert warmed == []
    a.new_utterance()
    a.feed("what am I")
    a.feed("what am I holding")
    a.feed("what am I holding right now")
    assert warmed == [1]
    a.new_utterance()
    a.feed("what is on my screen")
    assert warmed == [1]


def test_prewarm_without_the_vision_layer_is_quiet(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "neo2.core.vision", None)  # not installed
    assert camera_words.prewarm() is False


def test_prewarm_calls_the_answer_layer(monkeypatch):
    import sys
    import types

    calls: list[int] = []
    pkg = types.ModuleType("neo2.core.vision")
    ans = types.ModuleType("neo2.core.vision.answer")
    ans.prewarm = lambda: calls.append(1)
    pkg.answer = ans
    monkeypatch.setitem(sys.modules, "neo2.core.vision", pkg)
    monkeypatch.setitem(sys.modules, "neo2.core.vision.answer", ans)
    assert camera_words.prewarm() is True and calls == [1]


# ---- prompts ---------------------------------------------------------------------------------------


def test_agent_prompt_knows_the_tool_only_when_it_is_there():
    from neo2.core.agent import prompt

    assert "camera_look" in prompt.persona(["camera_look", "screenshot"])
    assert "camera_look" not in prompt.persona(["screenshot"])


def test_live_declares_and_explains_camera_look(cam_tool, monkeypatch):
    from neo2.core.voice import live as live_mod

    assert "camera_look" in live_mod._direct_tools()
    assert "camera_look" in [d.name for d in live_mod._declarations()]
    assert "camera_look" in live_mod._live_extra()


def test_live_extra_without_the_tool_never_mentions_it():
    from neo2.core.voice import live as live_mod

    prev = catalog().remove(fastpath.CAMERA_TOOL)
    try:
        assert "camera_look" not in live_mod._live_extra()
        assert "camera_look" not in [d.name for d in live_mod._declarations()]
    finally:
        if prev is not None:
            catalog().register(prev)


def _voice():
    from neo2.core.voice import live as live_mod

    v = live_mod.LiveVoice.__new__(live_mod.LiveVoice)
    v._taint, v._seen_calls, v._recent_idem = [], {}, {}
    return v


def test_live_call_runs_only_when_the_users_words_asked(cam_tool, monkeypatch):
    from neo2.core import events

    monkeypatch.setattr(events, "_bus", events.EventBus())
    monkeypatch.setattr(camera_words, "prewarm", lambda: True)
    v = _voice()
    fc = SimpleNamespace(name="camera_look", args={"question": "describe the room"}, id="c1")
    # the model's own idea: the user was talking about their screen
    out = asyncio.run(v._camera_call(fc, dict(fc.args), "what is on my screen", "c1", 1))
    assert out.startswith("NEEDS_USER") and cam_tool.calls == []
    out = asyncio.run(v._camera_call(fc, dict(fc.args), "", "c1", 1))
    assert out.startswith("NEEDS_USER") and cam_tool.calls == []
    # the user asked: their own words are the question, run as theirs
    fc2 = SimpleNamespace(name="camera_look", args={"question": "What is the user holding?"}, id="c2")
    out = asyncio.run(v._camera_call(fc2, dict(fc2.args), "mere haath mein kya hai", "c2", 2))
    assert out == "You're holding a coffee mug."
    (args, ctx), = cam_tool.calls
    assert args == {"question": "mere haath mein kya hai"}
    assert ctx.origin == "user" and ctx.get(USERS_WORDS) == "mere haath mein kya hai"
    assert v._own_utt == 2


def test_neo_apps_prompt_has_the_camera_line_only_while_the_tool_is_there(cam_tool):
    from neo2.core.agent import prompt

    assert "camera_look" in prompt.persona()
    catalog().remove(fastpath.CAMERA_TOOL)
    assert "camera_look" not in prompt.persona()
    assert "camera_look" not in prompt.PERSONA
