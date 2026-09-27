"""The Chess run's root cause: after Gemini was overloaded, a text-only brain (Groq's gpt-oss-120b)
got the screenshot's note but not its pixels — told "[screenshot attached]" — and clicked and
dragged at coordinates it never saw. Every step said ok; the board never moved.

Now a brain that can't see is told so plainly, a request working from a picture goes to a brain
that can see it first, and coordinates chosen by a brain that saw nothing are refused. Hermetic:
brains are fakes, nothing is sent anywhere, no event is posted."""

from __future__ import annotations

import asyncio
import types

import pytest

from neo2.core.agent.loop import run_agent
from neo2.core.agent.registry import ToolContext, ToolOutput, registry
from neo2.core.events import EventBus
from neo2.core.providers.base import (
    BLIND_PICTURE,
    ImagePart,
    Message,
    NoEyes,
    Overloaded,
    ToolCall,
    ToolResult,
    Turn,
)
from neo2.core.tools.computer import input as inp
from neo2.core.tools.computer import pointer, screen

PNG = ImagePart(b"\x89PNG fake", "image/png")


def _shot_history() -> list[Message]:
    return [
        Message.user("play e4"),
        Message.assistant("", [ToolCall("c1", "screenshot", {})]),
        Message.tool([ToolResult("c1", "screenshot", "Screenshot s1 attached (all of display 1); the image is 1512×982 px.", images=[PNG])]),
    ]


def test_a_text_only_brain_is_told_it_cannot_see_the_screenshot():
    from neo2.core.providers.openai_compat import _to_messages

    m = _to_messages(_shot_history(), "sys", vision=False)
    tool = next(x for x in m if x["role"] == "tool")
    assert BLIND_PICTURE in tool["content"] and "attached]" not in tool["content"].split("px.")[-1]
    assert not any(isinstance(x.get("content"), list) for x in m)  # and no pixels were sent


def test_a_fitted_request_tells_a_text_only_brain_it_cannot_see_not_that_it_was_trimmed():
    from neo2.core.providers.fit import fit

    msgs = _shot_history() + [Message.assistant("", [ToolCall("c2", "ax_tree", {})]),
                              Message.tool([ToolResult("c2", "ax_tree", "tree " * 2000)])]  # fmt: skip
    f = fit(msgs, system="sys", tools=None, budget=3000, max_tokens=512, vision=False, name="groq")
    shot = next(r for m in f.messages for r in m.tool_results if r.name == "screenshot")
    assert BLIND_PICTURE in shot.content and "picture dropped to fit" not in shot.content and not shot.images


def test_a_text_only_brains_replies_are_marked_as_not_seeing(monkeypatch):
    from neo2.core.providers.openai_compat import OpenAICompatProvider

    p = OpenAICompatProvider(name="groq", base_url="https://api.groq.test/v1", api_key="k", model="m")

    async def create(**kw):
        call = types.SimpleNamespace(id="t1", function=types.SimpleNamespace(name="click", arguments='{"x": 612, "y": 540}'))
        msg = types.SimpleNamespace(content="", tool_calls=[call])
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=None)

    monkeypatch.setattr(p._client.chat.completions, "create", create)
    turn = asyncio.run(p.complete(_shot_history(), system="sys"))
    assert turn.sees is False and turn.tool_calls[0].name == "click"


class Brain:
    def __init__(self, name, vision, error=None, answer="ok"):
        self.name, self.model, self.supports_vision, self.supports_tools = name, name, vision, True
        self.error, self.answer, self.asked = error, answer, 0

    def cooling_until(self):
        return 0.0

    async def complete(self, messages, **kw):
        self.asked += 1
        if self.error is not None:
            raise self.error
        return Turn(text=self.answer, sees=self.supports_vision)

    async def stream(self, messages, **kw):
        yield self.answer


def test_a_request_working_from_a_picture_goes_to_a_brain_that_can_see_it_first(monkeypatch):
    import neo2.core.providers as prov

    gemini, groq = Brain("gemini", True, answer="seen"), Brain("groq", False, answer="blind")
    monkeypatch.setattr(prov, "make", lambda n: {"gemini": gemini, "groq": groq}[n])
    monkeypatch.setattr(prov, "_demoted", {"gemini": 1e18})  # Gemini fell over a minute ago
    chain = prov.Chain(["gemini", "groq"])
    assert asyncio.run(chain.complete([Message.user("hi")])).text == "blind"  # no picture: demotion holds
    assert asyncio.run(chain.complete(_shot_history())).text == "seen"  # a board in view: eyes first
    gemini.error = Overloaded("busy")
    # Every brain that sees is busy: the run stops and says so rather than act blind (DECISIONS 39).
    with pytest.raises(NoEyes) as stop:
        asyncio.run(chain.complete(_shot_history()))
    assert groq.asked == 1 and "can see the screen is busy" in stop.value.spoken()


def test_a_picture_never_jumps_a_local_brain_the_user_put_first(monkeypatch):
    """Local-first: with brain=local the screen stays on the Mac — a cloud brain that can see is asked
    about a screenshot only after the local one, as the user ordered them."""
    import neo2.core.providers as prov
    from neo2.core.providers.openai_compat import local as make_local

    here, gemini, groq = Brain("local", False, answer="local"), Brain("gemini", True, answer="seen"), Brain("groq", False, answer="blind")
    here.on_device = True
    monkeypatch.setattr(prov, "make", lambda n: {"local": here, "gemini": gemini, "groq": groq}[n])
    monkeypatch.setattr(prov, "_demoted", {})
    assert asyncio.run(prov.Chain(["local", "gemini"]).complete(_shot_history())).text == "local" and gemini.asked == 0
    # a cloud brain that can't see still yields to one that can, and a local brain the user put last stays last
    assert asyncio.run(prov.Chain(["groq", "gemini", "local"]).complete(_shot_history())).text == "seen"
    assert make_local().on_device and not Brain("groq", False).__dict__.get("on_device")


class Script:
    """A brain that plays a fixed list of turns, and keeps what it was sent."""

    supports_vision = True
    supports_tools = True
    name = "script"

    def __init__(self, turns):
        self.turns, self.seen = list(turns), []

    async def complete(self, messages, **kw):
        self.seen.append(list(messages))
        return self.turns.pop(0)

    async def stream(self, messages, **kw):
        yield ""


def test_coordinates_from_a_brain_that_saw_nothing_are_refused_in_the_run(monkeypatch):
    clicked = []
    monkeypatch.setattr(inp, "click", lambda *a, **k: clicked.append(a) or "clicked")
    monkeypatch.setattr(inp, "move", lambda x, y: None)  # (the pointer resting on the spot before the look)
    blind = Script([
        Turn(tool_calls=[ToolCall("t1", "click", {"x": 612, "y": 540})], sees=False),
        Turn(text="I couldn't see the board.", sees=False),
    ])  # fmt: skip
    res = asyncio.run(run_agent("play e4", provider=blind, system="", ev=EventBus(), verify=False,
                                tools=[registry().get("click").spec()]))  # fmt: skip
    assert clicked == [] and "can't see the screen" in res.steps[0].result and not res.steps[0].ok
    sighted = Script([Turn(tool_calls=[ToolCall("t1", "click", {"x": 612, "y": 540})]), Turn(text="Done.")])
    asyncio.run(run_agent("play e4", provider=sighted, system="", ev=EventBus(), verify=False,
                          tools=[registry().get("click").spec()]))  # fmt: skip
    assert len(clicked) == 1


def test_the_screenshot_note_reaches_the_model_unfenced_and_a_look_is_not_an_act(monkeypatch):
    def capture(region=None, display=None):
        from io import BytesIO

        from PIL import Image

        buf = BytesIO()
        Image.new("RGB", (1512, 982)).save(buf, format="PNG")
        return ImagePart(buf.getvalue()), 1.0

    monkeypatch.setattr(screen, "capture", capture)
    b = Script([Turn(tool_calls=[ToolCall("t1", "screenshot", {})]), Turn(text="It's your move.")])
    res = asyncio.run(run_agent("whose move is it?", provider=b, system="", ev=EventBus(),
                                tools=[registry().get("screenshot").spec()]))  # fmt: skip
    note = res.messages[-2].tool_results[0].content
    assert note.startswith("Screenshot s") and "<untrusted" not in note  # NEO's own words, not screen text
    assert len(b.seen) == 2 and res.text == "It's your move."  # a look-only run pays no extra [verify] turn


def test_a_run_that_worked_from_the_screen_is_spoken():
    from neo2.core.agent.loop import AgentResult, Step
    from neo2.core.agent.session import _quietly
    from neo2.core.reflex.schema import Decision

    d = Decision("agent_task", 0.9, False, 0, False, 0, "laya+head", reply=False)
    steps = [Step("screenshot", {}, "Screenshot s1…", True), Step("click", {"x": 612, "y": 540, "shot": "s1"}, "Clicked…", True)]
    assert not _quietly(d, AgentResult("I moved your pawn to e4.", "done", steps))
    assert _quietly(d, AgentResult("Done.", "done", [Step("hotkey", {"keys": "cmd+n"}, "Pressed cmd+n", True)]))


def test_a_blind_refusal_names_what_still_works():
    with pytest.raises(pointer.Refused) as r:
        pointer.points({"x": 612, "y": 540}, ("x", "y"), ToolContext(user_text="play", extras={"blind": True}))
    assert "ax_tree" in str(r.value) and "click_text" in str(r.value)
    assert isinstance(ToolOutput("x").fence, bool)


class HostTextOnly:
    """A host's own text-only brain (neo2 providers): it declares no vision and has never heard of
    Turn.sees, so every reply it writes says sees=True."""

    name, model, supports_vision, supports_tools = "host", "m", False, True

    def __init__(self):
        self.turns = [Turn(tool_calls=[ToolCall("t1", "click", {"x": 612, "y": 540})]), Turn(text="Done.")]

    def cooling_until(self):
        return 0.0

    async def complete(self, messages, **kw):
        return self.turns.pop(0)

    async def stream(self, messages, **kw):
        yield ""


@pytest.mark.parametrize("chained", [True, False])
def test_a_brain_that_declares_no_vision_is_blind_whatever_its_turns_say(monkeypatch, chained):
    from neo2.core.providers import Chain

    clicked = []
    monkeypatch.setattr(inp, "click", lambda *a, **k: clicked.append(a) or "clicked")
    monkeypatch.setattr(inp, "move", lambda x, y: None)
    brain = Chain([HostTextOnly()]) if chained else HostTextOnly()
    res = asyncio.run(run_agent("play e4", provider=brain, system="", ev=EventBus(), verify=False,
                                tools=[registry().get("click").spec()]))  # fmt: skip
    assert clicked == [] and "can't see the screen" in res.steps[0].result


def test_only_the_users_own_words_give_a_blind_brain_its_numbers():
    """"Click at 612, 540" from the user is theirs to give; the same numbers in a model's goal, a
    Live relay's or an embedding host's words are not (whatever ctx.user_text holds)."""
    from neo2.core.agent.registry import USERS_WORDS

    goal = ToolContext(user_text="(a host model's goal) click 612 540", extras={"blind": True}, origin="agent")
    with pytest.raises(pointer.Refused):
        pointer.points({"x": 612, "y": 540}, ("x", "y"), goal)
    theirs = ToolContext(user_text="play", extras={"blind": True, USERS_WORDS: "click at 612, 540"}, origin="agent")
    assert pointer.points({"x": 612, "y": 540}, ("x", "y"), theirs)[0] == [(612.0, 540.0)]


def test_the_session_marks_the_users_own_words_and_nothing_else(monkeypatch):
    from test_v2_session import FakeChain, FakeReflex

    from neo2.core import runtime
    from neo2.core.agent import session as sess_mod
    from neo2.core.agent.loop import AgentResult
    from neo2.core.agent.registry import USERS_WORDS

    seen = []

    async def fake_run_agent(goal, **kw):
        seen.append(kw["ctx"].get(USERS_WORDS, ""))
        return AgentResult("ok", "done", [], [])

    monkeypatch.setattr(sess_mod, "run_agent", fake_run_agent)
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": FakeChain())
    s = sess_mod.Session(reflex=FakeReflex("agent_task"))
    said = "click at 612, 540"
    asyncio.run(s.handle(said, from_user=True))  # typed, or spoken to NEO.app
    asyncio.run(s.handle(said, origin="user"))  # Live handing over the user's own sentence
    asyncio.run(s.handle(said, origin="live"))  # Live's model's goal
    asyncio.run(s.handle(said, origin="live", extras={USERS_WORDS: said}))  # nobody else can say so
    monkeypatch.setattr(runtime, "library_bound", lambda: True)  # an embedded agent's request
    asyncio.run(s.handle(said, origin="user"))
    assert seen == [said, said, "", "", ""]
