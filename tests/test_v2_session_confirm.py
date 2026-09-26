"""Who asked, who answers: Hindi/Hinglish yes/no, cards adopted by the session (fast path and
re-staged by an approved call), `ToolOutput.say`, and the origin every tool call carries."""

from __future__ import annotations

import asyncio

import pytest

import neo.agent.session as sess_mod
from neo.agent import confirm
from neo.agent.registry import RegisteredTool, ToolContext, ToolOutput, registry, voiced
from neo.agent.session import Session, answer_of
from neo.events import EventBus, NeoState, bus
from neo.providers.base import ToolCall, Turn
from neo.reflex.schema import Decision

_OBJ = {"type": "object", "properties": {}}


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    from neo import config, events
    from neo.tools import load_all

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(events, "_bus", EventBus())
    confirm.cancel("test")
    load_all()
    yield
    confirm.cancel("test")


def _reflex(intent="agent_task"):
    class R:
        async def decide(self, t):
            return Decision(intent, 0.95, False, 0.0, False, 0.0, "laya+head", reply=False, reply_p=0.1)

    return R()


class Brain:
    name = "fake"
    supports_vision = False
    supports_tools = True

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = 0

    async def complete(self, messages, **kw):
        self.calls += 1
        return self.turns.pop(0)

    async def stream(self, messages, **kw):
        yield ""


def _add(monkeypatch, name, handler, **kw):
    monkeypatch.setitem(registry()._tools, name, RegisteredTool(name, "test tool", _OBJ, handler, **kw))


def _messenger(monkeypatch, log: list):
    """A send that can't reach the chat in the first app it was approved for and offers the
    other one on a new card — the shape of a real re-card. Each card is a pure function of its
    params, so the approved call rebuilds exactly the card the user saw."""

    def card(a):
        return (f"{a['why']}. " if a.get("why") else "") + f"send “{a['text']}” on {a['app']}"

    def zz_msg(a, c):
        log.append((a["app"], c.origin))
        p = confirm.gated("zz_msg", a, card(a))
        if isinstance(p, str):
            return p
        if a["app"] == "one":
            nxt = {**a, "app": "two", "why": "one couldn't open the chat"}
            return confirm.gated("zz_msg", nxt, card(nxt))
        return f"Sent on {a['app']}."

    _add(monkeypatch, "zz_msg", zz_msg, risk="destructive", quiet=True,
         fast_path=[(r"^zz message (?P<text>\w+)$", {"text": "<text>", "app": "one"})])


# ---- Hindi / Hinglish yes and no -------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["haan", "haan ji", "ji haan", "haa", "ji", "theek hai", "thik hai", "kar do", "bhej do", "haan bhej do",
     "haan, kar do", "Haan Bhej Do!", "ok bhej do", "हाँ", "हां", "जी", "ठीक है", "भेज दो", "हाँ भेज दो", "हाँ, कर दो।",
     "yes", "okay do it"],
)
def test_hindi_and_hinglish_yes(text):
    assert answer_of(text) is True


@pytest.mark.parametrize(
    "text",
    ["haan, mat bhejo", "nahi", "nahin", "mat bhejo", "ruko", "rehne do", "cancel karo", "haan ji ruko", "नहीं",
     "नही", "मत भेजो", "रुको", "रहने दो", "हाँ, मत भेजो", "नहीं।", "haan bhej do... nahi ruko", "yes, actually no"],
)
def test_hindi_and_hinglish_no_wins_over_a_leading_yes(text):
    assert answer_of(text) is False


@pytest.mark.parametrize("text", ["haan, text badal do", "मतलब क्या है", "format the disk", "haan but later"])
def test_neither_yes_nor_no_is_a_new_request(text):
    assert answer_of(text) is None


# ---- the session adopts the cards it has to answer ---------------------------------------------


def test_a_fast_path_card_is_the_sessions_and_its_answer_runs_it_without_a_model(monkeypatch):
    log: list = []
    _messenger(monkeypatch, log)
    brain = Brain([])
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": brain)

    async def go():
        s = Session(reflex=_reflex("quick_action"))
        r = await s.handle("zz message hello", from_user=True)
        p = confirm.peek()
        assert r.route == "quick" and not r.silent and "send “hello” on one" in r.text
        assert (s._pending_action, s._pending_origin) == (p.action_id, "direct")
        assert bus()._jobs.get(r.job) == NeoState.CONFIRMING  # the orb keeps asking
        # The approved call can't reach the chat and offers the other app: a new card, adopted.
        r = await s.handle("haan bhej do", from_user=True)
        p2 = confirm.peek()
        assert r.route == "confirm" and r.text.startswith(confirm.NEEDS_CONFIRM) and "on two" in r.text
        assert p2 is not None and p2.action_id != p.action_id
        assert (s._pending_action, s._pending_origin) == (p2.action_id, "direct")
        r = await s.handle("yes", from_user=True)
        assert r.route == "confirm" and r.text == "Sent on two." and r.silent  # done as approved
        assert confirm.peek() is None and s._pending_action == ""

    asyncio.run(go())
    assert log == [("one", "user"), ("one", "approval"), ("two", "approval")]
    assert brain.calls == 0  # nothing went to a model


def test_a_card_restaged_for_an_agent_stays_the_agents(monkeypatch):
    log: list = []
    _messenger(monkeypatch, log)
    brain = Brain([
        Turn(tool_calls=[ToolCall("1", "zz_msg", {"text": "hello", "app": "one"})]),
        Turn(text="Sent it on two."),
    ])
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": brain)

    async def go():
        s = Session(reflex=_reflex())
        r = await s.handle("tell zz hello", from_user=True)
        assert r.route == "confirm" and s._pending_origin == "agent"
        r = await s.handle("haan", from_user=True)
        assert r.route == "confirm" and "on two" in r.text and s._pending_origin == "agent"
        assert brain.calls == 1  # no follow-up yet: the user hasn't answered the new card
        r = await s.handle("yes", from_user=True)
        assert r.route == "agent" and r.text == "Sent it on two."  # the agent finishes the job

    asyncio.run(go())
    assert log == [("one", "agent"), ("one", "approval"), ("two", "approval")]
    assert brain.calls == 2


def test_something_staged_after_a_fast_path_card_is_never_approved_by_its_yes(monkeypatch):
    log: list = []
    _messenger(monkeypatch, log)

    async def go():
        s = Session(reflex=_reflex("quick_action"))
        await s.handle("zz message hello", from_user=True)
        confirm.stage("trash", {"path": "~/x"}, "move x to the Trash")  # not what the user heard
        return await s.handle("yes", from_user=True)

    r = asyncio.run(go())
    assert "didn't do anything" in r.text and confirm.peek() is None
    assert log == [("one", "user")]


def test_a_card_nobody_answers_lapses_for_fast_paths_too(monkeypatch):
    log: list = []
    _messenger(monkeypatch, log)
    monkeypatch.setattr(confirm, "_TTL_S", 0.05)

    async def go():
        s = Session(reflex=_reflex("quick_action"))
        await s.handle("zz message hello", from_user=True)
        await asyncio.sleep(0.15)
        return s

    s = asyncio.run(go())
    assert confirm.peek() is None and s._pending_action == ""


# ---- ToolOutput.say ---------------------------------------------------------------------------


def test_voiced_rules():
    assert not voiced(ToolOutput("Opened Notes"), quiet=True)
    assert voiced(ToolOutput("Set for 5 PM.", say=True), quiet=True)  # a guessed time is said
    assert not voiced(ToolOutput("Saved.", say=False), quiet=False)  # a habitual success isn't
    assert voiced(ToolOutput("Error: no", ok=False, say=False), quiet=True)  # failures always are
    assert voiced(ToolOutput("NEEDS_CONFIRM: send it — Should I go ahead?", say=False), quiet=True)
    assert voiced(ToolOutput("It's noon."), quiet=False)


@pytest.mark.parametrize(
    "quiet, out, silent",
    [
        (True, ToolOutput("Saved in Notes.", say=True), False),
        (True, ToolOutput("Saved."), True),
        (False, ToolOutput("Saved.", say=False), True),
        (True, ToolOutput(confirm.ask_user("which Priya?"), halt=True, say=False), False),
    ],
)
def test_the_quick_route_honours_say(monkeypatch, quiet, out, silent):
    _add(monkeypatch, "zz_say", lambda a, c: out, quiet=quiet, fast_path=[(r"^zz say$", {})])
    r = asyncio.run(Session(reflex=_reflex("quick_action")).handle("zz say"))
    assert r.route == "quick" and r.silent is silent


def test_an_approved_direct_card_speaks_when_the_tool_says_so(monkeypatch):
    def zz(a, c):
        p = confirm.gated("zz_first", a, "do the first thing")
        return p if isinstance(p, str) else ToolOutput("Done, in an app you haven't used before.", say=True)

    _add(monkeypatch, "zz_first", zz, quiet=True, fast_path=[(r"^zz first$", {})])

    async def go():
        s = Session(reflex=_reflex("quick_action"))
        await s.handle("zz first", from_user=True)
        return await s.handle("yes", from_user=True)

    r = asyncio.run(go())
    assert r.route == "confirm" and not r.silent and "haven't used" in r.text


# ---- origin ------------------------------------------------------------------------------------


def _recorder(monkeypatch, seen: list, **kw):
    def zz_rec(a, c):
        seen.append(c.origin)
        return "recorded"

    _add(monkeypatch, "zz_rec", zz_rec, fast_path=[(r"^zz record$", {})], **kw)


def test_origin_of_a_fast_path_and_of_a_model_call(monkeypatch):
    seen: list = []
    _recorder(monkeypatch, seen)
    asyncio.run(Session(reflex=_reflex("quick_action")).handle("zz record"))
    asyncio.run(Session(reflex=_reflex("quick_action")).command("zz record", [("zz_rec", {})]))
    brain = Brain([Turn(tool_calls=[ToolCall("1", "zz_rec", {})]), Turn(text="ok"), Turn(text="ok")])
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": brain)
    asyncio.run(Session(reflex=_reflex()).handle("please record something for me zz"))
    assert seen == ["user", "user", "agent"]


def test_a_model_call_is_agent_origin_whoever_started_the_run():
    from neo.agent.loop import run_agent

    seen: list = []
    reg = type(registry())()

    def page(a, c):
        seen.append(c.origin)
        return "some page text"

    reg.register(RegisteredTool("zz_page", "t", _OBJ, page, untrusted=True))
    ctx = ToolContext(user_text="x", origin="user")
    brain = Brain([Turn(tool_calls=[ToolCall("1", "zz_page", {})]), Turn(text="done")])
    asyncio.run(run_agent("x", provider=brain, system="", ctx=ctx, reg=reg, ev=EventBus()))
    assert seen == ["agent"]
    assert ctx.taint == ["zz_page"]  # still the caller's taint list: what the run read counts


def test_live_calls_are_live_origin_and_honour_say(monkeypatch):
    import threading
    from types import SimpleNamespace

    import neo.voice.live as live_mod

    seen: list = []

    def zz(a, c):
        seen.append(c.origin)
        return ToolOutput("Set for 5 PM.", say=True)

    _add(monkeypatch, "zz_live", zz, quiet=True)
    sent: list = []

    class FakeLive:
        async def send_tool_response(self, function_responses):
            sent.extend(function_responses)

    async def go():
        live = FakeLive()
        v = live_mod.LiveVoice.__new__(live_mod.LiveVoice)
        v._agent_session, v._tool_tasks, v._goals, v._server_cancelled = None, {}, {}, set()
        v._live, v._ui_lock, v._call_labels, v._taint = live, asyncio.Lock(), {}, []
        v._utt_started, v._last_answer, v._lookups, v._seen_calls, v._recent_idem = 0.0, (0.0, ""), {}, {}, {}
        v._speaking, v.spk = False, SimpleNamespace(playing=threading.Event())
        v._result_pending = v._result_speech = v._turn_open = False
        v._asked = v._voiced_action = None
        await v._run_tool(live, SimpleNamespace(id="c1", name="zz_live", args={}), "")

    asyncio.run(go())
    assert seen == ["live"]
    # A quiet action goes back SILENT; this one asked to be heard.
    assert sent and sent[0].scheduling == live_mod.gt.FunctionResponseScheduling.WHEN_IDLE


def test_the_early_actor_runs_the_users_words(monkeypatch):
    from neo.agent.early import EarlyActor

    seen: list = []
    _recorder(monkeypatch, seen)
    asyncio.run(EarlyActor()._run("zz_rec", {}, 0))
    assert seen == ["user"]
