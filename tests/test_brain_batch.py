"""Brain batch: several actions per model turn stopping at the first that went wrong, no separate
check turn when the model already saw the result, finishing with `finish` after actions NEO
verified, quiet only when nothing was a miss, an evidence-only lean screen prompt and tool list.

Hermetic: fake brains, fake tools in their own registries (never the Mac), no model calls."""

from __future__ import annotations

import asyncio
import re
import sys
import types

import pytest

from neo2.core.agent import batch
from neo2.core.agent.loop import AgentResult, Step, run_agent
from neo2.core.agent.registry import RegisteredTool, Registry, ToolContext, ToolOutput
from neo2.core.events import EventBus
from neo2.core.providers.base import ImagePart, Message, ToolCall, ToolResult, Turn

PNG = ImagePart(b"\x89PNG fake", "image/png")


def _fresh_settings(monkeypatch):
    """batch_actions / replay_flows are Settings fields: read again after the environment changed."""
    from neo2.core import config

    monkeypatch.setattr(config, "_settings", None)


def _eff(changed=True, *, scene=False, sheet="", looked=False, **more):
    e = batch.Effect(changed=changed, scene_changed=scene, sheet=sheet, looked=looked)
    for k, v in more.items():
        setattr(e, k, v)
    return e


def _out(text, *, ok=True, effect=None, images=()):
    o = ToolOutput(text, images=list(images), ok=ok)
    if effect is not None:
        o.effect = effect  # (ToolOutput.effect once Step 0's seam lands; the same attribute before)
    return o


class World:
    """A fake screen app: tools that record what ran and answer as scripted."""

    def __init__(self):
        self.ran: list[tuple[str, dict]] = []
        self.answers: dict[str, list] = {}

    def answer(self, tool, *outs):
        self.answers.setdefault(tool, []).extend(outs)

    def handler(self, name):
        async def h(a, ctx):
            self.ran.append((name, dict(a)))
            q = self.answers.get(name)
            if q:
                return q.pop(0) if len(q) > 1 else q[0]
            return _out(f"{name} ok", effect=_eff(True))

        return h


def _reg(world: World, *, extra=()) -> Registry:
    r = Registry()
    looks = {"ax_tree", "ax_find", "screenshot"}
    for name in ("ax_tree", "ax_find", "screenshot", "click", "drag", "scroll", "type_text", "hotkey", "ax_press",
                 "ax_set_value", "click_text", *extra):  # fmt: skip
        r.register(RegisteredTool(
            name, f"{name} tool", {"type": "object", "properties": {}}, world.handler(name),
            parallel_safe=name in ("ax_tree", "ax_find"), category="computer", untrusted=name in looks,
        ))  # fmt: skip
    return r


class Brain:
    """Replays scripted turns; records every request."""

    name = "fake"
    supports_vision = True
    supports_tools = True

    def __init__(self, *turns, sees=True):
        self.turns = list(turns)
        self.seen: list[list[Message]] = []
        self.systems: list[str] = []
        self.sees = sees

    async def complete(self, messages, *, system="", tools=None, effort="medium", **kw):
        self.seen.append(list(messages))
        self.systems.append(system)
        t = self.turns.pop(0) if self.turns else Turn(text="Done.")
        t.sees = self.sees
        return t

    async def stream(self, messages, **kw):
        yield "x"


def _call(i, name, **args):
    return ToolCall(f"c{i}", name, args)


def _run(reg, brain, **kw):
    tools = [t.spec() for t in reg.all()]
    return asyncio.run(run_agent("do it", provider=brain, system="s", reg=reg, ev=EventBus(), tools=tools, **kw))


def _results(res: AgentResult) -> list[ToolResult]:
    return [r for m in res.messages if m.role == "tool" for r in m.tool_results]


# ---- several actions per turn, stopping at the first that went wrong --------------------------------


@pytest.mark.parametrize(
    "first",
    [
        _out("Clicked. Nothing visibly changed there within 1s — it may have missed", ok=False),
        _out("Clicked.", effect=_eff(False)),  # ok, but NEO saw nothing change
        _out("Clicked.", effect=_eff(True, scene=True, sheet="a sheet “Start a New Game?”")),  # a sheet opened
        _out("Clicked. Changed: a sheet or dialog opened.\nnow showing a sheet"),  # (said in words, no Effect)
        _out("Clicked. Changed: Other is now in front."),
    ],
)
def test_the_actions_after_one_that_went_wrong_are_skipped(first):
    w = World()
    w.answer("click", first, _out("Clicked.", effect=_eff(True)))
    reg = _reg(w)
    b = Brain(Turn(tool_calls=[_call(1, "click", x=1, y=1), _call(2, "click", x=2, y=2), _call(3, "type_text", text="hi")]),
              Turn(text="Stopped to look."))  # fmt: skip
    res = _run(reg, b, verify=False)
    assert [n for n, _ in w.ran] == ["click"]  # the second click and the typing never ran
    rs = _results(res)
    assert [r.content.startswith("skipped: an earlier action in this turn") for r in rs] == [False, True, True]
    assert all(not r.ok for r in rs[1:]) and "look at its result first" in rs[1].content
    # the model saw why before its next step
    assert "skipped: an earlier action" in b.seen[1][-1].tool_results[1].content


def test_a_batch_that_works_runs_every_action_in_order():
    w = World()
    reg = _reg(w)
    b = Brain(Turn(tool_calls=[_call(1, "click", x=1, y=1), _call(2, "click", x=2, y=2), _call(3, "hotkey", keys="cmd+s")]),
              Turn(text="Done."))  # fmt: skip
    res = _run(reg, b, verify=False)
    assert [n for n, _ in w.ran] == ["click", "click", "hotkey"]
    assert all(r.ok for r in _results(res))


def test_a_look_after_the_stop_still_runs():
    w = World()
    w.answer("click", _out("Clicked.", effect=_eff(False)))
    reg = _reg(w)
    b = Brain(Turn(tool_calls=[_call(1, "click", x=1, y=1), _call(2, "click", x=2, y=2), _call(3, "ax_tree")]),
              Turn(text="ok"))  # fmt: skip
    _run(reg, b, verify=False)
    assert [n for n, _ in w.ran] == ["click", "ax_tree"]  # the look changes nothing: it is what's needed next


def test_the_last_action_going_wrong_skips_nothing_and_a_confirmation_still_halts_the_rest():
    from neo2.core.agent import confirm

    w = World()
    w.answer("click", _out("Clicked.", effect=_eff(False)))
    reg = _reg(w)
    res = _run(reg, Brain(Turn(tool_calls=[_call(1, "hotkey", keys="cmd+n"), _call(2, "click", x=1, y=1)]), Turn(text="x")),
               verify=False)  # fmt: skip
    assert [n for n, _ in w.ran] == ["hotkey", "click"]
    assert not any(r.content.startswith("skipped") for r in _results(res))

    w2 = World()
    w2.answer("ax_press", ToolOutput(confirm.Halt("NEEDS_CONFIRM: Press Send?", "act1"), halt=True))
    reg2 = _reg(w2)
    res2 = _run(reg2, Brain(Turn(tool_calls=[_call(1, "ax_press", id="e1"), _call(2, "click", x=1, y=1)])), verify=False)
    assert [n for n, _ in w2.ran] == ["ax_press"] and res2.stopped == "needs_confirm"
    assert _results(res2)[1].content == "skipped: waiting for the user's confirmation"


def test_the_effect_travels_with_the_result_and_the_step():
    w = World()
    eff = _eff(True, looked=True)
    w.answer("click", _out("Clicked.", effect=eff))
    res = _run(_reg(w), Brain(Turn(tool_calls=[_call(1, "click", x=1, y=1)]), Turn(text="Done.")))
    assert _results(res)[0].effect is eff and res.steps[0].effect is eff


# ---- the check turn ----------------------------------------------------------------------------------


def _verify_notes(b: Brain) -> list[Message]:
    return [m for msgs in b.seen for m in msgs[-1:] if m.role == "user" and m.text.startswith("[verify]")]


@pytest.mark.parametrize(
    "tool, out",
    [
        ("click", _out("Clicked.", effect=_eff(True, looked=True), images=[PNG])),  # an after-look came with it
        ("click", _out("Clicked.", images=[PNG])),  # a picture with the result, no Effect yet
        ("ax_set_value", _out("Set e3 to “Arnav”.")),  # read back by the tool itself
        ("ax_press", _out("Pressed e4.", effect=_eff(True))),  # NEO saw the element or window change
    ],
)
def test_no_check_turn_when_the_model_already_saw_the_state_after_its_action(tool, out):
    w = World()
    w.answer(tool, out)
    b = Brain(Turn(tool_calls=[_call(1, tool, id="e3")]), Turn(text="Done."))
    res = _run(_reg(w), b)
    assert res.stopped == "done" and len(b.seen) == 2 and not _verify_notes(b)


@pytest.mark.parametrize(
    "tool, out",
    [
        ("click", _out("Clicked. Changed: the screen changed around the point.")),  # pixels, no picture
        ("ax_press", _out("Pressed e4.")),  # no Effect: NEO couldn't tell
        ("ax_press", _out("Pressed e4.", effect=_eff(True, scene=True, sheet="a sheet"))),  # a sheet to answer
        ("hotkey", _out("Pressed cmd+n.", effect=_eff(True))),  # a keystroke: nothing checked
    ],
)
def test_a_check_turn_when_it_hasnt(tool, out):
    w = World()
    w.answer(tool, out)
    b = Brain(Turn(tool_calls=[_call(1, tool, id="e4")]), Turn(text="Done."), Turn(text="Done, checked."))
    res = _run(_reg(w), b)
    assert res.text == "Done, checked." and len(_verify_notes(b)) == 1


@pytest.fixture
def fake_observe(monkeypatch):
    """neo2.core.tools.computer.observe as the eyes add it: one read-only look (recorded here)."""
    calls = []

    async def observe(app=None, *, image=False, deadline_s=1.5, ctx=None):
        calls.append({"app": app, "image": image, "deadline_s": deadline_s, "blind": bool(ctx.get("blind")),
                      "origin": ctx.origin})  # fmt: skip
        ctx.taint.append("screen")  # (the real look taints the request like the tool does)
        text = 'Look (read-only):\n<untrusted source="screen">\napp: Boardy\n</untrusted>\ne1 Button "New Game"'
        return ToolOutput(text, images=[PNG] if image and not ctx.get("blind") else [], fence=False)

    mod = types.ModuleType("neo2.core.tools.computer.observe")
    mod.observe = observe
    monkeypatch.setitem(sys.modules, "neo2.core.tools.computer.observe", mod)
    import neo2.core.tools.computer as comp

    monkeypatch.setattr(comp, "observe", mod, raising=False)
    return calls


def test_the_check_turn_carries_a_look_so_it_costs_one_round_trip(fake_observe):
    w = World()
    w.answer("click", _out("Clicked. Changed: the screen changed around the point."))
    b = Brain(Turn(tool_calls=[_call(1, "click", x=5, y=5, shot="s1")]), Turn(text="Moved."), Turn(text="Moved e2 to e4."))
    ctx = ToolContext(user_text="move")
    tools = [t.spec() for t in _reg(w).all()]
    res = asyncio.run(run_agent("move", provider=b, system="s", reg=_reg(w), ev=EventBus(), tools=tools, ctx=ctx))
    [note] = _verify_notes(b)
    assert "NEO looked at the screen after your last action" in note.text and 'e1 Button "New Game"' in note.text
    assert note.images == [PNG]  # the last action was at pixels and the brain sees: a picture too
    assert fake_observe == [{"app": None, "image": True, "deadline_s": batch.LOOK_S, "blind": False, "origin": "agent"}]
    assert "screen" in ctx.taint  # attached, so read: the request is tainted like the tool would
    assert res.text == "Moved e2 to e4." and len(b.seen) == 3  # was 4: a look turn, then the answer


def test_a_blind_brains_check_gets_the_list_never_a_picture(fake_observe):
    w = World()
    w.answer("ax_press", _out("Pressed e4."))
    b = Brain(Turn(tool_calls=[_call(1, "ax_press", id="e4")]), Turn(text="ok"), Turn(text="ok"), sees=False)
    _run(_reg(w), b)
    [note] = _verify_notes(b)
    assert not note.images and fake_observe[0]["blind"] and not fake_observe[0]["image"]


def test_no_look_for_a_check_after_an_action_off_the_screen_or_without_a_look_tool(fake_observe):
    w = World()
    reg = _reg(w, extra=("notes_create",))
    reg.get("notes_create").category = "apps"
    b = Brain(Turn(tool_calls=[_call(1, "notes_create", title="x")]), Turn(text="ok"), Turn(text="ok"))
    _run(reg, b)
    assert _verify_notes(b)[0].text == batch.VERIFY_NOTE and not fake_observe
    # a run that was never given a way to look gets no look from NEO either
    w2 = World()
    reg2 = _reg(w2)
    b2 = Brain(Turn(tool_calls=[_call(1, "click", x=1, y=1)]), Turn(text="ok"), Turn(text="ok"))
    asyncio.run(run_agent("x", provider=b2, system="s", reg=reg2, ev=EventBus(), tools=[reg2.get("click").spec()], strict=True))
    assert _verify_notes(b2)[0].text == batch.VERIFY_NOTE and not fake_observe


def test_a_look_that_fails_leaves_the_plain_note(monkeypatch):
    async def observe(app=None, **kw):
        raise TimeoutError

    mod = types.ModuleType("neo2.core.tools.computer.observe")
    mod.observe = observe
    monkeypatch.setitem(sys.modules, "neo2.core.tools.computer.observe", mod)
    w = World()
    w.answer("click", _out("Clicked."))
    b = Brain(Turn(tool_calls=[_call(1, "click", x=1, y=1)]), Turn(text="ok"), Turn(text="ok"))
    _run(_reg(w), b)
    assert _verify_notes(b)[0].text == batch.VERIFY_NOTE


# ---- finishing without one more model call ------------------------------------------------------------


def _fin(i=99, answer="Filled in your name."):
    return _call(i, "finish", answer=answer)


def _finish_run(*outs, calls, text="", then=(), **kw):
    w = World()
    for tool, out in outs:
        w.answer(tool, out)
    b = Brain(Turn(text=text, tool_calls=calls), *then, Turn(text="(the model again)"), Turn(text="(and again)"))
    reg = _reg(w)
    tools = [t.spec() for t in reg.all()]
    res = asyncio.run(run_agent("do it", provider=b, system="s", reg=reg, ev=EventBus(), tools=tools, **kw))
    return res, b, w


def test_finish_with_actions_neo_verified_ends_the_run_without_another_model_call():
    res, b, w = _finish_run(
        ("ax_set_value", _out("Set e3 to “Arnav”.")), ("ax_press", _out("Pressed e9.", effect=_eff(True))),
        calls=[_call(1, "ax_set_value", id="e3", value="Arnav"), _call(2, "ax_press", id="e9"), _fin()],
        finish_verified=True,
    )  # fmt: skip
    assert res.stopped == "done" and res.text == "Filled in your name." and len(b.seen) == 1
    assert [n for n, _ in w.ran] == ["ax_set_value", "ax_press"]  # `finish` is never a tool that runs
    assert res.messages[-1].role == "assistant" and res.messages[-1].text == "Filled in your name."
    assert res.messages[-2].role == "tool" and len(res.messages[-2].tool_results) == 3  # every call answered
    assert [st.tool for st in res.steps] == ["ax_set_value", "ax_press"]  # the ending is not a step


class _ToolsSeen(Brain):
    def __init__(self, *turns):
        super().__init__(*turns)
        self.offered: list[list[str]] = []

    async def complete(self, messages, *, tools=None, **k):
        self.offered.append([t.name for t in tools or []])
        return await super().complete(messages, tools=tools, **k)


@pytest.mark.parametrize(
    "kw, offered", [({"finish_verified": True}, True), ({}, False), ({"finish_verified": True, "strict": True}, False)]
)
def test_finish_is_offered_only_to_a_run_that_may_end_that_way(kw, offered):
    reg = _reg(World())
    b = _ToolsSeen(Turn(text="ok"))
    asyncio.run(run_agent("x", provider=b, system="s", reg=reg, ev=EventBus(), tools=[t.spec() for t in reg.all()], **kw))
    assert ("finish" in b.offered[0]) is offered


@pytest.mark.parametrize(
    "outs, calls",
    [
        # a click at pixels: NEO can't verify it through Accessibility
        ([("click", _out("Clicked.", effect=_eff(True, looked=True), images=[PNG]))], [_call(1, "click", x=1, y=1)]),
        # a press whose effect NEO couldn't tell
        ([("ax_press", _out("Pressed."))], [_call(1, "ax_press", id="e9")]),
        # a sheet opened: there is more to do
        ([("ax_press", _out("Pressed.", effect=_eff(True, scene=True, sheet="a sheet")))], [_call(1, "ax_press", id="e9")]),
        # a look in the same reply: the answer was written before its result
        ([("ax_press", _out("Pressed.", effect=_eff(True)))], [_call(1, "ax_press", id="e9"), _call(2, "ax_tree")]),
        # a failed action, and the one after it skipped
        ([("ax_press", _out("Error: gone", ok=False))], [_call(1, "ax_press", id="e9"), _call(2, "ax_press", id="e8")]),
    ],
)  # fmt: skip
def test_finish_is_refused_when_neo_couldnt_verify_every_action(outs, calls):
    res, b, _w = _finish_run(*outs, calls=[*calls, _fin()], finish_verified=True)
    assert len(b.seen) >= 2 and res.text != "Filled in your name."
    [r] = [r for m in b.seen[1] if m.role == "tool" for r in m.tool_results if r.name == "finish"]
    assert r.content == batch.NOT_FINISHED  # the model hears why and carries on


def test_a_narrating_model_never_ends_a_run_by_its_words():
    res, b, _w = _finish_run(("ax_press", _out("Pressed.", effect=_eff(True))),
                             calls=[_call(1, "ax_press", id="e9")], text="I'll press Start, then Continue.",
                             finish_verified=True)  # fmt: skip
    assert len(b.seen) == 2 and res.text == "(the model again)"


def test_finish_alone_is_the_answer_after_the_same_check_as_text():
    # after an action NEO couldn't verify, `finish` alone gets the check note first, like text would
    res, b, _w = _finish_run(("hotkey", _out("Pressed cmd+s.")), calls=[_call(1, "hotkey", keys="cmd+s")],
                             then=[Turn(tool_calls=[_fin(5, "Saved.")]), Turn(tool_calls=[_fin(6, "Saved, checked.")])],
                             finish_verified=True)  # fmt: skip
    assert res.text == "Saved, checked." and len(_verify_notes(b)) == 1
    assert res.messages[-1].text == "Saved, checked." and res.messages[-2].tool_results[0].content == "Noted."


# ---- quiet only when nothing was a miss ------------------------------------------------------------------


def _d(reply=False):
    from neo2.core.reflex.schema import Decision

    return Decision("agent_task", 0.95, False, 0.0, False, 0.0, "laya+head", reply=reply, reply_p=0.1)


def test_quiet_requires_that_no_step_missed():
    from neo2.core.agent.session import _quietly

    ok = Step("ax_press", {"id": "e1"}, "Pressed e1.", True, _eff(True))
    miss = Step("click_text", {"label": "Start"}, "Clicked “Start”.", True, _eff(False))
    said = Step("ax_press", {"id": "e1"}, "Pressed e1.", True, _eff(False, missed=True))
    assert _quietly(_d(), AgentResult("Done.", "done", [ok]))
    assert not _quietly(_d(), AgentResult("Done.", "done", [miss]))  # "ok", but the click changed nothing
    assert not _quietly(_d(), AgentResult("Done.", "done", [said]))  # its Effect says it missed
    skipped = Step("hotkey", {}, "skipped: an earlier action in this turn changed the screen — look", False)
    assert _quietly(_d(), AgentResult("Done.", "done", [ok, skipped, ok]))  # a skip never ran: not a failure


def test_the_outcome_ledger_hears_misses_and_not_skips(monkeypatch):
    from neo2.core.agent import session as sess
    from neo2.core.reflex import outcomes

    got = {}
    monkeypatch.setattr(outcomes, "agent_done", lambda job, steps, **kw: got.setdefault("steps", steps))
    res = AgentResult("Done.", "done", [
        Step("click", {"x": 1, "y": 1}, "Clicked.", True, _eff(False)),
        Step("click", {}, "skipped: an earlier action in this turn changed nothing — look", False),
        Step("hotkey", {}, "Pressed cmd+n.", True),
    ])  # fmt: skip
    sess._learn_from("j1", res)
    assert got["steps"] == [("click", False), ("hotkey", True)]


def test_after_a_yes_done_means_nothing_changed_nothing(monkeypatch):
    """Session.answer's follow-up run: silent only when it worked, Effect included."""
    from neo2.core.agent.session import _worked

    assert _worked(AgentResult("x", "done", [Step("click", {}, "ok", True, _eff(True))]))
    assert not _worked(AgentResult("x", "done", [Step("click", {}, "ok", True, _eff(False))]))


# ---- the prompt ---------------------------------------------------------------------------------------

# The bench's own test for "the prompt invites several actions per reply" (bench/controls/scenarios.py)
_BATCH_WORDS = re.compile(r"\b(?:several|multiple|a batch of)\s+(?:actions|tool calls|calls|steps)\s+in\s+one\s+(?:turn|reply|step)", re.I)


def test_no_app_or_game_is_named_in_any_rule():
    from neo2.core.agent import prompt

    text = prompt.PERSONA + "".join(prompt._SCREEN_STEPS) + prompt.SCREEN_IDENTITY
    for name in ("Chess", "chess", "Safari", "Chrome", "Finder", "Notes", "Mail"):
        assert not re.search(rf"\b{name}\b", text), name
    assert "a board, a canvas, a map, a picture" in text  # the evidence the canvas rule goes by


def test_several_actions_per_reply_are_invited_only_with_the_setting(monkeypatch):
    from neo2.core.agent import prompt

    monkeypatch.delenv("NEO_BATCH_ACTIONS", raising=False)
    _fresh_settings(monkeypatch)
    assert not batch.enabled() and not _BATCH_WORDS.search(prompt.persona())
    assert "look, make one move, then check its result" in prompt.persona()
    monkeypatch.setenv("NEO_BATCH_ACTIONS", "1")
    _fresh_settings(monkeypatch)
    p = " ".join(prompt.persona().split())
    assert batch.enabled() and _BATCH_WORDS.search(p) and "make one move" not in p
    assert "stops at the first that fails, changes nothing or changes the screen" in p


def test_an_embedded_agent_never_reads_the_batch_switch_from_the_environment(monkeypatch, tmp_path):
    from neo_lib import make_runtime

    from neo2.core import runtime

    monkeypatch.setenv("NEO_BATCH_ACTIONS", "1")
    _fresh_settings(monkeypatch)
    rt = make_runtime(tmp_path / "a", "a", ["click", "screenshot"])
    with runtime.bind(rt.scope()):
        assert not batch.enabled()


def test_the_lean_screen_prompt_keeps_every_safety_rule_and_only_its_tools_guidance(monkeypatch):
    from neo2.core.agent import prompt

    monkeypatch.setattr(prompt, "system_context", lambda: "\n[CONTEXT] FIXED")
    full = prompt.build()
    lean = prompt.build(screen_tools=["memory", "ax_tree", "screenshot", "click", "ax_press", "type_text", "hotkey"])
    assert lean.startswith(prompt.SCREEN_IDENTITY) and prompt.SAFETY_RULES.strip() not in ("",)
    for line, about in prompt._HOW_TO_WORK:
        if about is None:
            assert line in lean  # every safety rule, word for word
    assert "search_site" not in lean and "System Events" not in lean and "Play <game>" not in lean
    assert "a board, a canvas" in lean and "Accessibility is off" in lean
    assert len(lean) < len(full) * 0.85


def test_a_hosts_own_persona_stays_on_a_screen_run(tmp_path):
    from neo_lib import make_runtime

    from neo2.core import runtime
    from neo2.core.agent import prompt

    rt = make_runtime(tmp_path / "p", "p", ["click", "screenshot"], persona="You are Acme's desk helper.")
    with runtime.bind(rt.scope()):
        text = prompt.build(screen_tools=["click", "screenshot", "web_search"])
    assert text.startswith("You are Acme's desk helper.") and "web_search" not in text


# ---- the lean screen tool list --------------------------------------------------------------------------


def _catalog_like() -> Registry:
    from neo2.core.agent.toolselect import SCREEN_CORE

    r = Registry()

    def t(name, desc, cat, safe=False):
        r.register(RegisteredTool(name, desc, {"type": "object", "properties": {}}, lambda a, c: name,
                                  parallel_safe=safe, category=cat))  # fmt: skip

    t("memory", "Long-term memory: remember and recall facts.", "memory", True)
    for n in SCREEN_CORE:
        if n not in ("look", "menu_command"):
            t(n, f"{n.replace('_', ' ')} in an app's window", "computer")
    t("dictate", "Write text from a brief and type it where the cursor is", "computer")
    t("apps_running", "List the running apps", "apps", True)
    t("shell", "Run a shell command to move files and more", "files")
    t("trash", "Move files to the trash", "files")
    for n in ("browser_open", "browser_read", "browser_click", "browser_scroll", "browser_type"):
        t(n, f"{n.replace('_', ' ')} down the page in NEO's browser", "browser")
    t("web_search", "Search the web", "web", True)
    return r


def test_a_screen_run_gets_the_lean_core_and_never_shell_trash_or_dictate_unasked(monkeypatch):
    from neo2.core.agent import toolselect

    monkeypatch.setattr(toolselect, "_names_an_app", lambda text: False)
    sel = toolselect.ToolSelector(_catalog_like()).select("move the file on the board and scroll", needs_screen=True)
    assert {"ax_tree", "click", "click_text", "ax_press", "ax_set_value", "type_text", "hotkey", "scroll", "drag"} <= set(sel.names)
    assert not {"shell", "trash", "dictate"} & set(sel.names)
    assert len(sel.names) <= toolselect._MAX_SCREEN_TOOLS


def test_asked_for_by_name_they_come_in(monkeypatch):
    from neo2.core.agent import toolselect

    monkeypatch.setattr(toolselect, "_names_an_app", lambda text: False)
    sel = toolselect.ToolSelector(_catalog_like()).select("run a shell command", needs_screen=True)
    assert "shell" in sel.names
    sel = toolselect.ToolSelector(_catalog_like()).select("scroll", needs_screen=True, required=("trash",))
    assert "trash" in sel.names


def test_neo_s_own_browser_is_left_out_when_the_request_names_an_app_on_this_mac(monkeypatch):
    from neo2.core.agent import toolselect

    reg = _catalog_like()
    monkeypatch.setattr(toolselect, "_names_an_app", lambda text: True)
    sel = toolselect.ToolSelector(reg).select("scroll down in its window", needs_screen=False)
    assert "scroll" in sel.names and not [n for n in sel.names if n.startswith("browser_")]
    monkeypatch.setattr(toolselect, "_names_an_app", lambda text: False)
    sel = toolselect.ToolSelector(reg).select("scroll down in its window", needs_screen=False)
    assert "browser_scroll" in sel.names  # no app named: NEO's browser may be what's meant


def test_names_an_app_reads_only_the_last_scan(monkeypatch):
    from neo2.core.agent import toolselect
    from neo2.core.decide import inventory

    monkeypatch.setattr(inventory, "snapshot", lambda: None)  # never scanned: can't tell, nothing left out
    assert not toolselect._names_an_app("scroll down in Boardy")
    app = types.SimpleNamespace(name="Boardy")
    monkeypatch.setattr(inventory, "snapshot", lambda: [app])
    monkeypatch.setattr(inventory, "named_span", lambda text, apps: (app, 15, 21) if "boardy" in text.lower() else None)
    assert toolselect._names_an_app("scroll down in Boardy") and not toolselect._names_an_app("scroll down")
    neo_app = types.SimpleNamespace(name="NEO")
    monkeypatch.setattr(inventory, "named_span", lambda text, apps: (neo_app, 0, 3))
    assert not toolselect._names_an_app("scroll down in neo's browser")  # NEO's own browser is meant


def test_a_run_off_the_screen_keeps_its_families():
    from neo2.core.agent import toolselect

    sel = toolselect.ToolSelector(_catalog_like()).select("search the web and open the page in the browser")
    assert {"browser_open", "browser_read", "browser_click"} <= set(sel.names)
    assert len(sel.names) <= toolselect._MAX_TOOLS


# ---- stale pictures -----------------------------------------------------------------------------------


def test_an_after_look_goes_stale_once_a_newer_picture_is_in_the_run():
    from neo2.core.providers.history import OBSERVATIONS

    old = ToolResult("a", "click", "Clicked.", images=[PNG])
    shot = ToolResult("b", "screenshot", "Screenshot s2.", images=[PNG])
    new = ToolResult("c", "click", "Clicked again.", images=[PNG])
    msgs = [
        Message.user("go"),
        Message.assistant("", [ToolCall("a", "click", {})]), Message.tool([old]),
        Message.assistant("", [ToolCall("b", "screenshot", {})]), Message.tool([shot]),
        Message.assistant("", [ToolCall("c", "click", {})]), Message.tool([new]),
    ]  # fmt: skip
    batch.drop_stale_after_looks(msgs, 0, OBSERVATIONS)
    assert not old.images and old.content == "Clicked."  # its words stay, its pixels go
    assert shot.images and new.images  # a look is the observations' business; the newest stays
    kept = ToolResult("d", "click", "Clicked.", images=[PNG])
    history = [Message.user("before"), Message.assistant("", [ToolCall("d", "click", {})]), Message.tool([kept])]
    batch.drop_stale_after_looks([*history, Message.user("now", images=[PNG])], len(history), OBSERVATIONS)
    assert kept.images  # before this run: not this run's to trim


# ---- review fixes ---------------------------------------------------------------------------------------


class _Events(EventBus):
    """A host's view of the run: every tool_start / tool_end it is shown."""

    def __init__(self):
        super().__init__()
        self.shown: list[tuple[str, str]] = []

    async def tool_start(self, name, args, *, job=""):
        self.shown.append(("start", name))
        await super().tool_start(name, args, job=job)

    async def tool_end(self, name, ok, summary, *, job=""):
        self.shown.append(("end", name))
        await super().tool_end(name, ok, summary, job=job)


def test_the_verify_look_never_sends_a_picture_to_a_run_without_screenshot_or_look(fake_observe):
    # the host left screenshot out: the check's look is the list only, never a picture of the screen
    w = World()
    w.answer("click", _out("Clicked. Changed: the screen changed around the point."))
    reg = _reg(w)
    b = Brain(Turn(tool_calls=[_call(1, "click", x=5, y=5)]), Turn(text="Moved."), Turn(text="Moved, checked."))
    tools = [reg.get("ax_tree").spec(), reg.get("click").spec()]
    asyncio.run(run_agent("x", provider=b, system="s", reg=reg, ev=EventBus(), tools=tools))
    [note] = _verify_notes(b)
    assert not note.images and fake_observe and fake_observe[0]["image"] is False
    assert "NEO looked at the screen" in note.text  # a text look: the run was given ax_tree


def test_the_check_turns_look_is_shown_to_the_host_like_any_tool(fake_observe):
    w = World()
    w.answer("click", _out("Clicked. Changed: the screen changed around the point."))
    reg = _reg(w)
    ev = _Events()
    b = Brain(Turn(tool_calls=[_call(1, "click", x=5, y=5, shot="s1")]), Turn(text="Moved."), Turn(text="ok"))
    asyncio.run(run_agent("x", provider=b, system="s", reg=reg, ev=ev, tools=[t.spec() for t in reg.all()]))
    assert ev.shown == [("start", "click"), ("end", "click"), ("start", "look"), ("end", "look")]


def test_the_check_turns_look_goes_through_a_registered_look_tool(fake_observe):
    from neo2.core.agent import grants

    w = World()
    w.answer("click", _out("Clicked. Changed: the screen changed around the point."))
    reg = _reg(w, extra=("look",))
    got = []

    async def look(a, ctx):
        got.append((dict(a), ctx.origin, bool(ctx.get("blind"))))
        return ToolOutput("Look (read-only):\ne1 Button", images=[PNG], fence=False)

    reg.get("look").handler = look
    reg.get("look").untrusted = True
    b = Brain(Turn(tool_calls=[_call(1, "click", x=5, y=5, shot="s1")]), Turn(text="Moved."), Turn(text="ok"))
    ctx = ToolContext(user_text="x")
    with grants.task(tid := grants.new_task()):
        asyncio.run(run_agent("x", provider=b, system="s", reg=reg, ev=EventBus(), ctx=ctx,
                              tools=[t.spec() for t in reg.all()]))  # fmt: skip
        reads = grants._book().reads[tid]
    grants.end(tid)
    [note] = _verify_notes(b)
    assert got == [({"image": True}, "agent", False)] and not fake_observe  # the run's own look tool
    assert note.images == [PNG] and "e1 Button" in note.text
    assert "look" in ctx.taint and reads >= 1  # read like the tool: taint, and a new reading for grants


def test_a_command_that_rightly_changes_nothing_is_no_miss(monkeypatch):
    """Effect.changed False from a command (a menu's Copy: nothing anyone sees changes) is no miss;
    from a pointer input (the eyes: the move didn't land) it is; an Effect that says it missed is."""
    from neo2.core.agent import session as sess
    from neo2.core.reflex import outcomes

    copy = Step("menu_command", {"path": "Edit > Copy"}, "Chose Edit > Copy.", True, _eff(False))
    assert sess._worked(AgentResult("Done.", "done", [copy]))
    assert sess._quietly(_d(), AgentResult("Done.", "done", [copy]))
    r = ToolResult("c1", "menu_command", "Chose Edit > Copy.")
    r.effect = _eff(False)
    assert batch.stop_reason(r) == ""
    click = Step("click", {"x": 1, "y": 1}, "Clicked.", True, _eff(False))
    assert not sess._worked(AgentResult("Done.", "done", [click]))
    said = Step("hotkey", {"keys": "cmd+n"}, "Pressed cmd+n.", True, _eff(False, missed=True))
    assert not sess._worked(AgentResult("Done.", "done", [said]))
    got = {}
    monkeypatch.setattr(outcomes, "agent_done", lambda job, steps, **kw: got.setdefault("steps", steps))
    sess._learn_from("j1", AgentResult("Done.", "done", [copy, click]))
    assert got["steps"] == [("menu_command", True), ("click", False)]


def test_a_command_that_changed_nothing_does_not_stop_the_batch():
    w = World()
    w.answer("menu_command", _out("Chose Edit > Copy.", effect=_eff(False)))
    reg = _reg(w, extra=("menu_command", "activate_app"))
    b = Brain(Turn(tool_calls=[_call(1, "menu_command", path="Edit > Copy"), ToolCall("c2", "activate_app", {"name": "Other"}),
                               _call(3, "hotkey", keys="cmd+v")]), Turn(text="Done."))  # fmt: skip
    _run(reg, b, verify=False)
    assert [n for n, _ in w.ran] == ["menu_command", "activate_app", "hotkey"]


def test_finish_with_duplicate_call_ids_pairs_each_result_with_its_own_call():
    # a local server numbering every call "call_0": each result still belongs to its own call
    w = World()
    w.answer("ax_press", _out("Pressed e1.", effect=_eff(None)), _out("Pressed e2.", effect=_eff(True)))
    reg = _reg(w)
    calls = [ToolCall("call_0", "ax_press", {"id": "e1"}), ToolCall("call_0", "ax_press", {"id": "e2"}),
             ToolCall("call_0", "finish", {"answer": "Pressed both."})]  # fmt: skip
    b = Brain(Turn(tool_calls=calls), Turn(text="(the model again)"), Turn(text="(and again)"))
    res = asyncio.run(run_agent("x", provider=b, system="s", reg=reg, ev=EventBus(),
                                tools=[t.spec() for t in reg.all()], finish_verified=True))  # fmt: skip
    assert res.text != "Pressed both." and len(b.seen) >= 2
    rs = b.seen[1][-1].tool_results
    assert [r.content for r in rs] == ["Pressed e1.", "Pressed e2.", batch.NOT_FINISHED]
    assert [(st.tool, st.args, st.result) for st in res.steps][:2] == [
        ("ax_press", {"id": "e1"}, "Pressed e1."), ("ax_press", {"id": "e2"}, "Pressed e2."),
    ]  # fmt: skip


def test_a_verified_last_action_does_not_skip_the_check_for_an_unverified_click_before_it():
    w = World()
    w.answer("click", _out("Clicked at (5, 5)."))  # no Effect: NEO couldn't tell
    w.answer("ax_set_value", _out("Set e2 to “Arnav”."))  # read back: verified
    reg = _reg(w)
    b = Brain(Turn(tool_calls=[_call(1, "click", x=5, y=5), _call(2, "ax_set_value", id="e2", value="Arnav")]),
              Turn(text="Done."), Turn(text="Done, checked."))  # fmt: skip
    res = _run(reg, b)
    assert len(_verify_notes(b)) == 1 and res.text == "Done, checked."


def test_finish_is_refused_while_an_earlier_action_was_never_seen():
    res, b, _w = _finish_run(
        ("click", _out("Clicked at (5, 5).")), ("ax_set_value", _out("Set e2 to “Arnav”.")),
        calls=[_call(1, "click", x=5, y=5)],
        then=[Turn(tool_calls=[_call(2, "ax_set_value", id="e2", value="Arnav"), _fin(3)])],
        finish_verified=True,
    )  # fmt: skip
    assert res.text != "Filled in your name." and len(b.seen) >= 3
    [r] = [r for r in b.seen[2][-1].tool_results if r.name == "finish"]
    assert r.content.startswith("Not finished")
    # …while a look since then covers it
    res, b, _w = _finish_run(
        ("click", _out("Clicked at (5, 5).")), ("ax_set_value", _out("Set e2 to “Arnav”.")),
        calls=[_call(1, "click", x=5, y=5), _call(2, "ax_tree")],
        then=[Turn(tool_calls=[_call(3, "ax_set_value", id="e2", value="Arnav"), _fin(4)])],
        finish_verified=True,
    )  # fmt: skip
    assert res.text == "Filled in your name." and len(b.seen) == 2


def test_a_refused_finish_carries_neos_look_so_the_check_costs_one_round_trip(fake_observe):
    res, b, _w = _finish_run(
        ("click", _out("Clicked at (5, 5).")), ("ax_set_value", _out("Set e2 to “Arnav”.")),
        calls=[_call(1, "click", x=5, y=5, shot="s1")],
        then=[Turn(tool_calls=[_call(2, "ax_set_value", id="e2", value="Arnav"), _fin(3)]), Turn(text="Done, checked.")],
        finish_verified=True,
    )  # fmt: skip
    [r] = [r for r in b.seen[2][-1].tool_results if r.name == "finish"]
    assert r.content.startswith(batch.NOT_FINISHED_LOOKED) and 'e1 Button "New Game"' in r.content
    assert r.images == [PNG] and fake_observe[0]["image"] is True  # the unseen click was at pixels
    assert res.text == "Done, checked." and len(b.seen) == 3 and not _verify_notes(b)  # no second check


def test_the_script_rule_comes_with_applescript_when_more_tools_brings_it_in(monkeypatch):
    """A lean screen prompt has no guidance for applescript; when the model pulls it in mid-run, the
    rule ("never drive an app's window through System Events scripts") arrives with it."""
    from neo2.core.agent import prompt, toolselect

    w = World()
    reg = _reg(w, extra=("applescript",))

    class Sel:
        def search(self, query, exclude=None, k=6):
            return [reg.get("applescript")]

    monkeypatch.setattr(toolselect, "selector", lambda: Sel())
    lean = prompt.build(screen_tools=["ax_tree", "screenshot", "click"])
    assert "System Events" not in lean
    b = Brain(Turn(tool_calls=[_call(1, "more_tools", query="applescript")]), Turn(text="ok"))
    tools = [reg.get(n).spec() for n in ("ax_tree", "screenshot", "click")]
    asyncio.run(run_agent("x", provider=b, system=lean, reg=reg, ev=EventBus(), tools=tools))
    [r] = b.seen[1][-1].tool_results
    assert r.content.startswith("Enabled: applescript.") and "System Events" in r.content
    # a full prompt already has the line: nothing repeated
    b2 = Brain(Turn(tool_calls=[_call(1, "more_tools", query="applescript")]), Turn(text="ok"))
    asyncio.run(run_agent("x", provider=b2, system=prompt.build(), reg=reg, ev=EventBus(), tools=tools))
    assert "System Events" not in b2.seen[1][-1].tool_results[0].content


def test_click_text_pressed_and_watched_through_accessibility_is_verified():
    """click_text that pressed its control through Accessibility and saw it change checked its own
    work, like ax_press: no check turn for it (its Effect says so). Clicked with the mouse (no
    Effect), or at words read off the screen (the pointer's outcome: a change at the point), it
    didn't."""
    watched = ToolResult("c1", "click_text", "Clicked 'Start' (Button)")
    watched.effect = _eff(True, verified=True)
    assert batch.verified("click_text", watched)
    mouse = ToolResult("c2", "click_text", "Clicked 'Start' (Button) at (10, 20)")
    assert not batch.verified("click_text", mouse)
    read = ToolResult("c4", "click_text", "Clicked the words 'Start' — found by reading the screen.")
    read.effect = _eff(True)
    assert not batch.verified("click_text", read) and not batch.seen("click_text", read)
    sheet = ToolResult("c3", "click_text", "Clicked 'New…' (Button)")
    sheet.effect = _eff(True, sheet="“Save?”", verified=True)
    assert not batch.verified("click_text", sheet)  # (a sheet opened: the model answers it first)
