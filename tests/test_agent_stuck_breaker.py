"""The stuck-call breaker (neo2.core.agent.failures, wired in neo2.core.agent.loop).

Within one agent run, an action that failed twice the same way gets one hint by kind of failure,
and a third identical call isn't run: it is answered "skipped: …" and the run goes on. Looks in
between don't make it a new attempt (the thrash guard misses exactly that: core.log 4080-4101, a
click by label failed again and again with a search between the tries); an action that worked
clears the count, since the screen moved on. NEO_STUCK_BREAKER=false is the loop as it was.

Every run also ends on one content-free line, `[loop] end=<why> steps=N s=…`, and a failed tool's
kind is in the tool log and the daily counters — numbers and fixed names only.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from neo2.core.agent import failures
from neo2.core.agent.loop import run_agent
from neo2.core.agent.registry import RegisteredTool, Registry, ToolOutput
from neo2.core.events import Event, EventBus
from neo2.core.providers.base import Message, ToolCall, Turn

_OBJ = {"type": "object", "properties": {}}
# NEO's own words when a label names nothing in the window (neo2.core.tools.computer.tools_label).
_MISSING = "Error: nothing in the front window of Google Chrome is called 'Add to cart'."


@pytest.fixture(autouse=True)
def _breaker_default(monkeypatch):
    """Today's setting, whatever the environment says: a test that wants it off says so."""
    from neo2.core import config

    monkeypatch.delenv("NEO_STUCK_BREAKER", raising=False)
    monkeypatch.setattr(config, "_settings", None)


def _off(monkeypatch):
    from neo2.core import config

    monkeypatch.setenv("NEO_STUCK_BREAKER", "false")
    monkeypatch.setattr(config, "_settings", None)


class Brain:
    """A scripted model: its replies in order (an exception is raised instead), and every
    transcript it was shown."""

    name = "scripted"
    supports_vision = False
    supports_tools = True

    def __init__(self, *turns):
        self.turns = list(turns)
        self.seen: list[list[Message]] = []

    async def complete(self, messages, **kw):
        self.seen.append(list(messages))
        t = self.turns.pop(0)
        if isinstance(t, Exception):
            raise t
        return t

    async def stream(self, messages, **kw):
        yield ""


class World:
    """The tools a run may call, and every call that reached one."""

    def __init__(self):
        self.reg = Registry()
        self.ran: list[tuple[str, dict]] = []

    def tool(self, name, answer, *, look=False, category="computer"):
        def handler(a, c):
            self.ran.append((name, dict(a)))
            return answer(a) if callable(answer) else answer

        self.reg.register(RegisteredTool(name, name, _OBJ, handler, parallel_safe=look, category=category))

    def calls(self, name):
        return [a for n, a in self.ran if n == name]


def _call(i, name, **args):
    return Turn(tool_calls=[ToolCall(str(i), name, args)])


def _world():
    w = World()
    w.tool(
        "click_text",
        lambda a: _MISSING if a.get("label") == "Add to cart" else f"Clicked '{a['label']}' (Button)",
    )
    w.tool("web_search", "1. a result", look=True, category="web")
    w.tool("scroll", "Scrolled down.")
    return w


def _results(res):
    return [r for m in res.messages if m.role == "tool" for r in m.tool_results]


def _run(w, brain, **kw):
    kw.setdefault("verify", False)
    return asyncio.run(
        run_agent(
            "add it to the cart", provider=brain, system="s", reg=w.reg, ev=kw.pop("ev", EventBus()), **kw
        )
    )


# ---- the log's run: a failing click with looks between the tries -----------------------------------


def _the_logs_run():
    return Brain(
        _call(1, "click_text", label="Add to cart"),
        _call(2, "web_search", query="mug"),
        _call(3, "click_text", label="Add to cart"),
        _call(4, "web_search", query="mug price"),
        _call(5, "click_text", label="Add to cart"),
        _call(6, "click_text", label="Add to cart", utterance="add the mug to the cart"),
        Turn(text="I couldn't find that button."),
    )


def test_a_hint_on_the_second_identical_failure_and_no_third_try():
    w = _world()
    brain = _the_logs_run()
    res = _run(w, brain)
    assert res.stopped == "done" and res.text == "I couldn't find that button."
    assert len(w.calls("click_text")) == 2  # the 3rd and 4th tries never reached the tool
    clicks = [r for r in _results(res) if r.name == "click_text"]
    assert clicks[0].content == _MISSING  # the first failure: as it always was
    assert clicks[1].content == _MISSING + "\n" + failures.advice("not_found")
    for skipped in clicks[2:]:  # (the utterance doesn't make it another step)
        assert not skipped.ok
        assert (
            skipped.content
            == "skipped: this exact step already failed twice (not found) — choose a different step"
        )
    # the run went on: the model saw the skip and answered
    assert brain.seen[-1][-1].role == "tool" and brain.seen[-1][-1].tool_results[0].content.startswith(
        "skipped: "
    )


def test_the_hint_names_tools_never_the_app_or_the_label():
    hint = failures.advice("not_found")
    assert hint.startswith("This exact step failed twice the same way.")
    assert "ax_find" in hint and "page_read" in hint
    assert "Chrome" not in hint and "cart" not in hint.lower()
    for kind in failures.KINDS:
        assert failures.advice(kind).startswith("This exact step failed twice the same way. ")
    assert "activate_app" in failures.advice("not_frontmost")
    assert failures.advice("something new") == failures.advice("other")


def test_a_skipped_call_is_never_shown_to_the_host():
    w = _world()
    ev = EventBus()
    shown: list[tuple[str, str]] = []
    ev.subscribe(
        lambda e: shown.append((e.type, e.data["name"])) if e.type in ("tool_start", "tool_end") else None
    )
    _run(w, _the_logs_run(), ev=ev)
    assert shown.count(("tool_start", "click_text")) == 2 and shown.count(("tool_end", "click_text")) == 2


def test_another_label_is_another_step():
    w = _world()
    brain = Brain(
        _call(1, "click_text", label="Add to cart"),
        _call(2, "web_search", query="mug"),
        _call(3, "click_text", label="Add to cart"),
        _call(4, "web_search", query="mug"),
        _call(5, "click_text", label="Add to basket"),
        Turn(text="Added."),
    )
    res = _run(w, brain)
    assert [a["label"] for a in w.calls("click_text")] == ["Add to cart", "Add to cart", "Add to basket"]
    assert _results(res)[-1].content == "Clicked 'Add to basket' (Button)"


def test_an_action_that_worked_clears_the_count():
    """A scroll that worked moved the screen: the same click after it is a fair try again."""
    w = _world()
    brain = Brain(
        _call(1, "click_text", label="Add to cart"),
        _call(2, "scroll", dy=5),
        _call(3, "click_text", label="Add to cart"),  # a first failure again: no hint
        _call(4, "web_search", query="mug"),
        _call(5, "click_text", label="Add to cart"),  # the second since the scroll: the hint
        _call(6, "web_search", query="mug"),
        _call(7, "click_text", label="Add to cart"),  # the third: skipped
        Turn(text="No luck."),
    )
    res = _run(w, brain)
    assert len(w.calls("click_text")) == 3
    clicks = [r.content for r in _results(res) if r.name == "click_text"]
    assert clicks[:2] == [_MISSING, _MISSING]
    assert clicks[2].endswith(failures.advice("not_found")) and clicks[3].startswith("skipped: ")


def test_a_look_never_counts():
    """A look that fails the same way again and again is still a look: it changes nothing, and it is
    what the model needs next (the thrash guard lets looks repeat too)."""
    w = World()
    w.tool("ax_find", "Error: app 'Mail' is not running.", look=True)
    res = _run(w, Brain(*[_call(i, "ax_find", query="inbox") for i in range(4)], Turn(text="ok")))
    assert len(w.calls("ax_find")) == 4
    assert [r.content for r in _results(res)] == ["Error: app 'Mail' is not running."] * 4


def test_a_call_that_worked_is_never_skipped():
    w = World()
    w.tool("hotkey", "Pressed cmd+r.")
    w.tool("web_search", "1. a result", look=True, category="web")
    turns = []
    for i in range(4):
        turns += [_call(2 * i, "hotkey", keys="cmd+r"), _call(2 * i + 1, "web_search", query="x")]
    _run(w, Brain(*turns, Turn(text="ok")))
    assert len(w.calls("hotkey")) == 4


def test_a_new_look_makes_a_stale_id_worth_another_try():
    """ "unknown element … call it again": the look the error asks for is the remedy, and the ids it
    gives may name that element again."""
    w = World()
    stale = "Error: unknown element e5 — ids come from the latest ax_tree; call it again."
    w.tool("ax_press", lambda a: stale if len(w.calls("ax_press")) <= 2 else "Pressed")
    w.tool("ax_tree", "e5 Button “Send”", look=True)
    w.tool("web_search", "1. a result", look=True, category="web")
    brain = Brain(
        _call(1, "ax_press", id="e5"),
        _call(2, "web_search", query="x"),
        _call(3, "ax_press", id="e5"),
        _call(4, "ax_tree"),
        _call(5, "ax_press", id="e5"),
        Turn(text="Sent."),
    )
    res = _run(w, brain)
    assert len(w.calls("ax_press")) == 3 and _results(res)[-1].content == "Pressed"


def test_failures_of_different_kinds_are_different_steps():
    w = World()
    said = iter(
        [
            "Error: Safari didn't come to the front (Notes stayed there), so I didn't do it.",
            _MISSING,
            _MISSING,
        ]
    )
    w.tool("click_text", lambda a: next(said, _MISSING))
    w.tool("web_search", "1. a result", look=True, category="web")
    brain = Brain(
        _call(1, "click_text", label="Go"),
        _call(2, "web_search", query="x"),
        _call(3, "click_text", label="Go"),  # not found once
        _call(4, "web_search", query="x"),
        _call(5, "click_text", label="Go"),  # not found twice: the hint, not a skip
        Turn(text="ok"),
    )
    res = _run(w, brain)
    clicks = [r.content for r in _results(res) if r.name == "click_text"]
    assert len(w.calls("click_text")) == 3
    assert failures.advice("not_frontmost") not in clicks[0] and clicks[1] == _MISSING
    assert clicks[2].endswith(failures.advice("not_found"))


def test_a_skip_in_a_batch_stops_the_actions_after_it():
    """A skipped action didn't run: what the model chose after it was chosen for a screen that
    isn't there (neo2.core.agent.batch). Every call still gets its result."""
    w = _world()
    two = Turn(
        tool_calls=[ToolCall("a", "click_text", {"label": "Add to cart"}), ToolCall("b", "scroll", {"dy": 3})]
    )
    brain = Brain(
        _call(1, "click_text", label="Add to cart"),
        _call(2, "web_search", query="x"),
        _call(3, "click_text", label="Add to cart"),
        _call(4, "web_search", query="x"),
        two,
        Turn(text="ok"),
    )
    res = _run(w, brain)
    last = res.messages[-2]
    assert last.role == "tool" and [r.call_id for r in last.tool_results] == ["a", "b"]
    assert last.tool_results[0].content.startswith("skipped: this exact step")
    assert last.tool_results[1].content.startswith("skipped: an earlier action")
    assert not w.calls("scroll")


def test_the_untrusted_part_stays_fenced_and_the_hint_is_outside_it():
    w = World()
    w.reg.register(
        RegisteredTool(
            "browser_click",
            "b",
            _OBJ,
            lambda a, c: "Error: no element matches “Buy” on the page",
            untrusted=True,
            category="browser",
        )
    )
    w.tool("web_search", "1. a result", look=True, category="web")
    brain = Brain(
        _call(1, "browser_click", target="Buy"),
        _call(2, "web_search", query="x"),
        _call(3, "browser_click", target="Buy"),
        Turn(text="ok"),
    )
    res = _run(w, brain)
    second = [r for r in _results(res) if r.name == "browser_click"][1].content
    assert second.startswith('<untrusted source="browser_click">')
    fenced, hint = second.rsplit("\n", 1)
    assert fenced.endswith("</untrusted>") and hint == failures.advice("other")


# ---- off: the loop as it was -----------------------------------------------------------------------


def _transcript(res):
    out = []
    for m in res.messages:
        out.append((m.role, m.text, [(c.id, c.name, c.args) for c in m.tool_calls],
                    [(r.call_id, r.name, r.content, r.ok) for r in m.tool_results]))  # fmt: skip
    return out


def test_off_every_call_runs_and_the_transcript_is_todays(monkeypatch):
    _off(monkeypatch)
    w = _world()
    res = _run(w, _the_logs_run())
    assert len(w.calls("click_text")) == 4
    expected = [("user", "add it to the cart", [], [])]
    for i, (name, args, said) in enumerate(
        [
            ("click_text", {"label": "Add to cart"}, _MISSING),
            ("web_search", {"query": "mug"}, "1. a result"),
            ("click_text", {"label": "Add to cart"}, _MISSING),
            ("web_search", {"query": "mug price"}, "1. a result"),
            ("click_text", {"label": "Add to cart"}, _MISSING),
            ("click_text", {"label": "Add to cart", "utterance": "add the mug to the cart"}, _MISSING),
        ],
        start=1,
    ):
        expected.append(("assistant", "", [(str(i), name, args)], []))
        expected.append(("tool", "", [], [(str(i), name, said, name != "click_text")]))
    expected.append(("assistant", "I couldn't find that button.", [], []))
    assert _transcript(res) == expected


def test_on_with_nothing_repeated_the_transcript_is_the_same_as_off(monkeypatch):
    def once():
        w = _world()
        brain = Brain(
            _call(1, "click_text", label="Add to cart"),
            _call(2, "scroll", dy=5),
            _call(3, "click_text", label="Add to cart"),
            _call(4, "click_text", label="Buy now"),
            Turn(text="Done."),
        )
        return _transcript(_run(w, brain))

    on = once()
    _off(monkeypatch)
    assert once() == on


# ---- what a failure is ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,kind",
    [
        # NEO's own error words (neo2/core/tools/computer/*, neo2/core/agent/registry.py)
        ("Error: nothing in the front window of Safari is called 'Next'.", "not_found"),
        ("Error: no menu item “File > Export” to press.", "not_found"),
        ("Error: no menu command of Pages matches “export as pdf”.", "not_found"),
        ("Error: no field called “Subject” in the front window of Mail.", "not_found"),
        ("Error: click what? 'it' isn't a label I can find.", "not_found"),
        ("Error: Notes has no window called “Inbox” — ax_tree lists its windows.", "not_found"),
        ("Error: 'Figmaa' isn't an installed app and figmaa.com doesn't exist", "not_found"),
        ("Error: unknown element e12 — ids come from the latest ax_tree; call it again.", "stale_id"),
        ("Error: unknown element e3 — call ax_tree first.", "stale_id"),
        ("Error: there's no screenshot 's4' any more — take a new one.", "stale_id"),
        (
            "Error: NEO last saw that board 31s ago — take a new screenshot first (it may have moved).",
            "stale_id",
        ),
        ("Error: that sheet in Pages isn't open any more — nothing was pressed.", "stale_id"),
        ("Error: 'Send' isn't on screen — scroll to it or move its window first.", "offscreen"),
        (
            "Error: e7 isn't on screen now (scrolled out of view, past the screen's edge, or gone) — scroll",
            "offscreen",
        ),
        ("Error: (2900, 40) is off every screen (display 1: 1512×982 at 0,0).", "offscreen"),
        (
            "Error: Safari didn't come to the front (NEO's own window is still there), so I didn't click it",
            "not_frontmost",
        ),
        ("Error: app 'Mail' is not running.", "not_frontmost"),
        ("Error: no app is in front to use the menus of.", "not_frontmost"),
        ("Error: Finder covers 'Save' — look again (ax_tree) before clicking.", "not_frontmost"),
        ("Error: Notes has no window in front.", "not_frontmost"),
        ("Error: “Export…” can't be chosen right now (it's greyed out).", "disabled"),
        ("Error: e4 can't be pressed (its actions: AXShowMenu) — click it by id instead.", "disabled"),
        ("Error: Pages is waiting on an open sheet or dialog — answer it first.", "disabled"),
        ("Error: that's a password field — NEO never fills those.", "secure_field"),
        ("Error: the cursor is in a password field — I won't type there.", "secure_field"),
        ("Error: click_text took longer than 30s and was stopped", "timeout"),
        ("Error: command timed out after 20s", "timeout"),
        (
            "Error: Mail isn't responding to Accessibility right now (busy or hung) — try again in a moment.",
            "timeout",
        ),
        ("Error: that's in the system menu (the Apple menu) — NEO never uses it.", "refused"),
        ("Error: that spot is NEO's own window or an app NEO never touches.", "refused"),
        (
            "Error: ~/.ssh/id_rsa holds keys, passwords or private app data, so NEO doesn't read it.",
            "refused",
        ),
        (
            "Error: Accessibility is still off for NEO, and the user has already been told how to turn it on.",
            "refused",
        ),
        ("Error: click_text failed: list index out of range", "other"),
        ("Error: several menu commands in Pages match “export” — which one?", "other"),
        ("Something went wrong", "other"),
        ("", "other"),
    ],
)
def test_kind_of_neos_own_error_words(text, kind):
    assert failures.kind_of("click_text", text) == kind


def test_a_tools_own_name_never_decides_the_kind():
    assert failures.kind_of("password_field_check", "Error: password_field_check failed: boom") == "other"
    assert (
        failures.kind_of("password_field_check", "Error: password_field_check took longer than 9s")
        == "timeout"
    )
    # …and only where the registry names it: the same word in NEO's sentence still counts
    said = "Error: NEO doesn't know a board here yet — take a screenshot of it first (NEO finds its squares)."
    assert failures.kind_of("screenshot", said) == "stale_id"


def test_the_kinds_are_plain_names():
    import re

    assert set(failures.KINDS) == {
        "not_found", "offscreen", "stale_id", "not_frontmost", "disabled", "secure_field", "timeout", "refused", "other"
    }  # fmt: skip
    assert all(re.fullmatch(r"[a-z_]{1,32}", k) for k in failures.KINDS)  # (the report's own check)


def test_the_target_key_leaves_out_the_words_that_dont_change_the_step():
    k = failures.target_key
    assert k("click_text", {"label": "Go"}) == k("click_text", {"utterance": "press go", "label": "Go"})
    assert k("click_text", {"label": "Go", "_plan": {"x": 1}}) == k("click_text", {"label": "Go"})
    assert k("click", {"x": 1, "y": 2}) == k("click", {"y": 2, "x": 1})
    assert k("click", {"x": 1, "y": 2}) != k("click", {"x": 1, "y": 3})
    assert k("click", {"id": "e1"}) != k("ax_press", {"id": "e1"})
    assert k("click", {"id": "e1", "shot": "s1"}) != k(
        "click", {"id": "e1", "shot": "s2"}
    )  # a new look's picture


def test_run_failures_counts_by_step_and_kind():
    f = failures.RunFailures()
    c = ToolCall("1", "click_text", {"label": "Go"})
    assert f.record(c, ok=False, kind="not_found") == 1 and not f.seen_twice(c)
    assert f.record(c, ok=False, kind="not_found") == 2 and f.seen_twice(c) == "not_found"
    assert not f.seen_twice(ToolCall("2", "click_text", {"label": "Stop"}))
    f.record(ToolCall("3", "scroll", {}), ok=True, kind="")
    assert not f.seen_twice(c)  # an action that worked: the screen moved on
    f.record(c, ok=False, kind="stale_id")
    f.record(c, ok=False, kind="stale_id")
    assert f.seen_twice(c) == "stale_id"
    f.looked()
    assert not f.seen_twice(c)  # a fresh look: new ids
    f.record(c, ok=False, kind="not_found")
    f.record(c, ok=False, kind="not_found")
    f.looked()
    assert f.seen_twice(c) == "not_found"  # a look doesn't make a missing control appear
    f.reset()
    assert not f.seen_twice(c)


# ---- the tool log and the counters ---------------------------------------------------------------


def test_a_failure_carries_its_kind_to_the_host_and_the_log(capsys):
    from neo2.core.app import _log_activity

    w = _world()
    ev = EventBus()
    ends: list[dict] = []
    ev.subscribe(lambda e: ends.append(dict(e.data)) if e.type == "tool_end" else None)
    _run(
        w,
        Brain(_call(1, "click_text", label="Add to cart"), _call(2, "scroll", dy=1), Turn(text="ok")),
        ev=ev,
    )
    assert ends[0]["kind"] == "not_found" and "kind" not in ends[1]  # a success: the event as it was
    capsys.readouterr()  # (only the tool log's own lines below)
    _log_activity(Event("tool_end", ends[0]))
    _log_activity(Event("tool_end", ends[1]))
    _log_activity(Event("tool_end", {**ends[0], "kind": "Add to cart"}))  # never anything but a kind
    out = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("  ◀")]
    assert out == [
        f"  ◀ click_text FAIL [not_found] ({len(_MISSING)} chars)",
        f"  ◀ scroll ok ({len('Scrolled down.')} chars)",
        f"  ◀ click_text FAIL ({len(_MISSING)} chars)",
    ]


def test_the_bus_publishes_a_kind_only_when_there_is_one():
    ev = EventBus()
    got: list[dict] = []
    ev.subscribe(lambda e: got.append(e.data))
    asyncio.run(ev.tool_end("click", False, "Error: x", job="j"))
    asyncio.run(ev.tool_end("click", False, "Error: x", job="j", kind="timeout"))
    assert got == [
        {"name": "click", "ok": False, "summary": "Error: x", "job": "j"},
        {"name": "click", "ok": False, "summary": "Error: x", "job": "j", "kind": "timeout"},
    ]


def test_the_daily_counters():
    from neo2.core.reflex import learn

    _run(_world(), _the_logs_run())
    assert learn._counts == {
        "tool_fail": 2,
        "tool_fail_not_found": 2,
        "tool_skip_repeat": 2,
        "agent_end_done": 1,
    }


def test_an_embedded_agents_runs_count_nothing(monkeypatch):
    from neo2.core import runtime
    from neo2.core.reflex import learn

    monkeypatch.setattr(runtime, "library_bound", lambda: True)
    _run(_world(), _the_logs_run())
    assert learn._counts == {}


# ---- how a run ended --------------------------------------------------------------------------------


def _ends(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("[loop] end=")]


@pytest.mark.parametrize(
    "why,stopped",
    [("done", "done"), ("max_steps", "max_steps"), ("deadline", "max_steps"), ("thrash", "thrash"),
     ("needs_confirm", "needs_confirm"), ("needs_user", "needs_user"), ("error", "error")],
)  # fmt: skip
def test_every_run_ends_on_one_line_saying_why(why, stopped, caplog):
    from neo2.core.reflex import learn

    w = World()
    w.tool("act", "did it")
    w.tool("peek", "seen", look=True)
    w.tool("ask", lambda a: ToolOutput("NEEDS_CONFIRM: send it?", ok=True, halt=True))
    w.tool("who", lambda a: ToolOutput("NEEDS_USER: which one?", ok=False, halt=True))
    kw: dict = {}
    if why == "done":
        brain = Brain(_call(1, "act"), Turn(text="Done."))
    elif why == "max_steps":
        brain, kw = Brain(_call(1, "peek"), _call(2, "peek")), {"max_steps": 2}
    elif why == "deadline":
        brain, kw = Brain(Turn(text="never")), {"max_seconds": 1e-9}
    elif why == "thrash":
        brain = Brain(_call(1, "act"), _call(2, "act"), _call(3, "act"), Turn(text="never"))
    elif why == "needs_confirm":
        brain = Brain(_call(1, "ask"))
    elif why == "needs_user":
        brain = Brain(_call(1, "who"))
    else:
        brain = Brain(RuntimeError("the brain fell over"))
    with caplog.at_level(logging.INFO, logger="neo"):
        res = _run(w, brain, **kw)
    assert res.stopped == stopped
    [line] = _ends(caplog)
    assert line.startswith(f"[loop] end={why} steps={len(res.steps)} s=")
    float(line.rsplit("s=", 1)[1])  # seconds, and nothing else
    assert learn._counts.get(f"agent_end_{why}") == 1


def test_the_end_line_stays_with_the_breaker_off(monkeypatch, caplog):
    _off(monkeypatch)
    with caplog.at_level(logging.INFO, logger="neo"):
        _run(_world(), _the_logs_run())
    assert [ln.split(" s=")[0] for ln in _ends(caplog)] == ["[loop] end=done steps=6"]
