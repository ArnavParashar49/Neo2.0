"""Regression tests for the findings of the adversarial review (2026-09-26)."""

from __future__ import annotations

import asyncio
import re

import pytest

from neo.agent import confirm
from neo.agent.loop import run_agent
from neo.agent.registry import RegisteredTool, Registry, ToolContext, registry
from neo.agent.session import _NO, _YES, Session, _match_fast_path
from neo.events import EventBus
from neo.providers.base import Message, ToolCall, Turn


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    from neo import config, events

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(events, "_bus", EventBus())
    confirm.cancel("test")
    from neo.tools import load_all

    load_all()


class Brain:
    name = "fake"
    supports_vision = False
    supports_tools = True

    def __init__(self, turns):
        self._turns = list(turns)
        self.seen = []

    async def complete(self, messages, **kw):
        self.seen.append(list(messages))
        return self._turns.pop(0)

    async def stream(self, messages, **kw):
        yield "x"


# ---- confirmation parsing ----------------------------------------------------------------


@pytest.mark.parametrize(
    "text", ["okay, cancel that", "ok never mind", "yes, actually no", "sure but don't send it yet", "wait"]
)
def test_retractions_are_not_confirmations(text):
    assert _NO.search(text) or not _YES.match(text)


@pytest.mark.parametrize(
    "text", ["yes", "Yes please", "ok", "okay do it", "sure thing", "go ahead.", "confirm"]
)
def test_pure_affirmatives_confirm(text):
    assert _YES.match(text) and not _NO.search(text)


# ---- two destructive calls in one turn -----------------------------------------------------


def test_second_destructive_call_never_runs_before_confirmation():
    executed = []
    reg = Registry()

    def trash(a, c):
        p = confirm.gated(f"trash:{a['path']}", "trash", a, f"move {a['path']} to the Trash")
        if isinstance(p, dict):
            executed.append(p["path"])
            return "trashed"
        return p

    reg.register(
        RegisteredTool("trash", "t", {"type": "object", "properties": {}}, trash, risk="destructive")
    )
    brain = Brain(
        [
            Turn(
                tool_calls=[
                    ToolCall("1", "trash", {"path": "a.txt"}),
                    ToolCall("2", "trash", {"path": "b.txt"}),
                ]
            )
        ]
    )
    res = asyncio.run(run_agent("trash both", provider=brain, system="s", reg=reg, ev=EventBus()))
    assert res.stopped == "needs_confirm" and "a.txt" in res.question
    assert res.pending_action == "trash:a.txt"  # the question and the staged action agree
    assert confirm.peek().params["path"] == "a.txt"
    assert executed == []
    # every tool_call has a result, the second one marked skipped
    tool_msg = res.messages[-1]
    assert [r.call_id for r in tool_msg.tool_results] == ["1", "2"]
    assert tool_msg.tool_results[1].content.startswith("skipped")


# ---- ordered execution -----------------------------------------------------------------------


def test_calls_run_in_emitted_order_with_readonly_batching():
    order = []
    reg = Registry()

    async def mk(name, safe):
        async def h(a, c):
            order.append(name)
            return name

        reg.register(RegisteredTool(name, name, {"type": "object", "properties": {}}, h, parallel_safe=safe))

    asyncio.run(mk("write", False))
    asyncio.run(mk("read", True))
    asyncio.run(mk("click", False))
    asyncio.run(mk("tree", True))
    brain = Brain(
        [
            Turn(
                tool_calls=[
                    ToolCall("1", "write", {}),
                    ToolCall("2", "read", {}),
                    ToolCall("3", "click", {}),
                    ToolCall("4", "tree", {}),
                ]
            ),
            Turn(text="done"),
        ]
    )
    asyncio.run(run_agent("go", provider=brain, system="s", reg=reg, ev=EventBus()))
    assert order == ["write", "read", "click", "tree"]


# ---- thrash guard --------------------------------------------------------------------------


def test_thrash_exit_keeps_transcript_valid_and_ignores_readonly_repeats():
    reg = Registry()
    reg.register(
        RegisteredTool(
            "look", "l", {"type": "object", "properties": {}}, lambda a, c: "same", parallel_safe=True
        )
    )
    reg.register(RegisteredTool("act", "a", {"type": "object", "properties": {}}, lambda a, c: "did"))
    look = Turn(tool_calls=[ToolCall("1", "look", {})])
    brain = Brain([look, look, look, look, Turn(text="fine")])
    res = asyncio.run(run_agent("x", provider=brain, system="s", reg=reg, ev=EventBus()))
    assert res.stopped == "done"  # repeated observations are allowed
    act = Turn(tool_calls=[ToolCall("1", "act", {"n": 1})])
    brain = Brain([act, act, act, Turn(text="unreachable")])
    res = asyncio.run(run_agent("x", provider=brain, system="s", reg=reg, ev=EventBus()))
    assert res.stopped == "thrash"
    # invariant: assistant tool_calls are always followed by matching tool results
    for i, m in enumerate(res.messages):
        if m.role == "assistant" and m.tool_calls:
            nxt = res.messages[i + 1]
            assert nxt.role == "tool" and [r.call_id for r in nxt.tool_results] == [
                c.id for c in m.tool_calls
            ]


# ---- fast paths --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("turn the volume up", ("volume", {"action": "up"})),
        ("volume down a bit", ("volume", {"action": "down"})),
        ("mute", ("volume", {"action": "mute"})),
        ("set volume to 0", ("volume", {"action": "set", "level": "0"})),
        ("turn up the brightness", ("brightness", {"direction": "up"})),
        ("lower the brightness", ("brightness", {"direction": "down"})),
        ("reduce the brightness please", ("brightness", {"direction": "down"})),
        ("start a timer for 5 minutes", ("timer", {"minutes": "5"})),
        ("set a 10 minute timer", ("timer", {"minutes": "10"})),
        ("open Safari", ("open_app", {"target": "Safari"})),
        ("launch vs code", ("open_app", {"target": "vs code"})),
        ("what time is it", ("clock", {})),
    ],
)
def test_fast_path_positive(text, expected):
    assert _match_fast_path(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "set brightness to 20%",
        "the silence of the lambs was on tv",
        "why is my mic muted in zoom",
        "turn up the heat in the living room",
        "open the file I was working on yesterday",
        "open a new tab and search for flights",
        "can you mute the notifications in slack for an hour",
        "is the volume too loud for you",
        "I need to dim the lights before the movie",
    ],
)
def test_fast_path_negative(text):
    assert _match_fast_path(text) is None


# ---- misc tool fixes --------------------------------------------------------------------


def test_volume_zero_is_not_fifty():
    import neo.tools.system as sysmod

    calls = []

    async def fake_osa(script, timeout=30):
        calls.append(script)
        return "50"

    orig = sysmod.osascript
    sysmod.osascript = fake_osa
    try:
        out = asyncio.run(registry().get("volume").handler({"action": "set", "level": 0}, ToolContext()))
    finally:
        sysmod.osascript = orig
    assert out == "Volume 0%." and "set volume output volume 0" in calls[-1]


def test_shell_gate_catches_bypasses_and_allows_dev_null():
    from neo.tools.computer.shell import looks_destructive as d

    assert (
        d("rm -fr ~/Library") and d("rm -r -f x") and d("rm notes.txt") and d("find . -name '*.log' -delete")
    )
    assert d("echo x | sh") and d("git branch -D main") and d("brew remove node") and d("osascript -e 'x'")
    assert not d("ls -la 2>/dev/null") and not d("grep -r foo . 2>/dev/null") and not d("rm -- --help")


def test_applescript_risky_scripts_are_gated():
    from neo.tools.computer.apps import script_is_risky

    assert script_is_risky('tell application "Finder" to delete file "x"')
    assert script_is_risky('do shell script "rm -rf ~"')
    assert script_is_risky('tell application "Mail" to send m')
    assert not script_is_risky('tell application "Safari" to get URL of current tab of window 1')


def test_memory_recall_survives_apostrophes(tmp_path, monkeypatch):
    from neo.memory import embed
    from neo.memory.store import Store

    monkeypatch.setattr(embed, "embed_texts", lambda texts: None)  # keyword path only; no model download
    st = Store(tmp_path / "m.sqlite3")
    st.remember("User's favourite editor is Zed", "profile", "editor")
    assert [m.text for m in st.recall("what's my editor?")]
    assert st.recall("???") == []


def test_read_file_start_zero(tmp_path):
    from neo.tools.computer.shell import read_file

    f = tmp_path / "a.txt"
    f.write_text("one\ntwo\n")
    assert "one" in read_file(str(f), start=0, lines=10)


def test_data_dir_expands_tilde(monkeypatch):
    from neo import config

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("NEO_DATA_DIR", "~/.neo-test-dir")
    assert "~" not in str(config.Settings().data_dir)
    monkeypatch.delenv("NEO_DATA_DIR")
    monkeypatch.setattr(config, "_settings", None)


def test_openai_skips_empty_assistant_and_claude_starts_with_user():
    from neo.providers.claude import _to_messages as claude_msgs
    from neo.providers.openai_compat import _to_messages as oai_msgs

    hist = [Message.assistant(""), Message.user("hi"), Message.assistant("")]
    assert [m["role"] for m in oai_msgs(hist, "", False)] == ["user"]
    hist2 = [Message.assistant("orphan"), Message.tool([]), Message.user("hi")]
    assert claude_msgs(hist2)[0]["role"] == "user"


def test_session_stale_confirmation_is_refused(monkeypatch):
    import neo.agent.session as sess_mod

    class R:
        async def decide(self, t):
            from neo.reflex.schema import Decision

            return Decision("agent_task", 0.9, True, 0.9, False, 0.0, "fake")

    s = Session(reflex=R())
    s._pending_tool, s._pending_args, s._pending_action = "trash", {"path": "a"}, "trash:a"
    confirm.stage("trash:b", "trash", {"path": "b"}, "move b")  # something else got staged meanwhile
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": Brain([Turn(text="never")]))
    r = asyncio.run(s.handle("yes"))
    assert r.route == "confirm" and "didn't do anything" in r.text
    assert confirm.peek() is None


_ = re
