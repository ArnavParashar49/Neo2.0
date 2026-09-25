"""SDK-free tests for the v2 core: registry, confirm gate, agent loop with a fake brain."""

from __future__ import annotations

import asyncio

from neo.agent import confirm
from neo.agent.loop import run_agent
from neo.agent.registry import RegisteredTool, Registry, ToolContext
from neo.events import EventBus, NeoState
from neo.providers.base import Message, ToolCall, Turn


class FakeBrain:
    name = "fake"
    supports_vision = False
    supports_tools = True

    def __init__(self, turns: list[Turn]):
        self._turns = list(turns)
        self.seen: list[list[Message]] = []

    async def complete(self, messages, *, system="", tools=None, effort="medium", max_tokens=4096):
        self.seen.append(list(messages))
        return self._turns.pop(0)

    async def stream(self, messages, *, system="", max_tokens=2048):
        yield "x"


def _reg() -> Registry:
    r = Registry()
    r.register(
        RegisteredTool(
            "echo",
            "echo",
            {"type": "object", "properties": {}},
            lambda a, c: a.get("t", ""),
            parallel_safe=True,
        )
    )

    async def boom(a, c):
        raise RuntimeError("kaboom")

    r.register(RegisteredTool("boom", "fails", {"type": "object", "properties": {}}, boom))

    def delete(a, c):
        p = confirm.gated("del-1", "delete", a, "delete the file")
        return "deleted" if isinstance(p, dict) else p

    r.register(
        RegisteredTool("delete", "delete", {"type": "object", "properties": {}}, delete, risk="destructive")
    )
    return r


def test_registry_invoke_and_errors():
    r = _reg()
    out = asyncio.run(r.invoke("echo", {"t": "hi"}, ToolContext()))
    assert out.ok and out.text == "hi"
    out = asyncio.run(r.invoke("boom", {}, ToolContext()))
    assert not out.ok and "kaboom" in out.text
    out = asyncio.run(r.invoke("nope", {}, ToolContext()))
    assert not out.ok


def test_confirm_gate_roundtrip(tmp_path, monkeypatch):
    from neo import config

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path))
    r = _reg()
    first = asyncio.run(r.invoke("delete", {"path": "x"}, ToolContext()))
    assert first.text.startswith(confirm.NEEDS_CONFIRM)
    # wrong action id can't release it
    assert confirm.consume("other") is None
    second = asyncio.run(r.invoke("delete", {"path": "x", "confirm": True}, ToolContext()))
    assert second.text == "deleted"
    assert (tmp_path / "audit.log").read_text().count("\n") == 2


def test_loop_runs_tools_then_answers():
    r = _reg()
    brain = FakeBrain(
        [
            Turn(tool_calls=[ToolCall("1", "echo", {"t": "a"}), ToolCall("2", "echo", {"t": "b"})]),
            Turn(text="done: a b"),
        ]
    )
    ev = EventBus()
    states: list[str] = []
    ev.subscribe(lambda e: states.append(e.data["state"]) if e.type == "state" else None)
    res = asyncio.run(run_agent("go", provider=brain, system="s", reg=r, ev=ev))
    assert res.stopped == "done" and res.text == "done: a b"
    assert [s.tool for s in res.steps] == ["echo", "echo"]
    assert (
        states[0] == NeoState.THINKING.value
        and NeoState.WORKING.value in states
        and states[-1] == NeoState.IDLE.value
    )
    # tool results were fed back
    assert brain.seen[1][-1].role == "tool" and len(brain.seen[1][-1].tool_results) == 2


def test_loop_halts_on_confirm(tmp_path, monkeypatch):
    from neo import config

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path))
    r = _reg()
    brain = FakeBrain([Turn(tool_calls=[ToolCall("1", "delete", {"path": "x"})]), Turn(text="never")])
    ev = EventBus()
    res = asyncio.run(run_agent("delete x", provider=brain, system="s", reg=r, ev=ev))
    assert res.stopped == "needs_confirm"
    assert "go ahead" in res.question.lower()
    assert ev.state == NeoState.CONFIRMING
    assert len(brain.seen) == 1


def test_loop_thrash_guard():
    r = _reg()
    # `echo` is read-only (parallel_safe) so it may repeat; `boom` has side effects and may not.
    same = Turn(tool_calls=[ToolCall("1", "boom", {"t": "z"})])
    brain = FakeBrain([same, same, same, Turn(text="unreachable")])
    res = asyncio.run(run_agent("loop", provider=brain, system="s", reg=r, ev=EventBus()))
    assert res.stopped == "thrash"


def test_loop_step_budget():
    r = _reg()
    brain = FakeBrain([Turn(tool_calls=[ToolCall(str(i), "echo", {"t": str(i)})]) for i in range(5)])
    res = asyncio.run(run_agent("x", provider=brain, system="s", reg=r, ev=EventBus(), max_steps=3))
    assert res.stopped == "max_steps" and len(res.steps) == 3
