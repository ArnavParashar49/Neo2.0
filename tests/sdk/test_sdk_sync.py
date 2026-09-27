"""Agent, the sync twin: a private loop on its own thread; never inside a running event loop."""

from __future__ import annotations

import asyncio
import threading

import pytest

import neo2
from neo2 import events as ev
from neo2.testing import ScriptedBrain, ToolCall, Turn


def _wire(calls):
    @neo2.host_tool(risk="destructive")
    def wire_money(to: str) -> str:
        """Send money."""
        calls.append((to, threading.current_thread().name))
        return "wired"

    return wire_money


def test_it_works_from_a_plain_thread():
    out: dict = {}
    calls: list = []

    def work():
        brain = ScriptedBrain([Turn(tool_calls=[ToolCall("1", "wire_money", {"to": "x"})]), "Wired."])
        with neo2.Agent(neo2.AgentConfig(tools="web"), provider=brain, host_tools=[_wire(calls)]) as a:
            r = a.ask("pay x")
            out["pending"] = a.pending()
            out["r"] = a.approve(r.approval.action_id, via=neo2.Approval("cli", "me"))
            out["history"] = a.history()
            out["stream"] = list(a.stream("what time is it"))

    t = threading.Thread(target=work)
    t.start()
    t.join(30)
    assert not t.is_alive()
    assert out["pending"].tool == "wire_money"
    assert out["r"].status == "done" and out["r"].text == "Wired." and [c[0] for c in calls] == ["x"]
    assert out["history"][0] == {"role": "user", "text": "pay x"}
    assert isinstance(out["stream"][0], ev.Started) and isinstance(out["stream"][-1], ev.Completed)
    assert out["stream"][-1].result.route == "quick"


def test_it_works_from_the_main_thread_too():
    with neo2.Agent(neo2.AgentConfig(tools="web"), provider=ScriptedBrain(["hello"])) as a:
        r = a.ask("say hello", conversation="c1")
        assert r.status == "done" and r.text == "hello" and r.conversation == "c1"
        assert a.name == "default" and "clock" in {t.name for t in a.tools}


async def test_it_refuses_to_run_inside_an_event_loop():
    with pytest.raises(RuntimeError, match="use AsyncAgent"):
        neo2.Agent(neo2.AgentConfig(tools="none"), provider=ScriptedBrain())
    a = await asyncio.to_thread(neo2.Agent, neo2.AgentConfig(tools="none"), provider=ScriptedBrain(["x"]))
    try:
        with pytest.raises(RuntimeError, match="use AsyncAgent"):
            a.ask("hi")
        with pytest.raises(RuntimeError, match="use AsyncAgent"):
            a.stream("hi")
        with pytest.raises(RuntimeError, match="use AsyncAgent"):
            a.approve("x:1", via=neo2.Approval("t", "u"))
        with pytest.raises(RuntimeError, match="use AsyncAgent"):
            a.call_tool("clock")
        assert (await asyncio.to_thread(a.ask, "hi")).text == "x"  # from a worker thread it's fine
    finally:
        await asyncio.to_thread(a.close)


def test_close_joins_the_thread_and_is_final():
    a = neo2.Agent(neo2.AgentConfig(tools="none"), provider=ScriptedBrain(["x"]))
    thread, loop, dd = a._thread, a._loop, a.async_agent.data_dir
    assert thread.is_alive() and thread.daemon and dd.exists()
    a.close()
    assert not thread.is_alive() and loop.is_closed() and not dd.exists()
    assert a.closed
    a.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        a.ask("hi")


def test_two_sync_agents_coexist():
    calls: list = []
    a = neo2.Agent(neo2.AgentConfig(name="a", tools="none"), provider=ScriptedBrain([Turn(tool_calls=[ToolCall("1", "wire_money", {"to": "a"})]), "A."]), host_tools=[_wire(calls)])
    b = neo2.Agent(neo2.AgentConfig(name="b", tools="none"), provider=ScriptedBrain([Turn(tool_calls=[ToolCall("1", "wire_money", {"to": "b"})]), "B."]), host_tools=[_wire(calls)])
    try:
        assert a._thread is not b._thread and a._loop is not b._loop
        results: dict = {}

        def run(agent, key):
            r = agent.ask(f"pay {key}")
            results[key] = (r, agent.approve(r.approval.action_id, via=neo2.Approval("t", key)))

        ts = [threading.Thread(target=run, args=(x, k)) for x, k in ((a, "a"), (b, "b"))]
        for t in ts:
            t.start()
        for t in ts:
            t.join(30)
        assert results["a"][1].text == "A." and results["b"][1].text == "B."
        assert (b.approve(results["a"][0].approval.action_id, via=neo2.Approval("t", "b"))).status == "expired"
        assert sorted(c[0] for c in calls) == ["a", "b"]
        assert all(c[1] != threading.main_thread().name for c in calls)  # off every caller's thread
    finally:
        a.close()
        b.close()


def test_a_sync_approver_calling_the_agent_is_refused():
    calls: list = []
    errors: list = []
    holder: dict = {}

    def approver(req):
        try:
            holder["agent"].ask("should I?")
        except neo2.ApprovalReentryError as e:
            errors.append(e)
        return None

    brain = ScriptedBrain([Turn(tool_calls=[ToolCall("1", "wire_money", {"to": "x"})])])
    with neo2.Agent(neo2.AgentConfig(tools="none"), provider=brain, approver=approver, host_tools=[_wire(calls)]) as a:
        holder["agent"] = a
        r = a.ask("pay x")
    assert r.status == "denied" and calls == [] and len(errors) == 1


def test_sync_host_tools_can_be_added_and_removed():
    with neo2.Agent(neo2.AgentConfig(tools="none"), provider=ScriptedBrain([Turn(tool_calls=[ToolCall("1", "ping", {})]), "pong!"])) as a:

        @a.tool(read_only=True, untrusted=False)
        def ping() -> str:
            """Ping."""
            return "pong"

        assert "ping" in {t.name for t in a.tools}
        r = a.ask("ping it")
        assert r.steps[0].summary == "pong" and r.text == "pong!"
        a.remove_tool("ping")
        assert "ping" not in {t.name for t in a.tools}
        a.reset()
        assert a.history() == []
        assert a.forget_all()["files"] == 0


def test_a_dropped_agent_stops_its_thread():
    import gc

    a = neo2.Agent(neo2.AgentConfig(tools="none"), provider=ScriptedBrain(["x"]))
    assert a.ask("hi").text == "x"
    thread, dd = a._thread, a.async_agent.data_dir
    del a
    gc.collect()
    thread.join(5)
    assert not thread.is_alive() and not dd.exists()
