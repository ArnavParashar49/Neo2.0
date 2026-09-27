"""An agent's life: what its requests leave in NEO.app's process-wide state (nothing), what they
start (no plain threads), how they stop (cancel), how it forgets (its own files only) and how
it closes (cards closed and audited, temp folder gone)."""

from __future__ import annotations

import asyncio
import sys
import threading

import pytest

import neo2
from neo2.testing import ScriptedBrain, ToolCall, Turn


def _wire(calls: list):
    @neo2.host_tool(risk="destructive")
    def wire_money(to: str) -> str:
        """Send money."""
        calls.append(to)
        return "wired"

    return wire_money


async def test_requests_leave_nothing_in_neo_apps_learning_state(tmp_path, monkeypatch):
    from neo.reflex import learn, outcomes

    calls: list = []
    brain = ScriptedBrain(
        [
            Turn(tool_calls=[ToolCall("1", "web_search", {"query": "neo"})]),
            "Found it.",
            Turn(tool_calls=[ToolCall("2", "wire_money", {"to": "x"})]),
            "Sent.",
        ]
    )
    dd = tmp_path / "agent"
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="web", data_dir=dd, memory=True), provider=brain, host_tools=[_wire(calls)]) as agent:
        agent._rt.registry.get("web_search").handler = lambda a, c: "NEO — an agent for your Mac"
        await agent.ask("what do people say about neo online?")
        await agent.ask("what time is it")  # a fast path
        r = await agent.ask("pay x")
        await agent.approve(r.approval.action_id, via=neo2.Approval("t", "u"))
        await agent.ask("stop")
    assert len(outcomes.ledger()) == 0 and learn._counts == {}
    left = {p.name for p in dd.iterdir()}
    assert left <= {".neo-agent", "audit.log", "memory.sqlite3", "memory.sqlite3-wal", "memory.sqlite3-shm"}
    assert calls == ["x"]


async def test_task_kinds_run_their_legacy_scripts_inside_an_agent(tmp_path):
    from neo.config import decide_mode

    modes: list = []

    @neo2.host_tool(read_only=True, untrusted=False)
    def which_modes() -> str:
        """Report the task modes."""
        modes.append({k: decide_mode(k) for k in ("web_search", "open_app", "add_event", "read_mail", "write_note")})
        return "ok"

    brain = ScriptedBrain([Turn(tool_calls=[ToolCall("1", "which_modes", {})]), "ok"])
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="none"), provider=brain, host_tools=[which_modes]) as agent:
        await agent.ask("modes?")
    assert modes and set(modes[0].values()) == {"legacy"}


@pytest.mark.skipif(sys.platform != "darwin", reason="search_site opens the Mac's browser")
async def test_search_site_runs_its_legacy_script_and_leaves_no_task_kind_files(tmp_path, monkeypatch):
    from neo.decide import shadow
    from neo.tools.computer import apps

    opened: list[str] = []
    dry_runs: list = []

    async def fake_open(name: str) -> str:
        opened.append(name)
        return f"Opened {name}"

    monkeypatch.setattr(apps, "open_app", fake_open)  # never the real `open`
    monkeypatch.setattr(shadow, "later", lambda *a, **kw: dry_runs.append(a))
    dd = tmp_path / "a"
    async with neo2.AsyncAgent(neo2.AgentConfig(tools=("web", "search_site"), data_dir=dd), provider=ScriptedBrain()) as agent:
        r = await agent.ask("search 2tb ssd on amazon")
    assert r.route == "quick" and [s.tool for s in r.steps] == ["search_site"] and r.steps[0].ok
    assert len(opened) == 1 and opened[0].startswith("https://") and "amazon" in opened[0]
    assert dry_runs == []  # legacy, not shadow: no dry run of the task kind's driver
    assert {p.name for p in dd.iterdir()} == {".neo-agent"}


async def test_a_request_starts_no_plain_threads(monkeypatch):
    """NEO.app's warm-ups, learning loops and task-kind drivers start plain threads, which don't
    carry the agent: an embedded agent's request must reach none of them. asyncio's own worker
    threads (to_thread) carry it and are fine."""
    started: list[str] = []
    real = threading.Thread.start

    def start(self):
        started.append(self.name)
        return real(self)

    monkeypatch.setattr(threading.Thread, "start", start)
    brain = ScriptedBrain([Turn(tool_calls=[ToolCall("1", "clock", {}), ToolCall("2", "weather", {"place": "Oslo"})]), "ok", "hi"])
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="web", fast_paths=False), provider=brain) as agent:
        agent._rt.registry.get("weather").handler = lambda a, c: "3°C"
        await agent.ask("time and weather")
        await agent.ask("hello", conversation="other", source="model")
    assert all(n.startswith("asyncio_") for n in started), started


async def test_cancel_stops_a_running_request():
    entered = asyncio.Event()

    class Hanging(ScriptedBrain):
        async def complete(self, messages, **kw):
            entered.set()
            await asyncio.sleep(60)
            return await super().complete(messages, **kw)

    async with neo2.AsyncAgent(neo2.AgentConfig(tools="none"), provider=Hanging(["x"])) as agent:
        task = asyncio.create_task(agent.ask("a long job"))
        await asyncio.wait_for(entered.wait(), 5)
        assert await agent.cancel() == 1
        r = await asyncio.wait_for(task, 5)
        assert r.status == "stopped" and r.route == "stop"
        assert await agent.cancel("nobody") == 0


async def test_cancel_closes_a_pending_card():
    calls: list = []
    brain = ScriptedBrain([Turn(tool_calls=[ToolCall("1", "wire_money", {"to": "x"})])])
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="none"), provider=brain, host_tools=[_wire(calls)]) as agent:
        r = await agent.ask("pay x")
        await agent.cancel()
        assert agent.pending() is None
        assert (await agent.approve(r.approval.action_id, via=neo2.Approval("t", "u"))).status == "expired"
    assert calls == []


async def test_forget_all_erases_only_this_agents_own_files(tmp_path, monkeypatch):
    from neo import privacy

    async def never(*a, **kw):
        raise AssertionError("an embedded agent must never run NEO.app's erase")

    monkeypatch.setattr(privacy, "forget_everything", never)
    dd, other_dd = tmp_path / "a", tmp_path / "b"
    brain = ScriptedBrain(["ok"])
    other_brain = ScriptedBrain()
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="web", data_dir=dd, memory=True), provider=brain) as agent, neo2.AsyncAgent(
        neo2.AgentConfig(tools="web", data_dir=other_dd, memory=True), provider=other_brain
    ) as other:
        await agent.ask("remember that my locker is 42")  # a fast path: straight into memory
        await agent.ask("hello")
        await other.ask("remember that my bike is blue")
        (dd / "browser").mkdir()
        (dd / "browser" / "Cookies").write_text("session=1")
        (dd / "host-notes.txt").write_text("the host's own file")  # added after NEO took the folder
        assert agent.history()
        counts = await agent.forget_all()
        assert counts["memory"] == 1 and counts["conversations"] == 1 and counts["files"] >= 2
        assert agent.history() == []
        assert sorted(p.name for p in dd.iterdir()) == [".neo-agent", "host-notes.txt"]
        brain.add(Turn(tool_calls=[ToolCall("1", "memory", {"action": "recall", "query": "locker"})]), "Nothing.")
        r = await agent.ask("what's my locker number?")
        assert r.steps[0].summary == "Nothing relevant in memory."  # works again, empty
        assert (other_dd / "memory.sqlite3").exists()
        from neo import runtime
        from neo.memory.store import store

        with runtime.bind(other._rt.scope()):
            assert [m.text for m in store().recall("bike")] == ["my bike is blue"]


async def test_forget_all_deletes_a_memory_file_only_while_its_neos_own(tmp_path):
    import sqlite3

    from neo.config import agent_memory_db

    mem = tmp_path / "db" / "u42.sqlite3"
    cfg = neo2.AgentConfig(tools="none", memory=neo2.MemoryConfig(path=mem))
    async with neo2.AsyncAgent(cfg, provider=ScriptedBrain()) as agent:
        await agent.ask("remember that my locker is 42")  # a fast path: straight into memory
        assert agent_memory_db(mem)  # NEO's own, marked so
        counts = await agent.forget_all()
        assert counts["memory"] == 1 and counts["files"] == 1 and list(mem.parent.iterdir()) == []
        await agent.ask("remember that my bike is blue")  # a new database: NEO's own again
        assert agent_memory_db(mem)
        host = sqlite3.connect(mem)
        host.execute("CREATE TABLE orders (id INTEGER)")  # the host kept something of its own in it
        host.execute("INSERT INTO orders VALUES (7)")
        host.commit()
        host.close()
        counts = await agent.forget_all()
        assert counts["memory"] == 1 and counts["files"] == 0 and mem.exists()  # NEO's rows only
        db = sqlite3.connect(mem)
        assert db.execute("SELECT id FROM orders").fetchall() == [(7,)]
        assert db.execute("SELECT count(*) FROM memory").fetchone() == (0,)
        db.close()
    with pytest.raises(ValueError, match="didn't make"):  # no longer only NEO's: not an agent's memory
        neo2.AsyncAgent(cfg, provider=ScriptedBrain())


async def test_close_closes_the_cards_and_is_final(tmp_path):
    calls: list = []
    audit: list = []
    brain = ScriptedBrain([Turn(tool_calls=[ToolCall("1", "wire_money", {"to": "x"})])])
    agent = neo2.AsyncAgent(neo2.AgentConfig(tools="none", data_dir=tmp_path / "a"), provider=brain, host_tools=[_wire(calls)], on_audit=audit.append)
    r = await agent.ask("pay x")
    await agent.aclose()
    await agent.aclose()  # idempotent
    assert agent.closed and [a["status"] for a in audit] == ["staged", "cancelled: closed"]
    for call in (agent.ask("hi"), agent.approve(r.approval.action_id, via=neo2.Approval("t", "u")), agent.call_tool("clock")):
        with pytest.raises(RuntimeError, match="closed"):
            await call
    with pytest.raises(RuntimeError, match="closed"):
        agent.stream("hi")
    assert calls == [] and (tmp_path / "a" / "audit.log").exists()
    from neo import runtime

    assert agent._rt not in runtime.live_runtimes()


async def test_reset_starts_a_conversation_over():
    calls: list = []
    brain = ScriptedBrain(["one", Turn(tool_calls=[ToolCall("1", "wire_money", {"to": "x"})]), "fresh"])
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="none"), provider=brain, host_tools=[_wire(calls)]) as agent:
        await agent.ask("hi", conversation="c")
        r = await agent.ask("pay x", conversation="c")
        assert agent.history("c") and agent.pending("c") is not None
        agent.reset("c")
        assert agent.history("c") == [] and agent.pending("c") is None
        assert (await agent.approve(r.approval.action_id, via=neo2.Approval("t", "u"))).status == "expired"
        r2 = await agent.ask("again", conversation="c")
        assert r2.text == "fresh" and brain.inputs[-1] == "again"
        agent.reset("never-used")  # nothing to do
    assert calls == []


async def test_history_is_plain_data():
    brain = ScriptedBrain([Turn(tool_calls=[ToolCall("1", "clock", {})]), "noon"])
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="web", fast_paths=False), provider=brain) as agent:
        await agent.ask("time?", context="the user is in Lisbon")
        h = agent.history()
    assert [m["role"] for m in h] == ["user", "assistant", "tool", "assistant"]
    assert h[0]["text"].startswith("time?") and "<recent_conversation" in h[0]["text"]  # quoted, never instructions
    assert h[1]["tool_calls"] == [{"id": "1", "name": "clock", "args": {}}]
    assert h[2]["tool_results"][0]["name"] == "clock" and h[3]["text"] == "noon"
