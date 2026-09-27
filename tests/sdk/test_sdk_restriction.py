"""An agent reaches only the tools it was given — by fast path, by a model's call, through
more_tools, and in the run that continues after an approval — however many tools NEO.app's
catalog holds in the same process."""

from __future__ import annotations

import importlib.util
import sys

import pytest

import neo2
from neo.agent.registry import catalog
from neo2.testing import ScriptedBrain, ToolCall, Turn

pytestmark = pytest.mark.portable  # Linux or macOS, any install (what needs an extra skips): CI's sdk-core job

WEB = {"web_search", "web_fetch", "weather", "clock"}
mac_only = pytest.mark.skipif(
    sys.platform != "darwin" or importlib.util.find_spec("AppKit") is None,
    reason="NEO's Mac tools: macOS with the [mac] extra",
)


@pytest.fixture
def mac_catalog(monkeypatch):
    """NEO.app's whole catalog loaded in this process, its Mac tools' handlers replaced by
    recorders: anything that reached one would show up in `ran`."""
    from neo.tools import load_all

    load_all()  # NEO.app's loader, before any agent exists (library mode is off)
    ran: list[str] = []
    for name in ("sleep_display", "hotkey", "shell", "click", "applescript", "open_app"):
        t = catalog().get(name)
        monkeypatch.setattr(t, "handler", lambda a, c, n=name: ran.append(n) or "ok")
    return ran


@mac_only
async def test_lock_the_screen_never_runs_a_mac_tool(mac_catalog):
    from neo.agent import fastpath

    assert fastpath.match("lock the screen")[0] == "sleep_display"  # NEO.app's own fast path
    brain = ScriptedBrain(["I can't lock the screen from here."])
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="web"), provider=brain) as agent:
        r = await agent.ask("lock the screen")
    assert r.route == "agent" and r.status == "done" and r.steps == ()
    assert mac_catalog == []
    assert set(brain.tools_seen[0]) <= WEB | {"more_tools"}


@mac_only
async def test_a_model_calling_shell_gets_unknown_tool(mac_catalog):
    brain = ScriptedBrain([Turn(tool_calls=[ToolCall("1", "shell", {"command": "id"})]), "Couldn't."])
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="web"), provider=brain) as agent:
        r = await agent.ask("run id in a shell")
    assert [(s.tool, s.ok, s.summary) for s in r.steps] == [("shell", False, "Unknown tool: shell")]
    assert mac_catalog == []


async def test_more_tools_enables_nothing_outside_the_preset():
    brain = ScriptedBrain(
        [
            Turn(tool_calls=[ToolCall("1", "more_tools", {"query": "run a shell command, click, send an email"})]),
            Turn(tool_calls=[ToolCall("2", "shell", {"command": "id"})]),
            "done",
        ]
    )
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="web"), provider=brain) as agent:
        r = await agent.ask("do something on my computer")
    assert all(set(seen) <= WEB | {"more_tools"} for seen in brain.tools_seen)
    assert r.steps[-1].summary == "Unknown tool: shell"


@mac_only
async def test_the_run_after_an_approval_offers_only_the_agents_tools(mac_catalog):
    assert len(catalog().all()) >= 50  # NEO.app's whole catalog is in this process
    calls: list = []

    @neo2.host_tool(risk="destructive")
    def wire_money(to: str) -> str:
        """Send money."""
        calls.append(to)
        return "wired"

    brain = ScriptedBrain(
        [
            Turn(tool_calls=[ToolCall("1", "wire_money", {"to": "x"})]),
            Turn(tool_calls=[ToolCall("2", "shell", {"command": "id"})]),
            "done",
        ]
    )
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="web"), provider=brain, host_tools=[wire_money]) as agent:
        r = await agent.ask("pay x")
        r = await agent.approve(r.approval.action_id, via=neo2.Approval("t", "u"))
    assert calls == ["x"] and mac_catalog == []
    after = brain.tools_seen[1]  # the continuation's first call: no tools= passed, the registry's own
    assert set(after) == WEB | {"wire_money", "more_tools"}  # more_tools: the agent's own registry only
    assert r.steps[-1].summary == "Unknown tool: shell"


async def test_the_agent_registry_is_a_strict_copy(monkeypatch):
    """Changing NEO.app's catalog entry later changes nothing in an agent built before."""
    from neo.tools import load

    load("web")
    t = catalog().get("clock")
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="web"), provider=ScriptedBrain()) as agent:
        mine = agent._rt.registry.get("clock")
        assert mine is not t and mine.handler is t.handler
        assert "shell" not in agent._rt.registry and "memory" not in agent._rt.registry
        assert {i.name for i in agent.tools} == WEB
        before = (mine.description, mine.handler, mine.risk, mine.parameters, list(mine.fast_path))
        monkeypatch.setattr(t, "description", "hijacked")
        monkeypatch.setattr(t, "handler", lambda a, c: "hijacked")
        monkeypatch.setattr(t, "risk", "destructive")
        t.fast_path.append((r"^hijack$", {}))  # in place: the agent's list is its own
        try:
            assert (mine.description, mine.handler, mine.risk, mine.parameters, list(mine.fast_path)) == before
            assert {i.name: i for i in agent.tools}["clock"].description != "hijacked"
            r = await agent.ask("what time is it")  # a fast path: the agent's own clock runs
            assert r.route == "quick" and r.steps[0].summary != "hijacked"
        finally:
            t.fast_path.pop()


async def test_a_fast_path_runs_only_the_agents_own_tool():
    brain = ScriptedBrain(["(unused)"])
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="web"), provider=brain) as agent:
        r = await agent.ask("what time is it")
        assert r.route == "quick" and [s.tool for s in r.steps] == ["clock"] and brain.calls == 0
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="web", fast_paths=False), provider=ScriptedBrain(["noon"])) as agent:
        r = await agent.ask("what time is it")
        assert r.route == "agent" and r.text == "noon"
