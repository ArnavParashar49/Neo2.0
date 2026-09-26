"""The process catalog vs an agent's own registry: an agent reaches only the tools it was given."""

from __future__ import annotations

import asyncio

import pytest

from neo import runtime
from neo.agent import fastpath
from neo.agent.loop import run_agent
from neo.agent.registry import RegisteredTool, Registry, ToolContext, catalog, registry, tool
from neo.events import EventBus
from neo.providers.base import ToolCall, Turn

WEB = ["web_search", "web_fetch", "weather", "clock"]


@pytest.fixture
def web_agent(tmp_path):
    from neo_lib import make_runtime

    return make_runtime(tmp_path / "web", "web", WEB, library=False)


class _Brain:
    name = "fake"
    supports_vision = False
    supports_tools = True

    def __init__(self, turns):
        self.turns = list(turns)
        self.tools_seen: list[list[str]] = []

    async def complete(self, messages, *, system="", tools=None, effort="medium", max_tokens=4096):
        self.tools_seen.append([t.name for t in tools or []])
        return self.turns.pop(0)

    async def stream(self, messages, *, system="", max_tokens=2048):
        yield "x"


def test_unbound_registry_is_the_catalog():
    assert registry() is catalog()


def test_a_tool_declared_while_bound_lands_in_the_catalog_only(web_agent):
    with runtime.bind(web_agent.scope()):
        assert registry() is web_agent.registry

        @tool("zz_declared_while_bound", "t")
        def zz(a, c):
            return "ran"

        assert "zz_declared_while_bound" not in registry()
    assert "zz_declared_while_bound" in catalog()
    catalog().remove("zz_declared_while_bound")


def test_an_agent_only_has_its_tools(web_agent):
    names = [t.name for t in web_agent.registry.all()]
    assert names == [t.name for t in catalog().all() if t.name in WEB]  # the catalog's order
    assert web_agent.registry.strict and not catalog().strict


@pytest.mark.parametrize("name", ["shell", "open_url", "click", "mail_send", "play_music"])
def test_invoking_anything_else_is_unknown(web_agent, name):
    assert name in catalog()
    with runtime.bind(web_agent.scope()):
        out = asyncio.run(registry().invoke(name, {"command": "id", "url": "https://x.org"}, ToolContext()))
    assert not out.ok and out.text == f"Unknown tool: {name}"


@pytest.mark.parametrize(
    "text, app_tool",
    [
        ("lock the screen", "sleep_display"),
        ("play 505", "play_music"),
        ("check my mail", "mail_unread"),
        ("turn the volume up", "volume"),
        ("open safari", "open_app"),
    ],
)
def test_fast_paths_only_match_the_agents_tools(web_agent, text, app_tool):
    assert fastpath.match(text)[0] == app_tool  # NEO.app: the whole catalog
    with runtime.bind(web_agent.scope()):
        assert fastpath.match(text) is None
        assert fastpath.plan(text) is None


def test_the_agents_own_fast_paths_still_work(web_agent):
    with runtime.bind(web_agent.scope()):
        assert fastpath.match("search for cats") == ("web_search", {"query": "cats"})
        assert fastpath.match("what time is it") == ("clock", {})


def test_the_loop_offers_and_runs_only_the_agents_tools(web_agent):
    brain = _Brain([Turn(tool_calls=[ToolCall("1", "shell", {"command": "id"})]), Turn(text="done")])
    with runtime.bind(web_agent.scope()):
        res = asyncio.run(run_agent("x", provider=brain, system="", ev=EventBus(), verify=False))
    assert brain.tools_seen[0] == [t.name for t in web_agent.registry.all()]
    assert res.steps[0].tool == "shell" and not res.steps[0].ok
    assert res.steps[0].result == "Unknown tool: shell"


def test_a_strict_registry_refuses_a_second_tool_of_the_same_name(web_agent):
    t = RegisteredTool("clock", "impostor", {"type": "object", "properties": {}}, lambda a, c: "x")
    with pytest.raises(ValueError, match="clock"):
        web_agent.registry.register(t)
    assert web_agent.registry.get("clock").description != "impostor"
    web_agent.registry.register(t, replace=True)  # only when asked for
    assert web_agent.registry.get("clock").description == "impostor"
    loose = Registry()
    loose.register(t)
    loose.register(RegisteredTool("clock", "second", {}, lambda a, c: "y"))  # the catalog: as today
    assert loose.get("clock").description == "second"


def test_copies_are_independent_of_the_catalog(web_agent):
    cat = catalog().get("clock")
    mine = web_agent.registry.get("clock")
    assert mine is not cat and mine.handler is cat.handler
    before = mine.description
    old = cat.description
    try:
        cat.description = "changed in the catalog"  # tasks.apply_modes rewrites catalog fields
        assert web_agent.registry.get("clock").description == before
    finally:
        cat.description = old


def test_task_modes_rewrite_the_catalog_not_an_agent(web_agent, monkeypatch, tmp_path):
    from neo.config import LibrarySettings
    from neo.tools import tasks

    agent = runtime.Runtime("mac", LibrarySettings(data_dir=tmp_path / "mac"), catalog().subset(["page_read"]))
    with runtime.bind(agent.scope()):
        assert registry().get("page_read").hidden
        monkeypatch.setattr(tasks, "mode", lambda kind: "on")
        tasks.apply_modes()
        assert registry().get("page_read").hidden  # the agent keeps what it was given
    assert not catalog().get("page_read").hidden
    monkeypatch.setattr(tasks, "mode", lambda kind: "legacy")
    tasks.apply_modes()
    assert catalog().get("page_read").hidden


def test_library_mode_unbound_registry_fails_closed():
    runtime.set_library_mode()
    with pytest.raises(runtime.NoRuntimeError):
        registry()
    catalog()  # the catalog itself is fine: registering tools is process-wide


def test_more_tools_only_finds_the_agents_tools(web_agent):
    from neo.agent.toolselect import selector

    with runtime.bind(web_agent.scope()):
        found = [t.name for t in selector().search("run a shell command, click a button, send an email", k=20)]
        picked = selector().select("run a shell command and lock the screen").names
    assert set(found) <= set(WEB) and set(picked) <= set(WEB)


def test_more_tools_in_a_run_enables_nothing_outside_the_agent(web_agent):
    brain = _Brain(
        [
            Turn(tool_calls=[ToolCall("1", "more_tools", {"query": "run a shell command"})]),
            Turn(tool_calls=[ToolCall("2", "shell", {"command": "id"})]),
            Turn(text="done"),
        ]
    )
    with runtime.bind(web_agent.scope()):
        res = asyncio.run(run_agent("x", provider=brain, system="", ev=EventBus(), verify=False, tools=[]))
    assert set(brain.tools_seen[1]) <= set(WEB)
    assert res.steps[1].result == "Unknown tool: shell"


def test_an_agent_gets_the_as_registered_surface_whatever_the_app_runs(decide_modes, tmp_path):
    """NEO.app's loader rewrites catalog entries for kinds that are "on" (descriptions, parameters,
    hidden): an agent, whose kinds run their legacy scripts, gets the tools as registered."""
    from neo_lib import make_runtime

    from neo.tools import tasks

    decide_modes(web_search="on", add_event="on", read_calendar="on")
    names = ["page_read", "safari_read", "search_site", "calendar_add", "calendar_today"]
    assert catalog().get("page_read").hidden is False and catalog().get("safari_read").hidden  # the app's "on"
    rt = make_runtime(tmp_path / "a", "a", names, library=False)
    for name in names:
        mine = rt.registry.get(name)
        for k, v in tasks._AS_REGISTERED[name].items():
            assert getattr(mine, k) == v, (name, k)
    assert rt.registry.get("page_read").hidden and not rt.registry.get("safari_read").hidden
