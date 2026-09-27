"""stream(): one request's events, in order, ending with Completed — never another request's."""

from __future__ import annotations

import asyncio

import pytest

import neo2
from neo2 import events as ev
from neo2.testing import ScriptedBrain, ToolCall, Turn


class ChunkBrain(ScriptedBrain):
    """Streams its chat answer in pieces, the way a real model does."""

    def __init__(self, chunks, *a, **kw):
        super().__init__(*a, **kw)
        self.chunks = chunks

    async def stream(self, messages, *, system="", max_tokens=2048):
        self.calls += 1
        for c in self.chunks:
            await asyncio.sleep(0)
            yield c


class SlowBrain(ScriptedBrain):
    async def complete(self, messages, **kw):
        await asyncio.sleep(0.01)
        return await super().complete(messages, **kw)


@pytest.fixture
def chat_reflex(monkeypatch):
    """The reflex reads every request as conversation (an embedded agent's is off by default)."""
    from neo.reflex import Reflex
    from neo.reflex.schema import Decision

    async def chat(self, text):
        return Decision("chat", 0.99, False, 0.0, False, 0.0, source="test")

    monkeypatch.setattr(Reflex, "_decide", chat)


async def _collect(agent, text, **kw) -> list[ev.Event]:
    return [e async for e in agent.stream(text, **kw)]


async def test_a_streamed_chat_answer_joins_to_the_result(chat_reflex):
    brain = ChunkBrain([" Hel", "lo, ", "world! ", "How can I help?  "])
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="web"), provider=brain) as agent:
        events = await _collect(agent, "hi there")
    assert isinstance(events[0], ev.Started) and isinstance(events[-1], ev.Completed)
    assert sum(isinstance(e, ev.Completed) for e in events) == 1
    result = events[-1].result
    deltas = [e.text for e in events if isinstance(e, ev.TextDelta)]
    assert len(deltas) >= 3 and "".join(deltas) == result.text == "Hello, world! How can I help?"
    route = next(e for e in events if isinstance(e, ev.RouteDecided))
    assert (route.intent, route.source) == ("chat", "test")
    stats = next(e for e in events if isinstance(e, ev.TurnStats))
    assert stats.route == "chat" and stats.brain == "scripted" and result.brain == "scripted"
    assert {e.request_id for e in events} == {result.request_id}
    assert all(e.conversation == "default" for e in events)


async def test_an_agent_run_streams_its_tools_and_answer():
    brain = ScriptedBrain([Turn(tool_calls=[ToolCall("1", "clock", {})]), "It's noon."])
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="web", fast_paths=False), provider=brain) as agent:
        events = await _collect(agent, "what's the time")
    types = [e.type for e in events]
    assert types[0] == "started" and types[-1] == "completed"
    assert types.index("tool_started") < types.index("tool_finished") < types.index("text_delta")
    fin = next(e for e in events if isinstance(e, ev.ToolFinished))
    assert fin.name == "clock" and fin.ok and not fin.untrusted
    assert "".join(e.text for e in events if isinstance(e, ev.TextDelta)) == events[-1].result.text == "It's noon."
    assert events[-1].result.steps[0].tool == "clock"
    for e in events:
        assert e.to_dict()["type"] == e.type  # JSON-ready (for server-sent events)


def _wire(calls):
    @neo2.host_tool(risk="destructive", confirm=lambda a: f"wire to {a['to']}")
    def wire_money(to: str) -> str:
        """Send money."""
        calls.append(to)
        return "wired"

    return wire_money


async def test_an_approval_required_pairs_with_its_resolution():
    calls: list = []

    async def approver(req):
        return neo2.Approval("dialog", "me")

    brain = ScriptedBrain([Turn(tool_calls=[ToolCall("1", "wire_money", {"to": "x"})]), "Wired."])
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="none"), provider=brain, approver=approver, host_tools=[_wire(calls)]) as agent:
        events = await _collect(agent, "pay x")
    req = [e for e in events if isinstance(e, ev.ApprovalRequired)]
    res = [e for e in events if isinstance(e, ev.ApprovalResolved)]
    assert len(req) == len(res) == 1
    assert req[0].request.action_id == res[0].action_id and res[0].outcome == "approved"
    types = [e.type for e in events]
    assert types.index("approval_required") < types.index("approval_resolved") < types.index("completed")
    result = events[-1].result
    assert result.status == "done" and calls == ["x"]
    # the question isn't the answer: only the final text streams
    assert "".join(e.text for e in events if isinstance(e, ev.TextDelta)) == result.text == "Wired."
    assert req[0].to_dict()["request"]["summary"] == "wire to x"


async def test_a_deferred_approval_streams_the_card_then_completes():
    calls: list = []
    brain = ScriptedBrain([Turn(tool_calls=[ToolCall("1", "wire_money", {"to": "x"})])])
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="none"), provider=brain, host_tools=[_wire(calls)]) as agent:
        events = await _collect(agent, "pay x")
        result = events[-1].result
        assert result.status == "needs_approval"
        req = next(e for e in events if isinstance(e, ev.ApprovalRequired)).request
        assert req == result.approval and not any(isinstance(e, ev.ApprovalResolved) for e in events)
        assert "".join(e.text for e in events if isinstance(e, ev.TextDelta)) == result.text
        assert result.text == "wire to x — Should I go ahead?"
        await agent.deny(req.action_id)
    assert calls == []


async def test_two_concurrent_requests_are_kept_apart():
    ba = SlowBrain([Turn(tool_calls=[ToolCall("1", "clock", {})]), Turn(tool_calls=[ToolCall("2", "clock", {})]), "A done."])
    bb = SlowBrain([Turn(tool_calls=[ToolCall("1", "weather", {"place": "Oslo"})]), "B done."])
    a = neo2.AsyncAgent(neo2.AgentConfig(tools="web", fast_paths=False), provider=ba)
    b = neo2.AsyncAgent(neo2.AgentConfig(tools="web", fast_paths=False), provider=bb)
    import neo.tools.weather as weather

    async def fake_weather(args, ctx):
        await asyncio.sleep(0.01)
        return "Oslo: 3°C"

    b._rt.registry.get("weather").handler = fake_weather
    async with a, b:
        ea, eb = await asyncio.gather(_collect(a, "time twice"), _collect(b, "weather in Oslo"))
    assert weather  # the real handler was never used
    for events, tools, text in ((ea, ["clock", "clock"], "A done."), (eb, ["weather"], "B done.")):
        rid = events[-1].result.request_id
        assert {e.request_id for e in events} == {rid}
        assert [e.name for e in events if isinstance(e, ev.ToolFinished)] == tools
        assert events[-1].result.text == text
    assert ea[-1].result.request_id != eb[-1].result.request_id


async def test_two_requests_in_one_conversation_are_kept_apart():
    brain = SlowBrain(
        [
            Turn(tool_calls=[ToolCall("1", "clock", {})]),
            Turn(tool_calls=[ToolCall("2", "web_search", {"query": "x"})]),
            "first",
            "second",
        ]
    )
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="web", fast_paths=False), provider=brain) as agent:
        agent._rt.registry.get("web_search").handler = lambda a, c: "results"
        e1, e2 = await asyncio.gather(_collect(agent, "one"), _collect(agent, "two"))
    names = sorted(e.name for es in (e1, e2) for e in es if isinstance(e, ev.ToolFinished))
    assert names == ["clock", "web_search"]
    for es in (e1, e2):
        assert len({e.request_id for e in es}) == 1
        assert sum(isinstance(e, ev.ToolFinished) for e in es) == 1
    by_tool = {next(e.name for e in es if isinstance(e, ev.ToolFinished)): es[-1].result for es in (e1, e2)}
    assert by_tool["web_search"].read_untrusted == ("web_search",) and by_tool["clock"].read_untrusted == ()


async def test_breaking_out_of_a_stream_stops_its_request():
    started = asyncio.Event()

    class Blocking(ScriptedBrain):
        async def complete(self, messages, **kw):
            started.set()
            await asyncio.sleep(30)
            return await super().complete(messages, **kw)

    async with neo2.AsyncAgent(neo2.AgentConfig(tools="none"), provider=Blocking(["x"])) as agent:
        gen = agent.stream("take your time")
        first = await gen.__anext__()
        assert isinstance(first, ev.Started)
        await asyncio.wait_for(started.wait(), 5)
        await gen.aclose()  # the host stopped listening: the request is cancelled, not left running
        sess = agent._rt.scopes["default"].session
        await asyncio.sleep(0)
        assert all(t.done() for t in sess.jobs.values())


async def test_a_failing_request_raises_from_the_stream():
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="none"), provider=ScriptedBrain()) as agent:
        with pytest.raises(ValueError):
            [e async for e in agent.stream("x", conversation="")]
        with pytest.raises(TypeError):
            [e async for e in agent.stream("x", tool_context=["not", "a", "mapping"])]
