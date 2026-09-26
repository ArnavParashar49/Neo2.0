"""Per-job orb state, the early actor, and concurrent jobs in the session."""

from __future__ import annotations

import asyncio
import time

import pytest

from neo.agent import early as early_mod
from neo.agent.early import EarlyActor
from neo.agent.registry import registry
from neo.agent.session import Session
from neo.events import EventBus, NeoState
from neo.providers.base import ToolCall, Turn
from neo.reflex.schema import Decision

# ---- event bus aggregation ------------------------------------------------------------------


def test_job_states_aggregate_by_priority_and_fall_back():
    ev = EventBus()
    seen: list[str] = []
    ev.subscribe(lambda e: seen.append(e.data["state"]) if e.type == "state" else None)

    async def run():
        await ev.set_state(NeoState.LISTENING)  # voice base
        await ev.set_state(NeoState.THINKING, job="a")
        await ev.set_state(NeoState.WORKING, job="b")
        assert ev.state == NeoState.WORKING
        await ev.set_state(NeoState.SPEAKING)  # NEO talks: wins over working
        assert ev.state == NeoState.SPEAKING
        await ev.set_state(NeoState.LISTENING)
        await ev.end_job("b")
        assert ev.state == NeoState.THINKING  # falls back to job a, not to idle
        await ev.end_job("a")
        assert ev.state == NeoState.LISTENING  # back to the voice base
        await ev.set_state(NeoState.IDLE)
        assert ev.state == NeoState.IDLE

    asyncio.run(run())
    assert "idle" not in seen[:-1]  # never flickered through idle while something was active


# ---- early actor ------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _tools(tmp_path, monkeypatch):
    from neo import config, events
    from neo.tools import load_all

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(events, "_bus", EventBus())
    monkeypatch.setattr(early_mod, "_actor", None)
    load_all()
    calls: list[tuple[str, dict]] = []

    async def fake_open(a, c):
        calls.append(("open_app", dict(a)))
        return f"Opened {a['target']}"

    async def fake_volume(a, c):
        calls.append(("volume", dict(a)))
        return "Volume up."

    monkeypatch.setattr(registry().get("open_app"), "handler", fake_open)  # restored after the test
    monkeypatch.setattr(registry().get("volume"), "handler", fake_volume)
    return calls


def test_early_actor_runs_a_finished_clause_and_leaves_the_rest(_tools):
    ea = EarlyActor()
    calls = _tools

    async def run():
        ea.feed("open youtube and")
        assert (
            await ea.tick() == []
        )  # "open youtube" is followed by "and" → clause finished, but stable window
        ea._changed -= 1.0  # pretend 1 s passed
        res = await ea.tick()
        assert res and calls == [("open_app", {"target": "youtube"})]
        ea.feed("open youtube and search for cat videos")
        ea._changed -= 1.0
        assert await ea.tick(final=True) == []  # the search belongs to YouTube: left for the agent…
        assert ea.remaining() == "open youtube and search for cat videos"  # …with its context
        assert (
            ea.recently_done("open_app", {"target": "YouTube "}) == "Opened youtube"
        )  # case/space-insensitive
        assert ea.recently_done("open_app", {"target": "spotify"}) is None

    asyncio.run(run())


def test_early_actor_waits_while_the_clause_is_still_being_spoken(_tools):
    ea = EarlyActor()

    async def run():
        ea.feed("turn the volume")
        ea._changed -= 1.0
        assert await ea.tick() == [] and _tools == []  # not a command yet
        ea.feed("turn the volume up")
        assert await ea.tick() == []  # just changed → not stable
        ea._changed -= 1.0
        assert await ea.tick() and _tools[-1][0] == "volume"
        ea.new_utterance()
        assert ea.remaining() == ""

    asyncio.run(run())


def test_early_actor_dedupes_within_window(_tools):
    ea = EarlyActor()

    async def run():
        ea.feed("open youtube")
        await ea.tick(final=True)
        ea.new_utterance()
        ea.feed("open youtube")
        await ea.tick(final=True)
        assert len(_tools) == 1  # second one answered from the cache
        ea._done = {k: (t - 100, r) for k, (t, r) in ea._done.items()}  # window expired
        ea.new_utterance()
        ea.feed("open youtube")
        await ea.tick(final=True)
        assert len(_tools) == 2

    asyncio.run(run())


# ---- concurrent jobs --------------------------------------------------------------------------


class SlowBrain:
    name = "slow"
    supports_vision = False
    supports_tools = True

    def __init__(self, delay: float, answer: str):
        self.delay, self.answer = delay, answer

    async def complete(self, messages, **kw):
        await asyncio.sleep(self.delay)
        return Turn(text=self.answer)

    async def stream(self, messages, **kw):
        await asyncio.sleep(self.delay)
        yield self.answer


def test_two_jobs_run_concurrently_and_history_stays_clean(monkeypatch):
    import neo.agent.session as sess_mod

    brains = iter([SlowBrain(0.4, "mail done"), SlowBrain(0.05, "it is noon")])
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": next(brains))

    class R:
        async def decide(self, t):
            return Decision("agent_task", 0.9, False, 0.0, False, 0.0, "fake")

    s = Session(reflex=R())

    async def run():
        t0 = time.time()
        a, b = await asyncio.gather(s.handle("summarize my mail"), s.handle("plan my week"))
        return a, b, time.time() - t0

    a, b, elapsed = asyncio.run(run())
    assert a.text == "mail done" and b.text == "it is noon" and a.job != b.job
    assert (
        elapsed < 0.6
    )  # they overlapped instead of queuing (0.4 + 0.05 would be ≥ 0.45 serial… and we allow slack)
    roles = [m.role for m in s.history]
    assert roles == ["user", "assistant", "user", "assistant"]  # two complete blocks, no interleaving
    assert s.history[0].text in ("plan my week", "summarize my mail")


def test_stop_cancels_running_jobs(monkeypatch):
    import neo.agent.session as sess_mod

    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": SlowBrain(5.0, "never"))
    decisions = iter(
        [
            Decision("agent_task", 0.9, False, 0.0, False, 0.0, "fake"),
            Decision("stop", 1.0, False, 0.0, False, 0.0, "rules"),
        ]
    )

    class R:
        async def decide(self, t):
            return next(decisions)

    s = Session(reflex=R())

    async def run():
        long = asyncio.create_task(s.handle("do something slow"))
        await asyncio.sleep(0.1)
        r = await s.handle("stop")
        assert r.route == "stop"
        with pytest.raises(asyncio.CancelledError):
            await long

    asyncio.run(run())
    assert not s.jobs


def test_quick_route_uses_early_result_when_already_done(monkeypatch, _tools):
    import neo.agent.session as sess_mod

    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": SlowBrain(0.0, "x"))

    class R:
        async def decide(self, t):
            return Decision("quick_action", 0.9, False, 0.0, False, 0.0, "fake")

    ea = early_mod.early()

    async def run():
        ea.feed("open youtube")
        await ea.tick(final=True)
        ea.new_utterance()
        r = await Session(reflex=R()).handle("open youtube")
        return r

    r = asyncio.run(run())
    assert r.route == "quick" and r.text == "Opened youtube"
    assert len(_tools) == 1  # not executed a second time


def test_loop_events_carry_job_ids():
    from neo.agent.loop import run_agent
    from neo.agent.registry import RegisteredTool, Registry

    reg = Registry()
    reg.register(RegisteredTool("t", "t", {"type": "object", "properties": {}}, lambda a, c: "ok"))

    class B:
        name = "b"
        supports_vision = False
        supports_tools = True
        turns = [Turn(tool_calls=[ToolCall("1", "t", {})]), Turn(text="done")]

        async def complete(self, messages, **kw):
            return self.turns.pop(0)

        async def stream(self, messages, **kw):
            yield ""

    ev = EventBus()
    events: list[tuple[str, str]] = []
    ev.subscribe(
        lambda e: (
            events.append((e.type, e.data.get("job", "?")))
            if e.type in ("tool_start", "tool_end", "state")
            else None
        )
    )
    asyncio.run(run_agent("x", provider=B(), system="s", reg=reg, ev=ev, job="j9", verify=False))
    assert events and all(j == "j9" for _, j in events), events
    assert ev.state == NeoState.IDLE and not ev.active_jobs
