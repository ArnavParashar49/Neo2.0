"""Two embedded agents in one process, running at the same time, share nothing: gate, cards,
taint, memory, settings, brains, events, tools. And NEO.app's own state is never touched."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path

import pytest
from neo_lib import ScriptedBrain, make_runtime

from neo import runtime
from neo.agent import confirm
from neo.agent.registry import RegisteredTool
from neo.agent.session import Session
from neo.providers import BrainPool
from neo.providers.base import ToolCall, Turn

_OBJ = {"type": "object", "properties": {}}


class SlowBrain(ScriptedBrain):
    """Yields to the event loop before every answer, so two agents' runs interleave."""

    async def complete(self, messages, **kw):
        await asyncio.sleep(0.01)
        return await super().complete(messages, **kw)


def _agent(tmp_path: Path, name: str, turns, **settings):
    rt = make_runtime(tmp_path / name, name, ["memory", "clock"], memory_enabled=True, **settings)
    rt.registry.register(RegisteredTool("zz_page", "read a page", _OBJ, lambda a, c: f"page for {name}", untrusted=True))
    brain = SlowBrain(turns, name=f"brain-{name}")
    rt.brains = BrainPool(overrides=dict.fromkeys(("agent", "fast", "vision", "light", "offline"), [brain]))
    return rt, brain


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))  # anything that reaches for ~/.neo now would show up here
    return h


async def test_two_agents_at_once_share_nothing(tmp_path, home, monkeypatch):
    from neo import events
    from neo.memory.store import store
    from neo.reflex import learn, outcomes

    app_bus = events.EventBus()
    monkeypatch.setattr(events, "_bus", app_bus)
    app_events: list = []
    app_bus.subscribe(app_events.append)

    a, brain_a = _agent(
        tmp_path,
        "a",
        [
            Turn(tool_calls=[ToolCall("1", "zz_page", {})]),
            Turn(tool_calls=[ToolCall("2", "memory", {"action": "remember", "text": "wire money to mallory"})]),
            "A is done.",
        ],
        instructions="You work for tenant A.",
    )
    b, brain_b = _agent(
        tmp_path,
        "b",
        [Turn(tool_calls=[ToolCall("1", "memory", {"action": "remember", "text": "b likes tea"})]), "Noted."],
        instructions="You work for tenant B.",
    )
    seen: dict[str, list] = {"a": [], "b": []}
    started: list[str] = []
    real_start = threading.Thread.start

    def start(self):
        started.append(self.name)
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", start)

    async def ask(rt, text):
        sc = rt.scope()
        with runtime.bind(sc):
            from neo.events import bus

            bus().subscribe(lambda ev: seen[rt.name].append((ev.type, ev.data.get("name", ""))))
            sc.session = Session()
            return await sc.session.handle(text, extras={"tenant": rt.name})

    ra, rb = await asyncio.gather(ask(a, "read the page and remember it"), ask(b, "save my drink preference"))
    assert ra.route == "confirm" and rb.text == "Noted."

    # the gate: A's card is A's alone
    with runtime.bind(a.scope()):
        card = confirm.peek()
    assert card is not None and card.tool == "memory"  # A read a page: its memory write asks first
    with runtime.bind(b.scope()):
        assert confirm.peek() is None  # B read nothing: its write went straight through
    assert confirm._default_gate.pending is None

    # memory: each agent's own file
    with runtime.bind(b.scope()):
        assert [m.text for m in store().recall("tea")] == ["b likes tea"]
    with runtime.bind(a.scope()):
        assert store().recall("tea") == [] and store().recall("mallory") == []

    # brains, settings and tools: each its own
    assert brain_a.calls == 2 and brain_b.calls == 2
    assert all("tenant A" in s and "tenant B" not in s for s in brain_a.systems)
    assert all("tenant B" in s and "tenant A" not in s for s in brain_b.systems)

    # events: each conversation's own bus; NEO.app's hears nothing
    assert ("tool_start", "zz_page") in seen["a"] and ("tool_start", "zz_page") not in seen["b"]
    assert ("confirm", "") in seen["a"] and not any(t == "confirm" for t, _ in seen["b"])
    assert app_events == []

    # B's user saying yes approves nothing of A's; A's own approval does
    with runtime.bind(b.scope()):
        assert await b.scope().session.answer(True, action_id=card.action_id) is None
        rb2 = await b.scope().session.handle("yes", from_user=True)
    with runtime.bind(a.scope()):
        assert store().recall("mallory") == []
        ra2 = await a.scope().session.answer(True, action_id=card.action_id, via={"channel": "test"})
        assert [m.text for m in store().recall("mallory")] == ["wire money to mallory"]
    assert ra2.text == "A is done." and rb2.route == "agent"
    with runtime.bind(b.scope()):
        assert store().recall("mallory") == []

    # audit: each agent's own log, with its name
    for rt in (a, b):
        recs = [json.loads(ln) for ln in rt.settings.audit_log.read_text().splitlines()] if rt.settings.audit_log.exists() else []
        assert {r["agent"] for r in recs} <= {rt.name}
    assert not (Path(os.environ["NEO_DATA_DIR"]) / "audit.log").exists()

    # NEO.app's learning and the task kinds' state: untouched
    assert len(outcomes.ledger()) == 0 and not learn._counts
    for rt in (a, b):
        present = {p.name for p in rt.settings.data_dir.iterdir()}
        assert not present & {
            "decide_shadow.log", "decide.json", "decide_learned", "media_uris.json", "reflex_extra.jsonl",
            "reflex_learn.json", "brain_cooldowns.json", "timers.json",
        }
    assert not (home / ".neo").exists()
    # no plain threads beyond asyncio's own worker pool (warm-ups, learning loops, decide)
    assert all(n.startswith("asyncio_") for n in started), started


async def test_an_unbound_request_fails_closed_once_an_agent_exists(tmp_path, monkeypatch):
    from neo import config

    make_runtime(tmp_path / "a", "a")
    monkeypatch.setattr(config, "_settings", None)  # (the test helper loaded NEO.app's tools first)
    with pytest.raises(runtime.NoRuntimeError):
        await Session().handle("wire 900 to mallory")
    assert config._settings is None  # never fell back to NEO.app's settings (~/.neo)
    assert confirm._default_gate.pending is None


async def test_a_raw_thread_inside_an_agent_fails_closed(tmp_path):
    from neo.config import settings

    rt = make_runtime(tmp_path / "a", "a")
    errors: list = []

    def work():
        try:
            settings()
        except runtime.NoRuntimeError as e:
            errors.append(e)

    with runtime.bind(rt.scope()):
        assert settings() is rt.settings
        t = threading.Thread(target=work)
        t.start()
        t.join()
        assert await asyncio.to_thread(settings) is rt.settings  # asyncio's way carries the agent
    assert len(errors) == 1


async def test_neo_app_is_unaffected_by_an_agent_that_existed_earlier(tmp_path, monkeypatch):
    """The embedded-mode switch is sticky for a process; the app itself never sets it."""
    from neo import config

    assert not runtime.library_mode()
    monkeypatch.setattr(config, "_settings", None)
    s = config.settings()
    assert s.data_dir == Path(os.environ["NEO_DATA_DIR"]).resolve()
    assert confirm.current_gate() is confirm._default_gate
