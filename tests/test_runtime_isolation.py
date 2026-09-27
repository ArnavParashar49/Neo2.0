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

from neo2.core import runtime
from neo2.core.agent import confirm
from neo2.core.agent.registry import RegisteredTool
from neo2.core.agent.session import Session
from neo2.core.providers import BrainPool
from neo2.core.providers.base import ToolCall, Turn

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
    from neo2.core import events
    from neo2.core.memory.store import store
    from neo2.core.reflex import learn, outcomes

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
            from neo2.core.events import bus

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

    # audit: A's card in A's own log, with its name and who approved; B staged nothing
    recs = [json.loads(ln) for ln in a.settings.audit_log.read_text().splitlines()]
    assert [(r["status"], r["action"], r["agent"], r.get("via")) for r in recs] == [
        ("staged", card.action_id, "a", None),
        ("confirmed", card.action_id, "a", {"channel": "test"}),
    ]
    assert not b.settings.audit_log.exists()
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
    from neo2.core import config

    make_runtime(tmp_path / "a", "a")
    monkeypatch.setattr(config, "_settings", None)  # (the test helper loaded NEO.app's tools first)
    with pytest.raises(runtime.NoRuntimeError):
        await Session().handle("wire 900 to mallory")
    assert config._settings is None  # never fell back to NEO.app's settings (~/.neo)
    assert confirm._default_gate.pending is None


async def test_a_raw_thread_inside_an_agent_fails_closed(tmp_path):
    from neo2.core.config import settings

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
    from neo2.core import config

    assert not runtime.library_mode()
    monkeypatch.setattr(config, "_settings", None)
    s = config.settings()
    assert s.data_dir == Path(os.environ["NEO_DATA_DIR"]).resolve()
    assert confirm.current_gate() is confirm._default_gate


async def test_task_kind_tools_run_their_legacy_scripts_with_no_dry_run(tmp_path, monkeypatch, decide_modes):
    """NEO.app ships these kinds in shadow (the legacy script, then a dry run of the new driver,
    logged to decide_shadow.log). An embedded agent runs the legacy script only."""
    from neo2.core.agent.registry import ToolContext, registry
    from neo2.core.decide import shadow
    from neo2.core.tools import web
    from neo2.core.tools.computer import apps

    decide_modes(web_search="shadow", open_app="shadow")  # what NEO.app would do
    opened: list[str] = []

    async def open_app(target):
        opened.append(target)
        return f"Opened {target}"

    dry_runs: list[str] = []
    monkeypatch.setattr(apps, "open_app", open_app)
    monkeypatch.setattr(web, "_country", lambda: "")
    monkeypatch.setattr(shadow, "later", lambda kind, facts, **kw: dry_runs.append(kind) or facts.close())
    rt = make_runtime(tmp_path / "mac", "mac", ["search_site", "open_app"])
    with runtime.bind(rt.scope()):
        out1 = await registry().invoke("search_site", {"site": "amazon", "query": "2tb ssd"}, ToolContext())
        out2 = await registry().invoke("open_app", {"target": "Safari"}, ToolContext())
    assert out1.ok and out2.ok, (out1.text, out2.text)
    assert opened == ["https://www.amazon.com/s?k=2tb+ssd", "Safari"]
    assert dry_runs == []
    assert not {p.name for p in rt.settings.data_dir.iterdir()} & {"decide_shadow.log", "decide.json", "decide_learned"}
    runtime.set_library_mode(False)  # the same tool in NEO.app, whose kinds are in shadow here
    out = await registry().invoke("open_app", {"target": "Notes"}, ToolContext())
    assert out.ok and dry_runs == ["open_app"]


async def test_one_task_asking_two_agents_carries_nothing_between_them(tmp_path):
    """A host often calls agent A, then agent B, from the same task: what A was told as context,
    the taint A's request started with, and A's tool extras must not ride into B's request."""
    seen: list[tuple[str, dict, list]] = []

    def probe(a, c):
        seen.append((c.user_text, dict(c.extras), list(c.taint)))
        return "ok"

    rts = []
    for name in ("a", "b"):
        rt = make_runtime(tmp_path / name, name)
        rt.registry.register(RegisteredTool("zz_probe", "probe", _OBJ, probe))
        brain = SlowBrain([Turn(tool_calls=[ToolCall("1", "zz_probe", {})]), "done"], name=f"brain-{name}")
        rt.brains = BrainPool(overrides=dict.fromkeys(("agent", "fast", "vision", "light", "offline"), [brain]))
        rts.append((rt, brain))
    (a, brain_a), (b, brain_b) = rts
    with runtime.bind(a.scope()):
        await Session().handle("do it", conversation="A's private notes", taint=("web_fetch",), extras={"t": "A"})
    with runtime.bind(b.scope()):
        await Session().handle("do it")
    assert seen == [("do it", {"t": "A"}, ["web_fetch"]), ("do it", {}, [])]
    assert any("A's private notes" in t for t in brain_a.inputs)  # A was told
    assert not any("A's private notes" in t for t in brain_b.inputs + brain_b.systems)  # B never hears it
    with runtime.bind(b.scope()):
        from neo2.core.agent.session import _with_conversation

        assert _with_conversation("x") == "x"


async def test_a_request_leaves_nothing_behind_in_its_callers_task(tmp_path):
    """handle() used to leave its extras and taint set in the calling task: approving a card of
    agent B that B's session didn't adopt (a direct call staged it), right after serving agent A
    from the same task, ran B's action with A's extras."""
    from neo2.core.agent import session as sess_mod
    from neo2.core.agent.registry import ToolContext, registry

    seen: list[dict] = []

    def wire(args, c):
        g = confirm.gated("zz_wire", args, "wire 900 to mallory")
        if isinstance(g, str):
            return g
        seen.append(dict(c.extras))
        return "wired"

    a, b = make_runtime(tmp_path / "a", "a"), make_runtime(tmp_path / "b", "b")
    for rt in (a, b):
        rt.registry.register(RegisteredTool("zz_wire", "wire", _OBJ, wire, risk="destructive"))
        rt.registry.register(RegisteredTool("zz_ping", "ping", _OBJ, lambda x, c: "pong", fast_path=[(r"^\s*ping\s*$", {})]))
        rt.brains = BrainPool(overrides={"agent": [ScriptedBrain([])]})
    session_b = Session()
    with runtime.bind(b.scope()):  # B's card, staged by a direct call (not adopted by B's session)
        aid = (await registry().invoke("zz_wire", {"amount": 900}, ToolContext(extras={"tenant": "B"}))).staged
    with runtime.bind(a.scope()):  # then the host serves A, in the same task
        await Session().handle("ping", taint=("web_fetch",), extras={"tenant": "A"})
    assert dict(sess_mod._request_extras.get()) == {} and sess_mod._request_taint.get() == ()
    assert sess_mod._conversation.get() == ""
    with runtime.bind(b.scope()):
        await session_b.answer(True, action_id=aid)
    assert seen == [{}]  # never A's


async def test_a_typed_yes_still_carries_its_own_requests_extras(tmp_path):
    """Inside a request (a typed "yes" answering a card no session adopted), the answering
    request's own extras still reach the approved call, as before."""
    from neo2.core.agent.registry import ToolContext, registry

    seen: list[dict] = []

    def wire(args, c):
        g = confirm.gated("zz_wire", args, "wire 5")
        if isinstance(g, str):
            return g
        seen.append(dict(c.extras))
        return "wired"

    rt = make_runtime(tmp_path / "a", "a")
    rt.registry.register(RegisteredTool("zz_wire", "wire", _OBJ, wire, risk="destructive"))
    with runtime.bind(rt.scope()):
        await registry().invoke("zz_wire", {"amount": 5}, ToolContext())
        await Session().handle("yes", from_user=True, extras={"tenant": "A"})
    assert seen == [{"tenant": "A"}]
