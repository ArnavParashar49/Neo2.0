"""What main added that is process-wide in NEO.app, kept per agent when NEO is embedded.

* Task grants ("yes for this task") and the untrusted-read counter live in each conversation's
  confirmation gate: they never cross conversations or agents, and an embedded agent's gate gives
  no grants at all (its cards offer no scope, every approval is one-off). NEO.app: as on main.
* The brains' learned state (quota windows and their file, demotion, the request-size scale) is
  the agent's own pool's, in its own data dir — never NEO.app's globals or ~/.neo.
* The effects switch refuses nothing an embedded agent does outside a test run.
* Accessibility off, in a host: the halt names the host's process, not NEO.app, and comes back as
  the SDK's needs_user.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from neo_lib import ScriptedBrain, make_runtime

from neo import effects, runtime
from neo import providers as prov
from neo.agent import confirm, grants
from neo.agent import session as sess_mod
from neo.agent.registry import RegisteredTool, ToolContext, catalog, registry
from neo.events import EventBus, bus
from neo.providers.base import QuotaExhausted, ToolCall, Turn
from neo.reflex.schema import Decision

ROOT = Path(__file__).resolve().parents[1]
CHESS = grants.Target("Chess", "", ("ax_tree",))
_OBJ = {"type": "object", "properties": {"app": {"type": "string"}, "step": {"type": "string"}}}


def _reflex():
    class R:
        async def decide(self, t):
            return Decision("agent_task", 0.95, False, 0.0, False, 0.0, "fake", reply=False, reply_p=0.1)

    return R()


@pytest.fixture
def grantable(monkeypatch):
    """zz_step: a tool whose calls a yes "for this task" may cover (like applescript on one app).
    Its handler only records the step: nothing real runs."""
    ran: list[str] = []

    def step(a, c):
        target = grants.Target(str(a.get("app") or ""), "", tuple(c.taint))
        gate = confirm.gated("zz_step", a, f"do {a['step']} in {a.get('app')}", target=target)
        if isinstance(gate, str):
            return gate
        ran.append(str(a["step"]))
        return f"did {a['step']}"

    monkeypatch.setitem(catalog()._tools, "zz_step", RegisteredTool("zz_step", "Do one step in an app.", _OBJ, step))
    monkeypatch.setattr(sess_mod.fastpath, "plan", lambda text: None)  # no fast path reaches a real tool
    return ran


def _steps(*names: str) -> list[Turn]:
    return [Turn(tool_calls=[ToolCall(str(i), "zz_step", {"app": "Chess", "step": n})]) for i, n in enumerate(names)]


# ---- (a) task grants --------------------------------------------------------------------------------


def test_an_agents_yes_for_this_task_stays_one_off_and_neo_app_keeps_its_grants(grantable, tmp_path, monkeypatch):
    """Two embedded agents in one process, then NEO.app in the same process: the same two Chess
    steps, the first card answered "for this task". An agent's card offers no scope and the second
    step asks again; NEO.app's grant covers it, as on main."""
    ran = grantable
    agents = [make_runtime(tmp_path / n, n, ["zz_step"], fast_paths=False) for n in ("a", "b")]

    async def embedded(rt):
        brain = ScriptedBrain([*_steps("new game", "pick white"), "done"])
        rt.brains = prov.BrainPool(overrides={"agent": [brain]})
        cards: list[dict] = []
        with runtime.bind(rt.scope()):
            bus().subscribe(lambda e: cards.append(e.data) if e.type == "confirm" else None)
            s = sess_mod.Session(reflex=_reflex())
            r = await s.handle("zz play chess as white")
            await asyncio.sleep(0)
            p = confirm.peek()
            assert r.route == "confirm" and p.task and p.target == grants.Target("Chess", "", ())
            assert cards[-1]["scopes"] == [] and cards[-1]["app"] == ""
            assert confirm.card(p)["scopes"] == [] == confirm.card(p, gate=confirm.current_gate())["scopes"]
            r = await s.answer(True, action_id=p.action_id, scope="task", via={"channel": "test"})
            assert grants.active(p.task) == []  # no grant was made
            assert r.route == "confirm" and confirm.peek().params["step"] == "pick white"  # asked again
            confirm.cancel("test")
        return ran[:]

    for rt in agents:
        ran.clear()
        assert asyncio.run(embedded(rt)) == ["new game"]

    # NEO.app, same process: the grant covers the rest of the task.
    runtime.set_library_mode(False)
    from neo import events

    monkeypatch.setattr(events, "_bus", EventBus())
    ran.clear()
    brain = ScriptedBrain([*_steps("new game", "pick white"), "done"])
    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": brain)

    async def app():
        cards: list[dict] = []
        bus().subscribe(lambda e: cards.append(e.data) if e.type == "confirm" else None)
        s = sess_mod.Session(reflex=_reflex())
        r = await s.handle("zz play chess as white", from_user=True)
        await asyncio.sleep(0)
        assert r.route == "confirm" and cards[-1]["scopes"] == ["task"] and cards[-1]["app"] == "Chess"
        return await s.handle("yes for this task", from_user=True)

    r = asyncio.run(app())
    assert ran == ["new game", "pick white"] and r.route == "agent" and confirm.peek() is None


def test_grants_and_untrusted_reads_are_their_own_gates(tmp_path):
    """Keyed by the gate: a task id (and its grant, its read count) from one conversation is no task
    in another — two conversations of one agent, another agent, or NEO.app — even with gates that
    could give grants."""
    t = grants.new_task()  # NEO.app's gate
    with grants.task(t):
        assert isinstance(confirm.gated("zz_step", {"s": 1}, "do 1 in Chess", target=CHESS), str)
        p = confirm.peek()
        assert grants.grant(t, "zz_step", p.target, p.action_id) is not None
        confirm.cancel("test")
        assert grants.covering("zz_step", CHESS) is not None
        grants.read_untrusted("zz_page")
        assert grants.reads(t) == 1 and grants.covering("zz_step", CHESS) is None  # new reading: as on main

    a, b = make_runtime(tmp_path / "a", "a"), make_runtime(tmp_path / "b", "b")
    one, two = a.scope("one"), a.scope("two")
    for sc in (one, two):  # as if both conversations' gates gave grants, like NEO.app's
        sc.gate = confirm.Gate(agent="a", audit_path=a.settings.audit_log)
        sc.gate.grants.grants_on = True
    for sc in (one, two, b.scope()):
        with runtime.bind(sc), grants.task(t):
            assert grants.active(t) == [] and grants.reads(t) == 0
            assert grants.covering("zz_step", CHESS) is None and grants.for_card(p.action_id) is None
            assert isinstance(confirm.gated("zz_step", {"s": 2}, "do 2 in Chess", target=CHESS), str)
            confirm.cancel("test")

    # A grant in conversation one: conversation two (the same task id) gets nothing of it.
    with runtime.bind(one), grants.task(t):
        confirm.gated("zz_step", {"s": 3}, "do 3 in Chess", target=CHESS)
        q = confirm.peek()
        assert grants.grant(t, "zz_step", q.target, q.action_id) is not None
        confirm.cancel("test")
        assert confirm.gated("zz_step", {"s": 4}, "do 4 in Chess", target=CHESS) == {"s": 4}
        grants.read_untrusted("zz_page")
        grants.read_untrusted("zz_page")
        assert grants.reads(t) == 2
    with runtime.bind(two), grants.task(t):
        assert grants.reads(t) == 0 and grants.covering("zz_step", CHESS) is None
    runtime.set_library_mode(False)
    with grants.task(t):  # NEO.app's own count and grant: untouched by all of that
        assert grants.reads(t) == 1 and len(grants.active(t)) == 1
    grants.end(t)


def test_an_approval_in_one_conversation_is_no_approving_in_another(tmp_path):
    a = make_runtime(tmp_path / "a", "a")
    one, two = a.scope("one"), a.scope("two")
    with runtime.bind(one):
        confirm.gated("zz_step", {"s": 1}, "do 1")
        aid = confirm.peek().action_id
        with confirm.approved(aid):
            assert confirm.approving() == aid
            with runtime.bind(two):
                assert confirm.approving() == "" and not confirm.approval_active()


# ---- (b) the brains' learned state ------------------------------------------------------------------


class _Spent:
    """A brain out of quota for an hour."""

    supports_vision = False
    supports_tools = True

    def __init__(self, name: str) -> None:
        self.name, self.model = name, "m"

    async def complete(self, messages, **kw):
        raise QuotaExhausted("429 out of quota", until=time.time() + 3600)

    async def stream(self, messages, **kw):
        raise QuotaExhausted("429 out of quota", until=time.time() + 3600)
        yield ""


def test_quota_demotion_is_the_agents_own(tmp_path, monkeypatch):
    monkeypatch.setattr(prov, "_demoted", {})
    a, b = make_runtime(tmp_path / "a", "a"), make_runtime(tmp_path / "b", "b")
    a.brains = prov.BrainPool(overrides={"agent": [_Spent("spent"), ScriptedBrain(["from ok"], name="ok")]})
    b.brains = prov.BrainPool(overrides={"agent": [_Spent("spent"), ScriptedBrain(["from ok"], name="ok")]})
    with runtime.bind(a.scope()):
        assert asyncio.run(prov.brain("agent").complete([])).text == "from ok"
        until = prov.pool().demoted()["spent"]
        assert until > time.time() + 3000  # a spent quota: demoted until it resets, not 5 minutes
        assert [prov._key(n) for n in prov.brain("agent")._order()] == ["ok", "spent"]
    with runtime.bind(b.scope()):
        assert prov.pool().demoted() == {}
        assert [prov._key(n) for n in prov.brain("agent")._order()] == ["spent", "ok"]
    assert prov._demoted == {}  # NEO.app's


def test_what_a_brain_learns_stays_in_the_agent_that_built_it(tmp_path, monkeypatch):
    """Groq's learned request size (its budget and estimate scale) and the Gemini quota file: each
    agent's own brain object and data dir, nothing in NEO.app's pool, data dir or ~/.neo."""
    pytest.importorskip("openai")
    gem = pytest.importorskip("neo.providers.gemini")

    class Client:
        def __init__(self, api_key, http_options=None):
            self.api_key = api_key

    monkeypatch.setattr(gem.genai, "Client", Client)
    monkeypatch.setattr(prov, "_cache", {})
    a = make_runtime(tmp_path / "a", "a", groq_api_key="g-a", gemini_api_key="k-a")
    b = make_runtime(tmp_path / "b", "b", groq_api_key="g-b", gemini_api_key="k-b")
    made = {}
    for rt in (a, b):
        with runtime.bind(rt.scope()):
            made[rt.name] = (prov.make("groq"), prov.make("gemini"))
    (ga, gema), (gb, gemb) = made["a"], made["b"]
    assert ga is not gb and gema is not gemb
    budget, scale = gb.token_budget, gb._scale
    ga._learn_size(RuntimeError("Limit 5000, Requested 9000"), SimpleNamespace(max_tokens=1000, tokens=4000))
    assert ga.token_budget == 5000 and ga._scale > 1.0
    assert (gb.token_budget, gb._scale) == (budget, scale)
    gema._bad_keys[gem._key_id("k-a")] = 9e18  # A's key refused
    assert gemb._bad_keys == {}
    gema._exhausted[("gemini-x", gem._key_id("k-a"))] = 9e18
    gema._save_cooldowns()
    assert (tmp_path / "a" / gem._COOLDOWN_FILE).exists() and not (tmp_path / "b" / gem._COOLDOWN_FILE).exists()
    assert prov._cache == {}
    assert not (Path(os.environ["NEO_DATA_DIR"]) / gem._COOLDOWN_FILE).exists()  # NEO.app's data dir


# ---- (c) the effects switch in a host ---------------------------------------------------------------


def test_outside_a_test_run_a_host_with_pytest_imported_keeps_its_effects(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("NEO_NO_EFFECTS", raising=False)
    assert effects.why_off() == "running under pytest"  # pytest loaded, nothing embedded: NEO's suite
    make_runtime(tmp_path / "a", "a")  # a host made an agent
    assert effects.why_off() == "" and effects.on()
    effects.guard("input", "post a key")  # passes: nothing refused
    monkeypatch.setenv("NEO_NO_EFFECTS", "1")
    assert effects.why_off() == "NEO_NO_EFFECTS=1"
    monkeypatch.delenv("NEO_NO_EFFECTS")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/x.py::test (call)")
    assert effects.why_off() == "running under pytest"  # a test is running: off, embedded or not


def test_the_suite_itself_runs_embedded_agents_with_effects_off(tmp_path):
    make_runtime(tmp_path / "a", "a")
    assert effects.why_off() == "running under pytest"
    with pytest.raises(effects.EffectRefused):
        effects.guard("input", "post a key")


def test_a_host_process_that_imported_pytest_runs_its_agents_approved_actions(tmp_path):
    """For real, not patched: a fresh process that imports pytest (no test running) and embeds an
    agent. Before the agent, NEO counts as its own test suite; once embedded, the host's approved
    action happens and nothing is refused. Nothing real is done: the checks are asked, not used."""
    home = tmp_path / "home"
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("PYTEST_CURRENT_TEST", "NEO_NO_EFFECTS", "NEO_DATA_DIR")}
    env.update(HOME=str(home), PYTHONDONTWRITEBYTECODE="1")
    code = textwrap.dedent(
        """
        import asyncio, json, sys
        import pytest  # a host that happens to have it loaded
        from neo import effects
        before = effects.why_off()
        import neo2
        from neo2.testing import ScriptedBrain, ToolCall, Turn

        done = []

        @neo2.host_tool(risk="destructive")
        def refund(order: str) -> str:
            \"\"\"Refund an order.\"\"\"
            effects.guard("process", "stand-in for the host's own effect")  # refused if effects were off
            done.append(order)
            return "refunded"

        async def main():
            brain = ScriptedBrain([Turn(tool_calls=[ToolCall("1", "refund", {"order": "812"})]), "Refunded."])
            cfg = neo2.AgentConfig(tools="none", data_dir=sys.argv[1])
            async with neo2.AsyncAgent(cfg, provider=brain, host_tools=[refund]) as agent:
                r = await agent.ask("refund 812")
                after = effects.why_off()
                r = await agent.approve(r.approval.action_id, via=neo2.Approval("test", "host"))
                return after, r.status

        after, status = asyncio.run(main())
        print(json.dumps([before, after, status, done, effects._hooked]))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "agent")], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr[-3000:]
    before, after, status, done, hooked = json.loads(out.stdout.strip().splitlines()[-1])
    assert before == "running under pytest" and hooked  # (the hook is in; it refuses nothing now)
    assert after == "" and status == "done" and done == ["812"]
    assert not (home / ".neo").exists()


# ---- (d) Accessibility off, in a host ---------------------------------------------------------------

_MAC = pytest.mark.skipif(
    sys.platform != "darwin" or importlib.util.find_spec("AppKit") is None, reason="NEO's Mac tools: macOS with pyobjc"
)


@_MAC
def test_accessibility_off_names_the_host_not_neo_app(tmp_path, monkeypatch):
    import neo2
    from neo.tools.computer import ax

    monkeypatch.setattr(ax, "is_trusted", lambda: False)
    monkeypatch.setattr(sys, "executable", "/Applications/Acme Desk.app/Contents/MacOS/python3")

    async def run():
        async with neo2.AsyncAgent(neo2.AgentConfig(tools=("ax_tree",), data_dir=tmp_path / "a"), provider=ScriptedBrain([])) as agent:
            direct = await agent.call_tool("ax_tree", {})
            brain = ScriptedBrain([Turn(tool_calls=[ToolCall("1", "ax_tree", {})]), "(never asked)"])
            agent._rt.brains = prov.BrainPool(overrides={"agent": [brain]})
            asked = await agent.ask("what's on screen?")
            return direct, asked, brain.calls

    direct, asked, calls = asyncio.run(run())
    for r in (direct, asked):
        assert r.status == "needs_user", r
        assert "Accessibility is off for Acme Desk, the app this agent runs in" in r.text
        assert "NEO needs" not in r.text and "turn NEO on" not in r.text
    assert calls == 1  # the run stopped at that question: no model looked for another way in


@_MAC
def test_accessibility_off_for_a_script_names_its_interpreter(tmp_path, monkeypatch):
    from neo.tools import computer

    monkeypatch.setattr(sys, "executable", "/opt/homebrew/bin/python3.12")
    rt = make_runtime(tmp_path / "a", "a")
    with runtime.bind(rt.scope()):
        ask, still = computer._ax_off_texts()
    assert "python3.12" in ask and "the terminal or app that started it" in ask and "NEO" not in ask
    assert still.startswith("Error: Accessibility is still off for the app this agent runs in")


@_MAC
def test_neo_apps_accessibility_question_is_unchanged(monkeypatch):
    from neo.tools import computer, load_all
    from neo.tools.computer import ax

    load_all()
    monkeypatch.setattr(ax, "is_trusted", lambda: False)
    assert computer._ax_off_texts() == (computer.AX_OFF, computer._AX_STILL_OFF)
    out = asyncio.run(registry().invoke("ax_tree", {}, ToolContext()))
    assert out.halt and out.text == f"{confirm.NEEDS_USER}: {computer.AX_OFF}"
