"""What main added that is process-wide in NEO.app, kept per agent when NEO is embedded.

* Task grants ("yes for this task") and the untrusted-read counter live in each conversation's
  confirmation gate: they never cross conversations or agents, and an embedded agent's gate gives
  no grants at all (its cards offer no scope, every approval is one-off). NEO.app: as on main.
* The brains' learned state (quota windows and their file, demotion, the request-size scale) is
  the agent's own pool's, in its own data dir — never NEO.app's globals or ~/.neo.
* The effects switch is NEO's own tests': a host with NEO installed keeps its effects, and its agents'
  approved actions happen, even under its own pytest run.
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


def test_the_suite_stays_off_whatever_a_test_does(tmp_path, monkeypatch):
    """NEO's own test process has its effects off from its conftest on: no test's environment or
    embedded agent turns them on, even for a moment (a thread another test left running could act)."""
    monkeypatch.delenv("NEO_NO_EFFECTS", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    make_runtime(tmp_path / "a", "a")
    assert effects.why_off() == "running under pytest" and effects._hooked
    with pytest.raises(effects.EffectRefused):
        effects.guard("input", "post a key")
    with pytest.raises(effects.EffectRefused):
        effects._audit("subprocess.Popen", ("open", ["open", "-a", "Safari"], None, None))


def test_the_suite_itself_runs_embedded_agents_with_effects_off(tmp_path):
    make_runtime(tmp_path / "a", "a")
    assert effects.why_off()  # "running under pytest", or "NEO_NO_EFFECTS=1" when that is exported too
    with pytest.raises(effects.EffectRefused):
        effects.guard("input", "post a key")


def _installed(tmp_path: Path) -> Path:
    """NEO as another app has it: installed into that app's environment (here its two packages,
    linked into a folder of their own), not run from NEO's source tree."""
    site = tmp_path / "site"
    site.mkdir()
    for pkg in ("neo", "neo2"):
        (site / pkg).symlink_to(ROOT / pkg, target_is_directory=True)
    return site


_HOST = textwrap.dedent(
    """
    import asyncio, json, socket, sys
    import pytest  # the host's own test run (or a notebook that has it loaded)
    import neo2
    from neo import effects
    from neo2.testing import ScriptedBrain, ToolCall, Turn

    assert "/site/neo/" in effects.__file__, effects.__file__  # the installed copy, not the source tree
    before = [effects.why_off(), effects._hooked]
    # The host's own program and its own service (a CI database): nothing of NEO's refuses them.
    # (Asked of the check itself: nothing is started or connected.)
    effects._audit("subprocess.Popen", ("/usr/bin/true", ["/usr/bin/true"], None, None))
    effects._audit("socket.connect", (socket.socket(socket.AF_INET, socket.SOCK_STREAM), ("postgres", 5432)))

    done = []

    @neo2.host_tool(risk="destructive")
    def refund(order: str) -> str:
        \"\"\"Refund an order.\"\"\"
        effects.guard("process", "stand-in for the host's own effect")  # refused if effects were off
        done.append(order)
        return "refunded"

    async def main():
        brain = ScriptedBrain([Turn(tool_calls=[ToolCall("1", "refund", {"order": "812"})]), "Refunded."])
        cfg = neo2.AgentConfig(tools="none", fast_paths=False, data_dir=sys.argv[1])
        async with neo2.AsyncAgent(cfg, provider=brain, host_tools=[refund]) as agent:
            r = await agent.ask("refund 812")
            r = await agent.approve(r.approval.action_id, via=neo2.Approval("test", "host"))
            return effects.why_off(), r.status

    after, status = asyncio.run(main())
    print(json.dumps([before, after, status, done, effects._hooked]))
    """
)


def _child(code: str, tmp_path: Path, *, pythonpath: Path, cwd: Path, **env: str):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    base = {k: v for k, v in os.environ.items() if k not in ("PYTEST_CURRENT_TEST", "NEO_NO_EFFECTS", "NEO_DATA_DIR")}
    base.update(HOME=str(home), PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(pythonpath), **env)
    with effects.allow("process"):  # a Python child: refused unless meant, as here
        out = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path / "agent")], cwd=cwd, env=base, capture_output=True, text=True, timeout=120
        )
    assert out.returncode == 0, out.stderr[-3000:]
    assert not (home / ".neo").exists()
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("in_a_test", [True, False])
def test_a_host_with_neo_installed_keeps_its_own_effects_and_its_agents(tmp_path, in_a_test):
    """For real, not patched: a fresh process with NEO installed, pytest imported and — in_a_test —
    one of the host's own tests running. NEO's switch is NEO's own tests': the host's programs and
    connections, before any agent as after, and its agents' approved actions are its business."""
    env = {"PYTEST_CURRENT_TEST": "host_tests/test_billing.py::test_refund (call)"} if in_a_test else {}
    before, after, status, done, hooked = _child(_HOST, tmp_path, pythonpath=_installed(tmp_path), cwd=tmp_path, **env)
    assert before == ["", False] and after == "" and not hooked  # no hook: nothing of the host's is refused
    assert status == "done" and done == ["812"]


def test_a_host_that_sets_neo_no_effects_gets_the_switch(tmp_path):
    """NEO_NO_EFFECTS=1 in a host's environment: NEO's switch, for that whole process."""
    code = textwrap.dedent(
        """
        import json
        import neo2
        from neo import effects
        try:
            effects._audit("subprocess.Popen", ("/usr/bin/true", ["/usr/bin/true"], None, None))
            refused = False
        except effects.EffectRefused:
            refused = True
        print(json.dumps([effects.why_off(), effects._hooked, refused]))
        """
    )
    got = _child(code, tmp_path, pythonpath=_installed(tmp_path), cwd=tmp_path, NEO_NO_EFFECTS="1")
    assert got == ["NEO_NO_EFFECTS=1", True, True]


def test_neo_from_its_source_tree_under_pytest_is_neos_own_test_run(tmp_path):
    """A scratch test run from NEO's source tree without the suite's conftest: pytest is loaded, so
    it is one of NEO's own test runs — off, and it stays off once it embeds an agent."""
    code = textwrap.dedent(
        """
        import asyncio, json, sys
        import pytest
        import neo2
        from neo import effects
        from neo2.testing import ScriptedBrain

        async def main():
            cfg = neo2.AgentConfig(tools="none", fast_paths=False, data_dir=sys.argv[1])
            async with neo2.AsyncAgent(cfg, provider=ScriptedBrain(["hi"])):
                return effects.why_off()

        print(json.dumps([effects.why_off(), effects._hooked, asyncio.run(main())]))
        """
    )
    got = _child(code, tmp_path, pythonpath=ROOT, cwd=ROOT)
    assert got == ["running under pytest", True, "running under pytest"]


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
