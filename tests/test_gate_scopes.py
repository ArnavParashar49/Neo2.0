"""One confirmation gate per conversation: a yes in one can never approve another's card."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import stat

import pytest
from neo_lib import make_runtime

from neo import runtime
from neo.agent import confirm
from neo.agent.registry import RegisteredTool, ToolContext, registry

_OBJ = {"type": "object", "properties": {}}


def _wire_tool(executed: list) -> RegisteredTool:
    def wire(a, c):
        gate = confirm.gated("zz_wire", a, f"wire {a['amount']} to {a['to']}", detail=json.dumps(a, sort_keys=True))
        if isinstance(gate, str):
            return gate
        executed.append((c.extras.get("who"), dict(gate)))
        return f"wired {a['amount']}"

    return RegisteredTool("zz_wire", "wire money", _OBJ, wire, risk="destructive")


@pytest.fixture
def agents(tmp_path):
    executed: list = []
    a = make_runtime(tmp_path / "a", "a")
    b = make_runtime(tmp_path / "b", "b")
    for rt in (a, b):
        rt.registry.register(_wire_tool(executed))
    return a, b, executed


def _invoke(scope, args, who="", *, approve: str = "", via=None):
    async def go():
        with runtime.bind(scope):
            ctx = ToolContext(extras={"who": who})
            if approve:
                with confirm.approved(approve, via=via):
                    return await registry().invoke("zz_wire", args, ctx)
            return await registry().invoke("zz_wire", args, ctx)

    return asyncio.run(go())


def _pending(scope):
    with runtime.bind(scope):
        return confirm.peek()


A_WIRE = {"amount": 900, "to": "mallory"}


def test_a_card_staged_in_one_conversation_is_invisible_to_others(agents):
    a, b, executed = agents
    out = _invoke(a.scope(), A_WIRE)
    assert out.halt and out.staged and out.text.startswith(confirm.NEEDS_CONFIRM)
    assert _pending(a.scope()).action_id == out.staged
    assert _pending(b.scope()) is None  # another agent
    assert _pending(a.scope("other thread")) is None  # another conversation of the same agent
    assert confirm._default_gate.pending is None  # and never NEO.app's gate
    assert executed == []


@pytest.mark.parametrize("other", ["agent", "conversation"])
def test_approving_another_conversations_id_runs_nothing(agents, other):
    a, b, executed = agents
    aid = _invoke(a.scope(), A_WIRE).staged
    elsewhere = b.scope() if other == "agent" else a.scope("thread-2")
    out = _invoke(elsewhere, A_WIRE, "intruder", approve=aid)
    assert executed == []  # the same operation, the other card's id: nothing ran
    assert out.halt and out.staged != aid  # it only asked, in its own conversation
    with runtime.bind(elsewhere):
        assert confirm.consume_now(aid) is None
        confirm.cancel("test")
    assert _pending(a.scope()).action_id == aid  # A's card is untouched
    out = _invoke(a.scope(), A_WIRE, "owner", approve=aid)
    assert out.ok and executed == [("owner", A_WIRE)]  # A's own yes runs it, exactly once
    assert _pending(a.scope()) is None


def test_an_approval_never_travels_into_another_conversation(agents):
    a, b, executed = agents
    aid = _invoke(a.scope(), A_WIRE).staged
    b_aid = _invoke(b.scope(), A_WIRE).staged  # B staged the very same operation (same key)

    async def go():
        with runtime.bind(a.scope()), confirm.approved(aid):
            assert confirm.approval_active()
            with runtime.bind(b.scope()):
                assert not confirm.approval_active()  # underscore args stay dropped in B
                return await registry().invoke("zz_wire", {**A_WIRE, "_plan": "x"}, ToolContext())

    out = asyncio.run(go())
    assert out.halt and out.staged == b_aid and executed == []
    assert _pending(b.scope()).action_id == b_aid  # B's card is still B's question
    assert _pending(a.scope()) is None  # A's approval ended unused: its card closed, as today


def test_the_same_conversation_approves_as_before(agents):
    a, _, executed = agents
    aid = _invoke(a.scope(), A_WIRE).staged
    changed = _invoke(a.scope(), {**A_WIRE, "amount": 901}, approve=aid)  # not what the card showed
    assert changed.halt and executed == []  # asks again, for the new amount
    aid2 = changed.staged
    assert aid2 != aid and _pending(a.scope()).action_id == aid2  # the old card is superseded
    assert _invoke(a.scope(), A_WIRE, approve=aid).halt  # a stale id approves nothing
    aid3 = _pending(a.scope()).action_id
    out = _invoke(a.scope(), A_WIRE, "me", approve=aid3)
    assert out.ok and executed == [("me", A_WIRE)]


def test_asking_the_same_card_again_keeps_its_id_and_is_collected(agents):
    a, _, _ = agents
    with runtime.bind(a.scope()):
        with confirm.collecting() as first:
            h1 = confirm.stage("zz_wire", A_WIRE, "wire 900")
        with confirm.collecting() as second:
            h2 = confirm.stage("zz_wire", A_WIRE, "wire 900")
        assert h1.action_id == h2.action_id and first == second == [h1.action_id]
        with confirm.collecting() as third:
            h3 = confirm.stage("zz_wire", {**A_WIRE, "amount": 1}, "wire 1")
        assert third == [h3.action_id] and h3.action_id != h1.action_id
        confirm.cancel("test")


async def test_the_sink_sees_what_parallel_read_only_tools_staged(tmp_path):
    """memory is read-only (parallel_safe) yet stages after an untrusted read; the loop runs it
    in an asyncio.gather child, whose ContextVar changes never reach the parent — the list does."""
    from neo.agent.loop import run_agent
    from neo.events import EventBus
    from neo.providers.base import ToolCall, Turn

    rt = make_runtime(tmp_path / "m", "m", ["memory", "clock"], memory_enabled=True)

    class Brain:
        name, supports_vision, supports_tools = "fake", False, True
        turns = [
            Turn(tool_calls=[ToolCall("1", "clock", {}), ToolCall("2", "memory", {"action": "remember", "text": "x y z"})]),
            Turn(text="done"),
        ]

        async def complete(self, messages, **kw):
            return self.turns.pop(0)

    with runtime.bind(rt.scope()), confirm.collecting() as sink:
        ctx = ToolContext(user_text="t", taint=["web_fetch"])
        res = await run_agent("t", provider=Brain(), system="", ctx=ctx, ev=EventBus(), verify=False)
        p = confirm.peek()
    assert res.stopped == "needs_confirm" and p is not None and p.tool == "memory"
    assert sink == [p.action_id]
    with runtime.bind(rt.scope()):
        confirm.cancel("test")


def test_library_mode_never_stages_into_a_shared_slot():
    runtime.set_library_mode()
    for fn in (
        lambda: confirm.gated("shell", {"command": "rm -rf ~"}, "run a command"),
        lambda: confirm.stage("shell", {"command": "x"}, "run"),
        confirm.peek,
        lambda: confirm.cancel("x"),
        confirm.ttl_s,
    ):
        with pytest.raises(runtime.NoRuntimeError):
            fn()
    assert confirm._default_gate.pending is None
    assert not confirm.approval_active()  # nothing approved anywhere


# ---- audit ------------------------------------------------------------------------------------------


def test_neo_apps_audit_lines_are_unchanged(tmp_path):
    from neo.config import settings

    old = os.umask(0o022)
    try:
        h = confirm.stage("shell", {"command": "ls -la", "confirm": True}, "run a command", detail="ls -la")
        with confirm.approved(h.action_id, via=None):
            pass
        confirm.stage("trash", {"path": "/tmp/x"}, "trash it")
        confirm.cancel()
    finally:
        os.umask(old)
    path = settings().audit_log
    lines = path.read_text().splitlines()
    assert [json.loads(ln)["status"] for ln in lines] == ["staged", "confirmed", "staged", "cancelled: user cancelled"]
    for ln in lines:
        rec = json.loads(ln)
        assert list(rec) == ["ts", "status", "action", "tool", "params"]  # no agent / via keys
        assert json.dumps(rec) == ln  # the same serialisation, byte for byte
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


async def test_neo_apps_audit_lines_match_the_released_ones(monkeypatch):
    """Byte for byte what NEO.app wrote before agents existed: the same calls, replayed with
    fixed ids and times, against lines captured at 4764b73 (fixtures/audit_golden.jsonl) —
    staging, the same card again, superseding, approval, cancel, a stray approval, expiry."""
    import secrets
    import time
    from pathlib import Path

    from neo.config import settings

    ids = iter(range(1, 100))
    monkeypatch.setattr(secrets, "token_hex", lambda n: f"{next(ids):016x}")
    monkeypatch.setattr(time, "strftime", lambda fmt, *a: "TS")
    mail = {"to": "a@b", "body": "hello there", "api_key": "sk-1"}
    confirm.stage("shell", {"command": "rm x", "password": "p", "confirm": True}, "delete x", "rm x")
    confirm.stage("shell", {"command": "rm x", "password": "p"}, "delete x", "rm x")  # the same card
    h = confirm.stage("mail_send", mail, "send", "")  # supersedes it
    with confirm.approved(h.action_id):
        assert confirm.gated("mail_send", mail, "send", "") == mail
    confirm.stage("trash", {"path": "/x"}, "trash x")
    confirm.cancel()
    confirm.stage("trash", {"path": "/y"}, "trash y")
    with confirm.approved("nope"):
        pass
    with confirm.approved(confirm.peek().action_id):
        pass
    confirm.stage("x", {"a": 1, "nested": {"token": "t", "n": [1, 2]}}, "x")
    confirm.cancel("stale")
    confirm.stage("y", {"a": 1}, "y")
    monkeypatch.setattr(confirm, "_TTL_S", -1)
    assert confirm.peek() is None
    golden = Path(__file__).parent / "fixtures" / "audit_golden.jsonl"
    assert settings().audit_log.read_text() == golden.read_text()


def test_an_agents_audit_lines_say_which_agent_and_who_approved(agents):
    a, _, _ = agents
    via = {"channel": "web-ui", "approver": "user:42"}
    aid = _invoke(a.scope(), A_WIRE).staged
    _invoke(a.scope(), A_WIRE, approve=aid, via=via)
    aid2 = _invoke(a.scope(), {**A_WIRE, "amount": 5}).staged
    with runtime.bind(a.scope()):
        confirm.cancel("denied", via={"channel": "none"})
    recs = [json.loads(ln) for ln in a.settings.audit_log.read_text().splitlines()]
    assert [(r["status"], r["action"], r.get("via")) for r in recs] == [
        ("staged", aid, None),
        ("confirmed", aid, via),
        ("staged", aid2, None),
        ("cancelled: denied", aid2, {"channel": "none"}),
    ]
    assert {r["agent"] for r in recs} == {"a"}
    assert stat.S_IMODE(a.settings.audit_log.stat().st_mode) == 0o600


def test_without_an_audit_file_records_go_to_the_sink_redacted(tmp_path, caplog):
    got: list[dict] = []
    rt = make_runtime(tmp_path / "s", "s")
    rt.audit_file, rt.audit_sink = False, got.append
    with runtime.bind(rt.scope()), caplog.at_level(logging.INFO, logger="neo.audit"):
        confirm.stage("mail_send", {"to": "a@b.c", "body": "my diagnosis", "api_key": "sk-1"}, "send mail")
        confirm.cancel("test")
    assert [r["status"] for r in got] == ["staged", "cancelled: test"]
    assert got[0]["params"] == {"to": "a@b.c", "body": "<12 chars>", "api_key": "***"}
    assert got[0]["agent"] == "s"
    assert not (tmp_path / "s" / "audit.log").exists()
    assert any("mail_send" in r.getMessage() and "diagnosis" not in r.getMessage() for r in caplog.records)


def test_a_failing_sink_never_blocks_the_gate(tmp_path):
    rt = make_runtime(tmp_path / "s", "s")

    def boom(rec):
        raise RuntimeError("host bug")

    rt.audit_sink = boom
    with runtime.bind(rt.scope()):
        h = confirm.stage("trash", {"path": "/x"}, "trash /x")
        assert confirm.peek().action_id == h.action_id
        confirm.cancel("test")
    assert "staged" in rt.settings.audit_log.read_text()


def test_each_audit_file_is_pruned(tmp_path, monkeypatch):
    old = json.dumps({"ts": "2020-01-01T00:00:00", "status": "confirmed", "action": "x", "tool": "shell", "params": {}})
    rts = [make_runtime(tmp_path / n, n) for n in ("p", "q")]
    monkeypatch.setattr(confirm, "_last_prune_at", {})
    for rt in rts:
        rt.settings.audit_log.write_text(old + "\n")
        with runtime.bind(rt.scope()):
            confirm.audit("y:1", "shell", {"command": "ls"}, "staged")
    for rt in rts:
        assert "2020-01-01" not in rt.settings.audit_log.read_text()


# ---- expiry -----------------------------------------------------------------------------------------


def test_the_ttl_is_read_at_call_time_and_per_gate(agents, monkeypatch):
    a, b, _ = agents
    with runtime.bind(a.scope()):
        assert confirm.ttl_s() == 120.0
        monkeypatch.setattr(confirm, "_TTL_S", 5.0)
        assert confirm.ttl_s() == 5.0
        confirm.current_gate().ttl_s = 0.0
        confirm.stage("zz_wire", A_WIRE, "wire")
        assert confirm.peek() is None  # lapsed at once, in this gate only
    with runtime.bind(b.scope()):
        confirm.stage("zz_wire", A_WIRE, "wire")
        assert confirm.peek() is not None and confirm.ttl_s() == 5.0
        confirm.cancel("test")


async def test_the_expiry_check_closes_the_card_in_its_own_conversation(agents):
    a, b, _ = agents
    with runtime.bind(a.scope()):
        confirm.current_gate().ttl_s = -0.95  # stage() schedules its check at ttl + 1 s = 0.05 s
        confirm.stage("zz_wire", A_WIRE, "wire")
    with runtime.bind(b.scope()):
        confirm.stage("zz_wire", A_WIRE, "wire")
    await asyncio.sleep(0.2)  # nobody calls peek(): the scheduled check does, in A's gate
    assert a.scope().gate.pending is None
    assert b.scope().gate.pending is not None
    with runtime.bind(b.scope()):
        confirm.cancel("test")


def test_an_odd_via_value_never_costs_the_audit_line(agents):
    import datetime

    a, _, _ = agents
    aid = _invoke(a.scope(), A_WIRE).staged
    _invoke(a.scope(), A_WIRE, approve=aid, via={"channel": "web-ui", "at": datetime.date(2026, 9, 27)})
    last = json.loads(a.settings.audit_log.read_text().splitlines()[-1])
    assert last["status"] == "confirmed" and last["via"] == {"channel": "web-ui", "at": "2026-09-27"}
