"""A card belongs to the agent and the conversation that staged it: another agent — or another
conversation of the same agent — can't see it, answer it, approve it or supersede it. (The
scratch repro behind the SDK: B's user's "yes" ran A's staged wire_money.)"""

from __future__ import annotations

import pytest

import neo2
from neo2.testing import ScriptedBrain, ToolCall, Turn

pytestmark = pytest.mark.portable  # no macOS or extra needed: CI's sdk-core job runs it on Linux too


def _wire(calls: list):
    @neo2.host_tool(risk="destructive", confirm=lambda a: f"wire {a['amount']} to {a['to']}")
    async def wire_money(to: str, amount: float) -> str:
        """Send money."""
        calls.append((to, amount))
        return f"wired {amount} to {to}"

    return wire_money


def _pay(to: str, n: str = "1") -> Turn:
    return Turn(tool_calls=[ToolCall(n, "wire_money", {"to": to, "amount": 1000})])


async def test_another_agent_can_neither_see_nor_approve_a_card():
    calls: list = []
    audit_a: list = []
    audit_b: list = []
    a = neo2.AsyncAgent(
        neo2.AgentConfig(name="a", tools="none"),
        provider=ScriptedBrain([_pay("mallory"), "A: sent."]),
        host_tools=[_wire(calls)],
        on_audit=audit_a.append,
    )
    b = neo2.AsyncAgent(
        neo2.AgentConfig(name="b", tools="none"),
        provider=ScriptedBrain(["B here.", "B again."]),
        host_tools=[_wire(calls)],
        on_audit=audit_b.append,
    )
    async with a, b:
        ra = await a.ask("pay mallory")
        assert ra.status == "needs_approval" and calls == []
        aid = ra.approval.action_id
        assert a.pending() == ra.approval
        assert b.pending() is None

        rb = await b.approve(aid, via=neo2.Approval("web-ui", "user:b"))
        assert rb.status == "expired" and calls == []
        assert (await b.deny(aid)).status == "expired"
        # B's user typing "yes" in B's own chat is just a message to B
        assert (await b.ask("yes")).status == "done" and calls == []
        assert a.pending() is not None and a.pending().action_id == aid

        r = await a.approve(aid, via=neo2.Approval("web-ui", "user:a"))
        assert r.status == "done" and r.text == "A: sent." and calls == [("mallory", 1000.0)]
        again = await a.approve(aid, via=neo2.Approval("web-ui", "user:a"))
        assert again.status == "expired" and calls == [("mallory", 1000.0)]  # exactly once

    assert [(x["status"], x["agent"]) for x in audit_a] == [("staged", "a"), ("confirmed", "a")]
    assert audit_a[1]["via"] == {"channel": "web-ui", "approver": "user:a"}
    assert audit_b == []


async def test_two_conversations_of_one_agent_have_their_own_cards():
    calls: list = []
    brain = ScriptedBrain([_pay("mallory"), "Hi Bob.", _pay("trent", "2"), "Alice: sent.", "Bob: sent."])
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="none"), provider=brain, host_tools=[_wire(calls)]) as agent:
        ra = await agent.ask("pay mallory", conversation="alice")
        aid = ra.approval.action_id
        assert ra.approval.conversation == "alice"
        assert agent.pending("bob") is None and agent.pending("alice").action_id == aid

        # Bob's "yes" answers nothing of Alice's; Bob's own card doesn't supersede hers
        assert (await agent.ask("yes", conversation="bob")).status == "done" and calls == []
        rb = await agent.ask("pay trent", conversation="bob")
        assert rb.status == "needs_approval" and rb.approval.action_id != aid
        assert agent.pending("alice").action_id == aid

        r1 = await agent.approve(aid, via=neo2.Approval("app", "alice"))
        assert r1.status == "done" and r1.conversation == "alice" and calls == [("mallory", 1000.0)]
        r2 = await agent.approve(rb.approval.action_id, via=neo2.Approval("app", "bob"))
        assert r2.status == "done" and r2.conversation == "bob"
        assert calls == [("mallory", 1000.0), ("trent", 1000.0)]


async def test_a_card_is_the_agents_own_even_with_the_same_tool_and_arguments():
    """Two agents staging the identical operation get different ids; each id approves only its own."""
    calls_a: list = []
    calls_b: list = []
    a = neo2.AsyncAgent(neo2.AgentConfig(name="a", tools="none"), provider=ScriptedBrain([_pay("x"), "ok"]), host_tools=[_wire(calls_a)])
    b = neo2.AsyncAgent(neo2.AgentConfig(name="b", tools="none"), provider=ScriptedBrain([_pay("x"), "ok"]), host_tools=[_wire(calls_b)])
    async with a, b:
        ra, rb = await a.ask("pay x"), await b.ask("pay x")
        assert ra.approval.action_id != rb.approval.action_id
        assert (await a.approve(rb.approval.action_id, via=neo2.Approval("t", "u"))).status == "expired"
        assert (await a.approve(ra.approval.action_id, via=neo2.Approval("t", "u"))).status == "done"
        assert calls_a == [("x", 1000.0)] and calls_b == []
        assert b.pending().action_id == rb.approval.action_id
