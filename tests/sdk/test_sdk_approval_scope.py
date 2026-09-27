"""An approval covers the one approved call, and nothing that call starts.

A host tool that, once approved, hands a follow-up to NEO (ask, stream or call_tool on the same
agent) or starts background work: that work never runs under the approval (asyncio copies the
approval's context variable into every task started inside it). A follow-up in the approved call's
own conversation is refused at once (ApprovalReentryError): it would run inside that conversation's
approval, and a card it staged could never be answered while the approved call holds it. Another
conversation's follow-up is a request of its own, asked afresh."""

from __future__ import annotations

import asyncio

import pytest

import neo2
from neo2.core.agent import confirm
from neo2.testing import ScriptedBrain, ToolCall, Turn

pytestmark = pytest.mark.portable

ARGS = {"to": "mallory", "amount": 1000}
PAY = Turn(tool_calls=[ToolCall("1", "wire_money", ARGS)])


class _Brain(ScriptedBrain):
    """Plays each request its own turns: those of the first key found in the latest user message
    (a request's words, or the "[User confirmed …]" note an approval continues with)."""

    def __init__(self, scripts: dict[str, list]) -> None:
        super().__init__([])
        self.scripts = {k: [Turn(text=t) if isinstance(t, str) else t for t in v] for k, v in scripts.items()}

    async def complete(self, messages, **kw):
        self.calls += 1
        last = next((m.text for m in reversed(messages) if m.role == "user" and m.text), "")
        for key, turns in self.scripts.items():
            if key in last:
                return turns.pop(0) if turns else Turn(text="(no more turns)")
        return Turn(text="(no script)")


def _script(**follow_ups: list) -> dict[str, list]:
    """The outer request pays once and ends; `follow_ups`: a follow-up's words -> its turns."""
    return {"User confirmed": ["Sent."], **{k.replace("_", " "): v for k, v in follow_ups.items()}, "pay mallory": [PAY]}


class _Host:
    """A host with one destructive tool, wire_money, whose approved run does `follow_up` once."""

    def __init__(self, scripts, follow_up=None, *, approver=None, extra_tools=()) -> None:
        self.wires: list[int] = []  # how many approvals had been given at each wire
        self.asked: list[str] = []
        self.errors: list[BaseException] = []
        self.results: list = []
        self.bg: list[asyncio.Task] = []
        host = self

        @neo2.host_tool(risk="destructive", confirm=lambda a: f"wire {a['amount']} to {a['to']}")
        async def wire_money(to: str, amount: float) -> str:
            """Send money."""
            host.wires.append(len(host.asked))
            if len(host.wires) == 1 and follow_up is not None:
                try:
                    host.results.append(await follow_up(host.agent))
                except neo2.ApprovalReentryError as e:
                    host.errors.append(e)
            return f"wired {amount} to {to}"

        async def inline(req):
            self.asked.append(req.summary)
            return neo2.Approval("test-ui", "u1")

        self.agent = neo2.AsyncAgent(
            neo2.AgentConfig(tools="none", fast_paths=False),
            provider=_Brain(scripts),
            approver=inline if approver is None else approver,
            host_tools=[wire_money, *extra_tools],
        )


async def _ask(host: _Host, text: str = "pay mallory 1000") -> neo2.Result:
    async with host.agent:
        return await asyncio.wait_for(host.agent.ask(text), timeout=5)  # never a hang


async def test_asking_again_in_its_own_conversation_is_refused():
    """The nested run would have let its model repeat the wire on the one approval (it did: three
    more wires) — and a card it staged would have waited forever for the approved call's lock."""

    async def follow_up(agent):
        return await agent.ask("let the customer know", source="model")

    host = _Host(_script(let_the_customer_know=[PAY, PAY, PAY, "told them"]), follow_up)
    r = await _ask(host)
    assert host.wires == [1] and len(host.asked) == 1  # one approval, one wire
    (e,) = host.errors
    assert "its own approved action" in str(e) and "default" in str(e)
    assert r.status == "done"


async def test_streaming_or_calling_a_tool_in_its_own_conversation_is_refused():
    async def follow_up(agent):
        with pytest.raises(neo2.ApprovalReentryError):
            agent.stream("and this?")
        async with agent.turn() as t:
            for _ in range(3):
                await agent.call_tool("wire_money", ARGS, turn=t)

    host = _Host(_script(), follow_up)
    r = await _ask(host)
    assert host.wires == [1] and len(host.asked) == 1 and len(host.errors) == 1 and r.status == "done"


async def test_a_card_the_follow_up_would_stage_never_hangs_the_request():
    """The reviewer's hang: the nested run staged send_mail, the approver said yes, and its answer
    waited for the lock the outer approved answer held. Now the follow-up is refused before it runs."""
    sent: list[str] = []

    @neo2.host_tool(risk="destructive", confirm=lambda a: f"mail {a['to']}")
    async def send_mail(to: str, body: str) -> str:
        """Send a mail."""
        sent.append(to)
        return "sent"

    async def follow_up(agent):
        return await agent.ask("tell the customer it's paid", source="model")

    mail = Turn(tool_calls=[ToolCall("2", "send_mail", {"to": "mallory", "body": "paid"})])
    host = _Host(_script(tell_the_customer=[mail, "mailed"]), follow_up, extra_tools=[send_mail])
    r = await _ask(host)
    assert r.status == "done" and sent == [] and len(host.errors) == 1 and host.wires == [1]


async def test_another_conversations_follow_up_is_asked_afresh():
    async def follow_up(agent):
        return await agent.ask("pay her again", conversation="other", source="model")

    scripts = _script(pay_her_again=[PAY])
    scripts["User confirmed"] = ["paid again", "Sent."]  # the follow-up's approval continues first
    host = _Host(scripts, follow_up)
    r = await _ask(host)
    assert host.wires == [1, 2] and len(host.asked) == 2  # its own approval, not the first one's
    assert host.results[0].status == "done" and r.status == "done" and host.errors == []


async def test_the_other_conversation_cannot_come_back_into_the_approved_one():
    """Approved in "default", the follow-up in "other" calls a tool that asks in "default" again:
    that is still inside the default conversation's approved call, and refused."""
    inner: list = []

    @neo2.host_tool
    async def notify(who: str) -> str:
        """Tell someone."""
        try:
            await host.agent.ask("wire it once more", source="model")
        except neo2.ApprovalReentryError as e:
            inner.append(e)
        return "noted"

    async def follow_up(agent):
        return await agent.ask("tell mallory", conversation="other", source="model")

    note = Turn(tool_calls=[ToolCall("2", "notify", {"who": "mallory"})])
    host = _Host(_script(tell_mallory=[note, "told"], wire_it_once_more=[PAY, "wired"]), follow_up, extra_tools=[notify])
    r = await _ask(host)
    assert len(inner) == 1 and host.wires == [1] and r.status == "done"


async def test_work_the_approved_call_leaves_running_gets_no_approval():
    """Fire and forget: a task started inside the approved call runs after it (and approve()) have
    returned. Its wire is a new card, asked for like any other."""
    later: list[str] = []
    started: list[asyncio.Task] = []

    async def follow_up(agent):
        async def retry_later():
            await asyncio.sleep(0.05)
            async with agent.turn() as t:
                later.append((await agent.call_tool("wire_money", ARGS, turn=t)).status)

        started.append(asyncio.create_task(retry_later()))

    host = _Host(_script(), follow_up, approver="defer")
    async with host.agent:
        r = await host.agent.ask("pay mallory")
        assert r.status == "needs_approval"
        r = await host.agent.approve(r.approval.action_id, via=neo2.Approval("ui", "u"))
        assert r.status == "done"
        await asyncio.wait_for(started[0], timeout=10)
    assert later == ["needs_approval"] and host.wires == [0]


async def test_an_approval_ends_with_its_block_for_every_task_that_holds_it(tmp_path):
    """NEO's own gate: a task started inside approved() keeps the approval in its context, but once
    the block is over it approves nothing there (consume, approval_active, approving)."""
    from neo2.core import runtime

    agent = neo2.AsyncAgent(neo2.AgentConfig(tools="none"), provider=ScriptedBrain([]))
    async with agent:
        sc = agent._scope("default")
        with runtime.bind(sc):
            key_args = ("zz_pay", {"to": "mallory"}, "pay mallory")
            staged = confirm.gated(*key_args)
            assert isinstance(staged, confirm.Halt)
            go = asyncio.Event()
            seen: list = []

            async def later():
                await go.wait()
                seen.append((confirm.approval_active(), confirm.approving(), confirm.gated(*key_args)))

            with confirm.approved(staged.action_id):
                assert confirm.gated(*key_args) == {"to": "mallory"}  # the approved call itself
                task = asyncio.create_task(later())
            go.set()
            await task
    ((active, approving, again),) = seen
    assert not active and approving == "" and isinstance(again, confirm.Halt)  # a new card, not a replay
