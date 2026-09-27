"""NEO embedded in another app, end to end, with a scripted brain (no API key, no network).

    python examples/embed_quickstart.py

1. A host tool that moves money (risk="destructive"): NEO stages it and asks first.
2. The host shows the card to its human and approves it from its own code, with who approved.
3. The same agent streams a second request's events.
4. An agent whose approver is the host's own dialog (here: a function that says yes).

Swap ScriptedBrain for a real provider (neo2.providers.OpenAICompatible, Gemini, Claude…) to use it
for real. The agent keeps nothing on disk here: with no data_dir it works in a private temp folder
that is removed when it closes.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from a checkout

import neo2
from neo2.testing import ScriptedBrain, ToolCall, Turn


async def main() -> None:
    # What a model would do: call the host's tool, then answer once it ran.
    brain = ScriptedBrain(
        [
            Turn(tool_calls=[ToolCall("c1", "refund", {"order_id": "812", "amount": 49.5})]),
            "Refunded 49.50 on order 812.",
        ]
    )
    audit: list[dict] = []
    async with neo2.AsyncAgent(neo2.AgentConfig(name="shop", tools="none"), provider=brain, on_audit=audit.append) as neo:

        @neo.tool(risk="destructive", confirm=lambda a: f"refund {a['amount']} on order {a['order_id']}")
        async def refund(order_id: str, amount: float, ctx: neo2.ToolContext) -> str:
            """Refund an order (in part or in full)."""
            return f"refund of {amount} issued on {order_id} for {ctx.extras['tenant']}"

        r = await neo.ask("Refund order 812 in full", tool_context={"tenant": "acme"})
        print(f"1. {r.status}: {r.approval.summary}  [shows: {r.approval.detail}]")

        # ...the host's human clicks "Approve" in the host's own UI:
        r = await neo.approve(r.approval.action_id, via=neo2.Approval("web-ui", "user:42"))
        print(f"2. {r.status}: {r.text}  (steps: {[s.tool for s in r.steps]})")
        print(f"   audit: {[(a['status'], a.get('via', {}).get('approver')) for a in audit]}")

        brain.add("Hello from NEO.")
        print("3.", " ".join(e.type for e in [e async for e in neo.stream("say hello", conversation="other")]))

    async def my_dialog(req: neo2.ApprovalRequest) -> neo2.Approval | None:
        ok = True  # a real host asks its human here, showing req.summary and req.detail verbatim
        return neo2.Approval("desktop-dialog", "demo-user") if ok else None

    brain2 = ScriptedBrain([Turn(tool_calls=[ToolCall("c1", "send_invoice", {"to": "ada@example.com"})]), "Sent."])
    async with neo2.AsyncAgent(neo2.AgentConfig(tools="none"), provider=brain2, approver=my_dialog) as neo:

        @neo.tool(risk="destructive", untrusted=False)
        def send_invoice(to: str) -> str:
            """Email this month's invoice."""
            return f"invoice sent to {to}"

        r = await neo.ask("send Ada her invoice")
        print(f"4. {r.status}: {r.text}")


if __name__ == "__main__":
    asyncio.run(main())
