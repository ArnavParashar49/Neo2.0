"""The one agent loop.

Model picks tool calls → we run them (parallel when safe) → results go back → repeat, until
the model answers in text. Guardrails are structural: step budget, thrash guard, and a hard
stop on NEEDS_CONFIRM / NEEDS_USER that hands control to the human.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Literal

from neo.agent.confirm import NEEDS_CONFIRM, NEEDS_USER
from neo.agent.registry import Registry, ToolContext, ToolOutput
from neo.agent.registry import registry as default_registry
from neo.config import settings
from neo.events import EventBus, NeoState
from neo.events import bus as default_bus
from neo.providers.base import Message, Provider, ToolCall, ToolResult, ToolSpec, Turn

Stopped = Literal["done", "max_steps", "needs_confirm", "needs_user", "thrash", "error"]

_VERIFY_NOTE = (
    "[verify] Before you report, confirm the goal was actually achieved with a read-only check "
    "(ax_tree, read_file, browser_read, mail_unread, calendar_today…). If it is done, give the final "
    "answer. If something is off, fix it first. Do not mention this note."
)

_SEARCHY = ("search", "fetch", "web", "lookup", "find")


@dataclass
class Step:
    tool: str
    args: dict
    result: str
    ok: bool


@dataclass
class AgentResult:
    text: str
    stopped: Stopped = "done"
    steps: list[Step] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)  # full transcript for continuation
    question: str = ""  # populated on needs_confirm / needs_user
    pending_action: str = ""  # confirm.Pending.action_id the question refers to


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    return text[:head] + f"\n…[{len(text) - limit} chars trimmed]…\n" + text[-(limit - head) :]


def _halt_kind(text: str) -> Stopped | None:
    t = text.lstrip()
    if t.startswith(NEEDS_CONFIRM):
        return "needs_confirm"
    if t.startswith(NEEDS_USER):
        return "needs_user"
    return None


async def _settle(ev: EventBus, idle: bool) -> None:
    """End of a run: go idle unless a voice session is driving the orb (it decides what's next)."""
    if idle:
        await ev.set_state(NeoState.IDLE)


async def _run_calls(
    calls: list[ToolCall], reg: Registry, ctx: ToolContext, ev: EventBus, limit: int
) -> list[ToolResult]:
    async def one(c: ToolCall) -> ToolResult:
        await ev.tool_start(c.name, c.args)
        t = reg.get(c.name)
        if t and any(k in c.name for k in _SEARCHY):
            await ev.set_state(NeoState.SEARCHING)
        out: ToolOutput = await reg.invoke(c.name, c.args, ctx)
        await ev.tool_end(c.name, out.ok, out.text)
        return ToolResult(
            call_id=c.id, name=c.name, content=_truncate(out.text, limit), ok=out.ok, images=out.images
        )

    # Keep the model's order: consecutive read-only (parallel_safe) calls run together, any
    # side-effect call runs at its own position. Once a call asks the human, nothing after it
    # runs — the rest get synthetic results so every tool_call still has a matching result.
    results: list[ToolResult] = []
    halted = False
    i = 0
    while i < len(calls):
        if halted:
            c = calls[i]
            results.append(ToolResult(c.id, c.name, "skipped: waiting for the user's confirmation", ok=False))
            i += 1
            continue
        t = reg.get(calls[i].name)
        if t and t.parallel_safe:
            j = i
            while j < len(calls) and (tj := reg.get(calls[j].name)) and tj.parallel_safe:
                j += 1
            batch = await asyncio.gather(*(one(c) for c in calls[i:j]))
            results.extend(batch)
            i = j
        else:
            results.append(await one(calls[i]))
            i += 1
        if any(_halt_kind(r.content) for r in results):
            halted = True
    return results


async def run_agent(
    goal: str,
    *,
    provider: Provider,
    system: str,
    history: list[Message] | None = None,
    ctx: ToolContext | None = None,
    reg: Registry | None = None,
    ev: EventBus | None = None,
    max_steps: int | None = None,
    effort: Literal["low", "medium", "high"] = "medium",
    images: list | None = None,
    verify: bool = True,
    idle_on_finish: bool = True,
    max_seconds: float | None = None,
    tools: list[ToolSpec] | None = None,
) -> AgentResult:
    s = settings()
    reg = reg or default_registry()
    ev = ev or default_bus()
    ctx = ctx or ToolContext(user_text=goal)
    max_steps = max_steps or s.max_steps
    import time as _time

    deadline = _time.monotonic() + (max_seconds or s.max_seconds)

    messages: list[Message] = list(history or [])
    if goal:
        messages.append(Message.user(goal, images=images or []))
    steps: list[Step] = []
    last_key, repeats = "", 0
    tools = list(tools) if tools is not None else reg.specs()
    enabled = {t.name for t in tools}
    effectful = False  # a side-effect tool ran this run
    last_readonly = True  # the most recent tool call was an observation
    verified = False

    for _ in range(max_steps):
        if _time.monotonic() > deadline:
            await _settle(ev, idle_on_finish)
            summary = "; ".join(f"{st.tool}: {'ok' if st.ok else 'failed'}" for st in steps[-4:])
            return AgentResult(
                f"This is taking longer than I allow myself ({int(max_seconds or s.max_seconds)}s), so I stopped. "
                f"Last steps — {summary or 'none'}.",
                "max_steps",
                steps,
                messages,
            )
        await ev.set_state(NeoState.THINKING)
        try:
            turn: Turn = await provider.complete(messages, system=system, tools=tools, effort=effort)
        except Exception as e:  # noqa: BLE001
            await _settle(ev, idle_on_finish)
            return AgentResult(f"I hit a problem talking to the model: {e}", "error", steps, messages)

        messages.append(Message.assistant(turn.text, turn.tool_calls, raw=turn.raw))
        if not turn.tool_calls:
            # Acted without looking afterwards? Make it check its work once before reporting.
            if verify and effectful and not last_readonly and not verified:
                verified = True
                messages.append(Message.user(_VERIFY_NOTE))
                continue
            await _settle(ev, idle_on_finish)
            return AgentResult(turn.text.strip(), "done", steps, messages)

        # Thrash guard — the same side-effect call three turns in a row means the model is stuck.
        # Read-only observations (ax_tree, read_file…) legitimately repeat in observe/act loops.
        key = "|".join(
            c.name + json.dumps(c.args, sort_keys=True, default=str)
            for c in turn.tool_calls
            if not ((t := reg.get(c.name)) and t.parallel_safe)
        )
        repeats = repeats + 1 if key and key == last_key else 0
        last_key = key
        if repeats >= 2:
            # Keep the transcript valid: every tool_call must have a matching result.
            messages.append(
                Message.tool(
                    [
                        ToolResult(c.id, c.name, "skipped: repeated call, loop stopped", ok=False)
                        for c in turn.tool_calls
                    ]
                )
            )
            await _settle(ev, idle_on_finish)
            name = turn.tool_calls[0].name
            return AgentResult(
                f"I kept repeating the same step ({name}) without progress, so I stopped.",
                "thrash",
                steps,
                messages,
            )

        # `more_tools` is answered here, not by the registry: it grows this run's tool list.
        meta = [c for c in turn.tool_calls if c.name == "more_tools"]
        real = [c for c in turn.tool_calls if c.name != "more_tools"]
        meta_results: list[ToolResult] = []
        for c in meta:
            from neo.agent.toolselect import selector

            found = selector().search(str(c.args.get("query", "")), exclude=enabled)
            for t in found:
                tools.append(t.spec())
                enabled.add(t.name)
            listing = ", ".join(t.name for t in found) or "nothing new"
            meta_results.append(
                ToolResult(c.id, "more_tools", f"Enabled: {listing}. They are available from your next step.")
            )
        await ev.set_state(NeoState.WORKING)
        results = meta_results + (
            await _run_calls(real, reg, ctx, ev, s.max_tool_result_chars) if real else []
        )
        results.sort(key=lambda r: [c.id for c in turn.tool_calls].index(r.call_id))
        messages.append(Message.tool(results))
        for c, r in zip(turn.tool_calls, results, strict=False):
            steps.append(Step(c.name, c.args, r.content, r.ok))
            t = reg.get(c.name)
            readonly = bool(t and t.parallel_safe)
            effectful = effectful or (not readonly and r.ok)
            last_readonly = readonly

        for r in results:
            kind = _halt_kind(r.content)
            if kind:
                question = r.content.split(":", 1)[-1].strip()
                await ev.set_state(NeoState.CONFIRMING if kind == "needs_confirm" else NeoState.IDLE)
                from neo.agent.confirm import peek

                p = peek()
                return AgentResult(
                    question,
                    kind,
                    steps,
                    messages,
                    question=question,
                    pending_action=p.action_id if p else "",
                )

    await _settle(ev, idle_on_finish)
    summary = "; ".join(f"{st.tool}: {'ok' if st.ok else 'failed'}" for st in steps[-4:])
    return AgentResult(
        f"I reached my step limit before finishing. Last steps — {summary}.", "max_steps", steps, messages
    )
