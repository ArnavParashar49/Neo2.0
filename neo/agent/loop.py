"""The one agent loop.

Model picks tool calls → we run them (parallel when safe) → results go back → repeat, until
the model answers in text. Guardrails are structural: step budget, thrash guard, and a hard
stop on NEEDS_CONFIRM / NEEDS_USER that hands control to the human.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Literal

from neo.agent.confirm import NEEDS_CONFIRM, NEEDS_USER
from neo.agent.registry import Registry, ToolContext, ToolOutput, fence
from neo.agent.registry import registry as default_registry
from neo.config import settings
from neo.events import EventBus, NeoState
from neo.events import bus as default_bus
from neo.providers.base import AllBrainsFailed, Message, Provider, ToolCall, ToolResult, ToolSpec, Turn
from neo.providers.history import OBSERVATIONS

Stopped = Literal["done", "max_steps", "needs_confirm", "needs_user", "thrash", "error"]

_VERIFY_NOTE = (
    "[verify] Before you report, confirm the goal was actually achieved with a read-only check "
    "(ax_tree, read_file, browser_read, mail_unread, calendar_today…). If it is done, give the final "
    "answer. If something is off, fix it first. Do not mention this note."
)

_SEARCHY = ("search", "fetch", "web", "lookup", "find")

# Time spent waiting for the user rather than working — NEO's browser held while the user drives it
# — is not a run's time: each clock says how many seconds of that there have been so far.
_pause_clocks: list[Callable[[], float]] = []


def add_pause_clock(fn: Callable[[], float]) -> None:
    if fn not in _pause_clocks:
        _pause_clocks.append(fn)


def paused() -> float:
    total = 0.0
    for fn in list(_pause_clocks):
        try:
            total += float(fn())
        except Exception:  # noqa: BLE001 — a broken clock counts nothing
            pass
    return total


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


def all_tools(reg: Registry) -> list[ToolSpec]:
    """Every tool, the ones that look at and act on the screen first. A brain too small to be
    shown them all gets back as many as fit in the order given (neo.providers.fit), and the order
    tools were registered in is only the order their modules happened to be imported in."""
    from neo.agent.toolselect import SCREEN

    return sorted(reg.specs(), key=lambda t: t.name not in SCREEN)


def _model_trouble(e: Exception, steps: list[Step]) -> str:
    """What the user hears when no brain could take the next step: which brain, and whether it's
    out of quota, can't take a request this long, isn't running — not a raw stack of errors."""
    if not isinstance(e, AllBrainsFailed):
        return f"I hit a problem talking to the model: {e}"
    said = e.spoken()
    if steps:
        n = len(steps)
        said += f" I'd done {n} step{'s' if n != 1 else ''} of it before that."
    return said


def _halt_kind(text: str) -> Stopped | None:
    t = text.lstrip()
    if t.startswith(NEEDS_CONFIRM):
        return "needs_confirm"
    if t.startswith(NEEDS_USER):
        return "needs_user"
    return None


async def _settle(ev: EventBus, idle: bool, job: str = "") -> None:
    """End of a run. With a job id this only clears that job's state (the aggregate falls back
    to whatever else is active); without one it drops the base state to idle when asked."""
    if job:
        await ev.end_job(job)
    elif idle:
        await ev.set_state(NeoState.IDLE)


# Screen observations go stale the moment the next one is taken; keeping every ax_tree dump of a
# run in the prompt is pure token cost (and on an 8K-TPM brain, pure waiting).
_OBSERVATIONS = OBSERVATIONS
_SUPERSEDED_KEEP = 200
# A browsing job compares pages ("the cheapest of these three"): the latest view of this many
# different pages stays whole; a newer view of the same page, or more pages than this, trims it.
_BROWSER_PAGES = 3
_PAGE_URL = re.compile(r"^URL: (\S+)$", re.M)
_PAGE_FIND = re.compile(r"^\(only what mentions “(.*)”\)$", re.M)


def _view(r: ToolResult) -> tuple[str, str]:
    """(what this observation is a view of, the page's URL — "" for anything but a browser page).
    A find= read of a page is its own view: it doesn't replace the whole page."""
    if not r.name.startswith("browser_"):
        return r.name, ""
    m = _PAGE_URL.search(r.content[:2000])
    if not m:
        return r.name, ""
    f = _PAGE_FIND.search(r.content[:2000])
    return f"browser:{m.group(1)}" + (f"|{f.group(1)}" if f else ""), m.group(1)


def _supersede_observations(messages: list[Message], *, start: int = 0) -> None:
    """Replace all but the newest result of each observation in this run with a stub (for NEO's
    browser: of each page, keeping a few pages whole). Only the newest picture of the browser
    keeps its pixels."""
    seen: set[str] = set()
    pages: set[str] = set()
    browser_shot = False
    for m in reversed(messages[start:]):
        if m.role != "tool":
            continue
        for r in m.tool_results:
            if r.name not in _OBSERVATIONS:
                continue
            key, url = _view(r)
            stale = key in seen or (url and url not in pages and len(pages) >= _BROWSER_PAGES)
            if stale and (len(r.content) > _SUPERSEDED_KEEP or r.images):
                r.content = (
                    r.content[:_SUPERSEDED_KEEP].rstrip()
                    + f" …[older {r.name} result trimmed; a newer one follows]"
                )
                r.images = []  # an old screenshot is the most expensive stale thing of all
            if r.name.startswith("browser_") and r.images:
                if browser_shot:
                    r.images = []
                browser_shot = True
            seen.add(key)
            if url:
                pages.add(url)


async def _run_calls(
    calls: list[ToolCall],
    reg: Registry,
    ctx: ToolContext,
    ev: EventBus,
    limit: int,
    job: str = "",
    allowed: set[str] | None = None,
) -> list[ToolResult]:
    """`allowed`: the only tools that may run (None = any the model names)."""
    # A model chose these calls, whoever started the run (the taint list stays shared).
    ctx = ctx if ctx.origin == "agent" else replace(ctx, origin="agent")

    async def one(c: ToolCall) -> ToolResult:
        if allowed is not None and c.name not in allowed:
            return ToolResult(
                c.id, c.name, f"Error: {c.name[:60]} isn't available here — use only the tools you were given.", ok=False
            )
        await ev.tool_start(c.name, c.args, job=job)
        t = reg.get(c.name)
        if t and any(k in c.name for k in _SEARCHY):
            await ev.set_state(NeoState.SEARCHING, job=job)
        out: ToolOutput = await reg.invoke(c.name, c.args, ctx)
        await ev.tool_end(c.name, out.ok, out.text, job=job)
        content = _truncate(out.text, limit)
        if t and t.untrusted and not out.halt:
            # Third-party words — a failure too can quote them (a page's element names, a report
            # that starts with "Error"): data for the model, never orders.
            content = fence(c.name, content)
        return ToolResult(call_id=c.id, name=c.name, content=content, ok=out.ok, images=out.images)

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
    job: str = "",
    strict: bool = False,
) -> AgentResult:
    """`strict`: only the `tools` given ever run — whatever else the model names gets an error,
    and `more_tools` isn't answered (a run fenced off from the rest of NEO, like `browse`).

    A brain with little room may be shown only some of the tools (neo.providers.fit). A strict
    run's tools are all essential — nothing could get one back. Any other run always offers
    `more_tools`, which hands back what that brain wasn't shown."""
    s = settings()
    reg = reg or default_registry()
    ev = ev or default_bus()
    ctx = ctx or ToolContext(user_text=goal)
    max_steps = max_steps or s.max_steps
    import time as _time

    deadline = _time.monotonic() + (max_seconds or s.max_seconds)
    paused0 = paused()

    messages: list[Message] = list(history or [])
    if goal:
        messages.append(Message.user(goal, images=images or []))
    steps: list[Step] = []
    last_key, repeats = "", 0
    from neo.agent.toolselect import MORE_TOOLS

    tools = list(tools) if tools is not None else all_tools(reg)
    if strict:
        tools = [t if t.essential else replace(t, essential=True) for t in tools]
    elif not any(t.name == MORE_TOOLS.name for t in tools):
        tools.append(MORE_TOOLS)
    enabled = {t.name for t in tools}
    effectful = False  # a side-effect tool ran this run
    last_readonly = True  # the most recent tool call was an observation
    verified = False

    for _ in range(max_steps):
        if _time.monotonic() - (paused() - paused0) > deadline:
            await _settle(ev, idle_on_finish, job)
            summary = "; ".join(f"{st.tool}: {'ok' if st.ok else 'failed'}" for st in steps[-4:])
            return AgentResult(
                f"This is taking longer than I allow myself ({int(max_seconds or s.max_seconds)}s), so I stopped. "
                f"Last steps — {summary or 'none'}.",
                "max_steps",
                steps,
                messages,
            )
        await ev.set_state(NeoState.THINKING, job=job)
        _supersede_observations(messages, start=len(history or []))
        try:
            turn: Turn = await provider.complete(messages, system=system, tools=tools, effort=effort)
        except Exception as e:  # noqa: BLE001
            await _settle(ev, idle_on_finish, job)
            return AgentResult(_model_trouble(e, steps), "error", steps, messages)

        messages.append(Message.assistant(turn.text, turn.tool_calls, raw=turn.raw))
        if not turn.tool_calls:
            # Acted without looking afterwards? Make it check its work once before reporting.
            if verify and effectful and not last_readonly and not verified:
                verified = True
                messages.append(Message.user(_VERIFY_NOTE))
                continue
            await _settle(ev, idle_on_finish, job)
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
            await _settle(ev, idle_on_finish, job)
            name = turn.tool_calls[0].name
            return AgentResult(
                f"I kept repeating the same step ({name}) without progress, so I stopped.",
                "thrash",
                steps,
                messages,
            )

        # `more_tools` is answered here, not by the registry: it grows this run's tool list.
        meta = [] if strict else [c for c in turn.tool_calls if c.name == "more_tools"]
        real = [c for c in turn.tool_calls if strict or c.name != "more_tools"]
        meta_results: list[ToolResult] = []
        # What the brain was shown: a small one may have been sent only some of `tools`, and a
        # tool it wasn't shown is one it can ask for — named here, the next fitted request keeps it.
        shown = set(turn.offered) if turn.offered is not None else enabled
        for c in meta:
            from neo.agent.toolselect import selector

            found = selector().search(str(c.args.get("query", "")), exclude=shown)
            for t in found:
                if t.name not in enabled:
                    tools.append(t.spec())
                    enabled.add(t.name)
            listing = ", ".join(t.name for t in found) or "nothing new"
            meta_results.append(
                ToolResult(c.id, "more_tools", f"Enabled: {listing}. They are available from your next step.")
            )
        await ev.set_state(NeoState.WORKING, job=job)
        results = meta_results + (
            await _run_calls(real, reg, ctx, ev, s.max_tool_result_chars, job, set(enabled) if strict else None)
            if real
            else []
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
                await ev.set_state(NeoState.CONFIRMING if kind == "needs_confirm" else NeoState.IDLE, job=job)
                from neo.agent.confirm import peek

                p = peek()
                if kind == "needs_confirm" and p is not None:
                    question = f"{p.summary} — Should I go ahead?"  # the staged action's own words
                return AgentResult(
                    question,
                    kind,
                    steps,
                    messages,
                    question=question,
                    pending_action=p.action_id if p else "",
                )

    await _settle(ev, idle_on_finish, job)
    summary = "; ".join(f"{st.tool}: {'ok' if st.ok else 'failed'}" for st in steps[-4:])
    return AgentResult(
        f"I reached my step limit before finishing. Last steps — {summary}.", "max_steps", steps, messages
    )
