"""selftest_controls: "Neo, check your computer controls" (neo.selftest.controls).

Only the user starts it, with those words (its fast path, and a reflex rule that keeps the sentence
from being taken for conversation). It is hidden from every model — the agent's tool list, the
tool picker, Gemini Live's declarations and Laya's single-tool lane never offer it — and a call a
model makes up anyway is refused: the test moves the pointer and types for half a minute.

Once per request: the same words handed over again with nothing new from the user (Gemini Live
mapping a call of its own onto the user's last sentence, a retry after the report) don't start a
second run. And never under another job: while NEO is still clicking or typing for something else,
it says so instead of taking the mouse and keyboard from under that job.
"""

from __future__ import annotations

import asyncio

from neo.agent.registry import ToolContext, ToolOutput, tool
from neo.selftest import phrases

_running = asyncio.Lock()
_answered: tuple[object, int] | None = None  # (the event bus, the user's turn) last answered: run or refused


def _others_busy(own_job: str) -> str:
    """Why not now: other NEO jobs still at work ("" when none is)."""
    from neo.events import bus

    others = bus().running_jobs(besides=own_job)
    if not others:
        return ""
    return "another NEO job is still running" if len(others) == 1 else f"{len(others)} other NEO jobs are still running"


@tool(
    "selftest_controls",
    "NEO checks its own mouse, keyboard and screen controls in a test window of its own. Only when "
    "the user asks for it themselves ('check your computer controls').",
    {"type": "object", "properties": {}},
    category="selftest",
    hidden=True,  # never offered to a model
    slow=True,
    chain=False,  # the first thing asked, never a later clause
    timeout=90.0,  # the test keeps itself under a minute
    fast_path=[(phrases.PATTERN, {})],
    requires=frozenset({"macos"}),
)
async def selftest_controls(a: dict, c: ToolContext):
    if c.origin == "agent":
        return "Error: selftest_controls isn't available — only the user can start it, by asking NEO to check its computer controls."
    if c.origin != "user" or not phrases.is_request(c.user_text):
        # Not the user's own words, or not those words alone: answered, never handed to a model.
        return ToolOutput(
            "I check my computer controls only when you ask for just that yourself: say “check your computer controls”.",
            say=True,
        )
    from neo.events import bus, current_job
    from neo.tools import computer

    global _answered
    if err := computer._needs_ax():
        return err
    if _running.locked():
        return ToolOutput("The controls check is already running.", say=True)
    turn = (bus(), bus().user_turns)
    if _answered == turn:
        return ToolOutput(
            "I've already answered that request; say “check your computer controls” again to run the check once more.",
            say=True,
        )
    _answered = turn
    own_job = current_job.get()
    if why := _others_busy(own_job):
        return ToolOutput(
            f"I'm in the middle of something else ({why}), and the controls check needs the mouse and "
            "keyboard to itself. Ask me again when it's done.",
            say=True,
        )
    async with _running:
        from neo.selftest import controls

        try:
            report = await controls.run(notify=bus().note, busy=lambda: _others_busy(own_job))
        except Exception as e:  # noqa: BLE001 - said to the user, never handed to a model to "fix"
            return ToolOutput(f"The controls check broke before it could finish: {e!r:.160}", say=True)
        saved = await asyncio.to_thread(controls.save, report)
    return ToolOutput(report.reply(saved), say=True)
