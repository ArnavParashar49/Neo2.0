"""selftest_controls: "Neo, check your computer controls" (neo.selftest.controls).

Only the user starts it, with those words (its fast path, and a reflex rule that keeps the sentence
from being taken for conversation). It is hidden from every model — the agent's tool list, the
tool picker, Gemini Live's declarations and Laya's single-tool lane never offer it — and a call a
model makes up anyway is refused: the test moves the pointer and types for half a minute.
"""

from __future__ import annotations

import asyncio

from neo.agent.registry import ToolContext, ToolOutput, tool
from neo.selftest import phrases

_running = asyncio.Lock()


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
    from neo.tools import computer

    if err := computer._needs_ax():
        return err
    if _running.locked():
        return ToolOutput("The controls check is already running.", say=True)
    async with _running:
        from neo.events import bus
        from neo.selftest import controls

        report = await controls.run(notify=bus().note)
        saved = await asyncio.to_thread(controls.save, report)
    return ToolOutput(report.reply(saved), say=True)
