"""search_screen: open Search Screen, so the user can circle or drag over what they mean.

The user's own words ("search my screen", "screen pe search karo") open it through the Session's
lane (fastpath.lens), Gemini Live's command lane and the on-device voice — no model in between. A
model may call it too, when the user asks about something only they can point at on the screen
("find these shoes" while a video plays): opening the overlay sends nothing anywhere — only the area
the user then selects goes to their brains. Never on words NEO read on a page, in a mail or a file
(a tainted request), and never for an embedded agent (it has no overlay).

It asks NEO.app's shell to open the overlay (neo2.core.lens.live.open_overlay: one every 1.5 s). A
question in the user's words ("search my screen for the price") waits there for the first area.
Hidden from every model while Search Screen is off (NEO_LENS) or no NEO.app runs this core.
"""

from __future__ import annotations

from neo2.core import logger, runtime
from neo2.core.agent.registry import ToolContext, ToolOutput, catalog, tool
from neo2.core.config import settings

log = logger(__name__)

OPENED = "Circle or drag what you mean."
NO_APP = "Search Screen needs the NEO app."
OFF = "Search Screen is off in your settings (NEO_LENS)."
BUSY = "Search Screen is opening."
NO_SC = " NEO needs Screen Recording for that — the overlay shows how to allow it."
NOT_FOR_MODELS = "Search Screen opens only when you ask for it yourself."


def start(has_shell: bool) -> None:
    """NEO.app is starting: hidden from every model while Search Screen is off or no shell runs
    this core (read at start, like the setting itself)."""
    t = catalog().get("search_screen")
    if t is not None:
        t.hidden = not (bool(getattr(settings(), "lens", False)) and has_shell)


def refusal(ctx: ToolContext) -> str:
    """Why this call may not open Search Screen ("" = it may)."""
    if runtime.library_mode() or runtime.library_bound():
        return NO_APP
    if not getattr(settings(), "lens", False):
        return OFF
    if ctx.origin != "user" and ctx.tainted():
        return NOT_FOR_MODELS  # a model's call after reading a page, a mail or a file: never its idea
    return ""


@tool(
    "search_screen",
    "Open Search Screen: a glowing overlay where the user circles or drags over part of their screen, and "
    "NEO says what it is (text, a product and where to buy it, a place, a translation, a sum). Use when the "
    "user wants to search or ask about something they can see on screen but you can't name (\"find these "
    "shoes\", \"what's this plant in the video\"). `question`: what they want to know, in their words.",
    {"type": "object", "properties": {"question": {"type": "string", "description": "the user's question, if any"}}},
    category="system",
    risk="safe",
    timeout=10.0,
    requires=("macos",),
)
async def search_screen(args: dict, ctx: ToolContext) -> ToolOutput:
    why = refusal(ctx)
    if why:
        return ToolOutput(why, ok=False)
    from neo2.core.lens.live import lens_live
    from neo2.core.tools.computer import screen

    question = " ".join(str(args.get("question") or "").split())[:200]
    got = lens_live.open_overlay(question)
    log.info(f"[lens] search_screen: {got}")
    if got == "off":
        return ToolOutput(OFF, ok=False)
    if got == "no_app":
        return ToolOutput(NO_APP, ok=False)
    if got == "busy":
        return ToolOutput(BUSY, say=False)
    try:
        allowed = bool(screen.can_capture())
    except Exception:  # noqa: BLE001
        allowed = True
    return ToolOutput(OPENED + ("" if allowed else NO_SC), say=True)
