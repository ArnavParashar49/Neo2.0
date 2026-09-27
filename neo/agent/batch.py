"""Several actions in one model turn, and when a run needs no more model turns.

A model may answer with several tool calls at once. The loop runs them in order (neo.agent.loop),
and this module says, from what each action did, when to stop and when the model has already seen
enough:

* **stop at the first action that went wrong** — it failed, it changed nothing, or it changed the
  screen under the rest of the batch (a sheet or dialog opened or closed, another window or app came
  forward). The actions after it are skipped with a note; looks still run (they change nothing).
  This is how batched computer actions are run elsewhere too: in order, stopping at the first
  failure, so a batch never runs blind past a miss;
* **no separate check turn** when the model has already seen the state after its last action: the
  result carried a picture of it (an after-look), or NEO checked it through Accessibility (a value
  read back, a press whose element or window visibly changed). When a check is still needed, a
  look is attached to the note, so it costs one round trip instead of two;
* **finishing without another model call**: the model wrote its final answer in the same reply as
  its last actions, and NEO confirmed through Accessibility that every one of them worked;
* whether several actions per reply are invited at all (``batch_actions``, off until the user's
  A/B on the self-test window says otherwise — DECISIONS 47).

What an action did is its **Effect** (Step 0's seam: ``ToolOutput.effect``). Until that lands, a
tool reports it in its words, and those are read instead. Everything here only reads results: it
never acts, and never lets an action past a gate — each call still passes its own.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

from neo import runtime
from neo.providers.base import ImagePart, Message, ToolCall, ToolResult

log = logging.getLogger(__name__)

try:  # Step 0's seam (ToolOutput.effect): the same fields, whichever lands first
    from neo.agent.registry import Effect  # type: ignore[attr-defined]
except ImportError:  # until the seam lands

    @dataclass
    class Effect:  # type: ignore[no-redef]
        """What an input did, as NEO saw it."""

        changed: bool | None = None  # something visibly changed (None: couldn't tell)
        scene_changed: bool = False  # the front app, the window, a sheet or the focus changed
        sheet: str = ""  # a sheet or dialog open after it (its words: screen data)
        looked: bool = False  # the result carries a picture of the state after it


# ---- what an action did ------------------------------------------------------------------------------

# A tool that reports no Effect says it in words (neo.tools.computer.pointer / ax): the same words.
_MISS_WORDS = ("Nothing visibly changed",)
_SCENE_WORDS = (
    "a sheet or dialog opened",
    "the sheet or dialog closed",
    "is now in front",
    "the window changed",
)
# NEO's actions that check their own work through Accessibility: a value read back after it was
# set (the tool fails otherwise), a press or a menu item whose element, window or sheet changed.
_READ_BACK = frozenset({"ax_set_value"})
_AX_PRESS = frozenset({"ax_press", "menu_command"})
# What a skipped action's result says (history.halted() knows "skipped: " is no result).
SKIPPED = "skipped: an earlier action in this turn {why} — look at its result first, then decide"
_HALTS = ("NEEDS_CONFIRM", "NEEDS_USER", "skipped: ")
# A call that aims at pixels: its arguments name a position on a picture (or on the screen).
_PIXEL_ARGS = frozenset({"x", "y", "x1", "y1", "x2", "y2", "shot", "box_2d", "nx", "ny", "cell", "mark"})


def effect(r: Any) -> Any | None:
    """The Effect a result (a ToolOutput or a ToolResult) carries, or None."""
    return getattr(r, "effect", None)


def halted(r: ToolResult) -> bool:
    """A step that stopped to ask the user, or was skipped: no result of an action."""
    return r.content.lstrip().startswith(_HALTS)


def changed_nothing(r: Any) -> bool:
    """The action ran and NEO saw that nothing changed (not: couldn't tell)."""
    eff = effect(r)
    if eff is not None:
        return getattr(eff, "changed", None) is False
    text = getattr(r, "content", None) or getattr(r, "text", "") or ""
    return any(w in text for w in _MISS_WORDS)


def moved_scene(r: Any) -> bool:
    """The action changed the screen under whatever comes next: a sheet or dialog opened or
    closed, another window or app came forward."""
    eff = effect(r)
    if eff is not None:
        return bool(getattr(eff, "scene_changed", False))
    text = getattr(r, "content", None) or getattr(r, "text", "") or ""
    return any(w in text for w in _SCENE_WORDS)


def stop_reason(r: ToolResult) -> str:
    """Why the actions after this one in the same turn must not run ("" = they may)."""
    if not r.ok:
        return "failed"
    if changed_nothing(r):
        return "changed nothing"
    if moved_scene(r):
        return "changed the screen"
    return ""


def skipped(c: ToolCall, why: str) -> ToolResult:
    return ToolResult(c.id, c.name, SKIPPED.format(why=why), ok=False)


def looked(r: ToolResult) -> bool:
    """The result shows the state after the action: its Effect says so, or it carries a picture."""
    eff = effect(r)
    return bool((eff is not None and getattr(eff, "looked", False)) or r.images)


def verified(tool: str, r: ToolResult) -> bool:
    """NEO checked through Accessibility that this action did what it was for: never for an
    action at pixels, a keystroke or anything NEO couldn't check."""
    if not r.ok or halted(r):
        return False
    eff = effect(r)
    said = getattr(eff, "verified", None) if eff is not None else None
    if said is not None:  # an Effect that says so itself
        return bool(said) and getattr(eff, "changed", None) is not False and not getattr(eff, "sheet", "")
    if tool in _READ_BACK:
        return eff is None or getattr(eff, "changed", None) is not False
    if tool in _AX_PRESS:
        return eff is not None and getattr(eff, "changed", None) is True and not getattr(eff, "sheet", "")
    return False


def seen(tool: str, r: ToolResult) -> bool:
    """Has the model already seen the state after this action (so a check turn adds nothing)?"""
    return looked(r) or verified(tool, r)


def at_pixels(c: ToolCall) -> bool:
    return bool(_PIXEL_ARGS & set(c.args or {}))


def finished(calls: Sequence[ToolCall], results: Sequence[ToolResult]) -> bool:
    """Can the run end on the model's own final words written with these calls? Only when every
    call was an action NEO verified through Accessibility (none skipped, none asked the user, no
    look whose answer the words couldn't have known, no sheet left open)."""
    if not calls or len(calls) != len(results):
        return False
    by_id = {r.call_id: r for r in results}
    for c in calls:
        r = by_id.get(c.id)
        if r is None or not verified(c.name, r):
            return False
    return True


# ---- the settings ---------------------------------------------------------------------------------------


def _setting(name: str, app_default: Any, library_default: Any) -> Any:
    """A setting the integrator adds to neo.config (``batch_actions``, ``replay_flows``). Until it's
    there: NEO.app reads ``NEO_<NAME>`` itself; an embedded agent never reads the environment."""
    from neo.config import settings

    v = getattr(settings(), name, None)
    if v is not None:
        return v
    if runtime.library_bound():
        return library_default
    raw = os.environ.get(f"NEO_{name.upper()}")
    if raw is None or not raw.strip():
        return app_default
    return raw.strip().lower() not in ("0", "false", "off", "no")


def enabled() -> bool:
    """Is the model invited to send several actions per reply (and to finish with its last ones)?
    Off by default: the default changes only after the user's A/B (DECISIONS 47)."""
    return bool(_setting("batch_actions", False, False))


# ---- the check turn, with a look attached ---------------------------------------------------------------

VERIFY_NOTE = (
    "[verify] Before you report, confirm the goal was actually achieved with a read-only check "
    "(ax_tree, read_file, browser_read, mail_unread, calendar_today…; after clicks or drags on a "
    "board, canvas or picture, a new screenshot of the same area, compared with the one before). If "
    "it is done, give the final answer. If something is off, fix it first. Do not mention this note."
)
VERIFY_WITH_LOOK = (
    "[verify] Before you report, confirm the goal was actually achieved: NEO looked at the screen "
    "after your last action (below). If it is done, give the final answer. If something is off, fix "
    "it first. Do not mention this note."
)
LOOK_S = 0.6  # the attached look's element walk; the whole look is dropped past LOOK_S + 1 s
_LOOK_TOOLS = frozenset({"look", "ax_tree", "screenshot"})
_SCREEN_CATEGORY = "computer"


async def _observe(*, image: bool, ctx: Any) -> Any | None:
    """neo.tools.computer.observe.observe (the eyes' one-call look), when it is there."""
    try:
        from neo.tools.computer import observe as obs  # type: ignore[attr-defined]
    except ImportError:
        return None
    fn = getattr(obs, "observe", None)
    if fn is None:
        return None
    return await asyncio.wait_for(fn(None, image=image, deadline_s=LOOK_S, ctx=ctx), LOOK_S + 1.0)


async def verify_message(
    reg: Any, enabled_tools: Collection[str], ctx: Any, last: ToolCall | None, *, sees: bool
) -> Message:
    """The note that asks the model to check its work before it reports — with a look at the app
    it last acted on attached, when that action worked on the screen and the run may look at it
    (it was given a look tool). The look is read-only and passes the same checks as the look tool
    (never NEO's own window or a protected app, never a picture for a brain that can't see); it is
    NEO's words with the screen's fenced, and it taints the request like the tool does."""
    t = reg.get(last.name) if last is not None else None
    if t is None or t.category != _SCREEN_CATEGORY or not (_LOOK_TOOLS & set(enabled_tools)):
        return Message.user(VERIFY_NOTE)
    from dataclasses import replace

    # NEO takes this look for a model's run (the taint list stays shared: what it reads counts).
    look_ctx = replace(ctx, origin="agent", extras={**ctx.extras, **({} if sees else {"blind": True})})
    try:
        out = await _observe(image=sees and last is not None and at_pixels(last), ctx=look_ctx)
    except Exception as e:  # noqa: BLE001 - no look: the plain note, and the model looks itself
        log.info(f"[loop] no look for the check: {str(e)[:80]}")
        out = None
    if out is None or not getattr(out, "ok", False) or not getattr(out, "text", ""):
        return Message.user(VERIFY_NOTE)
    from neo.agent.registry import fence

    text = out.text if not getattr(out, "fence", True) else fence("look", out.text)
    images: list[ImagePart] = list(getattr(out, "images", []) or []) if sees else []
    return Message.user(VERIFY_WITH_LOOK + "\n" + text, images=images)


# ---- pictures that went stale ---------------------------------------------------------------------------

def drop_stale_after_looks(messages: list[Message], start: int, observations: Collection[str]) -> None:
    """An action's after-look (a picture on a click's result) is stale once any newer picture is in
    the run: its words stay, its pixels go (only the newest picture is worth its tokens)."""
    newer = False
    for m in reversed(messages[start:]):
        if m.role == "user" and m.images:
            newer = True
            continue
        if m.role != "tool":
            continue
        for r in reversed(m.tool_results):
            if not r.images:
                continue
            if newer and r.name not in observations:
                r.images = []
            newer = True

