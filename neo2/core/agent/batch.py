"""Several actions in one model turn, and when a run needs no more model turns.

A model may answer with several tool calls at once. The loop runs them in order (neo2.core.agent.loop),
and this module says, from what each action did, when to stop and when the model has already seen
enough:

* **stop at the first action on the screen that went wrong** — it failed, it missed (a click,
  drag or scroll that changed nothing where it acted), or it changed the screen under the rest of
  the batch (a sheet or dialog opened or closed, another window or app came forward). The actions
  after it are skipped with a note; looks still run (they change nothing). This is how batched
  computer actions are run elsewhere too: in order, stopping at the first failure, so a batch never
  runs blind past a miss. A command that rightly changes nothing anyone sees (a menu's Copy) is no
  miss, and a failed call off the screen (a reminder, a mail) stops nothing;
* **no separate check turn** when the model has already seen the state after every action since
  it last looked: each result carried a picture of it (an after-look), or NEO checked it through
  Accessibility (a value read back, a press whose element or window visibly changed). When a check
  is still needed, a look is attached to the note, so it costs one round trip instead of two — a
  look the run could have taken itself (a picture only for a run given ``screenshot`` or ``look``),
  shown to the host like any tool call;
* **finishing without another model call**: the model called ``finish`` with its final answer in
  the same reply as its last actions, NEO confirmed through Accessibility that every one of them
  worked, and nothing done earlier is still unseen;
* whether several actions per reply are invited at all (``batch_actions``, off until the user's
  A/B on the self-test window says otherwise — DECISIONS 47).

What an action did is its **Effect** (``ToolOutput.effect``, copied onto the loop's result). A
tool that sets none reports it in its words, and those are read instead. Everything here only
reads results: it never acts, and never lets an action past a gate — each call still passes its own.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection, Sequence
from typing import Any

from neo2.core import logger, runtime
from neo2.core.agent.registry import Effect  # noqa: F401 - what an action did (ToolOutput.effect)
from neo2.core.providers.base import ImagePart, Message, ToolCall, ToolResult, ToolSpec

log = logger(__name__)



# ---- what an action did ------------------------------------------------------------------------------

# A tool that reports no Effect says it in words (neo2.core.tools.computer.pointer / ax): the same words.
_MISS_WORDS = ("Nothing visibly changed",)
# The pointer inputs: their Effect is the eyes' outcome, where "nothing changed" means the move
# didn't land. Anything else may rightly change nothing anyone sees (a menu's Copy, a keystroke):
# its changed=False is no miss (an Effect with a ``missed`` of its own would say otherwise).
_POINTER = frozenset({"click", "drag", "scroll", "click_text"})
_SCREEN_CATEGORY = "computer"  # the tools that work the screen (a registered tool's category)
_SCENE_WORDS = (
    "a sheet or dialog opened",
    "the sheet or dialog closed",
    "is now in front",
    "the window changed",
)
# NEO's actions that check their own work through Accessibility: a value read back after it was
# set (the tool fails otherwise), a press or a menu item whose element, window or sheet changed.
# A tool that may act either way says which in its Effect (``verified``): click_text is verified
# only when it pressed the control through Accessibility and saw it change — a click it made with
# the mouse (at the element, or at words read off the screen) carries the pointer's outcome, and a
# change at the point (a press highlight) is no check: not verified.
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


def missed(r: Any, tool: str = "") -> bool:
    """The action ran and NEO saw that it didn't do what it was for (not: couldn't tell): its
    Effect says it missed, or a pointer input changed nothing where it acted. `r`: a ToolResult,
    a ToolOutput (with `tool`) or a loop Step."""
    tool = tool or getattr(r, "name", "") or getattr(r, "tool", "") or ""
    eff = effect(r)
    if eff is not None:
        said = getattr(eff, "missed", None)
        if isinstance(said, bool):
            return said
        return getattr(eff, "changed", None) is False and tool in _POINTER
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


def stop_reason(r: ToolResult, category: str = _SCREEN_CATEGORY) -> str:
    """Why the actions after this one in the same turn must not run ("" = they may). Only an action
    on the screen (`category`: its tool's) stops them: they were chosen for the screen it failed on,
    missed or changed. Any other call (a reminder, a mail, a file) that failed says so in its own
    result and the others run as asked, each passing its own gates."""
    if category != _SCREEN_CATEGORY:
        return ""
    if not r.ok:
        return "failed"
    if missed(r):
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


# The model says its last actions finish the task, in the same reply as those actions. An explicit
# call, never text: a model that narrates ("I'll press Start…") alongside a call can't end a run.
FINISH = ToolSpec(
    "finish",
    "Call this in the same reply as your last actions when they complete the task, with your short "
    "final answer. NEO runs the actions first: when it confirms each one worked, this is the answer "
    "and the task ends there; if not, you get their results and carry on.",
    {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
    essential=True,
)
NOT_FINISHED = (
    "Not finished: NEO couldn't confirm that every action in this reply worked (their results say "
    "why). Check them, then carry on or answer."
)
NOT_FINISHED_EARLIER = (
    "Not finished: nothing has shown the screen since an earlier action NEO couldn't confirm. Check "
    "it with a read-only look, then carry on or answer."
)
NOT_FINISHED_LOOKED = (
    "Not finished yet: nothing had shown the screen since an earlier action NEO couldn't confirm, so "
    "NEO looked now (below). If the task is done, answer; if something is off, fix it first."
)


def finished(calls: Sequence[ToolCall], results: Sequence[ToolResult]) -> bool:
    """Can the run end with these actions? Only when every one was an action NEO verified through
    Accessibility (none skipped, none asked the user, no look whose answer the model couldn't have
    known, no sheet left open). `results` are the calls' own, in the same order: paired by position,
    never by id (a local server may number every call "call_0")."""
    if not calls or len(calls) != len(results):
        return False
    return all(verified(c.name, r) for c, r in zip(calls, results, strict=True))


# ---- the settings ---------------------------------------------------------------------------------------


def _setting(name: str, app_default: Any, library_default: Any) -> Any:
    """A setting of neo2.core.config (``batch_actions``, ``replay_flows``: NEO_<NAME> in NEO.app's
    environment and .env). None — not said — is ``app_default`` in NEO.app and ``library_default``
    for an embedded agent."""
    from neo2.core.config import settings

    v = getattr(settings(), name, None)
    if v is not None:
        return v
    return library_default if runtime.library_bound() else app_default


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
LOOK = "look"  # the eyes' look tool (neo2.core.tools.computer.observe), when the run has it
LOOK_TOOL_S = 2.5  # the look tool's whole call (its own walk is bounded at 1.5 s)
_LOOK_TOOLS = frozenset({LOOK, "ax_tree", "screenshot"})  # a run given one of these may be shown a look
_PICTURE_TOOLS = frozenset({LOOK, "screenshot"})  # …and a picture only with one of these


def _observer() -> Any | None:
    """neo2.core.tools.computer.observe.observe (the eyes' one-call look), when it is there."""
    try:
        from neo2.core.tools.computer import observe as obs  # type: ignore[attr-defined]
    except ImportError:
        return None
    return getattr(obs, "observe", None)


async def _observe(*, image: bool, ctx: Any) -> Any | None:
    fn = _observer()
    if fn is None:
        return None
    return await asyncio.wait_for(fn(None, image=image, deadline_s=LOOK_S, ctx=ctx), LOOK_S + 1.0)


async def _look(reg: Any, enabled: Collection[str], *, image: bool, ctx: Any, ev: Any, job: str) -> Any | None:
    """NEO's look for the check, as a look call of the run's own: the run's registered look tool
    (its checks, and what reading the screen counts for) when it has one, else the eyes' look with
    the same bookkeeping — shown to the host (tool_start / tool_end) either way."""
    args = {"image": image}
    tool = LOOK in enabled and reg.get(LOOK) is not None
    if not tool and _observer() is None:
        return None
    if ev is not None:
        await ev.tool_start(LOOK, args, job=job)
    out = None
    try:
        if tool:
            out = await asyncio.wait_for(reg.invoke(LOOK, dict(args), ctx), LOOK_TOOL_S)
        else:
            out = await _observe(image=image, ctx=ctx)
            if out is not None and getattr(out, "ok", False):
                from neo2.core.agent import grants

                grants.read_untrusted(LOOK)  # the screen's words, read for the model: a new reading
    except Exception as e:  # noqa: BLE001 - no look: the plain note, and the model looks itself
        log.info(f"[loop] no look for the check: {str(e)[:80]}")
        out = None
    if ev is not None:
        ok = bool(out is not None and getattr(out, "ok", False))
        await ev.tool_end(LOOK, ok, str(getattr(out, "text", "") or "no look"), job=job)
    return out


async def check_look(
    reg: Any,
    enabled_tools: Collection[str],
    ctx: Any,
    actions: Sequence[ToolCall],
    *,
    sees: bool,
    ev: Any = None,
    job: str = "",
) -> tuple[str, list[ImagePart]] | None:
    """NEO's look for a check, as (its words, its picture): taken when an action the model hasn't
    seen the result of (`actions`) worked on the screen and the run may look at it (it was given a
    look tool). A picture only for a brain that sees, after an action at pixels, in a run given
    ``screenshot`` or ``look``: a run given only ``ax_tree`` gets the list. The look is read-only and
    passes the same checks as the look tool (never NEO's own window or a protected app); it is NEO's
    words with the screen's fenced, and it taints the request like the tool does. None: no look."""
    enabled = set(enabled_tools)
    on_screen = [c for c in actions if (t := reg.get(c.name)) is not None and t.category == _SCREEN_CATEGORY]
    if not on_screen or not (_LOOK_TOOLS & enabled):
        return None
    from dataclasses import replace

    picture = sees and bool(_PICTURE_TOOLS & enabled) and any(at_pixels(c) for c in on_screen)
    # NEO takes this look for a model's run (the taint list stays shared: what it reads counts).
    look_ctx = replace(ctx, origin="agent", extras={**ctx.extras, **({} if sees else {"blind": True})})
    out = await _look(reg, enabled, image=picture, ctx=look_ctx, ev=ev, job=job)
    if out is None or not getattr(out, "ok", False) or not getattr(out, "text", ""):
        return None
    from neo2.core.agent.registry import fence

    text = out.text if not getattr(out, "fence", True) else fence(LOOK, out.text)
    return text, (list(getattr(out, "images", []) or []) if picture else [])


async def verify_message(
    reg: Any,
    enabled_tools: Collection[str],
    ctx: Any,
    actions: Sequence[ToolCall],
    *,
    sees: bool,
    ev: Any = None,
    job: str = "",
) -> Message:
    """The note that asks the model to check its work before it reports — with NEO's look at the
    app attached when it can take one (check_look)."""
    got = await check_look(reg, enabled_tools, ctx, actions, sees=sees, ev=ev, job=job)
    if got is None:
        return Message.user(VERIFY_NOTE)
    text, images = got
    return Message.user(VERIFY_WITH_LOOK + "\n" + text, images=images)


def carries_look(m: Message) -> bool:
    """A check note with NEO's look attached: the model now sees the state after its actions."""
    return m.role == "user" and m.text.startswith(VERIFY_WITH_LOOK)


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

