"""click_text: click the element with this label in the front window (pressed through
Accessibility, else clicked where it is). Every gate it passes is in gates.py."""

from __future__ import annotations

import asyncio
import re

from neo2.core.agent import confirm
from neo2.core.agent.registry import Choice, Effect, Handoff, ToolContext, ToolOutput, fence
from neo2.core.reflex import outcomes
from neo2.core.tools.computer import ax, gates, ocr, phrases, pointer, resolve, wait
from neo2.core.tools.computer import input as inp
from neo2.core.tools.computer._catalog import _p, tool


@tool(
    "click_text",
    "Click the on-screen element with this label (button, link, tab, menu item…) in the front "
    "window.",
    _p(
        {
            "label": {"type": "string"},
            "utterance": {"type": "string", "description": "the user's whole request, when it names the control"},
        },
        ["label"],
    ),
    category="computer",
    quiet=True,
    fast_path=phrases._CLICK_PATHS,
    payload=True,
    by_utterance=True,
)
async def click_text(a: dict, c: ToolContext) -> str | ToolOutput:
    label = " ".join(str(a.get("label") or "").lower().split())
    if not label or len(label) < 2 or all(w in phrases._VAGUE_LABELS for w in label.split()):
        return f"Error: click what? '{a.get('label')}' isn't a label I can find."
    if err := gates._needs_input():
        return err
    if err := await gates._guard_front():
        return err
    app, frame, els = await asyncio.to_thread(ax.window_elements)
    if gates._protected(app):
        return gates._refuse_protected()
    if frame is None:
        return f"Error: {app or 'the front app'} has no window in front."
    # Which control the words mean (resolve.py: the exact label live; the later stages in shadow
    # unless promoted), among the window's elements outside the system menu — the one gate helper
    # every candidate passes first; the confirmation gate judges the one picked, below.
    allowed = {id(e) for e in await asyncio.to_thread(gates._outside_system_menu, els)}
    pid, bundle = await asyncio.to_thread(_owner, els)
    if pid and ax.is_neo(pid, app):
        return gates._refuse_protected()
    res = await resolve.ground(a, c, els, frame, bundle=bundle, skip=lambda e: id(e) not in allowed)
    if isinstance(res, resolve.Ambiguous):
        if res.reason in ("nothing", "no_match"):
            # Nothing Accessibility names so: the window's words read off the screen (a canvas, a
            # board, a dialog the app draws itself), clicked through the same checks as any click.
            if (out := await ocr.click_text_fallback(a, c, app=app, window=frame, els=els)) is not None:
                return out
            if c.origin != "user" or not res.shortlist:
                return f"Error: nothing in the front window of {app} is called '{a.get('label')}'."
        # Not sure which (the same label twice, several, a request that sounds like it deletes):
        # a multiple choice for a model — ids and the app's labels (fenced), never a place.
        h = resolve.handoff(res, resolve.utterance_of(a, c.origin), ctx=c, tool="click_text")
        return ToolOutput(
            h.text, ok=False, handoff=Handoff(res.reason, [Choice(x["id"], x["label"]) for x in h.choices if x.get("id")])
        )
    e = res.element
    name = (e.title or e.value)[:40]
    staged = resolve.staged_args(a, res)  # (a yes presses the control resolved now, not a new guess)
    where = f" in {app}" if app else ""
    if res.ask:  # a label the gate can't read, or one that commits by meaning: the user's call
        gate = confirm.gated("click_text", staged, f"click “{res.label[:60]}”{where}")
        if isinstance(gate, str):
            return gate
    if not app and await gates._protected_at("", *e.center):  # a window accessibility couldn't name
        return gates._refuse_protected()
    if gate := await gates._ask_first("click_text", staged, e.title or e.value, app, els):
        return gate
    mine = await asyncio.to_thread(ax.pid_of, e.ref) if e.ref is not None else 0
    if "AXPress" in e.actions:
        watch = await asyncio.to_thread(ax.press_watch, e, mine) if mine and c.origin in ("agent", "user") else None
        pressed = await asyncio.to_thread(ax.press_element, e)
        if not pressed.startswith("Error"):
            effect = None
            text = f"Clicked '{name}' ({e.role.removeprefix('AX')})"
            if watch is not None:
                ch = await wait.in_thread(ax.press_change, watch, e)
                effect = Effect(changed=ch.changed, scene_changed=ch.scene_changed, sheet=ch.sheet)
                if ch.changed and c.origin == "agent":
                    text += " " + ch.said
                    if ch.details:
                        gates._taint(c, "screen")
                        text += "\n" + fence("screen", ch.details)
            outcomes.signature("", resolve.after_press(res, verified=effect.changed if effect else None, bundle=bundle, ctx=c))
            return ToolOutput(text, effect=effect)
        code = _AX_CODE.search(pressed)
        # Only a press the app says it doesn't support falls back to the mouse. "Cannot complete"
        # (-25204) comes back when the press happened but the app is busy (it opened a dialog):
        # a second, mouse click would press it twice or land on that dialog.
        if code and int(code.group(1)) == ax._CANNOT_COMPLETE:
            return f"Clicked '{name}' ({e.role.removeprefix('AX')}), but the app didn't answer in time — look before clicking again."
        if code and int(code.group(1)) not in ax._UNSUPPORTED:
            return f"Error: couldn't press '{name}' ({pressed.removeprefix('Error: ')})."
    spot = await asyncio.to_thread(pointer.spot, (e.x, e.y, e.w, e.h))
    if spot is None:
        return f"Error: '{name}' isn't on screen — scroll to it or move its window first."
    try:
        t = await pointer.probe([spot])  # never NEO's own window over it
    except pointer.Refused as r:
        return gates._refused(c, r)
    # The click lands on whatever is at the spot now: that must be the element's app, and what is
    # there is gated like the element was (another control can sit over part of it).
    h = t.hits[0]
    if (h.pid and mine and h.pid != mine) or (not (h.pid and mine) and h.app and app and h.app != app):
        return ToolOutput(f"Error: {h.app or 'another window'} covers '{name}' — look again (ax_tree) before clicking.", ok=False)
    if gate := await gates._gate_click(staged, [(h.label, h.app or app)]):
        return gate
    await asyncio.to_thread(inp.click, *spot)
    outcomes.signature("", resolve.after_press(res, verified=None, bundle=bundle, ctx=c))
    at = "" if c.get("blind") else f" at {tuple(map(int, spot))}"  # (no coordinates for a brain that can't see)
    return f"Clicked '{name}' ({e.role.removeprefix('AX')}){at}" + gates._input_doubt()


def _owner(els: list) -> tuple[int, str]:
    """(pid, bundle id) of the app whose window these elements are ((0, "") when unknown): the
    resolver's per-app memory key."""
    pid = 0
    for e in els[:5]:
        if e.ref is not None and (pid := ax.pid_of(e.ref)):
            break
    if not pid:
        return 0, ""
    try:
        app = ax._running(pid)
        return pid, str(app.bundleIdentifier() or "") if app is not None else ""
    except Exception:  # noqa: BLE001
        return pid, ""


_AX_CODE = re.compile(r"\((-?\d+)\)")
