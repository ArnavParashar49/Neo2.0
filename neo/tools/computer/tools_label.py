"""click_text: click the element with this label in the front window (pressed through
Accessibility, else clicked where it is). Every gate it passes is in gates.py."""

from __future__ import annotations

import asyncio
import re

from neo.agent.registry import ToolContext, ToolOutput
from neo.tools.computer import ax, gates, phrases, pointer
from neo.tools.computer import input as inp
from neo.tools.computer._catalog import _p, tool

_CLICK_ROLES = ("AXButton", "AXLink", "AXMenuItem", "AXTab", "AXCheckBox", "AXRadioButton", "AXPopUpButton", "AXMenuButton", "AXCell", "AXRow", "AXStaticText", "AXImage")


@tool(
    "click_text",
    "Click the on-screen element with this label (button, link, tab, menu item…) in the front "
    "window.",
    _p({"label": {"type": "string"}}, ["label"]),
    category="computer",
    quiet=True,
    fast_path=phrases._CLICK_PATHS,
    payload=True,
)
async def click_text(a: dict, c: ToolContext) -> str | ToolOutput:
    label = " ".join(str(a["label"]).lower().split())
    if not label or len(label) < 2 or all(w in phrases._VAGUE_LABELS for w in label.split()):
        return f"Error: click what? '{a['label']}' isn't a label I can find."
    if err := gates._needs_input():
        return err
    if err := await gates._guard_front():
        return err
    app, frame, els = await asyncio.to_thread(ax.window_elements)
    if gates._protected(app):
        return gates._refuse_protected()
    if frame is None:
        return f"Error: {app or 'the front app'} has no window in front."
    fx, fy, fw, fh = frame
    word = re.compile(rf"(?<![\w]){re.escape(label)}(?![\w])", re.I)

    def visible(e) -> bool:
        return e.w > 0 and e.h > 0 and e.x + e.w > fx and e.y + e.h > fy and e.x < fx + fw and e.y < fy + fh

    cands = await asyncio.to_thread(gates._outside_system_menu, [e for e in els if e.enabled and visible(e)])
    exact = [e for e in cands if e.title.lower().strip() == label or (not e.title and e.value.lower().strip() == label)]
    hits = exact or [e for e in cands if word.search(e.title) or word.search(e.value)]
    if not hits:
        return f"Error: nothing in the front window of {app} is called '{a['label']}'."
    if not exact and len({(e.title or e.value).lower() for e in hits}) > 1:
        names = ", ".join(sorted({(e.title or e.value)[:30] for e in hits})[:4])
        return f"Error: several things match '{a['label']}' ({names}) — which one?"
    hits.sort(
        key=lambda e: (
            _CLICK_ROLES.index(e.role) if e.role in _CLICK_ROLES else len(_CLICK_ROLES),
            e.w * e.h,
        )
    )
    e = hits[0]
    name = (e.title or e.value)[:40]
    if not app and await gates._protected_at("", *e.center):  # a window accessibility couldn't name
        return gates._refuse_protected()
    if gate := await gates._ask_first("click_text", a, e.title or e.value, app, els):
        return gate
    if "AXPress" in e.actions:
        pressed = await asyncio.to_thread(ax.press_element, e)
        if not pressed.startswith("Error"):
            return f"Clicked '{name}' ({e.role.removeprefix('AX')})"
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
    mine = await asyncio.to_thread(ax.pid_of, e.ref) if e.ref is not None else 0
    if (h.pid and mine and h.pid != mine) or (not (h.pid and mine) and h.app and app and h.app != app):
        return ToolOutput(f"Error: {h.app or 'another window'} covers '{name}' — look again (ax_tree) before clicking.", ok=False)
    if gate := await gates._gate_click(a, [(h.label, h.app or app)]):
        return gate
    await asyncio.to_thread(inp.click, *spot)
    return f"Clicked '{name}' ({e.role.removeprefix('AX')}) at {tuple(map(int, spot))}" + gates._input_doubt()


_AX_CODE = re.compile(r"\((-?\d+)\)")
