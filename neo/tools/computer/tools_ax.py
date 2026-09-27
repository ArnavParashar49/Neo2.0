"""The Accessibility tools: read an app's UI as a tree (ax_tree, ax_find), press an element and
set a text field's value by its id (ax_press, ax_set_value). Every gate they pass is in gates.py."""

from __future__ import annotations

import asyncio

from neo.agent.registry import ToolContext
from neo.tools.computer import apps, ax, gates
from neo.tools.computer._catalog import _p, tool


@tool(
    "ax_tree",
    "Read the frontmost (or named) app's UI as a text tree of elements with ids, roles, titles, "
    "values and screen positions. Use this FIRST to see what's on screen — it's fast and exact. "
    "Element ids (e12) can be used with ax_press, ax_set_value, and click. When it shows little "
    "(a game board like Chess, a map, a canvas), take a screenshot and click at pixel positions in "
    "it (with its shot id) instead.",
    _p({"app": {"type": "string", "description": "App name; omit for the frontmost app"}}),
    parallel_safe=True,
    category="computer",
    untrusted=True,
    read_only=True,
)
async def ax_tree(a: dict, c: ToolContext) -> str:
    if err := gates._needs_ax():
        return err
    return await asyncio.to_thread(ax.snapshot, a.get("app"))


@tool(
    "ax_find",
    "Search the current UI tree for elements whose title or value contains the text.",
    _p({"query": {"type": "string"}, "app": {"type": "string"}}, ["query"]),
    parallel_safe=True,
    category="computer",
    untrusted=True,
    read_only=True,
)
async def ax_find(a: dict, c: ToolContext) -> str:
    if err := gates._needs_ax():
        return err
    return await asyncio.to_thread(ax.find, a["query"], a.get("app"))


def _element_pid(e: ax.Element) -> int:
    return ax.pid_of(e.ref) if e.ref is not None else 0


@tool(
    "ax_press",
    "Activate a UI element by id from ax_tree (button, link, menu item, checkbox, tab).",
    _p({"id": {"type": "string"}}, ["id"]),
    category="computer",
    quiet=True,
)
async def ax_press(a: dict, c: ToolContext) -> str:
    if err := gates._needs_ax():
        return err
    e = ax.get(str(a.get("id") or ""))  # resolved once: the element gated is the element pressed
    if not e:
        return f"Error: unknown element {a.get('id')} — ids come from the latest ax_tree; call it again."
    pid = await asyncio.to_thread(_element_pid, e)
    owner = await asyncio.to_thread(ax.app_of, e.ref) if e.ref is not None else ""
    # The app that owns the element, not whatever is in front: clicking NEO's own card to
    # approve brings NEO forward, and the question must stay the same.
    if await gates._off_limits_el(e, pid, owner):
        return gates._protected_out()
    if gate := await gates._ask_first("ax_press", a, e.title or e.value, owner):
        return gate
    before = await gates._watch(c, pid, [])
    out = await asyncio.to_thread(ax.press_el, e)
    if not out.startswith("Error") and owner and await asyncio.to_thread(ax.regular, pid):
        apps.set_target(owner)  # "press the field, then type": the typing goes to that app
    if before is not None and not out.startswith("Error"):
        out += await gates._changes(before, pid, c)
    return out


@tool(
    "ax_set_value",
    "Put text into a text field or area by element id (read back; typed into it when the field "
    "ignores a direct set). Only for text inputs — press or click checkboxes, sliders and popups.",
    _p({"id": {"type": "string"}, "value": {"type": "string"}}, ["id", "value"]),
    category="computer",
    quiet=True,
)
async def ax_set_value(a: dict, c: ToolContext) -> str:
    if err := gates._needs_input():
        return err
    e = ax.get(str(a.get("id") or ""))
    if not e:
        return f"Error: unknown element {a.get('id')} — ids come from the latest ax_tree; call it again."
    pid = await asyncio.to_thread(_element_pid, e)
    owner = await asyncio.to_thread(ax.app_of, e.ref) if e.ref is not None else ""
    if await gates._off_limits_el(e, pid, owner):  # NEO's own field, or a protected app's
        return gates._protected_out()
    return await asyncio.to_thread(ax.set_el_value, e, str(a.get("value") or ""))
