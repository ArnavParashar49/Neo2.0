"""The pointer tools: screenshot (with its shot id), click, drag and scroll — by element id, by a
screenshot's pixels or by screen points (pointer.py maps and checks them). Every gate they pass is
in gates.py."""

from __future__ import annotations

import asyncio

from neo2.core.agent import confirm
from neo2.core.agent.registry import ToolContext, ToolOutput
from neo2.core.tools.computer import ax, gates, phrases, pointer, screen, tools_ax
from neo2.core.tools.computer import input as inp
from neo2.core.tools.computer._catalog import _p, tool


@tool(
    "screenshot",
    "Take a screenshot: a whole display (display: its number, 1 = main), or a region x,y,w,h in screen "
    "points. Use when ax_tree is not enough (game boards, maps, drawing and other canvas "
    "apps, Electron apps, images, layout questions). Its note gives the image size and a shot id: to "
    "click, drag or scroll at something in the picture, pass its pixel position in the image with that "
    "shot id and NEO maps it to the screen.",
    _p(
        {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "w": {"type": "integer"},
            "h": {"type": "integer"},
            "display": {"type": "integer"},
        }
    ),
    category="computer",
    slow=True,
    untrusted=True,
    read_only=True,
)
async def screenshot(a: dict, c: ToolContext) -> ToolOutput:
    import io

    from PIL import Image

    region = None
    if all(a.get(k) is not None for k in ("x", "y", "w", "h")):
        try:
            region = tuple(int(float(a[k])) for k in ("x", "y", "w", "h"))
        except (TypeError, ValueError):
            return ToolOutput("Error: x, y, w, h must be integers", ok=False)
        if region[2] <= 0 or region[3] <= 0:
            return ToolOutput("Error: region width and height must be positive", ok=False)
    shown = await asyncio.to_thread(screen.displays)
    display = shown[0]
    if region:
        # Only the part on a screen exists in the picture: cut the region to the display it starts on,
        # so its pixels always span exactly the region the note names.
        home = next((d for d in shown if d.contains(region[0], region[1])), None)
        if home is None:
            return ToolOutput(f"Error: ({region[0]}, {region[1]}) is off every screen ({'; '.join(d.words() for d in shown)})", ok=False)
        w = int(min(region[2], home.x + home.w - region[0]))
        h = int(min(region[3], home.y + home.h - region[1]))
        region = (region[0], region[1], w, h)
    if a.get("display") not in (None, "") and not region:
        try:
            want = int(a["display"])
        except (TypeError, ValueError):
            return ToolOutput("Error: display must be a number (1 = the main display)", ok=False)
        display = next((d for d in shown if d.index == want), None)
        if display is None:
            return ToolOutput(f"Error: there's no display {want} ({'; '.join(d.words() for d in shown)})", ok=False)
    try:
        img, scale = await asyncio.to_thread(screen.capture, region, None if region else display)
    except PermissionError as e:
        return ToolOutput(confirm.ask_user(str(e)), ok=False)
    w, h = Image.open(io.BytesIO(img.data)).size
    x0, y0 = (region[0], region[1]) if region else (display.x, display.y)
    shot = pointer.remember(x0, y0, scale, w, h)
    if region:
        what = f"region {region[2]}×{region[3]} pt at screen ({region[0]},{region[1]})"
        others: list = []
    else:
        what = f"all of {display.words()}"
        others = [d for d in shown if d.index != display.index]
    # NEO's own words (the mapping) aren't fenced; the picture is what came from the screen.
    return ToolOutput(pointer.note(shot, what, others), images=[img], fence=False)


async def _element_spot(eid: str, verb: str = "click") -> tuple[ax.Element, int, str, tuple[float, float]] | str | ToolOutput:
    """(element, its app's pid, its app's name, where to act on it now) for an ax_tree id — the
    middle of its part that is on screen now, not where the tree saw it — or why not."""
    e = ax.get(str(eid))
    if not e:
        return f"Error: unknown element {eid} — call ax_tree first."
    pid = await asyncio.to_thread(tools_ax._element_pid, e)
    owner = await asyncio.to_thread(ax.app_of, e.ref) if e.ref is not None else ""
    if await gates._off_limits_el(e, pid, owner):  # NEO's own window, or a protected app's: never brought forward
        return gates._protected_out()
    frame = await asyncio.to_thread(ax.visible_frame, e.ref) if e.ref is not None else (e.x, e.y, e.w, e.h)
    at = await asyncio.to_thread(pointer.spot, frame) if frame is not None else None
    if at is None:
        return (
            f"Error: {e.id} isn't on screen now (scrolled out of view, past the screen's edge, or gone) — scroll "
            "to it or move its window, then look again (ax_tree)."
        )
    return e, pid, owner, at


_CLICK_ARGS = {
    "x": {"type": "number"},
    "y": {"type": "number"},
    "shot": {"type": "string", "description": "screenshot id: x,y are pixels of that image"},
    "id": {"type": "string", "description": "element id from ax_tree, instead of x,y"},
    "button": {"type": "string", "enum": ["left", "right", "middle"]},
    "count": {"type": "integer", "description": "1, 2 (double-click) or 3"},
    "modifiers": {"type": "array", "items": {"type": "string", "enum": ["cmd", "shift", "alt", "ctrl"]}},
}


def _at(x: float, y: float) -> str:
    return f"({int(round(x))}, {int(round(y))})"


@tool(
    "click",
    "Click on an element id from ax_tree, or at a point: pixels of a screenshot with its shot id, or "
    "screen points. Brings the app there to the front first, never clicks NEO's own window, and "
    "says what changed. button: left|right|middle; count: 1-3; modifiers: cmd, shift, alt, ctrl.",
    _p(_CLICK_ARGS),
    category="computer",
    quiet=True,
)
async def click(a: dict, c: ToolContext) -> str | ToolOutput:
    if err := gates._needs_input():
        return err
    button, count, mods = str(a.get("button") or "left"), a.get("count", 1), a.get("modifiers") or []
    try:
        inp.mods_flags(mods)
        if button not in ("left", "right", "middle"):
            raise ValueError(f"button must be left, right or middle, not {button!r}")
        count = inp.click_count(1 if count in (None, "") else count)
    except ValueError as e:
        return f"Error: {e}"
    try:
        shot = None
        if a.get("id"):
            got = await _element_spot(a["id"])
            if isinstance(got, (str, ToolOutput)):
                return got
            e, pid, owner, at = got
            pts = [at]
            t = await pointer.probe(pts)
            labels = [(e.title or e.value, owner), (t.hits[0].label, t.hits[0].app or owner)]
            head = f"{pointer.click_word(button, count)} {e.id} ({e.role.removeprefix('AX')}) at {_at(*pts[0])}"
        else:
            pid = 0
            pts, shot = pointer.points(a, ("x", "y"), c)
            t = await pointer.probe(pts)
            labels = [(t.hits[0].label, t.hits[0].app)]
            head = f"{pointer.click_word(button, count)} at {_at(*pts[0])}"
            if shot is not None:
                head += f" (pixel ({float(a['x']):g}, {float(a['y']):g}) of {shot.id})"
        if gate := await gates._gate_click(a, labels):
            return gate
        await pointer.bring(t, pts, want_pid=pid)
        # A point nothing could be told about before now shows something: judged like the rest.
        if t.relabeled and (gate := await gates._gate_click(a, [(t.hits[0].label, t.hits[0].app)])):
            return gate
    except pointer.Refused as r:
        return gates._refused(c, r)
    h = t.hits[0]
    pid = pid or h.pid
    if h.app:
        head += f" in {h.app}"
    if t.brought and not t.cold:
        head += f" (brought {t.brought} to the front first)"
    head += "."
    before = await gates._watch(c, pid, pts, hover=pts[0])
    await asyncio.to_thread(inp.click, *pts[0], button=button, count=count, mods=mods)
    if before is None:
        if await pointer.woke(t, pid):  # nothing looked at, but the click itself brought its app forward
            said = pointer.brought_only(h.app or t.brought or "the app", looked=False).said
            return ToolOutput(f"{head} {said}{gates._input_doubt()}")
        return head + gates._input_doubt()
    out = await pointer.outcome(before, pid, pts, [_at(*pts[0])], target=t)
    if not pointer.at_the_point(h, shot is not None):
        out = pointer.maybe_elsewhere(out, _at(*pts[0]))
    under = [pointer.words(h)] if not a.get("id") else []
    text, ok = pointer.report(head, out, under, gates._input_doubt())
    if any(under) or out.details:
        gates._taint(c, "screen")
    return ToolOutput(text, ok=ok)


@tool(
    "drag",
    "Drag from one point to another: each end an element id from ax_tree (id1, id2), or x1,y1 / x2,y2 — "
    "pixels of a screenshot with its shot id, or screen points. Brings the app there to the front "
    "first (not when the ends are in two different apps) and says what changed at both ends.",
    _p(
        {
            "x1": {"type": "number"},
            "y1": {"type": "number"},
            "x2": {"type": "number"},
            "y2": {"type": "number"},
            "id1": {"type": "string", "description": "element id from ax_tree to drag from, instead of x1,y1"},
            "id2": {"type": "string", "description": "element id from ax_tree to drop on, instead of x2,y2"},
            "shot": {"type": "string", "description": "screenshot id: the points are pixels of that image"},
        }
    ),
    category="computer",
    quiet=True,
)
async def drag(a: dict, c: ToolContext) -> str | ToolOutput:
    if err := gates._needs_input():
        return err
    try:
        ends: list[tuple[float, float] | None] = [None, None]
        for i, n in enumerate(("1", "2")):
            if a.get(f"id{n}"):
                got = await _element_spot(a[f"id{n}"], "drag")
                if isinstance(got, (str, ToolOutput)):
                    return got
                ends[i] = got[3]
        keys = tuple(k for i, n in enumerate(("1", "2")) if ends[i] is None for k in (f"x{n}", f"y{n}"))
        coords, shot = pointer.points(a, keys, c, "drag") if keys else ([], None)
        pts = [e if e is not None else coords.pop(0) for e in ends]
        t = await pointer.probe(pts)
        if gate := await gates._gate_drop(a, t):
            return gate
        src, dst = t.hits
        # Two apps: neither is brought forward — raising one could put its window over the other
        # end (the usual drag goes from a window behind to one in front), and file views let a drag
        # start in a background window. One app (a board, a list): it comes forward, and both ends
        # are looked at again.
        between_apps = bool(src.pid and dst.pid and src.pid != dst.pid)
        if not between_apps:
            await pointer.bring(t, pts)
            if t.relabeled and (gate := await gates._gate_drop(a, t)):
                return gate
    except pointer.Refused as r:
        return gates._refused(c, r)
    src, dst = t.hits
    pid = src.pid
    before = await gates._watch(c, pid, pts, hover=pts[1])  # the drop leaves the pointer at the end
    await asyncio.to_thread(inp.drag, *pts[0], *pts[1])
    head = f"Dragged {_at(*pts[0])} → {_at(*pts[1])}"
    if shot is not None:
        head += f" (pixels of {shot.id})"
    if src.app:
        head += f" in {src.app}" + (f" onto {dst.app}" if dst.app and dst.pid != src.pid else "")
    if t.brought and not t.cold:
        head += f" (brought {t.brought} to the front first)"
    head += "."
    if before is None:
        if await pointer.woke(t, pid):  # nothing looked at, but the button press itself brought its app forward
            said = pointer.brought_only(src.app or t.brought or "the app", looked=False, what="drag").said
            return ToolOutput(f"{head} {said}{gates._input_doubt()}")
        return head + gates._input_doubt()
    out = await pointer.outcome(before, pid, pts, [f"the start {_at(*pts[0])}", f"the end {_at(*pts[1])}"], between_apps, target=t)
    if between_apps and out.changed is False:
        out.said += (
            f" (Neither app was brought forward for a drag between two apps; if {src.app or 'the app it starts in'} "
            "only came forward, bring it to the front yourself, look again, and drag again.)"
        )
    under = [f"start {pointer.words(src)}" if pointer.words(src) else "", f"end {pointer.words(dst)}" if pointer.words(dst) else ""]
    text, ok = pointer.report(head, out, under, gates._input_doubt())
    if any(under) or out.details:
        gates._taint(c, "screen")
    return ToolOutput(text, ok=ok)


@tool(
    "scroll",
    "Scroll the content under the pointer, or at an element id from ax_tree (a list, a pane), or at "
    "x,y: screenshot pixels with its shot id, or screen points. dy<0 scrolls down, dy>0 up (lines).",
    _p(
        {
            "x": {"type": "number"},
            "y": {"type": "number"},
            "id": {"type": "string", "description": "element id from ax_tree to scroll, instead of x,y"},
            "shot": {"type": "string", "description": "screenshot id: x,y are pixels of that image"},
            "dy": {"type": "integer"},
            "dx": {"type": "integer"},
            "at_pointer": {"type": "boolean"},
        }
    ),
    category="computer",
    fast_path=phrases._SCROLL,
    quiet=True,
)
async def scroll(a: dict, c: ToolContext) -> str | ToolOutput:
    if err := gates._needs_input():
        return err
    try:
        dy = int(float(a.get("dy", -5) if a.get("dy") not in (None, "") else -5))
        dx = int(float(a.get("dx", 0) or 0))
    except (TypeError, ValueError):
        return "Error: dy and dx must be whole numbers (lines)."
    try:
        if not a.get("at_pointer") and a.get("id"):
            got = await _element_spot(a["id"], "scroll")
            if isinstance(got, (str, ToolOutput)):
                return got
            pts = [got[3]]
        elif not a.get("at_pointer") and a.get("x") is not None and a.get("y") is not None:
            pts, _shot = pointer.points(a, ("x", "y"), c, "scroll")
        else:
            pts = [await asyncio.to_thread(inp.mouse_position)]  # "scroll down": wherever the pointer is
            h = await asyncio.to_thread(ax.hit, *pts[0])
            if h.pid and ax.is_neo(h.pid, h.app):
                # The pointer rests on NEO's window (the user just clicked the orb): scroll the app
                # they're working in, not NEO's chat.
                frame = await asyncio.to_thread(ax.work_window_frame)
                if frame is None:
                    return ToolOutput("Error: the pointer is over NEO's own window, and no other window is in front to scroll.", ok=False)
                pts = [(frame[0] + frame[2] / 2, frame[1] + frame[3] / 2)]
        t = await pointer.probe(pts)
    except pointer.Refused as r:
        return gates._refused(c, r)
    pid = t.hits[0].pid
    before = await gates._watch(c, pid, pts, hover=pts[0])
    text = await asyncio.to_thread(inp.scroll, pts[0][0], pts[0][1], dy, dx)
    if before is None:
        return str(text) + gates._input_doubt()
    out = await pointer.outcome(before, pid, pts, [_at(*pts[0])])
    text, ok = pointer.report(str(text) + (f" in {t.hits[0].app}." if t.hits[0].app else "."), out, [], gates._input_doubt())
    return ToolOutput(text, ok=ok)
