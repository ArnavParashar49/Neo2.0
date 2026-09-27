"""The pointer tools: screenshot (with its shot id), click, drag and scroll — by element id, by a
screenshot's pixels or by screen points (pointer.py maps and checks them). Every gate they pass is
in gates.py."""

from __future__ import annotations

import asyncio
import math

from neo2.core.agent import confirm
from neo2.core.agent.registry import ToolContext, ToolOutput
from neo2.core.tools.computer import apps, ax, focus, gates, marks, ocr, phrases, pointer, screen, tools_ax
from neo2.core.tools.computer._catalog import _p, tool

SCREENSHOT_PARAMS = _p(
    {
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "w": {"type": "integer"},
        "h": {"type": "integer"},
        "display": {"type": "integer"},
        "app": {"type": "string", "description": "an app whose front window to take (default: the app NEO works in)"},
        "zoom": {
            "type": "boolean",
            "description": "a close look at full detail around x,y (screen points); w,h its size (default 320×200)",
        },
    }
)


@tool(
    "screenshot",
    pointer.SCREENSHOT_DESCRIPTION
    + " By default it is the front window of the app NEO works in (app: another app's); zoom with x,y (and "
    "w,h) for a close look at small things.",
    SCREENSHOT_PARAMS,
    category="computer",
    slow=True,
    untrusted=True,
    read_only=True,
)
async def screenshot(a: dict, c: ToolContext) -> ToolOutput:
    region = None
    zoom = bool(a.get("zoom"))
    if zoom:
        try:
            zx, zy = float(a["x"]), float(a["y"])
            zw = float(a["w"]) if a.get("w") not in (None, "") else float(screen.ZOOM_W)
            zh = float(a["h"]) if a.get("h") not in (None, "") else float(screen.ZOOM_H)
        except (KeyError, TypeError, ValueError):
            return ToolOutput("Error: a zoom needs x and y (screen points; w and h optional)", ok=False)
        if not all(math.isfinite(v) for v in (zx, zy, zw, zh)) or zw <= 0 or zh <= 0:
            return ToolOutput("Error: a zoom's x, y, w and h must be finite, and w and h positive", ok=False)
    elif all(a.get(k) is not None for k in ("x", "y", "w", "h")):
        try:
            region = tuple(int(float(a[k])) for k in ("x", "y", "w", "h"))
        except (TypeError, ValueError):
            return ToolOutput("Error: x, y, w, h must be integers", ok=False)
        if region[2] <= 0 or region[3] <= 0:
            return ToolOutput("Error: region width and height must be positive", ok=False)
    shown = await asyncio.to_thread(screen.displays)
    display = shown[0]
    if zoom:
        home = screen.display_at(zx, zy) if hasattr(screen, "display_at") else None
        home = home or next((d for d in shown if d.contains(zx, zy)), None)
        if home is None:
            return ToolOutput(f"Error: ({zx:g}, {zy:g}) is off every screen ({'; '.join(d.words() for d in shown)})", ok=False)
    if region:
        # Only the part on a screen exists in the picture: cut the region to the display it starts on,
        # so its pixels always span exactly the region the note names.
        home = next((d for d in shown if d.contains(region[0], region[1])), None)
        if home is None:
            return ToolOutput(f"Error: ({region[0]}, {region[1]}) is off every screen ({'; '.join(d.words() for d in shown)})", ok=False)
        w = int(min(region[2], home.x + home.w - region[0]))
        h = int(min(region[3], home.y + home.h - region[1]))
        region = (region[0], region[1], w, h)
    display_asked = a.get("display") not in (None, "")
    if display_asked and not region and not zoom:
        try:
            want = int(a["display"])
        except (TypeError, ValueError):
            return ToolOutput("Error: display must be a number (1 = the main display)", ok=False)
        display = next((d for d in shown if d.index == want), None)
        if display is None:
            return ToolOutput(f"Error: there's no display {want} ({'; '.join(d.words() for d in shown)})", ok=False)
    # The app NEO works in, when one is known and nothing else was asked for: its front window (fewer
    # of other apps' pixels, more detail per token). Never NEO's own or a protected app's window.
    win = None if (region or zoom or display_asked) else await _working_window(str(a.get("app") or ""))
    # The controls in the picture, from Accessibility, walked while the picture is taken (no wait).
    if zoom:
        area = (zx - zw / 2, zy - zh / 2, zw, zh)
    elif region:
        area = tuple(float(v) for v in region)
    else:
        area = None if win is not None else (display.x, display.y, display.w, display.h)
    walk = asyncio.ensure_future(marks.collect(area, c, pid=win[2] if win is not None else 0))
    try:
        if zoom:
            cap = await asyncio.to_thread(screen.zoom, zx, zy, zw, zh, display=home)
        elif win is not None:
            cap = await asyncio.to_thread(screen.window_shot, win[0])
        else:
            cap = await asyncio.to_thread(screen.shot, region, None if region else display)
    except PermissionError as e:
        walk.cancel()
        return ToolOutput(confirm.ask_user(str(e)), ok=False)
    except BaseException:
        walk.cancel()
        raise
    if zoom:
        what = f"a zoom on {cap.w}×{cap.h} px of {zw:g}×{zh:g} pt around screen ({zx:g},{zy:g})"
        others: list = []
    elif win is not None:
        what = f"{win[1]}'s front window"
        others = []
    elif region:
        what = f"region {region[2]}×{region[3]} pt at screen ({region[0]},{region[1]})"
        others = []
    else:
        what = f"all of {display.words()}"
        others = [d for d in shown if d.index != display.index]
    collected = await walk
    blind = bool(c.get("blind"))
    ocr.warm()  # (NEO.app: Vision loads in the background for the next read)
    # The words on the picture where Accessibility names little of it (always for a brain that
    # can't see): text marks, clicked by id (click id=t…) through the same gates.
    frame = getattr(cap, "frame", None)
    tm = []
    if frame is not None and cap.clickable:
        tm = await ocr.look_marks(
            frame, (cap.x0, cap.y0, cap.w * cap.ppp, cap.h * cap.ppp), named=collected.marks,
            window=collected.window, blind=blind, app=collected.app, pid=collected.pid,
        )  # fmt: skip
        if tm:
            gates._taint(c, "screen")
    if blind:
        # A brain that can't see gets no picture's numbers in any form (no mapping, no sizes, no
        # places): what the picture holds, by element id only.
        text = (
            "Screenshot taken, but you can't see it (this model has no vision), and NEO gives no coordinates to "
            "act on: use the element ids below, or ax_tree's (click id=…, ax_press), or tell the user the model "
            "that can see the screen is busy."
        )
        return ToolOutput(text + marks.for_shot(collected, None, c) + ocr.marks_block(tm, None, blind=True), images=[cap.part], fence=False)
    if not cap.clickable:
        # The window server's bounds didn't match the picture (a shadow, a window that moved, one
        # off screen): no mapping to click through — for looking at only.
        text = (
            f"Screenshot attached ({what}, for looking only: its pixels can't be mapped to the screen — for a "
            "place to click, take a screenshot of the display or a region)."
        )
        return ToolOutput(text + marks.for_shot(collected, None, c), images=[cap.part], fence=False)
    shot = pointer.remember(cap.x0, cap.y0, cap.ppp, cap.w, cap.h, covered=getattr(cap, "covered", ()))
    # NEO's own words (the mapping) aren't fenced; the picture is what came from the screen, and so
    # are the controls' labels (fenced by marks.for_shot, which taints the request).
    text = pointer.note(shot, what, others) + marks.for_shot(collected, shot, c) + ocr.marks_block(tm, shot, blind=False)
    return ToolOutput(text, images=[cap.part], fence=False)


async def _working_window(asked: str) -> tuple[int, str, int] | None:
    """(window id, app name, pid) of the front window of ``asked`` — else of the app NEO works in —
    when it has one on screen; None for NEO's own and a protected app's (the whole display then)."""
    name = asked or apps.target() or ""
    if not name:
        return None
    try:
        run = await asyncio.to_thread(apps._running, name)
        if run is None:
            return None
        pid, real = int(run.processIdentifier()), str(run.localizedName() or name)
        if not pid or ax.is_neo(pid, real) or gates._protected(real, pid):
            return None
        wid = await asyncio.to_thread(focus.front_window_id, pid)
    except Exception:  # noqa: BLE001 - can't tell: the whole display
        return None
    return (wid, real, pid) if wid else None


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


def _at(x: float, y: float) -> str:
    return f"({int(round(x))}, {int(round(y))})"


# click, drag and scroll: pointer.py holds them whole (one checked function for every form of a
# place — an element id, a board square, a helper's mark, pixels, 0–1000 of a shot, screen points —
# snapped, gated again, brought forward and looked at after), calling the gates through this package.


@tool("click", pointer.CLICK_DESCRIPTION, pointer.CLICK_PARAMETERS, category="computer", quiet=True)
async def click(a: dict, c: ToolContext) -> str | ToolOutput:
    return await pointer.click_tool(a, c)


@tool("drag", pointer.DRAG_DESCRIPTION, pointer.DRAG_PARAMETERS, category="computer", quiet=True)
async def drag(a: dict, c: ToolContext) -> str | ToolOutput:
    return await pointer.drag_tool(a, c)


@tool(
    "scroll",
    pointer.SCROLL_DESCRIPTION,
    pointer.SCROLL_PARAMETERS,
    category="computer",
    fast_path=phrases._SCROLL,
    quiet=True,
)
async def scroll(a: dict, c: ToolContext) -> str | ToolOutput:
    return await pointer.scroll_tool(a, c)


# Text marks (words read off the screen, ids t…) are clicked and dragged like any element id, their
# words judged by the click gate with what is under the point.
ocr.install()
