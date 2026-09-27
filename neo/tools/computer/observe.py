"""One look at an app, read-only: which app and window, what blocks it (an open sheet or dialog —
its buttons as ids), its on-screen elements as ids, and — for a brain that can see, when asked — a
picture of its window with a shot id the pointer tools map from. One call instead of ax_tree plus a
screenshot; the ``look`` tool and the early first look both use it.

Never NEO's own window or a protected app's (nothing is read there, nothing sent anywhere), never
a picture for a brain that can't see, and never longer than its deadline: a busy app's window is
listed as far as the walk got.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from neo.agent.registry import ToolContext, ToolOutput, fence
from neo.decide._objc import pooled
from neo.tools.computer import ax, marks, pointer, screen

LOOK_MARKS = 60  # a look lists more than a screenshot's marks: it is the whole answer for a blind brain

# The ``look`` tool, for registration beside ax_tree (see the wiring notes): read-only, untrusted.
LOOK_NAME = "look"
LOOK_DESCRIPTION = (
    "Look at an app in one call (read-only): which app and window, anything blocking it (an open sheet "
    "or dialog, with its buttons), and its on-screen controls with element ids to click or press. "
    "image=true also attaches a picture of its window with a shot id, when you can see — use it for "
    "boards, canvases and anything the list can't name. app: omit for the app being worked in."
)
LOOK_PARAMETERS = {
    "type": "object",
    "properties": {
        "app": {"type": "string", "description": "App name; omit for the app being worked in"},
        "image": {"type": "boolean", "description": "also a picture of its window (for a model that can see)"},
    },
    "required": [],
}

# NEO's own notes about a picture (the board helper: "board 8×8 at …, squares a1–h8"): each
# ``fn(shot, image, ctx)`` → text (NEO's words, or "" for nothing to say).
Note = Callable[[pointer.Shot, object, ToolContext], str]
_notes: list[Note] = []


def add_note(fn: Note) -> None:
    if fn not in _notes:
        _notes.append(fn)


@pooled
def _window(pid: int) -> tuple[str, tuple[float, float, float, float] | None]:
    """(title, frame) of the app's focused window: two quick questions, so the picture can be taken
    while the elements are still being listed."""
    root = ax._app_el(pid)
    win = ax._attr(root, "AXFocusedWindow") or ax._attr(root, "AXMainWindow")
    if win is None:
        return "", None
    f = ax._point_size(win)
    return ax._str(ax._attr(win, "AXTitle")), (f if f[2] > 0 and f[3] > 0 else None)


def _on_display(frame: tuple[float, float, float, float]) -> tuple[int, int, int, int] | None:
    """The window's part on the display showing most of it (a capture never spans two)."""
    best, most = None, 0.0
    for d in screen.displays():
        left, top = max(frame[0], d.x), max(frame[1], d.y)
        right, bottom = min(frame[0] + frame[2], d.x + d.w), min(frame[1] + frame[3], d.y + d.h)
        if right - left >= 8 and bottom - top >= 8 and (right - left) * (bottom - top) > most:
            best, most = (int(left), int(top), int(round(right - left)), int(round(bottom - top))), (right - left) * (bottom - top)
    return best


async def _picture(frame: tuple[float, float, float, float] | None, what: str):
    """(image, shot, NEO's note) of the window, or (None, None, why not)."""
    if frame is None:
        return None, None, "(No picture: the app has no window on screen.)"
    area = _on_display(frame)
    if area is None:
        return None, None, "(No picture: its window is off every screen.)"
    if await asyncio.to_thread(pointer.protected_in, area):
        return None, None, "(No picture: a window of an app NEO never reads is over it.)"
    try:
        img, x0, y0, scale, w, h = await asyncio.to_thread(pointer.capture_area, area, "window")
    except PermissionError as e:
        return None, None, f"(No picture: {e})"
    except Exception as e:  # noqa: BLE001 - the list still answers
        return None, None, f"(No picture: the capture failed — {str(e)[:80]}.)"
    s = pointer.remember(x0, y0, scale, w, h)
    return img, s, pointer.note(s, f"{what}: {area[2]}×{area[3]} pt at screen ({area[0]},{area[1]})", [])


def _started(coro) -> asyncio.Task:
    """A task whose end (or error) nobody may wait for: never reported as unheard."""
    task = asyncio.ensure_future(coro)
    task.add_done_callback(lambda t: t.cancelled() or t.exception())
    return task


async def _by(task: asyncio.Task, end: float) -> bool:
    """Did ``task`` finish well by ``end`` (the clock's)? It is never waited for past it."""
    if not task.done():
        await asyncio.wait({task}, timeout=max(0.0, end - time.monotonic()))
    return task.done() and not task.cancelled() and task.exception() is None


async def observe(
    app: str | None = None, *, image: bool = False, deadline_s: float = 1.5, ctx: ToolContext | None = None
) -> ToolOutput:
    """The look (see the module's docstring). ``deadline_s`` bounds the whole look (the element
    walk, the window's place, the scene; the picture is taken alongside the walk)."""
    from neo.tools import computer as _c  # (the gate helpers, looked up at call time)

    ctx = ctx or ToolContext()
    if err := _c._needs_ax():
        return ToolOutput(err, ok=False)
    if app:
        pid = await asyncio.to_thread(ax.pid_for_app, str(app))
        if not pid:
            return ToolOutput(f"Error: app '{str(app)[:40]}' is not running.", ok=False)
        name = str(app)
    else:
        name, pid = await asyncio.to_thread(ax.working_app)
    if not pid or ax.is_neo(pid, name) or _c._protected(name, pid):
        return ToolOutput(
            "Error: that's NEO's own window or an app NEO never touches — I won't look at it.", ok=False
        )
    blind = bool(ctx.get("blind"))
    want_picture = image and not blind
    # Everything at once, and nothing waited for past the deadline (+ marks.GRACE_S): a busy app's
    # questions are left to finish on their own threads, and the look says what it has.
    end = time.monotonic() + deadline_s + marks.GRACE_S
    walk = _started(asyncio.to_thread(marks.collect_sync, None, pid=pid, name=name, deadline_s=deadline_s, limit=LOOK_MARKS))
    scene = _started(asyncio.to_thread(ax.scene, pid))
    img, shot, said = (None, None, "")
    title, frame = "", None
    if want_picture:
        win = _started(asyncio.to_thread(_window, pid))
        if await _by(win, end):
            title, frame = win.result()
            img, shot, said = await _picture(frame, "its window")
        else:
            said = "(No picture: the app didn't say where its window is in time — it may be busy; try again in a moment.)"
    got = walk.result() if await _by(walk, end) else marks.Collected(app=name, pid=pid, refused="slow")
    sc = scene.result() if await _by(scene, end) else ax.Scene()
    title = title or got.title
    in_front = sc.known and sc.front_pid == pid
    head = ["Look (read-only) at " + ("the app in front" if in_front else "an app that isn't in front") + ", as named below:"]
    # Names and titles are the apps' own words (any app may call itself anything): fenced with the rest.
    screen_words = [f"app: {name}"]
    if sc.known and sc.front and not in_front:
        screen_words.append(f"in front instead: {sc.front}")
    if title:
        screen_words.append(f"window: “{title[:80]}”")
    if sc.sheet:
        screen_words.append(f"blocking: {sc.sheet}")
    if got.blocker or sc.sheet:
        head.append(
            f"A {got.blocker or 'sheet or dialog'} is open: it blocks the rest of the window — answer it first "
            "(its controls are marked [sheet] / [dialog] below)."
        )
    if image and blind:
        head.append("(No picture: this model can't see. The ids below work — click id=…, ax_press.)")
    text = "\n".join(head)
    if screen_words:
        text += "\n" + fence("screen", "; ".join(screen_words))
    if said:
        text += "\n" + said
    listed = marks.block(got, shot, blind=blind)
    if listed:
        text += listed
    elif got.refused == "slow":
        text += "\n(The app didn't list its controls in time — it may be busy; try again in a moment, or ax_tree.)"
    elif not got.refused:
        text += "\n(Accessibility shows no controls here — " + (
            "the picture is the way in: act at its pixels with its shot id.)" if shot is not None else "a picture (image=true) or ax_tree may show more.)"
        )
    if shot is not None:
        for fn in list(_notes):
            try:
                extra = fn(shot, img, ctx)
            except Exception:  # noqa: BLE001 - a helper's note is never the look's failure
                extra = ""
            if extra:
                text += "\n" + extra
    if title or listed or img is not None or sc.sheet:
        _c._taint(ctx, "screen")
    # NEO's own words aren't fenced; everything read off the screen already is.
    return ToolOutput(text, images=[img] if img is not None else [], fence=False)


async def look_tool(a: dict, c: ToolContext) -> ToolOutput:
    """The ``look`` tool's handler."""
    return await observe(a.get("app") or None, image=bool(a.get("image")), deadline_s=1.5, ctx=c)
