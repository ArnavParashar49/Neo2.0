"""Coordinate input the model can trust: it lands where the model aimed, in the app it aimed at,
never on NEO's own window — and the result says what it hit and whether anything changed.

The Chess run that prompted this: every click and drag reported "ok" while the board never
moved. A click on a window whose app isn't frontmost may only activate it (a view that doesn't
accept the first mouse), a screenshot's pixels are not always screen points, NEO's overlay is on
top but invisible in screenshots, and a bare "ok" hid all of it. So, for each click, drag and
scroll a model makes:

1. the numbers: plain screen points, or pixels of a screenshot (``shot``) that NEO maps itself;
   on some display; never picked by a brain that couldn't see the screen;
2. the target: what's under each point (one AX lookup) — never NEO's own window, never a window
   a sheet is blocking; the owner brought to the front (and its window made key) first, and every
   point looked at again after that: an input whose target changed on the way isn't made (a drag
   between two apps brings neither forward: raising one could cover the other's drop spot);
3. the outcome: the owner app and label at the point, and a before/after look (the picture
   around each point, taken with the pointer already resting where it will be — a hover highlight
   is no change — plus the front app, a sheet or dialog, the focused window and element). A move on
   a board or canvas after which nothing visibly changed is reported as a miss; a click on a control
   whose effect may show elsewhere says it saw no change near the point, and to check before
   repeating it.
"""

from __future__ import annotations

import asyncio
import itertools
import re
import secrets
from dataclasses import dataclass

from neo2.core.agent.registry import USERS_WORDS, ToolContext, fence
from neo2.core.tools.computer import apps, ax, screen, wait

SETTLE_S = 0.12  # after bringing an app forward (or raising a window), before the input
KEY_S = 1.0  # at most this long for an app brought forward to be active with its window key
HOVER_S = 0.1  # the pointer rests where the input will be this long before the first look
LOOK_S = 0.15  # after the input, the first look at what changed (most changes start at once)…
LOOK_AGAIN_S = 0.85  # …and once more when nothing had yet (a page that loads, a slow redraw)
POLL_S = 0.1  # after a key or a press: a look at the window, sheet and focus this often…
SCENE_LOOKS = 4  # …at most this many times, done at the first change


class Refused(Exception):
    """Nothing was posted; the message says why (and what to do instead). ``screen``: the message
    quotes words read off the screen (fenced) — the request has read untrusted text. ``protected``:
    the target is NEO's own window or an app NEO never touches — the run stops and the user is told."""

    def __init__(self, message: str, *, screen: bool = False, protected: bool = False) -> None:
        super().__init__(message)
        self.screen = screen
        self.protected = protected


# ---- screenshots and their pixels ---------------------------------------------------------------


@dataclass(frozen=True)
class Shot:
    """How one screenshot's pixels map to the screen: screen = origin + pixel × ppp."""

    id: str
    x0: float
    y0: float
    ppp: float  # screen points per image pixel
    w: int  # image size, pixels
    h: int

    def to_screen(self, ix: float, iy: float) -> tuple[float, float]:
        return self.x0 + ix * self.ppp, self.y0 + iy * self.ppp


_shots: dict[str, Shot] = {}
_shot_ids = itertools.count(1)
_KEEP_SHOTS = 12
# Ids carry this run's mark: an id a model reads in a conversation from before NEO restarted can
# never name a different, newer screenshot.
_RUN = secrets.token_hex(2)


def remember(x0: float, y0: float, ppp: float, w: int, h: int) -> Shot:
    s = Shot(f"s{next(_shot_ids)}-{_RUN}", float(x0), float(y0), float(ppp), int(w), int(h))
    _shots[s.id] = s
    while len(_shots) > _KEEP_SHOTS:
        _shots.pop(next(iter(_shots)))
    return s


def get_shot(sid: str) -> Shot | None:
    return _shots.get(str(sid).strip())


def note(s: Shot, what: str, others: list[screen.Display]) -> str:
    """NEO's own words about a screenshot (not fenced: nothing in them came from the screen)."""
    exact = abs(s.ppp - 1.0) < 1e-6
    text = (
        f"Screenshot {s.id} attached ({what}); the image is {s.w}×{s.h} px. To act on something in it, "
        f'give its pixel position in this image with shot="{s.id}" — click x,y · drag x1,y1,x2,y2 · '
        "scroll x,y — and NEO maps it to the screen."
    )
    if exact and s.x0 == 0 and s.y0 == 0:
        text += " (Here image pixels are also screen points.)"
    else:
        text += (
            f" Without shot, x and y are screen points: screen_x = {s.x0:g} + image_x × {s.ppp:.4f}, "
            f"screen_y = {s.y0:g} + image_y × {s.ppp:.4f}."
        )
    if others:
        text += " Other displays (not in this picture): " + "; ".join(
            d.words() + f' — screenshot display={d.index} to see it' for d in others
        ) + "."
    return text


# ---- the numbers --------------------------------------------------------------------------------

_BY_ID = {
    "click": "click id=…, ax_press or click_text",
    "drag": "drag id1=… id2=…",
    "scroll": "scroll id=…",
}


def blind(tool: str = "click") -> str:
    """Why a brain that can't see may not act at coordinates, and what it can use instead."""
    return (
        "Error: you can't see the screen — this model has no vision, so no screenshot ever reached you — "
        f"and I won't {tool} at coordinates nobody saw. Use element ids from ax_tree ({_BY_ID.get(tool, _BY_ID['click'])}), "
        "or tell the user the model that can see the screen is busy right now."
    )


def _numbers(a: dict, keys: tuple[str, ...]) -> list[float]:
    out = []
    for k in keys:
        v = a.get(k)
        if v is None or isinstance(v, bool):
            raise Refused(f"Error: {' and '.join(keys)} are needed (numbers).")
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise Refused(f"Error: {k} must be a number, not {str(v)[:20]!r}.") from None
        if f != f or f in (float("inf"), float("-inf")):
            raise Refused(f"Error: {k} must be a finite number.")
        out.append(f)
    return out


def _said(ctx: ToolContext, values: list[float]) -> bool:
    """Did the user's own words give these numbers ("click at 300, 400")? Only what the Session
    marked as theirs (USERS_WORDS) — never a model's goal, a Live relay's or an embedding host's
    words, whatever ctx.user_text holds."""
    said = {float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", str(ctx.get(USERS_WORDS, "") or ""))}
    return bool(values) and all(v in said or round(v) in said for v in values)


def points(
    a: dict, keys: tuple[str, ...], ctx: ToolContext, tool: str = "click"
) -> tuple[list[tuple[float, float]], Shot | None]:
    """The screen points a call means, checked: pairs of ``keys`` ("x", "y", or "x1".."y2")."""
    vals = _numbers(a, keys)
    if ctx.get("blind") and not _said(ctx, vals):
        raise Refused(blind(tool))
    s = None
    if a.get("shot") not in (None, ""):
        s = get_shot(str(a["shot"]))
        if s is None:
            raise Refused(f"Error: there's no screenshot {str(a['shot'])[:12]!r} any more — take a new one.")
        for i in range(0, len(vals), 2):
            ix, iy = vals[i], vals[i + 1]
            if not (0 <= ix <= s.w and 0 <= iy <= s.h):
                raise Refused(
                    f"Error: ({ix:g}, {iy:g}) is outside screenshot {s.id}, which is {s.w}×{s.h} px — give "
                    "the pixel position in that image (from its top-left corner)."
                )
    pts = []
    for i in range(0, len(vals), 2):
        x, y = s.to_screen(vals[i], vals[i + 1]) if s else (vals[i], vals[i + 1])
        pts.append((x, y))
    shown = screen.displays()
    for x, y in pts:
        if not any(d.contains(x, y) for d in shown):
            raise Refused(
                f"Error: ({x:g}, {y:g}) is off every screen ({'; '.join(d.words() for d in shown)}). "
                "Screen points are what click takes; for a spot in a screenshot, pass its pixel position with shot."
            )
    return pts, s


def spot(frame: tuple[float, float, float, float]) -> tuple[float, float] | None:
    """Where to act on something with this frame (screen points): the middle of its part on a
    display — the display showing most of it. None when none of it is on any (a window hanging
    past the screen's edge: macOS would pin the pointer to the edge, onto something else)."""
    x, y, w, h = frame
    best, most = None, 0.0
    for d in screen.displays():
        left, top = max(x, d.x), max(y, d.y)
        right, bottom = min(x + w, d.x + d.w), min(y + h, d.y + d.h)
        if right - left >= 1 and bottom - top >= 1 and (right - left) * (bottom - top) > most:
            best, most = ((left + right) / 2, (top + bottom) / 2), (right - left) * (bottom - top)
    return best


# ---- the target ---------------------------------------------------------------------------------


@dataclass
class Target:
    hits: list[ax.Hit]
    brought: str = ""  # the app brought to the front for this input ("" = it already was)
    raised: bool = False  # a window was raised over the app's others for it
    relabeled: bool = False  # a point nothing could be told about before now shows something
    window: ax.Win | None = None  # the app's own normal window at the first point, when it had to come forward
    cold: bool = False  # the app still wasn't seen active with that window key when the input went


def _off_limits() -> Refused:
    """NEO's own window (it floats on top and never shows in screenshots) or a protected app's: never
    touched, never brought forward — the user is told (the package's protected refusal)."""
    return Refused("Error: that spot is NEO's own window or an app NEO never touches.", protected=True)


async def _protected(h: ax.Hit, x: float, y: float) -> bool:
    from neo2.core.tools.computer import gates  # (the frozen gate helpers, looked up at call time)

    return await gates._protected_hit(h, x, y)


async def _protected_pid(pid: int, name: str = "") -> bool:
    from neo2.core.tools.computer import gates

    return bool(pid) and (ax.is_neo(pid, name) or gates._protected(name, pid))


def _sheet_refusal(h: ax.Hit) -> Refused:
    return Refused(
        f"Error: {h.app or 'that app'} is waiting on an open sheet or dialog — "
        + fence("screen", h.sheet)
        + "\nIt blocks clicks on the window behind it: answer it first (click one of its buttons).",
        screen=True,
    )


def _label(h: ax.Hit) -> str:
    return " ".join(h.label.split())


async def probe(pts: list[tuple[float, float]]) -> Target:
    """What's under each point. Refuses NEO's own window, and a window a sheet is blocking."""
    hits = [await asyncio.to_thread(ax.hit, x, y) for x, y in pts]
    for (x, y), h in zip(pts, hits, strict=True):
        if await _protected(h, x, y):  # NEO's own window or a protected app: before anything else
            raise _off_limits()
    if hits[0].sheet:
        raise _sheet_refusal(hits[0])
    return Target(hits)


async def bring(t: Target, pts: list[tuple[float, float]], *, want_pid: int = 0) -> None:
    """Bring the app acted on — the owner of the first point, or ``want_pid`` (an element's app) —
    to the front and make its window key, then look at every point again: an input whose target
    changed on the way (another window came over a point, the drop spot is now something else) isn't
    made, and the hits are the new ones. With ``want_pid`` the point must be that app's once it is in
    front (not covered by another window).

    What comes forward is the app whose own normal window is at the point, whatever its activation
    policy (a menu-bar utility's window, a helper's, NEO's self-test window): never for the menu bar,
    a status item, the Dock, a menu, a popover or a system overlay, nor for a window another process
    owns (:func:`_own_window`). An app LaunchServices calls frontmost that isn't active with a window
    key (a window ordered in front, its app never activated) is made active too: a click there would
    only activate it. The wait for it is a deadline, over at the first look that sees it."""
    first = here = t.hits[0]
    pid = want_pid or first.pid
    if pid:
        from neo2.core.tools.computer.focus import front_app

        front, front_pid = await asyncio.to_thread(front_app)
        mine = pid == first.pid  # (else an element's app, and another app's window may be at the point)
        seen = mine and first.window is not None  # Accessibility can say whether that window is key
        cold = front_pid != pid or (seen and await asyncio.to_thread(ax.key_window, pid) is None)
        if cold:
            if await _protected_pid(pid, first.app if mine else ""):
                raise _off_limits()  # never brought forward, even to refuse it afterwards
            t.window = await _own_window(first if mine else None, pid, pts[0])
        if cold and t.window is not None:
            name = await _come_forward(pid, front, front_pid, first.app if mine else "")
            t.brought = name
            if not await wait.until(lambda: asyncio.to_thread(_active, pid, seen), KEY_S):
                now, now_pid = await asyncio.to_thread(front_app)
                if now_pid != pid:
                    raise _stayed_behind(name, now)
                t.cold = True  # in front, never seen active with a window key: the click may only activate it
            again = await asyncio.to_thread(ax.hit, *pts[0])  # (the window there now: the one to raise)
            if (again.pid or again.app) and await _protected(again, *pts[0]):
                raise _off_limits()
            _check_first(again, pid, want_pid, name)
            here = again if again.pid else first
        if (not cold or t.brought) and here.pid == pid and await asyncio.to_thread(ax.raise_window, here):
            t.raised = True  # (never another app's window)
            if not await wait.until(lambda: asyncio.to_thread(ax.is_key, here), KEY_S):
                t.cold = True
                t.window = t.window or await _own_window(here, pid, pts[0])
        if (not cold or t.brought or front_pid == pid) and here.app and here.pid == pid:
            apps.set_target(here.app)  # "click the field, then type": the typing goes here too
    if t.brought or t.raised:
        await _look_again(t, pts, pid, want_pid)
    final = t.hits[0]
    if want_pid and final.pid and final.pid != want_pid:
        raise Refused(
            f"Error: {final.app or 'another window'} covers that element — look again (ax_tree) before clicking."
        )


# Never brought forward, however the window server layers them: a menu, a popover, the menu bar, the Dock.
_NOT_WINDOWS = frozenset({"AXMenu", "AXMenuItem", "AXMenuBar", "AXMenuBarItem", "AXPopover", "AXDockItem"})


async def _own_window(h: ax.Hit | None, pid: int, pt: tuple[float, float]) -> ax.Win | None:
    """The app's own normal window under the point: the one thing an input brings an app forward
    for. The window server says whose window is there and on what layer — 0 for a normal app window;
    the menu bar and status items, the Dock, menus, Notification Center, Control Center and Spotlight
    are all on others. None for anything else, a window another process owns, or when the window
    server can't say."""
    if h is not None and h.role in _NOT_WINDOWS:
        return None
    wins = await asyncio.to_thread(ax.windows_at, *pt)
    w = next((w for w in wins or [] if w.pid == pid), None)  # (its frontmost one there)
    return w if w is not None and w.layer == 0 else None


async def _come_forward(pid: int, front: str, front_pid: int, app: str) -> str:
    """Bring the app to the front the way apps.bring_pid_front does; its name. One LaunchServices
    already calls frontmost gets the window server's switch that path starts with (bring_pid_front
    stops at "it is frontmost"): whether it took is looked at next."""
    if front_pid != pid:
        ok, real = await apps.bring_pid_front(pid)
        name = real or app or "that app"
        if not ok:
            raise _stayed_behind(name, front)
        return name
    from neo2.core.tools.computer.focus import force_front

    await asyncio.to_thread(force_front, pid)
    return app or front or "that app"


def _active(pid: int, seen: bool) -> bool:
    """Is the app in front — and, where Accessibility can say (``seen``), active with a window key?"""
    from neo2.core.tools.computer.focus import front_app

    return front_app()[1] == pid and (not seen or ax.key_window(pid) is not None)


def _stayed_behind(name: str, front: str) -> Refused:
    return Refused(
        f"Error: {name} didn't come to the front ({front or 'another app'} stayed there), so I "
        f"didn't touch it — a click on a window in the background may only bring it forward. "
        f"Ask the user to click {name} once, then try again."
    )


def _check_first(again: ax.Hit, pid: int, want_pid: int, name: str) -> None:
    if again.pid and ax.is_neo(again.pid, again.app):
        raise _off_limits()
    if again.pid and again.pid != pid and not want_pid:
        raise Refused(
            f"Error: with {name} in front, {again.app or 'another window'} is still over that spot — "
            "look again (screenshot or ax_tree) before clicking."
        )
    if again.sheet:
        raise _sheet_refusal(again)


async def _look_again(t: Target, pts: list[tuple[float, float]], pid: int, want_pid: int) -> None:
    """After an app came forward or a window rose: what is under every point now, against what the
    probe saw there (``t.hits``). A point whose owner or label changed is refused — the gate and the
    model judged what was there before."""
    for i, p in enumerate(pts):
        was = t.hits[i]
        now = await asyncio.to_thread(ax.hit, *p)
        if (now.pid or now.app) and await _protected(now, *p):
            raise _off_limits()
        if i == 0:
            _check_first(now, pid, want_pid, t.brought or was.app or "that app")
        if not now.pid:
            continue  # nothing can be told there now: what was seen before stands
        if not was.pid or (i == 0 and want_pid and (now.pid != was.pid or _label(now) != _label(was))):
            t.relabeled = True  # (an element's own app came over it: expected, but its label is judged again)
        elif now.pid != was.pid or _label(now) != _label(was):
            what = "the drop spot" if i == len(pts) - 1 and len(pts) > 1 else "that spot"
            raise Refused(
                f"Error: bringing {t.brought or was.app or 'the app'} forward changed what's at {what} "
                f"{tuple(int(round(v)) for v in p)} — it's now "
                + fence("screen", f"{now.app}: {words(now) or 'something else'}")
                + "\nso I didn't do it. Look again (screenshot) before trying again.",
                screen=True,
            )
        t.hits[i] = now


# ---- the outcome --------------------------------------------------------------------------------


@dataclass
class Look:
    scene: ax.Scene
    pixels: list | None


async def look(pid: int, pts: list[tuple[float, float]]) -> Look:
    sc = await asyncio.to_thread(ax.scene, pid)
    px = await asyncio.to_thread(screen.sample_points, pts) if pts else None
    return Look(sc, px)


async def before(pid: int, pts: list[tuple[float, float]], hover: tuple[float, float] | None = None) -> Look:
    """The look before an input, with the pointer already resting at ``hover`` (where the input
    leaves it): a view that highlights under the pointer looks the same before and after, so only
    what the input did counts as a change."""
    if hover is not None:
        from neo2.core.tools.computer import input as inp

        await asyncio.to_thread(inp.move, *hover)
        await asyncio.sleep(HOVER_S)
    return await look(pid, pts)


async def outcome(
    before: Look,
    pid: int,
    pts: list[tuple[float, float]],
    where: list[str],
    only_at_points: bool = False,
    target: Target | None = None,
) -> Outcome:
    """What the input changed: looked at once, and again a moment later if nothing had yet. With a
    click's ``target``: a click that itself brought its app forward and did nothing else anyone can
    see only activated it, most likely (:func:`brought_only`) — not "Changed: … is now in front"."""
    if before.pixels is None and not before.scene.known:
        return compare(before, None, where)
    await asyncio.sleep(LOOK_S)
    now = await look(pid, pts)
    out = compare(before, now, where, only_at_points)
    woke = target is not None and not only_at_points and await _woke(target, pid, before, now)
    if out.changed is False or (woke and not _did_more(before, now, pts, target)):
        await asyncio.sleep(LOOK_AGAIN_S)
        now = await look(pid, pts)
        out = compare(before, now, where, only_at_points)
    if woke and target is not None and not _did_more(before, now, pts, target):
        return brought_only(target.hits[0].app or now.scene.front or "the app", out.details)
    return out


# ---- a click that only brought its app forward ----------------------------------------------------
# A view that doesn't take the first mouse (the Chess board) spends a click on a window that isn't
# key, in an app that isn't active, on activating it. bring() makes the app active first; when that
# couldn't be seen to work, a click that itself brought its app forward and changed nothing else is
# no success: the front app, its window's title, the focus, the title bar all change with the
# activation alone.

ONLY_BROUGHT = "only brought"  # (the words the controls self-test knows an honest answer by)
_CHROME_PT = 52.0  # a window's title bar and toolbar, from its top edge: redrawn when it becomes key


async def woke(t: Target, pid: int) -> bool:
    """After a click on an app that wasn't seen active with its window key (``t.cold``): is it now?
    Then the click itself brought it forward. (For a click nobody looks at before and after.)"""
    return bool(t.cold and await wait.until(lambda: asyncio.to_thread(_active_key, pid, t.hits[0]), KEY_S))


async def _woke(t: Target, pid: int, before: Look, now: Look) -> bool:
    """Did the click itself bring its app forward? The front app changed to it during the click, or
    it wasn't seen active with its window key before (``t.cold``) and is now."""
    b, a = before.scene, now.scene
    if b.known and a.known and b.front_pid != pid and a.front_pid == pid:
        return True
    return t.cold and await asyncio.to_thread(_active_key, pid, t.hits[0])


def _active_key(pid: int, h: ax.Hit) -> bool:
    from neo2.core.tools.computer.focus import front_app

    return front_app()[1] == pid and (h.window is None or h.pid != pid or ax.is_key(h))


def _did_more(before: Look, now: Look, pts: list[tuple[float, float]], t: Target) -> bool:
    """After a click that brought its app forward: did anything else change? A sheet or dialog that
    came, or the picture at a point inside the window's content — its frame without the title bar
    and toolbar, which redraw as the window becomes key."""
    b, a = before.scene, now.scene
    if b.known and a.known and a.sheet != b.sheet:
        return True
    if before.pixels is None or now.pixels is None:
        return False  # (nothing shows the click did more than that)
    frame = t.window.frame if t.window is not None else None
    return any(
        screen.differs(p, _content_only(q, p, pt, frame))
        for pt, p, q in zip(pts, before.pixels, now.pixels, strict=False)
    )


def _content_only(now, before, pt: tuple[float, float], frame: tuple[float, float, float, float] | None):
    """``now`` with every cell of the look around ``pt`` that is outside the window's content (its
    ``frame`` less the title bar and toolbar band at the top) taken from ``before``."""
    if frame is None or now.size != before.size:
        return now
    x, y, w, h = frame
    r = screen._SAMPLE_R
    step = 2 * r / now.size[0]
    out = now.copy()
    for j in range(now.size[1]):
        cy = pt[1] - r + (j + 0.5) * step
        for i in range(now.size[0]):
            cx = pt[0] - r + (i + 0.5) * step
            if not (x <= cx < x + w and y + _CHROME_PT <= cy < y + h):
                out.putpixel((i, j), before.getpixel((i, j)))
    return out


def brought_only(app: str, details: str = "", looked: bool = True) -> Outcome:
    """A click that itself brought its app forward and did nothing else anyone can see (``looked``:
    NEO looked at the point before and after; else it didn't look at all): unknown, not a miss — and
    never clicked again by NEO (had it registered, a second click would do it twice)."""
    said = (
        f"The click {ONLY_BROUGHT} {app} to the front, it seems — it wasn't active with that window key before "
        "the click, and nothing else changed at the point (a first click on such a window often just activates it)."
        if looked
        else f"The click may have {ONLY_BROUGHT} {app} to the front — it wasn't active with that window key before "
        "the click, and a first click on such a window often just activates it."
    )
    return Outcome(
        said + " Take a screenshot to check before clicking again: if it did register, a second click would do it twice.",
        details,
        None,
    )


async def scene_change(before: Look, pid: int) -> Outcome:
    """What a key or an AX press changed in the scene (the front app, the window, a sheet, the
    focus): looked at every POLL_S, done at the first change."""
    if not before.scene.known:
        return compare(before, None, [])
    out = compare(before, before, [])
    for _ in range(SCENE_LOOKS):
        await asyncio.sleep(POLL_S)
        out = compare(before, await look(pid, []), [])
        if out.changed:
            break
    return out


@dataclass
class Outcome:
    said: str  # NEO's own words: what changed, or that nothing visibly did
    details: str  # words read from the screen (titles, sheets, labels): fenced for the model
    changed: bool | None  # None: couldn't tell


def compare(before: Look, now: Look | None, where: list[str], only_at_points: bool = False) -> Outcome:
    """What differs between two looks. ``only_at_points``: judge by the pictures at the points and a
    sheet or dialog only — not by which app is in front, its window or its focus (a drag between two
    apps: the source coming forward on the button press says nothing about the drop)."""
    if now is None:
        return Outcome("I couldn't check whether anything changed (no Screen Recording or Accessibility answer).", "", None)
    changes: list[str] = []
    details: list[str] = []
    b, a = before.scene, now.scene
    moved = []
    if before.pixels is not None and now.pixels is not None:
        moved = [w for w, p, q in zip(where, before.pixels, now.pixels, strict=False) if screen.differs(p, q)]
        if moved:
            changes.append("the screen changed around " + " and ".join(moved))
    if b.known and a.known:
        if a.front_pid != b.front_pid and not only_at_points:
            changes.append(f"{a.front or 'another app'} is now in front")
        if a.sheet != b.sheet:
            changes.append("a sheet or dialog opened" if a.sheet else "the sheet or dialog closed")
            if a.sheet:
                details.append(f"now showing {a.sheet}")
        if a.title != b.title and (a.title or b.title) and not only_at_points:
            changes.append("the window changed")
            details.append(f"window: “{a.title[:80]}”")
        if a.focus != b.focus and a.focus and not only_at_points:
            changes.append("the keyboard focus moved")
            details.append(f"focus: {a.focus}")
    if changes:
        return Outcome("Changed: " + "; ".join(changes) + ".", "; ".join(details), True)
    if before.pixels is None or now.pixels is None:
        return Outcome("I couldn't compare the screen (no Screen Recording), and nothing else changed.", "", None)
    return Outcome(
        f"Nothing visibly changed there within {LOOK_S + LOOK_AGAIN_S:g}s — it may have missed, been swallowed, "
        "or the app needs something else first. Look (screenshot) before trying again; don't assume it worked.",
        "",
        False,
    )


# Where a click's effect shows: a board, a canvas, a picture or a bare area answers at the point (a
# piece moves, a stroke appears); a control (a button, a link, a menu item, a row) may act anywhere
# — a track changes, a badge in the header counts up, the other pane of a split view opens.
_SURFACES = frozenset({"", "AXGroup", "AXImage", "AXUnknown", "AXLayoutArea", "AXLayoutItem", "AXScrollArea", "AXWebArea"})


def at_the_point(h: ax.Hit, from_picture: bool) -> bool:
    """Should a click here change the picture around the point? A click aimed at a screenshot's
    pixels, or at a surface rather than a control: yes."""
    return from_picture or h.role in _SURFACES


def maybe_elsewhere(out: Outcome, where: str) -> Outcome:
    """A click on a control after which nothing changed near the point: unknown, not a miss — its
    effect may be elsewhere, and repeating a click that did work does it twice."""
    if out.changed is not False:
        return out
    return Outcome(
        f"No change was seen near {where} within {LOOK_S + LOOK_AGAIN_S:g}s (NEO looks only around the point, and "
        "at the front app, the window, a sheet and the focus) — its effect may be elsewhere, or it missed. Take a "
        "screenshot to check before clicking it again: repeating a click that worked would do it twice.",
        out.details,
        None,
    )


def click_word(button: str, count: int) -> str:
    return f"{('', 'Double-', 'Triple-')[max(1, min(3, count)) - 1]}{button.capitalize()}-clicked"


def words(h: ax.Hit) -> str:
    """What was under the pointer, as read from the screen."""
    role = h.role.removeprefix("AX")
    label = " ".join(h.label.split())[:60]
    return f"{role} “{label}”" if label and role else (label or role)


def report(head: str, out: Outcome, under: list[str], doubt: str = "") -> tuple[str, bool]:
    """(the tool's text, ok). Third-party words go inside an untrusted fence. ``doubt``: macOS's own
    warning that it may have dropped the events, said when nothing is known to have changed."""
    text = f"{head} {out.said}" + (doubt if out.changed is not True else "")
    bits = [f"under the pointer: {u}" for u in under if u] + ([out.details] if out.details else [])
    if bits:
        text += "\n" + fence("screen", "; ".join(bits))
    return text, out.changed is not False
