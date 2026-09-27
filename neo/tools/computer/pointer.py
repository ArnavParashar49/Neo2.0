"""Coordinate input the model can trust: it lands where the model aimed, in the app it aimed at,
never on NEO's own window — and the result says what it hit and whether anything changed.

The run that prompted this (a board game): every click and drag reported "ok" while the board never
moved. A click on a window whose app isn't frontmost may only activate it (a view that doesn't
accept the first mouse), a screenshot's pixels are not always screen points, NEO's overlay is on
top but invisible in screenshots, and a bare "ok" hid all of it. So, for each click, drag and
scroll a model makes:

1. the numbers: plain screen points, a screenshot's pixels (``shot``), 0–1000 of a screenshot
   (nx,ny, or a box_2d [ymin, xmin, ymax, xmax] acted on at its middle: Gemini's own scale), an
   element id or a board square — all through one checked function (:func:`numeric` / :func:`aim`),
   which NEO does the arithmetic in; on some display; never numbers picked by a brain that couldn't
   see the screen, in any form. A point a model picked that lies inside a control (a button, a
   link, a tab) is moved to its middle, and that spot is looked at and gated again (:func:`snap`);
2. the target: what's under each point (one AX lookup) — never NEO's own window, never a window
   a sheet is blocking; the owner brought to the front (and its window made key) first, and every
   point looked at again after that: an input whose target changed on the way isn't made (a drag
   between two apps brings neither forward: raising one could cover the other's drop spot);
3. the outcome: the owner app and label at the point, and a before/after look (the picture
   around each point, taken with the pointer already resting where it will be — a hover highlight
   is no change — plus the front app, a sheet or dialog, the focused window and element, watched
   meanwhile: done at the first change). A move on a board or canvas after which nothing visibly
   changed is reported as a miss; a click on a control whose effect may show elsewhere says it saw
   no change near the point, and to check before repeating it. The result carries an Effect (what
   changed) and, for a brain that can see that aimed at a picture, the picture after it — so no
   separate "take a screenshot to check" turn is needed.
"""

from __future__ import annotations

import asyncio
import itertools
import re
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from neo.agent.registry import ToolContext, fence
from neo.decide._objc import pooled
from neo.tools.computer import apps, ax, screen

SETTLE_S = 0.12  # after bringing an app forward (or raising a window), before the input
HOVER_S = 0.1  # the pointer rests where the input will be this long before the first look
LOOK_S = 0.15  # after the input, the first look at what changed (most changes start at once)…
LOOK_AGAIN_S = 0.85  # …and once more when nothing had yet (a page that loads, a slow redraw)
POLL_S = 0.04  # after a key or a press: a look at the window, sheet and focus this often…
SCENE_WAIT_S = 0.4  # …for at most this long, done at the first change
SCENE_POLL_S = 0.04  # while waiting for a look at the picture: the scene (cheap) this often


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
_shots_lock = threading.Lock()
_shot_ids = itertools.count(1)
_KEEP_SHOTS = 12
# Ids carry this run's mark: an id a model reads in a conversation from before NEO restarted can
# never name a different, newer screenshot.
_RUN = secrets.token_hex(2)


def remember(x0: float, y0: float, ppp: float, w: int, h: int) -> Shot:
    s = Shot(f"s{next(_shot_ids)}-{_RUN}", float(x0), float(y0), float(ppp), int(w), int(h))
    with _shots_lock:
        _shots[s.id] = s
        while len(_shots) > _KEEP_SHOTS:
            _shots.pop(next(iter(_shots)))
    return s


def get_shot(sid: str) -> Shot | None:
    """The screenshot with this id — and, being in use, the last one to be let go: the after-looks
    of the inputs a model makes from one screenshot never push that screenshot out."""
    key = str(sid).strip()
    with _shots_lock:
        s = _shots.pop(key, None)
        if s is not None:
            _shots[key] = s
    return s


def note(s: Shot, what: str, others: list[screen.Display]) -> str:
    """NEO's own words about a screenshot (not fenced: nothing in them came from the screen): the
    image's size, its shot id, and every way to say where to act in it."""
    exact = abs(s.ppp - 1.0) < 1e-6
    text = (
        f"Screenshot {s.id} attached ({what}); the image is {s.w}×{s.h} px. To act on something in it, "
        f'give its pixel position in this image with shot="{s.id}" — click x,y · drag x1,y1,x2,y2 · '
        "scroll x,y — and NEO maps it to the screen. " + FORMS_NOTE
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


# Every way a call may say where, besides pixels (NEO does all the arithmetic, never the model).
FORMS_NOTE = (
    "Or, with the same shot: nx,ny — a point on 0–1000 of the image (0,0 top-left, 1000,1000 bottom-right) — "
    "or box_2d [ymin, xmin, ymax, xmax] on 0–1000 (NEO acts at the box's middle); for a drag nx1,ny1,nx2,ny2 or "
    "box_2d1, box_2d2. An element id (from the list with a screenshot, a look or ax_tree) is better still: exact."
)


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
        f"and I won't {tool} at coordinates nobody saw, in any form (x,y, nx,ny, box_2d). Use element ids from "
        f"ax_tree or a look ({_BY_ID.get(tool, _BY_ID['click'])}), "
        "or tell the user the model that can see the screen is busy right now."
    )


def _numbers(a: dict, keys: tuple[str, ...]) -> list[float]:
    out = []
    for k in keys:
        v = a.get(k)
        if v is None or isinstance(v, bool):
            raise Refused(f"Error: {' and '.join(keys)} are needed (numbers).")
        out.append(_number(k, v))
    return out


def _number(k: str, v) -> float:
    if isinstance(v, bool):
        raise Refused(f"Error: {k} must be a number, not {str(v)[:20]!r}.")
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise Refused(f"Error: {k} must be a number, not {str(v)[:20]!r}.") from None
    if f != f or f in (float("inf"), float("-inf")):
        raise Refused(f"Error: {k} must be a finite number.")
    return f


def _said(ctx: ToolContext, values: list[float]) -> bool:
    """Did the user's own words give these numbers ("click at 300, 400")?"""
    said = {float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", ctx.user_text or "")}
    return bool(values) and all(v in said or round(v) in said for v in values)


# ---- every way a call says where: one checked function -----------------------------------------
# x,y (screen points, or a screenshot's pixels with shot), nx,ny (0–1000 of a screenshot, Gemini's
# own scale), box_2d [ymin, xmin, ymax, xmax] (0–1000, Gemini's own boxes: NEO acts at the middle),
# an element id (NEO found it: Accessibility), a board cell (NEO found it: the board helper). Every
# form a model picks numbers for is refused to a brain that saw nothing — in any form, whatever the
# numbers — and every form maps through NEO's own record of the screenshot (never the model's sums).

NORM = 1000.0  # the 0–1000 scale of nx,ny and box_2d

# The pointer tools' words for the model (the handlers' registrations use these; see the wiring
# notes): every form named, no app named — when a picture is needed is told by evidence.
CLICK_DESCRIPTION = (
    "Click an element id (from a look, ax_tree or a screenshot's list — exact, best), or a place in a "
    "screenshot with its shot id: its pixels x,y, or nx,ny on 0–1000 of it, or box_2d [ymin, xmin, ymax, xmax] "
    "on 0–1000 (its middle is clicked); or screen points x,y. A point inside a button, link or tab is clicked "
    "at its middle. Brings the app there to the front first, never clicks NEO's own window, and says what "
    "changed (with the picture after it, when you can see). button: left|right|middle; count: 1-3; "
    "modifiers: cmd, shift, alt, ctrl."
)
DRAG_DESCRIPTION = (
    "Drag from one place to another: each end an element id (id1, id2), or a place in a screenshot with its "
    "shot id — pixels x1,y1 / x2,y2, nx1,ny1 / nx2,ny2 on 0–1000, or box_2d1 / box_2d2 — or screen points. "
    "Brings the app there to the front first (not when the ends are in two different apps) and says what "
    "changed at both ends."
)
SCROLL_DESCRIPTION = (
    "Scroll the content under the pointer, or at an element id (a list, a pane), or at a place in a "
    "screenshot with its shot id (x,y pixels, nx,ny or box_2d on 0–1000), or screen points x,y. dy<0 scrolls "
    "down, dy>0 up (lines)."
)
SCREENSHOT_DESCRIPTION = (
    "Take a screenshot: a whole display (display: its number, 1 = main), or a region x,y,w,h in screen "
    "points. Use it when a look or ax_tree shows too little to act on — a large area of a window with no "
    "elements in it (a board, a map, a canvas, a picture) — or for a layout question. Its note gives the "
    "image size, a shot id and the controls in it with element ids: click an id when the target is listed; "
    "otherwise give its place in the image with the shot id (pixels x,y, nx,ny on 0–1000, or box_2d "
    "[ymin, xmin, ymax, xmax] on 0–1000) and NEO maps it to the screen."
)

# The schema of each form, to merge into click / scroll (end "") and drag (ends "1", "2") parameters.
FORM_ARGS = {
    "nx": {"type": "number", "description": "0–1000 across the screenshot (with shot), instead of x"},
    "ny": {"type": "number", "description": "0–1000 down the screenshot (with shot), instead of y"},
    "box_2d": {
        "type": "array",
        "items": {"type": "number"},
        "description": "[ymin, xmin, ymax, xmax] on 0–1000 of the screenshot (with shot): acts at its middle",
    },
}
DRAG_FORM_ARGS = {
    **{f"{k}{n}": {"type": "number", "description": f"0–1000 of the screenshot (with shot): {'start' if n == '1' else 'end'}"}
       for n in ("1", "2") for k in ("nx", "ny")},
    "box_2d1": {"type": "array", "items": {"type": "number"}, "description": "start: [ymin, xmin, ymax, xmax] on 0–1000"},
    "box_2d2": {"type": "array", "items": {"type": "number"}, "description": "end: [ymin, xmin, ymax, xmax] on 0–1000"},
}

# The pointer tools' whole parameter schemas (the registrations swap theirs for these).
_SHOT_ARG = {"type": "string", "description": "screenshot id: x,y are pixels of that image; nx,ny and box_2d are 0–1000 of it"}
CLICK_PARAMETERS = {
    "type": "object",
    "properties": {
        "x": {"type": "number"},
        "y": {"type": "number"},
        **FORM_ARGS,
        "shot": _SHOT_ARG,
        "id": {"type": "string", "description": "element id (a look, ax_tree, a screenshot's list), instead of a place"},
        "button": {"type": "string", "enum": ["left", "right", "middle"]},
        "count": {"type": "integer", "description": "1, 2 (double-click) or 3"},
        "modifiers": {"type": "array", "items": {"type": "string", "enum": ["cmd", "shift", "alt", "ctrl"]}},
    },
    "required": [],
}
DRAG_PARAMETERS = {
    "type": "object",
    "properties": {
        "x1": {"type": "number"},
        "y1": {"type": "number"},
        "x2": {"type": "number"},
        "y2": {"type": "number"},
        **DRAG_FORM_ARGS,
        "id1": {"type": "string", "description": "element id to drag from, instead of a place"},
        "id2": {"type": "string", "description": "element id to drop on, instead of a place"},
        "shot": _SHOT_ARG,
    },
    "required": [],
}
SCROLL_PARAMETERS = {
    "type": "object",
    "properties": {
        "x": {"type": "number"},
        "y": {"type": "number"},
        **FORM_ARGS,
        "id": {"type": "string", "description": "element id to scroll, instead of a place"},
        "shot": _SHOT_ARG,
        "dy": {"type": "integer"},
        "dx": {"type": "integer"},
        "at_pointer": {"type": "boolean"},
    },
    "required": [],
}

# A board's squares (neo.tools.computer.board registers the resolver): ``cell`` for a click or a
# scroll, ``from`` / ``to`` for a drag's two ends.
_CELL_KEYS = {"": "cell", "1": "from", "2": "to"}
# fn(cell, shot, *, blind) → (x, y): see set_cell_resolver.
CellResolver = Callable[..., tuple[float, float]]
_cell_resolver: CellResolver | None = None
# Is there a board (a grid NEO found) at this screen point? Nothing snaps inside one.
GridCheck = Callable[[float, float], bool]
_grid_check: GridCheck | None = None


def _hears(fn: Callable, name: str) -> bool:
    """Does ``fn`` take the keyword argument ``name`` (by name, or in a **kwargs)?"""
    import inspect

    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        p.kind is p.VAR_KEYWORD or (p.name == name and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)) for p in params
    )


def set_cell_resolver(fn: CellResolver | None) -> None:
    """The board helper's seam: ``fn(cell, shot, *, blind)`` → the screen point of that square, or
    raises (unknown square, no board, a stale grid — any exception: nothing is done, its words are
    the refusal, so they name no point). ``blind``: the brain acting can't see — the helper gives
    it squares only on a board NEO found itself from its own capture (never one mapped from a
    seeing model's corners). A resolver that can't be told ``blind`` is not taken (TypeError)."""
    global _cell_resolver
    if fn is not None and not _hears(fn, "blind"):
        raise TypeError("a cell resolver takes blind= (whether the brain acting can see): fn(cell, shot, *, blind)")
    _cell_resolver = fn


# Other ids NEO grounds itself, by their prefix (OCR's text marks "t1"…: neo.tools.computer.ocr
# registers "t"): ``fn(id)`` → the screen point of that mark, or raises (any exception: nothing is
# done); ``labels(id)`` → the mark's words, which the click gate judges beside what's under the point.
IdResolver = Callable[[str], tuple[float, float]]
IdLabels = Callable[[str], list[str]]
_id_resolvers: dict[str, IdResolver] = {}
_id_labels: dict[str, IdLabels] = {}


def set_id_resolver(prefix: str, fn: IdResolver | None, *, labels: IdLabels | None = None) -> None:
    """A helper's seam for its own marks (ids with ``prefix`` and a number: "t3"). Their points were
    read off NEO's own capture, so a brain that can't see may use them, like Accessibility's ids.
    ``labels``: the mark's own words (a text mark's text) — the click gate asks the user before a
    click on words that send, delete or buy, whatever is under the point."""
    if not re.fullmatch(r"[a-z]{1,3}", prefix) or prefix in ("e", "w"):
        raise ValueError(f"{prefix!r}: a mark prefix is 1–3 lowercase letters, never Accessibility's own (e, w)")
    if fn is None:
        _id_resolvers.pop(prefix, None)
        _id_labels.pop(prefix, None)
        return
    _id_resolvers[prefix] = fn
    if labels is None:
        _id_labels.pop(prefix, None)
    else:
        _id_labels[prefix] = labels


def _helper_error(e: Exception) -> Refused:
    """A helper's own error (a board's, a mark's) as the refusal: its words, once "Error:"."""
    msg = " ".join(str(e).split()) or type(e).__name__
    return Refused(msg if msg.startswith("Error") else f"Error: {msg}")


def _own_mark(eid: str, blind: bool = False) -> Aim | None:
    for prefix, fn in _id_resolvers.items():
        if re.fullmatch(rf"{re.escape(prefix)}\d+", eid):
            try:
                pt = fn(eid)
                x, y = float(pt[0]), float(pt[1])
                words = [str(w) for w in (_id_labels[prefix](eid) if prefix in _id_labels else []) if w]
            except Refused:
                raise
            except Exception as e:  # noqa: BLE001 - a helper's refusal (or its failure): nothing is done
                raise _helper_error(e) from None
            _on_screen([(x, y)], blind=blind, what=f"mark {eid}")
            return Aim((x, y), "mark", None, f"mark {eid}", labels=words)
    return None


def set_grid_check(fn: GridCheck | None) -> None:
    """The board helper's seam: ``fn(x, y)`` — is that screen point inside a grid NEO found?"""
    global _grid_check
    _grid_check = fn


@dataclass
class Aim:
    """Where one end of an input goes (screen points) and how the call said it."""

    point: tuple[float, float]
    form: str  # "points" | "pixels" | "normalized" | "box" | "id" | "cell" | "mark"
    shot: Shot | None = None
    said: str = ""  # NEO's own words for how it was named: "pixel (612, 540) of s3"
    element: Any = None  # the ax.Element (form "id")
    pid: int = 0  # its app's process (form "id")
    owner: str = ""  # its app's name (form "id")
    labels: list[str] = field(default_factory=list)  # a helper's mark's own words (form "mark"): for the gate

    @property
    def numbers(self) -> bool:
        """A model picked these numbers itself (not an element or a square NEO found)."""
        return self.form in ("points", "pixels", "normalized", "box")

    @property
    def pictured(self) -> bool:
        """Aimed at a picture: a screenshot's pixels, a board's square or a mark read off a capture
        (a change shows at the point)."""
        return self.shot is not None or self.form in ("cell", "mark")


def _present(v) -> bool:
    return v is not None and v != "" and v != []


def _given(a: dict, end: str) -> list[str]:
    forms = []
    if _present(a.get(f"x{end}")) or _present(a.get(f"y{end}")):
        forms.append("x,y")
    if _present(a.get(f"nx{end}")) or _present(a.get(f"ny{end}")):
        forms.append("nx,ny")
    if _present(a.get(f"box_2d{end}")):
        forms.append("box_2d")
    if _present(a.get(_CELL_KEYS.get(end, ""))):
        forms.append(_CELL_KEYS[end])
    return forms


def _shot_of(a: dict) -> Shot | None:
    if a.get("shot") in (None, ""):
        return None
    s = get_shot(str(a["shot"]))
    if s is None:
        raise Refused(f"Error: there's no screenshot {str(a['shot'])[:12]!r} any more — take a new one.")
    return s


def _box(v) -> list[float]:
    """box_2d as 4 numbers: a list, or text like "88,370,150,470" / "[88, 370, 150, 470]"."""
    if isinstance(v, str):
        v = [p for p in re.split(r"[\s,;\[\]()]+", v) if p]
    if isinstance(v, dict):  # {"ymin":…, "xmin":…, "ymax":…, "xmax":…}
        v = [v.get(k) for k in ("ymin", "xmin", "ymax", "xmax")]
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        raise Refused("Error: box_2d is 4 numbers, [ymin, xmin, ymax, xmax] on 0–1000 of the screenshot.")
    return [_number("box_2d", x) for x in v]


def _norm_to_screen(s: Shot, nx: float, ny: float) -> tuple[float, float]:
    """A 0–1000 point of screenshot ``s`` on the screen: a share of the area it shows, so it holds
    however the image was resized on the way to the model. 1000 is the last pixel, not past it."""
    ix = min(nx / NORM * s.w, s.w - 0.5)
    iy = min(ny / NORM * s.h, s.h - 0.5)
    return s.to_screen(max(ix, 0.0), max(iy, 0.0))


def _on_screen(pts: list[tuple[float, float]], *, blind: bool = False, what: str = "") -> None:
    """Every point on some display, else refused — naming the point and the displays, except to a
    brain that can't see (no coordinates reach it in any form: the square or mark by its name)."""
    shown = screen.displays()
    for x, y in pts:
        if not any(d.contains(x, y) for d in shown):
            if blind:
                raise Refused(
                    f"Error: {what or 'that spot'} is off every screen now — look again (a look or ax_tree) before acting."
                )
            raise Refused(
                f"Error: ({x:g}, {y:g}) is off every screen ({'; '.join(d.words() for d in shown)}). "
                "Screen points are what click takes; for a spot in a screenshot, pass its pixel position with shot."
            )


def numeric(a: dict, ctx: ToolContext, tool: str = "click", end: str = "") -> Aim | None:
    """The one checked function for every form a call gives numbers in, for one end ("" or "1"/"2"):
    x,y · nx,ny · box_2d, and a board cell. None: no such form for that end (an id, the pointer).

    - a brain that can't see is refused every numeric form (only x,y the user's own words gave pass,
      as before); a cell goes to the board helper with ``blind`` (it gives such a brain squares only
      on a board NEO found itself, from its own capture);
    - nx,ny and box_2d need a shot, and are 0–1000 of it; x,y with a shot are its pixels; out of
      range is refused with a hint; every point must be on a display.
    """
    forms = _given(a, end)
    if not forms:
        return None
    if len(forms) > 1:
        raise Refused(f"Error: say where one way only — {' and '.join(forms)} were both given" + (f" for end {end}." if end else "."))
    form = forms[0]
    blind_ctx = bool(ctx.get("blind"))
    if form == _CELL_KEYS.get(end):
        return _cell(a, form, blind_ctx)
    shot = None
    if form == "x,y":
        vals = _numbers(a, (f"x{end}", f"y{end}"))
        if blind_ctx and not _said(ctx, vals):
            raise Refused(blind(tool))
        shot = _shot_of(a)
        ix, iy = vals
        if shot is not None:
            if not (0 <= ix <= shot.w and 0 <= iy <= shot.h):
                raise Refused(
                    f"Error: ({ix:g}, {iy:g}) is outside screenshot {shot.id}, which is {shot.w}×{shot.h} px — give "
                    "the pixel position in that image (from its top-left corner), or nx,ny on 0–1000 of it."
                )
            pt = shot.to_screen(ix, iy)
            aimed = Aim(pt, "pixels", shot, f"pixel ({ix:g}, {iy:g}) of {shot.id}")
        else:
            aimed = Aim((ix, iy), "points", None, "")
    else:
        if blind_ctx:  # never a picture's numbers from a brain that saw no picture, whatever they are
            raise Refused(blind(tool))
        shot = _shot_of(a)
        if shot is None:
            raise Refused(
                f"Error: {form} is on 0–1000 of a screenshot — pass its shot id too (shot=\"s…\"), or give "
                "screen points as x,y."
            )
        if form == "nx,ny":
            nx, ny = _numbers(a, (f"nx{end}", f"ny{end}"))
            if not (0 <= nx <= NORM and 0 <= ny <= NORM):
                raise Refused(
                    f"Error: nx,ny ({nx:g}, {ny:g}) must each be 0–1000 of screenshot {shot.id} (0,0 its top-left, "
                    "1000,1000 its bottom-right) — for its pixels, give x,y instead."
                )
            aimed = Aim(_norm_to_screen(shot, nx, ny), "normalized", shot, f"({nx:g}, {ny:g}) of 1000 in {shot.id}")
        else:
            ymin, xmin, ymax, xmax = _box(a.get(f"box_2d{end}"))
            if not all(0 <= v <= NORM for v in (ymin, xmin, ymax, xmax)):
                raise Refused(
                    f"Error: box_2d values must each be 0–1000 of screenshot {shot.id}, in the order [ymin, xmin, "
                    "ymax, xmax] — for its pixels, give x,y instead."
                )
            if ymin > ymax or xmin > xmax:
                raise Refused(
                    "Error: box_2d is [ymin, xmin, ymax, xmax] — y first: "
                    f"[{ymin:g}, {xmin:g}, {ymax:g}, {xmax:g}] has a min past its max."
                )
            pt = _norm_to_screen(shot, (xmin + xmax) / 2, (ymin + ymax) / 2)
            aimed = Aim(pt, "box", shot, f"the middle of box_2d [{ymin:g}, {xmin:g}, {ymax:g}, {xmax:g}] of {shot.id}")
    _on_screen([aimed.point], blind=blind_ctx)
    return aimed


def _cell(a: dict, key: str, blind: bool) -> Aim:
    """A board's square, from the board helper — told whether the brain acting can see (it refuses
    such a brain a board NEO didn't find itself). Any error of the helper's refuses: nothing is done."""
    name = str(a.get(key)).strip()[:12]
    resolve = _cell_resolver
    if resolve is None:
        raise Refused(f"Error: there's no board to name squares on — take a screenshot of it first ({key}={name!r}).")
    shot = _shot_of(a)
    try:
        pt = resolve(name, shot, blind=blind)
        x, y = float(pt[0]), float(pt[1])
    except Refused:
        raise
    except Exception as e:  # noqa: BLE001 - the helper's refusal (a BoardError…) or its failure: nothing is done
        raise _helper_error(e) from None
    _on_screen([(x, y)], blind=blind, what=f"square {name}")
    return Aim((x, y), "cell", shot, f"square {name}")


async def aim(a: dict, ctx: ToolContext, tool: str = "click", end: str = "") -> Aim | None:
    """Where one end of an input goes, from whichever form the call used: an element id first (as
    before: exact, the element's live frame), else :func:`numeric`. None: nothing for this end."""
    eid = a.get(f"id{end}")
    if _present(eid) and (own := _own_mark(str(eid).strip(), bool(ctx.get("blind")))) is not None:
        return own
    if _present(eid):
        got = await _gates()._element_spot(str(eid), tool)
        if isinstance(got, str):
            raise Refused(got)
        if not isinstance(got, tuple):  # (NEO's own window or a protected app's element)
            raise _off_limits()
        e, pid, owner, at = got
        return Aim(at, "id", None, f"{e.id} ({e.role.removeprefix('AX')})", e, pid, owner)
    return numeric(a, ctx, tool, end)


def points(
    a: dict, keys: tuple[str, ...], ctx: ToolContext, tool: str = "click"
) -> tuple[list[tuple[float, float]], Shot | None]:
    """The screen points a call means, checked: pairs of ``keys`` ("x", "y", or "x1".."y2") — or,
    for each pair's end, the same end's nx,ny / box_2d / cell (see :func:`numeric`)."""
    pts: list[tuple[float, float]] = []
    shot = None
    for i in range(0, len(keys), 2):
        end = keys[i][1:]
        got = numeric(a, ctx, tool, end)
        if got is None:
            _numbers(a, keys[i : i + 2])  # (says which numbers are needed)
            raise Refused(f"Error: {' and '.join(keys)} are needed (numbers).")
        pts.append(got.point)
        shot = shot or got.shot
    return pts, shot


def _gates():
    """The package's gate helpers (the confirmation gate, protected apps, NEO's own window, the
    look before an input), looked up at call time — never bound here, so a test's monkeypatch and
    the frozen gates module (Step 0: neo.tools.computer.gates, re-exported by the package) both reach
    every path."""
    from neo.tools import computer as _c

    return _c


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


def _off_limits() -> Refused:
    """NEO's own window (it floats on top and never shows in screenshots) or a protected app's: never
    touched, never brought forward — the user is told (the package's protected refusal)."""
    return Refused("Error: that spot is NEO's own window or an app NEO never touches.", protected=True)


async def _protected(h: ax.Hit, x: float, y: float) -> bool:
    from neo.tools import computer as _c  # (the gate helpers, looked up at call time)

    return await _c._protected_hit(h, x, y)


async def _protected_pid(pid: int, name: str = "") -> bool:
    from neo.tools import computer as _c

    return bool(pid) and (ax.is_neo(pid, name) or _c._protected(name, pid))


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


async def bring(t: Target, pts: list[tuple[float, float]], *, want_pid: int = 0, blind: bool = False) -> None:
    """Bring the app acted on — the owner of the first point, or ``want_pid`` (an element's app) —
    to the front and make its window key, then look at every point again: an input whose target
    changed on the way (another window came over a point, the drop spot is now something else) isn't
    made, and the hits are the new ones. With ``want_pid`` the point must be that app's once it is in
    front (not covered by another window). ``blind``: the brain acting can't see — a refusal names
    no point."""
    first = here = t.hits[0]
    pid = want_pid or first.pid
    if pid and (first.regular if pid == first.pid else await asyncio.to_thread(ax.regular, pid)):
        from neo.tools.computer.focus import front_app

        front, front_pid = await asyncio.to_thread(front_app)
        if front_pid != pid:
            if await _protected_pid(pid, first.app if pid == first.pid else ""):
                raise _off_limits()  # never brought forward, even to refuse it afterwards
            ok, real = await apps.bring_pid_front(pid)
            name = real or first.app or "that app"
            if not ok:
                raise Refused(
                    f"Error: {name} didn't come to the front ({front or 'another app'} stayed there), so I "
                    f"didn't touch it — a click on a window in the background may only bring it forward. "
                    f"Ask the user to click {name} once, then try again."
                )
            t.brought = name
            await asyncio.sleep(SETTLE_S)
            again = await asyncio.to_thread(ax.hit, *pts[0])  # (the window there now: the one to raise)
            if (again.pid or again.app) and await _protected(again, *pts[0]):
                raise _off_limits()
            _check_first(again, pid, want_pid, name)
            here = again if again.pid else first
        if here.pid == pid and await asyncio.to_thread(ax.raise_window, here):  # (never another app's window)
            t.raised = True
            await asyncio.sleep(SETTLE_S)
        if here.app and here.pid == pid:
            apps.set_target(here.app)  # "click the field, then type": the typing goes here too
    if t.brought or t.raised:
        await _look_again(t, pts, pid, want_pid, blind)
    final = t.hits[0]
    if want_pid and final.pid and final.pid != want_pid:
        raise Refused(
            f"Error: {final.app or 'another window'} covers that element — look again (ax_tree) before clicking."
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


async def _look_again(t: Target, pts: list[tuple[float, float]], pid: int, want_pid: int, blind: bool = False) -> None:
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
            if not blind:  # (a brain that can't see is never told a point)
                what += f" {tuple(int(round(v)) for v in p)}"
            raise Refused(
                f"Error: bringing {t.brought or was.app or 'the app'} forward changed what's at {what} — it's now "
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
    cost: float = 0.0  # how long the look at the scene took (a busy app answers slowly)


async def look(pid: int, pts: list[tuple[float, float]]) -> Look:
    t0 = time.monotonic()
    sc = await asyncio.to_thread(ax.scene, pid)
    cost = time.monotonic() - t0
    px = await asyncio.to_thread(screen.sample_points, pts) if pts else None
    return Look(sc, px, cost)


async def before(pid: int, pts: list[tuple[float, float]], hover: tuple[float, float] | None = None) -> Look:
    """The look before an input, with the pointer already resting at ``hover`` (where the input
    leaves it): a view that highlights under the pointer looks the same before and after, so only
    what the input did counts as a change."""
    if hover is not None:
        from neo.tools.computer import input as inp

        await asyncio.to_thread(inp.move, *hover)
        await asyncio.sleep(HOVER_S)
    return await look(pid, pts)


class _Clock:
    """The time since an input, for the waits after it: by the clock and by the sleeps asked for,
    whichever says more has passed — a slow app's answers count against a wait, and a wait never
    outlasts the sleeps it was given."""

    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.slept = 0.0

    def left(self, at: float) -> float:
        """Seconds left until ``at`` seconds after the input."""
        return min(at - (time.monotonic() - self.t0), at - self.slept)

    async def sleep(self, s: float) -> None:
        if s > 0:
            await asyncio.sleep(s)
            self.slept += s


def _gap(poll: float, cost: float) -> float:
    """The pause before the next look at the scene: every ``poll`` seconds, but never looking more
    than half the time (a look that takes long is spaced out, not repeated back to back)."""
    return max(poll - cost, cost)


async def _scene_meanwhile(
    before: Look, pid: int, clock: _Clock, at: float, only_at_points: bool, cost: float
) -> tuple[Outcome | None, float]:
    """Until ``at`` seconds after the input, look at the scene (the front app, the window, a sheet,
    the focus: a few quick Accessibility questions) every SCENE_POLL_S: (the outcome at the first
    change — no picture needed to know something happened — else None, the last look's cost). A
    look that couldn't end by ``at`` (by the last one's cost) isn't started: the wait ends on time."""
    if not before.scene.known:
        await clock.sleep(clock.left(at))
        return None, cost
    while True:
        gap, left = _gap(SCENE_POLL_S, cost), clock.left(at)
        if left < gap + cost:
            await clock.sleep(left)
            return None, cost
        await clock.sleep(gap)
        t0 = time.monotonic()
        sc = await asyncio.to_thread(ax.scene, pid)
        cost = time.monotonic() - t0
        out = compare(Look(before.scene, None), Look(sc, None), [], only_at_points)
        if out.changed:
            return out, cost


async def outcome(
    before: Look,
    pid: int,
    pts: list[tuple[float, float]],
    where: list[str],
    only_at_points: bool = False,
    *,
    settle: bool = False,
) -> Outcome:
    """What the input changed: the scene watched meanwhile (a sheet that opens, an app that comes
    forward, the focus that moves), the picture looked at LOOK_S after the input, and both again
    until LOOK_S + LOOK_AGAIN_S after it if nothing had changed yet (a page that loads, a slow
    redraw) — never later than that and one look, however slowly the app answers. A change seen in
    the scene ends the wait, confirmed by one look at the picture (the result names both) — taken
    LOOK_S after the change when ``settle`` (a picture of the result follows: the redraw, a sheet
    sliding in, has had its moment)."""
    if before.pixels is None and not before.scene.known:
        return compare(before, None, where)
    clock = _Clock()
    early, _cost = await _scene_meanwhile(before, pid, clock, LOOK_S, only_at_points, before.cost)
    if early is None:
        first = await look(pid, pts)
        out = compare(before, first, where, only_at_points)
        if out.changed is not False:
            return out
        early, _cost = await _scene_meanwhile(before, pid, clock, LOOK_S + LOOK_AGAIN_S, only_at_points, first.cost)
        if early is None:
            return compare(before, await look(pid, pts), where, only_at_points)
    if settle:
        await clock.sleep(LOOK_S)
    seen = compare(before, await look(pid, pts), where, only_at_points)
    return seen if seen.changed else early  # (a change that went back by now still happened)


async def scene_change(before: Look, pid: int) -> Outcome:
    """What a key or an AX press changed in the scene (the front app, the window, a sheet, the
    focus): looked at every POLL_S (spaced out when a look takes long), done at the first change,
    and by SCENE_WAIT_S after the input — by the clock: a look that couldn't end by then isn't
    started (the first always is)."""
    if not before.scene.known:
        return compare(before, None, [])
    out = compare(before, before, [])
    clock, cost, looked = _Clock(), 0.0, False
    while True:
        gap = _gap(POLL_S, cost) if looked else POLL_S
        if looked and clock.left(SCENE_WAIT_S) < gap + cost:
            break
        await clock.sleep(gap)
        now = await look(pid, [])
        cost, looked = now.cost, True
        out = compare(before, now, [])
        if out.changed:
            break
    return out


@dataclass
class Outcome:
    said: str  # NEO's own words: what changed, or that nothing visibly did
    details: str  # words read from the screen (titles, sheets, labels): fenced for the model
    changed: bool | None  # None: couldn't tell
    scene_changed: bool = False  # the front app, the window, a sheet or the focus changed
    sheet: str = ""  # a sheet or dialog open after the input (its words: screen data)


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
    scened = len(changes) > (1 if moved else 0)
    sheet = a.sheet if a.known else ""
    if changes:
        return Outcome("Changed: " + "; ".join(changes) + ".", "; ".join(details), True, scened, sheet)
    if before.pixels is None or now.pixels is None:
        return Outcome("I couldn't compare the screen (no Screen Recording), and nothing else changed.", "", None, False, sheet)
    return Outcome(
        f"Nothing visibly changed there within {LOOK_S + LOOK_AGAIN_S:g}s — it may have missed, been swallowed, "
        "or the app needs something else first. Look (screenshot) before trying again; don't assume it worked.",
        "",
        False,
        False,
        sheet,
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
        out.scene_changed,
        out.sheet,
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


# ---- what an input did: the Effect, and the after-look for a brain that can see -------------------

try:  # Step 0's seam (ToolOutput.effect): the same fields, whichever lands first
    from neo.agent.registry import Effect  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - until the seam lands

    @dataclass
    class Effect:  # type: ignore[no-redef]
        """What an input did, for the loop (a batch stops at the first input that changed nothing,
        a verify turn is skipped when the result already carried a look)."""

        changed: bool | None = None  # something visibly changed (None: couldn't tell)
        scene_changed: bool = False  # the front app, the window, a sheet or the focus changed
        sheet: str = ""  # a sheet or dialog open after it (its words: screen data)
        looked: bool = False  # the result carries an after-picture


def with_effect(out, eff: Effect):
    """``out`` (a ToolOutput) carrying ``eff`` — the ToolOutput.effect field once Step 0 adds it,
    the same attribute before (a plain dataclass takes it)."""
    try:
        out.effect = eff
    except Exception:  # noqa: BLE001 - (a frozen or slotted output: nothing to carry it)
        pass
    return out


def effect_of(out: Outcome | None, looked: bool = False) -> Effect:
    if out is None:
        return Effect(changed=None, looked=looked)
    return Effect(changed=out.changed, scene_changed=out.scene_changed, sheet=out.sheet, looked=looked)


def detail(img, level: str):
    """``img`` with ImagePart.detail set (Step 0's seam: "auto" by default) — unchanged before it lands."""
    import dataclasses

    try:
        if any(f.name == "detail" for f in dataclasses.fields(img)):
            return dataclasses.replace(img, detail=level)
    except TypeError:
        pass
    return img


_LOOK_BOX = (480.0, 320.0)  # the after-look around the point when no window is known (points)
_LOOK_MARGIN = 24.0  # every point acted on, with this much around it, is in the after-look
_MIN_WINDOW = 50.0


def _display_shot(s: Shot) -> bool:
    """Does screenshot ``s`` show a whole display (not a region or a window)?"""
    return any(
        abs(d.x - s.x0) < 1 and abs(d.y - s.y0) < 1 and abs(d.w - s.w * s.ppp) < 2 for d in screen.displays()
    )


@pooled
def _window_frame(h: ax.Hit) -> tuple[float, float, float, float] | None:
    if h.window is None:
        return None
    try:
        f = ax._point_size(h.window)
    except Exception:  # noqa: BLE001
        return None
    return f if f[2] >= _MIN_WINDOW and f[3] >= _MIN_WINDOW else None


def look_area(
    pts: list[tuple[float, float]], shot: Shot | None, window: tuple[float, float, float, float] | None
) -> tuple[int, int, int, int] | None:
    """Where the after-look is taken (screen points, on the display of the first point): the same
    area as the screenshot acted on when that was a region or a window (the model compares like with
    like), else the window acted in, else a box around the point — always with every point in it."""
    home = next((d for d in screen.displays() if d.contains(*pts[0])), None) if pts else None
    if home is None:
        return None
    if shot is not None and not _display_shot(shot):
        x, y, w, h = shot.x0, shot.y0, shot.w * shot.ppp, shot.h * shot.ppp
    elif window is not None:
        x, y, w, h = window
    else:
        x, y, w, h = pts[0][0] - _LOOK_BOX[0] / 2, pts[0][1] - _LOOK_BOX[1] / 2, _LOOK_BOX[0], _LOOK_BOX[1]
    left, top, right, bottom = x, y, x + w, y + h
    for px, py in pts:
        left, top = min(left, px - _LOOK_MARGIN), min(top, py - _LOOK_MARGIN)
        right, bottom = max(right, px + _LOOK_MARGIN), max(bottom, py + _LOOK_MARGIN)
    left, top = max(left, home.x), max(top, home.y)
    right, bottom = min(right, home.x + home.w), min(bottom, home.y + home.h)
    if right - left < 8 or bottom - top < 8:
        return None
    return int(left), int(top), int(round(right - left)), int(round(bottom - top))


def capture_area(area: tuple[int, int, int, int], kind: str = "after"):
    """(picture, x0, y0, points per pixel, width, height) of a screen area (on one display): the
    capture pipeline's own shot when it has one (neo.tools.computer.screen.shot: its format, size
    limit and detail per ``kind``), else the screenshot capture (detail set here)."""
    shot_fn = getattr(screen, "shot", None)
    if callable(shot_fn):
        c = shot_fn(area, kind=kind)
        return c.part, c.x0, c.y0, c.ppp, c.w, c.h
    import io

    from PIL import Image

    img, scale = screen.capture(area)
    w, h = Image.open(io.BytesIO(img.data)).size
    return detail(img, "medium" if kind == "after" else "high"), float(area[0]), float(area[1]), scale, w, h


@pooled
def protected_in(area: tuple[float, float, float, float]) -> bool:
    """Does any window on screen over ``area`` belong to an app NEO never touches (a protected app:
    a host's approval dialog, a vault)? Then no picture of it is taken on NEO's own initiative. When
    the window server can't say: yes (fail closed). NEO's own overlay never shows in a capture."""
    try:
        import Quartz as Q

        wins = Q.CGWindowListCopyWindowInfo(
            Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements, Q.kCGNullWindowID
        )
    except Exception:  # noqa: BLE001
        return True
    if wins is None:
        return True
    g = _gates()
    ax0, ay0, aw, ah = area
    for w in wins:
        b = w.get("kCGWindowBounds") or {}
        wx, wy, ww, wh = (float(b.get(k, 0) or 0) for k in ("X", "Y", "Width", "Height"))
        if ww <= 0 or wh <= 0 or float(w.get("kCGWindowAlpha", 1) or 0) <= 0.01:
            continue
        if wx >= ax0 + aw or wy >= ay0 + ah or wx + ww <= ax0 or wy + wh <= ay0:
            continue
        pid, name = int(w.get("kCGWindowOwnerPID", 0) or 0), str(w.get("kCGWindowOwnerName", "") or "")
        if pid and ax.is_neo(pid, name):
            continue
        if g._protected(name, pid):
            return True
    return False


def wants_after_look(ctx: ToolContext, where: list[Aim]) -> bool:
    """Will the result carry the picture after the input? For a brain that can see, the model's own
    input (not the user's command), aimed at a picture, a board's square or a mark — not an element
    id from ax_tree, whose text outcome is enough."""
    if ctx.get("blind") or ctx.origin != "agent":
        return False
    from neo.tools.computer import marks

    return any(w.pictured or w.numbers or (w.form == "id" and marks.is_mark(getattr(w.element, "id", ""))) for w in where)


async def after_look(ctx: ToolContext, where: list[Aim], pts: list[tuple[float, float]], hit: ax.Hit):
    """The picture after an input (see :func:`wants_after_look`): (image, its Shot, NEO's note) —
    registered as a new shot the next input may use — or None (no Screen Recording, a protected
    app's window over the area before or once it was taken). It replaces the separate "take a
    screenshot to check" turn."""
    if not pts or not wants_after_look(ctx, where):
        return None
    if not await asyncio.to_thread(screen.can_capture):
        return None  # (never asks for Screen Recording: that question is the screenshot tool's)
    shot = next((w.shot for w in where if w.shot is not None), None)
    window = await asyncio.to_thread(_window_frame, hit)
    area = look_area(pts, shot, window)
    if area is None or await asyncio.to_thread(protected_in, area):
        return None
    try:
        img, x0, y0, scale, w, h = await asyncio.to_thread(capture_area, area, "after")
    except Exception:  # noqa: BLE001 - no Screen Recording, a failed capture: the text outcome stands
        return None
    if await asyncio.to_thread(protected_in, area):
        return None  # (a protected app's window came up over it while it was taken: dropped unseen)
    s = remember(x0, y0, scale, w, h)
    said = (
        f"After-look {s.id} attached ({area[2]}×{area[3]} pt at screen ({area[0]},{area[1]})); the image is {w}×{h} px. "
        f'To act on it next, use shot="{s.id}" (x,y pixels · nx,ny · box_2d, as before) — no separate screenshot needed.'
    )
    return detail(img, "medium"), s, said


# ---- snapping a model's point onto the control under it ----------------------------------------------
# A point a model picked (pixels, nx,ny, box_2d) that lies inside a control's live frame — a button,
# a link, a tab — is moved to that control's middle: an edge of a button can miss where its frame
# doesn't, and the result names what was hit. Never onto a neighbour (only the control the point is
# already inside), never in a canvas or a board (a surface has no control under it; a grid NEO found
# says so), never for a brain that saw nothing or the user's own command, and the snapped spot is
# looked at, protected-checked and gated again before anything is posted.

_SNAP_ROLES = frozenset(
    {"AXButton", "AXLink", "AXCheckBox", "AXRadioButton", "AXPopUpButton", "AXMenuItem", "AXMenuBarItem", "AXTab", "AXDisclosureTriangle"}
)
_SNAP_UP = 3  # the control is the element under the point or one of its parents this far up
_STOP_ROLES = frozenset({"AXWindow", "AXSheet", "AXApplication", "AXScrollArea", "AXWebArea"})
# An input of its own, where the place of a click matters (a caret's, a thumb's, a cell's) or which
# holds controls of its own: a point in one is never moved out of it onto a control around it.
_OWN_ROLES = frozenset(
    {
        "AXTextField", "AXTextArea", "AXSearchField", "AXSecureTextField", "AXComboBox", "AXSlider",
        "AXIncrementor", "AXMenuButton", "AXColorWell", "AXCell", "AXRow", "AXList", "AXOutline",
        "AXTable", "AXTabGroup", "AXToolbar", "AXScrollBar", "AXSplitter", "AXValueIndicator",
    }
)
# What makes an element pressable by itself (an icon with its own click, a web span with a handler):
# a click aimed at it stays there. (Not AXShowMenu: nearly every element offers a context menu.)
_OWN_ACTIONS = frozenset({"AXPress", "AXPick", "AXOpen"})


@dataclass
class Control:
    """A control as Accessibility tells it: its role, label, live frame (its part on screen now)."""

    role: str
    label: str
    frame: tuple[float, float, float, float]
    ref: Any = None

    @property
    def middle(self) -> tuple[float, float]:
        x, y, w, h = self.frame
        return round((x + w / 2) * 2) / 2, round((y + h / 2) * 2) / 2


def _actions(el) -> list[str] | None:
    """The actions an element offers by itself; None when its app can't say (then nothing snaps)."""
    try:
        err, names = ax.AS.AXUIElementCopyActionNames(el, None)
    except Exception:  # noqa: BLE001
        return None
    if err == ax._CANNOT_COMPLETE:
        return None
    return [str(n) for n in (names or [])] if err == 0 else []


@pooled
def control_of(h: ax.Hit) -> Control | None:
    """The control the element under a point is, or is inside (a button's own text): None for a
    surface (a canvas, a board, a picture), a field or anything else a click position matters in —
    and None when the element aimed at, or one passed on the way up, is pressable by itself (an
    icon inside a link, a web span with its own click): the click stays where the model aimed."""
    el = h.el
    for _ in range(_SNAP_UP):
        if el is None:
            return None
        role = ax._str(ax._attr(el, "AXRole"))
        if role in _STOP_ROLES or role in _OWN_ROLES:
            return None
        if role in _SNAP_ROLES:
            frame = ax.visible_frame(el)
            if frame is None:
                return None
            label = ax._str(ax._attr(el, "AXTitle")) or ax._str(ax._attr(el, "AXDescription"))
            return Control(role, " ".join(label.split())[:60], frame, el)
        acts = _actions(el)
        if acts is None or _OWN_ACTIONS.intersection(acts):
            return None
        el = ax._attr(el, "AXParent")
    return None


def _inside(p: tuple[float, float], frame: tuple[float, float, float, float]) -> bool:
    x, y, w, h = frame
    return x <= p[0] <= x + w and y <= p[1] <= y + h


def may_snap(where: Aim, ctx: ToolContext) -> bool:
    """A model's own numbers (never a brain's that saw nothing, never the user's own command) — and
    the same again when the user's yes re-runs the model's click: the approved click is the one the
    gate judged, at the control's middle."""
    return where.numbers and ctx.origin in ("agent", "approval") and not ctx.get("blind")


@dataclass
class Snap:
    point: tuple[float, float]
    control: Control
    raw: tuple[float, float]

    def words(self) -> str:
        """What it snapped to, as read from the screen (fenced by the caller)."""
        role = self.control.role.removeprefix("AX")
        return f"snapped to {role} “{self.control.label}”" if self.control.label else f"snapped to a {role}"


async def snap(p: tuple[float, float], h: ax.Hit) -> Snap | None:
    """The control ``p`` is inside, and its middle — or None: no control there, the point isn't in
    its live frame, a grid NEO found is there, or its middle shows something else now."""
    if _grid_check is not None and _grid_check(*p):
        return None
    c = await asyncio.to_thread(control_of, h)
    if c is None or not _inside(p, c.frame):
        return None
    m = c.middle
    if abs(m[0] - p[0]) < 1 and abs(m[1] - p[1]) < 1:
        return None  # (already there)
    if _grid_check is not None and _grid_check(*m):
        return None
    again = await asyncio.to_thread(ax.hit, *m)
    there = await asyncio.to_thread(control_of, again)
    if there is None or not ax._same(there.ref, c.ref) or again.pid != h.pid:
        return None  # its middle is another element (a wrapped link's gap, something over it)
    return Snap(m, c, p)


# ---- the pointer tools, whole (the handlers call these; see the wiring notes) -----------------------


def _at(x: float, y: float) -> str:
    return f"({int(round(x))}, {int(round(y))})"


def _spot(where: Aim | None, p: tuple[float, float], blind: bool) -> str:
    """How a result names where an input went: its point — or, to a brain that can't see, only the
    square, mark or element it named (no coordinates reach such a brain in any form)."""
    if not blind:
        return _at(*p)
    return (where.said if where is not None and where.said else "") or "the point the user gave"


def where_needed(tool: str) -> str:
    ends = "x1,y1 and x2,y2 (or id1/id2, nx1,ny1…, box_2d1/box_2d2)" if tool == "drag" else "x,y (or id, nx,ny, box_2d)"
    return f"Error: say where to {tool}: {ends} — pixels and 0–1000 forms with a screenshot's shot id."


async def click_tool(a: dict, c: ToolContext):
    """click: an element id, a board square, or a point in any numeric form — one checked function
    for all of them — snapped onto the control it is inside (and gated again), its app brought
    forward, the outcome and, for a brain that can see, the picture after it."""
    from neo.agent.registry import ToolOutput
    from neo.tools.computer import input as inp

    g = _gates()
    if err := g._needs_input():
        return err
    button, count, mods = str(a.get("button") or "left"), a.get("count", 1), a.get("modifiers") or []
    try:
        inp.mods_flags(mods)
        if button not in ("left", "right", "middle"):
            raise ValueError(f"button must be left, right or middle, not {button!r}")
        count = inp.click_count(1 if count in (None, "") else count)
    except ValueError as e:
        return f"Error: {e}"
    snapped: Snap | None = None
    blind_ctx = bool(c.get("blind"))
    try:
        where = await aim(a, c, "click")
        if where is None:
            raise Refused(where_needed("click"))
        pts = [where.point]
        t = await probe(pts)
        if where.form == "id":
            e = where.element
            labels = [(e.title or e.value, where.owner), (t.hits[0].label, t.hits[0].app or where.owner)]
            head = f"{click_word(button, count)} {e.id} ({e.role.removeprefix('AX')})"
            if not blind_ctx:
                head += f" at {_at(*pts[0])}"
        else:
            # A helper's mark's own words (a text mark's text) are judged with what's under the point.
            labels = [(w, t.hits[0].app) for w in where.labels] + [(t.hits[0].label, t.hits[0].app)]
            if blind_ctx:
                head = f"{click_word(button, count)} {_spot(where, pts[0], True)}"
            else:
                head = f"{click_word(button, count)} at {_at(*pts[0])}"
                if where.said:
                    head += f" ({where.said})"
        if may_snap(where, c) and (snapped := await snap(pts[0], t.hits[0])) is not None:
            # The control's middle: looked at again (NEO's window, a protected app, a sheet in the
            # way), and everything there judged by the gate like the spot the model picked.
            pts = [snapped.point]
            t = await probe(pts)
            labels += [(snapped.control.label, t.hits[0].app), (t.hits[0].label, t.hits[0].app)]
            head = f"{click_word(button, count)} at {_at(*pts[0])} ({where.said + '; ' if where.said else ''}moved from {_at(*snapped.raw)} to the middle of the control there)"
        if gate := await g._gate_click(a, labels):
            return gate
        await bring(t, pts, want_pid=where.pid, blind=blind_ctx)
        # A point nothing could be told about before now shows something: judged like the rest.
        if t.relabeled and (gate := await g._gate_click(a, [(t.hits[0].label, t.hits[0].app)])):
            return gate
    except Refused as r:
        return g._refused(c, r)
    h = t.hits[0]
    pid = where.pid or h.pid
    if h.app:
        head += f" in {h.app}"
    if t.brought:
        head += f" (brought {t.brought} to the front first)"
    head += "."
    before = await g._watch(c, pid, pts, hover=pts[0])
    await asyncio.to_thread(inp.click, *pts[0], button=button, count=count, mods=mods)
    if before is None:
        return with_effect(ToolOutput(head + g._input_doubt()), effect_of(None))
    spot = _spot(where, pts[0], blind_ctx)
    out = await outcome(before, pid, pts, [spot], settle=wants_after_look(c, [where]))
    if not at_the_point(h, where.pictured):
        out = maybe_elsewhere(out, spot)
    under = [words(h)] if where.form != "id" else []
    if snapped is not None:
        under.append(snapped.words())
    text, ok = report(head, out, under, g._input_doubt())
    if any(under) or out.details:
        g._taint(c, "screen")
    return await _finish(c, ToolOutput(text, ok=ok), out, [where], pts, h)


async def drag_tool(a: dict, c: ToolContext):
    """drag: each end an element id, a board square, or a point in any numeric form (the same
    checked function) — never snapped; the drop is gated where it lands."""
    from neo.agent.registry import ToolOutput
    from neo.tools.computer import input as inp

    g = _gates()
    if err := g._needs_input():
        return err
    blind_ctx = bool(c.get("blind"))
    try:
        ends: list[Aim] = []
        for n in ("1", "2"):
            got = await aim(a, c, "drag", n)
            if got is None:
                raise Refused(where_needed("drag"))
            ends.append(got)
        pts = [e.point for e in ends]
        t = await probe(pts)
        if gate := await g._gate_drop(a, t):
            return gate
        # A helper's mark's own words at either end are judged too (dragging "Delete" onto something).
        marked = [(w, h.app) for e, h in zip(ends, t.hits, strict=True) for w in e.labels]
        if marked and (gate := await g._gate_click(a, marked, "drag")):
            return gate
        src, dst = t.hits
        between_apps = bool(src.pid and dst.pid and src.pid != dst.pid)
        if not between_apps:
            await bring(t, pts, blind=blind_ctx)
            if t.relabeled and (gate := await g._gate_drop(a, t)):
                return gate
    except Refused as r:
        return g._refused(c, r)
    src, dst = t.hits
    pid = src.pid
    before = await g._watch(c, pid, pts, hover=pts[1])  # the drop leaves the pointer at the end
    await asyncio.to_thread(inp.drag, *pts[0], *pts[1])
    starts, stops = _spot(ends[0], pts[0], blind_ctx), _spot(ends[1], pts[1], blind_ctx)
    head = f"Dragged {starts} → {stops}"
    said = [e.said for e in ends if e.said]
    if said and not blind_ctx:  # (a brain that can't see was told the names instead of the points)
        head += f" ({' → '.join(said)})"
    if src.app:
        head += f" in {src.app}" + (f" onto {dst.app}" if dst.app and dst.pid != src.pid else "")
    if t.brought:
        head += f" (brought {t.brought} to the front first)"
    head += "."
    if before is None:
        return with_effect(ToolOutput(head + g._input_doubt()), effect_of(None))
    out = await outcome(
        before, pid, pts, [f"the start {starts}", f"the end {stops}"], between_apps, settle=wants_after_look(c, ends)
    )
    if between_apps and out.changed is False:
        out.said += (
            f" (Neither app was brought forward for a drag between two apps; if {src.app or 'the app it starts in'} "
            "only came forward, bring it to the front yourself, look again, and drag again.)"
        )
    under = [f"start {words(src)}" if words(src) else "", f"end {words(dst)}" if words(dst) else ""]
    text, ok = report(head, out, under, g._input_doubt())
    if any(under) or out.details:
        g._taint(c, "screen")
    return await _finish(c, ToolOutput(text, ok=ok), out, ends, pts, src)


async def scroll_tool(a: dict, c: ToolContext):
    """scroll: at an element id, a board square, a point in any numeric form (the same checked
    function), or wherever the pointer is (never NEO's own window: the app behind it)."""
    from neo.agent.registry import ToolOutput
    from neo.tools.computer import input as inp

    g = _gates()
    if err := g._needs_input():
        return err
    try:
        dy = int(float(a.get("dy", -5) if a.get("dy") not in (None, "") else -5))
        dx = int(float(a.get("dx", 0) or 0))
    except (TypeError, ValueError):
        return "Error: dy and dx must be whole numbers (lines)."
    where: list[Aim] = []
    try:
        got = None if a.get("at_pointer") else await aim(a, c, "scroll")
        if got is not None:
            where = [got]
            pts = [got.point]
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
        t = await probe(pts)
    except Refused as r:
        return g._refused(c, r)
    pid = t.hits[0].pid
    blind_ctx = bool(c.get("blind"))
    before = await g._watch(c, pid, pts, hover=pts[0])
    text = str(await asyncio.to_thread(inp.scroll, pts[0][0], pts[0][1], dy, dx))
    spot = _spot(where[0] if where else None, pts[0], blind_ctx)
    if blind_ctx:  # (its words name the point: a brain that can't see gets the square, mark or id)
        text = re.sub(r"\s+at\s*\(\s*-?[\d.]+\s*,\s*-?[\d.]+\s*\)", "", text)
        if where and where[0].said:
            text += f" at {where[0].said}"
    if before is None:
        return with_effect(ToolOutput(text + g._input_doubt()), effect_of(None))
    out = await outcome(before, pid, pts, [spot if where or not blind_ctx else "the pointer"], settle=wants_after_look(c, where))
    text, ok = report(text + (f" in {t.hits[0].app}." if t.hits[0].app else "."), out, [], g._input_doubt())
    return await _finish(c, ToolOutput(text, ok=ok), out, where, pts, t.hits[0])


async def _finish(c: ToolContext, res, out: Outcome, where: list[Aim], pts: list[tuple[float, float]], hit: ax.Hit):
    """The result with its Effect, and the after-look when the brain can see and aimed at a picture."""
    looked = await after_look(c, where, pts, hit)
    if looked is not None:
        img, _s, said = looked
        res.images = [*res.images, img]
        res.text += "\n" + said
        _gates()._taint(c, "screen")  # (a picture of the screen went into the result)
    return with_effect(res, effect_of(out, looked=looked is not None))
