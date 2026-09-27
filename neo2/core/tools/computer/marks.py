"""Numbered element marks: the controls Accessibility sees in a screenshot's area, each with the
id a model clicks by — so a model picks an element from a list with exact boxes instead of reading
pixel positions off a picture.

For a model that can see, each line also says where the element is in the image, in Gemini's own
box order (``e31 Button “Start” [sheet] box_2d 88,370,150,470``: [ymin, xmin, ymax, xmax] on 0–1000),
which is how it matches an id to what it sees. For a model that can't, ids only: numbers it never
saw would only be copied. The labels are the screen's words (fenced as data); NEO's own words about
the list stay outside the fence.

What is listed: the visible interactive elements of the app's focused window — an open sheet or
dialog's first (they block the rest) — capped (about 470 tokens per 40) and ranked: what can be
pressed or typed into before what only shows, enabled before disabled among the same kind, then in
reading order. Never NEO's own window's, never a protected app's. The walk issues real ids (``click
id=…`` resolves them), stops at a deadline, and starts nothing in library mode (no web-tree switch,
no timers).
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field

from neo2.core.agent.registry import ToolContext, fence
from neo2.core.decide._objc import pooled
from neo2.core.tools.computer import ax
from neo2.core.tools.computer.pointer import NORM, Shot

MAX_MARKS = 40
DRAW_MAX = 30  # boxes with id tags drawn on a copy of the picture: only this few (off by default)
_MIN_PT = 4.0  # less of an element than this on screen (points, either side) isn't a target
_LABEL = 50

# What a model acts on, best first (0) — an image or a cell only when it is what there is.
_RANK = {
    "AXButton": 0,
    "AXTextField": 0,
    "AXSearchField": 0,
    "AXSecureTextField": 0,
    "AXComboBox": 0,
    "AXCheckBox": 0,
    "AXRadioButton": 0,
    "AXPopUpButton": 0,
    "AXMenuButton": 0,
    "AXTab": 0,
    "AXMenuItem": 1,
    "AXLink": 1,
    "AXTextArea": 1,
    "AXSlider": 1,
    "AXDisclosureTriangle": 1,
    "AXIncrementor": 2,
    "AXCell": 2,
    "AXRow": 2,
    "AXImage": 3,  # only one that can be pressed
    "AXMenuBarItem": 3,
}
_PRESSABLE = ("AXPress", "AXPick", "AXOpen", "AXShowMenu")


@dataclass(frozen=True)
class Mark:
    id: str  # the ax id a click takes (issued by this walk)
    role: str
    label: str  # the screen's words (fenced wherever they go)
    frame: tuple[float, float, float, float]  # screen points: its part in the area
    where: str = ""  # "sheet" / "dialog": in the window that blocks the rest
    enabled: bool = True


@dataclass
class Collected:
    """One walk of an app's focused window, for a picture of ``area`` (screen points)."""

    app: str = ""
    pid: int = 0
    title: str = ""
    window: tuple[float, float, float, float] | None = None
    blocker: str = ""  # "sheet" / "dialog" when one is open
    marks: list[Mark] = field(default_factory=list)
    more: int = 0  # elements left off the list
    refused: str = ""  # why nothing is listed: "protected", "Accessibility is off", "slow" (not in time)
    # Every walked element's frame and role (screen points: x, y, w, h, role): where a window is drawn
    # in one piece with no accessible parts (the board helper's canvas evidence — never a second walk).
    frames: list[tuple[float, float, float, float, str]] = field(default_factory=list)


# Ids issued for marks lately (an input "by mark" gets the picture after it).
_marked: dict[str, None] = {}
_MARKED_MAX = 2000


def is_mark(eid: str) -> bool:
    return bool(eid) and eid in _marked


def _remember(ids: list[str]) -> None:
    for i in ids:
        _marked[i] = None
    while len(_marked) > _MARKED_MAX:
        _marked.pop(next(iter(_marked)))


# ---- choosing and writing marks (pure: tested and benchmarked without a Mac) -------------------------


def _clip(f, box) -> tuple[float, float, float, float] | None:
    if box is None:
        return f
    left, top = max(f[0], box[0]), max(f[1], box[1])
    right, bottom = min(f[0] + f[2], box[0] + box[2]), min(f[1] + f[3], box[1] + box[3])
    if right - left < _MIN_PT or bottom - top < _MIN_PT:
        return None
    return (left, top, right - left, bottom - top)


def _label(e: ax.Element) -> str:
    text = e.title or ("" if e.role == "AXSecureTextField" else e.value)
    text = " ".join(str(text).replace("“", '"').replace("”", '"').split())
    return text[:_LABEL]


def pick(
    elements: list[ax.Element],
    area: tuple[float, float, float, float] | None,
    *,
    window: tuple[float, float, float, float] | None = None,
    blocked: dict[str, str] | None = None,
    limit: int = MAX_MARKS,
) -> tuple[list[Mark], int]:
    """(the marks to list, how many more there were): the interactive elements with a part in
    ``area`` (and the window), ranked, capped at ``limit``, in the order shown — the blocking
    sheet's first, then reading order. ``blocked``: element id → "sheet" / "dialog"."""
    blocked = blocked or {}
    seen: set[tuple] = set()
    found: list[tuple[tuple, Mark]] = []
    for i, e in enumerate(elements):
        rank = _RANK.get(e.role)
        if rank is None:
            continue
        if e.role in ("AXImage", "AXCell", "AXRow") and not any(a in e.actions for a in _PRESSABLE) and not e.title:
            continue  # a picture or a row nothing can be done to by name
        label = _label(e)
        if not label and e.role not in ("AXTextField", "AXTextArea", "AXSearchField", "AXComboBox", "AXSecureTextField"):
            continue  # an unnamed button: nothing to match it by (the picture still shows it)
        f = _clip((e.x, e.y, e.w, e.h), window)
        f = _clip(f, area) if f is not None else None
        if f is None or f[2] < _MIN_PT or f[3] < _MIN_PT:
            continue
        key = (e.role, label, round(f[0]), round(f[1]), round(f[2]), round(f[3]))
        if key in seen:
            continue
        seen.add(key)
        where = blocked.get(e.id, "")
        found.append(((0 if where else 1, rank, 0 if e.enabled else 1, i), Mark(e.id, e.role, label, f, where, e.enabled)))
    found.sort(key=lambda t: t[0])
    chosen = [m for _, m in found[:limit]]
    chosen.sort(key=lambda m: (0 if m.where else 1, round(m.frame[1] / 8), m.frame[0]))
    return chosen, max(0, len(found) - len(chosen))


def box_2d(m: Mark, s: Shot) -> tuple[int, int, int, int]:
    """Where a mark is in screenshot ``s``: [ymin, xmin, ymax, xmax] on 0–1000 of the image."""
    wpt, hpt = s.w * s.ppp, s.h * s.ppp
    x, y, w, h = m.frame

    def n(v: float, size: float) -> int:
        return int(round(min(NORM, max(0.0, v / size * NORM)))) if size > 0 else 0

    return n(y - s.y0, hpt), n(x - s.x0, wpt), n(y + h - s.y0, hpt), n(x + w - s.x0, wpt)


def line(m: Mark, s: Shot | None, *, blind: bool) -> str:
    role = m.role.removeprefix("AX")
    bits = [m.id, role]
    if m.label:
        bits.append(f"“{m.label}”")
    if m.where:
        bits.append(f"[{m.where}]")
    if not m.enabled:
        bits.append("(disabled)")
    if s is not None and not blind:  # never numbers for a brain that can't see: it would copy them
        bits.append("box_2d " + ",".join(str(v) for v in box_2d(m, s)))
    return " ".join(bits)


def block(c: Collected, s: Shot | None, *, blind: bool) -> str:
    """The marks as a result's text: NEO's words (unfenced) around the screen's (fenced). "" when
    there is nothing to list."""
    if c.refused or not c.marks:
        return ""
    if blind:
        head = (
            "Controls NEO found by Accessibility (you can't see the picture, but these ids work: click id=…, "
            "ax_press):"
        )
    elif s is None:
        head = "Controls NEO found by Accessibility (click id=…, ax_press):"
    else:
        head = (
            "Elements in this picture, from Accessibility — exact, so prefer click id=… over pixels when the target "
            "is one of them. box_2d is where each is in the image: [ymin, xmin, ymax, xmax] on 0–1000."
        )
    if c.blocker:
        head += f" A {c.blocker} is open: it blocks the rest of the window until one of its buttons is pressed."
    tail = f"\n({c.more} more not listed — ax_tree lists them all.)" if c.more else ""
    return "\n" + head + "\n" + fence("screen", "\n".join(line(m, s, blind=blind) for m in c.marks)) + tail


def draw(img, s: Shot, marks: list[Mark]):
    """A copy of the picture with each mark's box and id drawn on it (at most DRAW_MAX of them; off
    by default — the text list is the default, bench A/B-tests this). The original is never changed."""
    from PIL import Image, ImageDraw

    if not marks or len(marks) > DRAW_MAX:
        return img
    pic = Image.open(io.BytesIO(img.data)).convert("RGB")
    d = ImageDraw.Draw(pic)
    k = 1.0 / s.ppp
    for m in marks:
        x, y, w, h = m.frame
        box = ((x - s.x0) * k, (y - s.y0) * k, (x + w - s.x0) * k, (y + h - s.y0) * k)
        d.rectangle(box, outline=(255, 0, 170), width=2)
        d.rectangle((box[0], box[1], box[0] + 7 * len(m.id) + 4, box[1] + 12), fill=(255, 0, 170))
        d.text((box[0] + 2, box[1]), m.id, fill=(255, 255, 255))
    buf = io.BytesIO()
    fmt = "JPEG" if img.mime == "image/jpeg" else "PNG"
    pic.save(buf, format=fmt, **({"quality": 85} if fmt == "JPEG" else {"compress_level": 1}))
    import dataclasses

    return dataclasses.replace(img, data=buf.getvalue())


# ---- the walk (Accessibility: a thread, a deadline) -----------------------------------------------------


class _Budget(list):
    """ax._walk's node budget (``[n]``), which also runs out at a deadline: a busy app's window
    is listed as far as it got, never waited on."""

    def __init__(self, n: int, deadline: float) -> None:
        super().__init__([n])
        self.deadline = deadline

    def __getitem__(self, i):  # type: ignore[override]
        if time.monotonic() > self.deadline:
            return -1
        return super().__getitem__(i)


def _register(els: list[ax.Element]) -> None:
    """Make the walk's ids clickable (ax.get resolves them), as ax_tree's are."""
    with ax._lock:
        for e in els:
            ax._issued[e.id] = e
        while len(ax._issued) > ax._ISSUED_MAX:
            ax._issued.pop(next(iter(ax._issued)))


class _Late(Exception):
    """The walk's deadline passed before it got to the window's elements."""


def _answers(root) -> bool:
    """Does the app answer Accessibility at all? A busy or hung app says kAXErrorCannotComplete to
    its first question — then nothing more is asked of it (each would wait out the timeout)."""
    try:
        err, _role = ax.AS.AXUIElementCopyAttributeValue(root, "AXRole", None)
    except Exception:  # noqa: BLE001 - can't tell: go on, the deadline still holds
        return True
    return err != ax._CANNOT_COMPLETE


def walk_window(pid: int, deadline: float) -> tuple[str, tuple | None, list[ax.Element], dict[str, str]]:
    """(window title, window frame, elements, blocked ids) of an app's focused window: an open
    sheet walked first (its elements are "sheet"), a dialog window's all "dialog". The deadline is
    looked at before every question about the window, and per element in the walk: a hung app
    costs at most the one answer being waited on (ax-core's walk, once merged, also stops at an app
    that stopped answering)."""
    from neo2.core import runtime

    title, frame = "", None

    def on_time() -> None:
        if time.monotonic() > deadline:
            raise _Late

    root = ax._app_el(pid)
    try:
        on_time()
        if not _answers(root):
            return "", None, [], {}
        on_time()
        win = ax._attr(root, "AXFocusedWindow")
        if win is None:
            on_time()
            win = ax._attr(root, "AXMainWindow")
        if win is None:
            return "", None, [], {}
        on_time()
        if not runtime.library_bound() and ax._expose_web(int(pid), root):
            # (a Chromium app's web content was just switched on for Accessibility: a moment to fill in)
            time.sleep(min(ax._WEB_SETTLE_S, max(0.0, deadline - time.monotonic() - 0.2)))
        on_time()
        f = ax._point_size(win)
        frame = f if f[2] > 0 and f[3] > 0 else None
        on_time()
        title = ax._str(ax._attr(win, "AXTitle"))
        on_time()
        sheet = ax._sheet_of(win)
        dialog = False
        if sheet is None:
            on_time()
            dialog = ax._str(ax._attr(win, "AXSubrole")) in ax._DIALOG_SUBROLES
            if not dialog:
                on_time()
                dialog = ax._attr(win, "AXModal") is True
        on_time()
    except _Late:
        return title, frame, [], {}
    budget = _Budget(ax._MAX_NODES, deadline)
    walk_kw = _walk_limits(deadline)
    blocked: dict[str, str] = {}
    first: list[ax.Element] = []
    if sheet is not None:
        ax._walk(sheet, 1, first, ax._new_id, budget=budget, **walk_kw)
        blocked = {e.id: "sheet" for e in first}
    rest: list[ax.Element] = []
    ax._walk(win, 1, rest, ax._new_id, budget=budget, skip=sheet, **walk_kw)
    els = first + rest
    if dialog:
        blocked = {e.id: "dialog" for e in els}
    _register(els)
    return title, frame, els, blocked


def _walk_limits(deadline: float) -> dict:
    """ax._walk's own limits when it has them (ax-core's ``_Walk``: a deadline, and an end at an app
    that stopped answering), else none beyond the budget's deadline."""
    make = getattr(ax, "_Walk", None)
    if make is None:
        return {}
    try:
        return {"w": make(deadline_s=max(0.01, deadline - time.monotonic()))}
    except Exception:  # noqa: BLE001 - (another shape of it: the budget's deadline still holds)
        return {}


def _target(area: tuple[float, float, float, float] | None, pid: int) -> tuple[str, int]:
    """Whose window to list: ``pid``, else the app owning the middle of a region, else the app
    being worked in (never NEO's own window: the one behind it)."""
    if pid:
        app = ax._running(pid)
        return (str(app.localizedName() or "") if app is not None else ""), pid
    if area is not None and not _whole_display(area):
        got = ax.window_owner_at(area[0] + area[2] / 2, area[1] + area[3] / 2)
        if got and got[1] and not ax.is_neo(got[1], got[0]):
            return got
    return ax.working_app()


def _whole_display(area: tuple[float, float, float, float]) -> bool:
    from neo2.core.tools.computer import screen

    return any(abs(d.x - area[0]) < 1 and abs(d.y - area[1]) < 1 and abs(d.w - area[2]) < 2 for d in screen.displays())


def _off_limits(name: str, pid: int) -> bool:
    from neo2.core.tools import computer as _c  # (the gate helpers, looked up at call time)

    return not pid or ax.is_neo(pid, name) or _c._protected(name, pid)


@pooled  # (a thread's Objective-C objects are released when it is done)
def collect_sync(
    area: tuple[float, float, float, float] | None, *, pid: int = 0, name: str = "", deadline_s: float = 0.8, limit: int = MAX_MARKS
) -> Collected:
    """The marks for a picture of ``area`` (None: the whole window). Never raises: a look that
    can't list elements lists none."""
    deadline = time.monotonic() + deadline_s
    try:
        if not ax.is_trusted():
            return Collected(refused="Accessibility is off")
        if not pid:
            name, pid = _target(area, 0)
        elif not name:
            name = _target(None, pid)[0]
        if _off_limits(name, pid):  # NEO's own window, a protected app: never read, never listed
            return Collected(app=name, pid=pid, refused="protected")
        title, window, els, blocked = walk_window(pid, deadline)
        if not els and time.monotonic() > deadline:
            return Collected(name, pid, title, window, refused="slow")  # (it didn't get to the elements in time)
        chosen, more = pick(els, area, window=window, blocked=blocked, limit=limit)
        _remember([m.id for m in chosen])  # (an input by one of these gets the picture after it)
        kind = next(iter(set(blocked.values())), "")
        frames = [(e.x, e.y, e.w, e.h, e.role) for e in els]
        return Collected(name, pid, title, window, kind, chosen, more, frames=frames)
    except Exception:  # noqa: BLE001 - marks are a help, never a failure of the look
        return Collected()


GRACE_S = 0.1  # past its deadline, a walk's result is waited for this much longer at most


async def collect(
    area: tuple[float, float, float, float] | None, ctx: ToolContext | None = None, *, pid: int = 0, deadline_s: float = 0.8
) -> Collected:
    """:func:`collect_sync` off the event loop — start it alongside the capture: it costs no wait,
    and never holds the picture past ``deadline_s`` (+ GRACE_S): a walk not done by then (an app
    stuck on one answer) is left to finish on its own, and the picture goes without marks
    (``refused="slow"``)."""
    import asyncio

    task = asyncio.ensure_future(asyncio.to_thread(collect_sync, area, pid=pid, deadline_s=deadline_s))
    task.add_done_callback(_quiet)
    done, _ = await asyncio.wait({task}, timeout=deadline_s + GRACE_S)  # (cancelled with the picture: the walk ends on its own)
    if task in done and not task.cancelled() and task.exception() is None:
        return task.result()
    return Collected(pid=pid, refused="slow")


def _quiet(task) -> None:
    """(A walk left to finish on its own: its end, or its error, is nobody's to hear.)"""
    if not task.cancelled():
        task.exception()


def area_of(s: Shot) -> tuple[float, float, float, float]:
    return (s.x0, s.y0, s.w * s.ppp, s.h * s.ppp)


def for_shot(c: Collected, s: Shot, ctx: ToolContext) -> str:
    """The text to append to a screenshot's note: the marks in its area, ids only for a brain that
    can't see. Taints the request (the screen's words went into the result)."""
    text = block(c, s, blind=bool(ctx.get("blind")))
    if text:
        from neo2.core.tools import computer as _c

        _c._taint(ctx, "screen")
    return text
