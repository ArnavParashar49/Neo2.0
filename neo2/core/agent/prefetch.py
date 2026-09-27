"""NEO's first look at the screen, taken while the request is still being routed, and the
resolver's Laya warmed while the user is still talking.

**The early first look** (decision 41). On nearly every screen task the model's first step is a
look (ax_tree, then often a screenshot): one whole model round trip, 2-5 s on a healthy Gemini and
far more on a throttled brain, spent before anything happens. So when the evidence the request
already carries says the task works on the screen, NEO reads the working window itself, in the
time it spends on playbook recall and tool selection anyway, and hands the result to the first
model call. It is read-only, and it only happens where the model would look anyway:

* the evidence: Laya's needs_screen, a screen tool picked by the reflex, or a game to play — and
  the run must then be given the computer tools (else nothing leaves this Mac);
* the app: the one in front, and only when it is the request's app — named in the request (by
  the installed apps' own names: evidence, no list of apps here), the game being played, or, when
  the request names no app, the one NEO last worked in. The front app is read only for a request
  that names one, plays a game, or finds NEO has worked in an app lately. Never NEO's own window,
  never a protected app (the gate helpers decide, fail closed), never without Accessibility;
* what: the window's elements as a compact list — ids, roles, labels, no coordinates, so a brain
  that can't see is never handed any — and a picture of that window only when Laya says the screen
  itself is needed, the brain can see, and Screen Recording is already allowed (it is never asked
  for from here). It is the window's own pixels, as the screenshot tool pictures that window —
  never another app's, not even where another window is in front of part of it (a click at such a
  part is refused: it would land on that window) — and none when the window server can't say where
  the window is or what is in front of it;
* how long: a hard deadline (``prefetch_look_s``); a look that isn't ready by then is dropped —
  except that a Chromium or Electron app's first look in a while also waits out the moment its web
  tree needs once switched on (a one-time cost, not a slow app).

What is attached is fenced as untrusted, and taints the request exactly as the tools would —
only when it is attached. It goes stale as soon as the model looks again or an action changes the
screen (:func:`supersede`), and a finished run keeps only a stub of it (:func:`forget`). The
``prefetch_look`` setting turns all of it off.

**Prewarm.** The resolver's zero-shot Laya checkpoint (neo2.core.reflex.zeroshot, typed-decisions) is
let go after ten idle minutes, and a question that finds it unloaded pays a 0.4-0.7 s load. The
early actor (neo2.core.agent.early) calls :func:`prewarm` when the user starts talking to NEO (wake word,
push-to-talk, the first words of an utterance), so the load happens while they speak and the
resolver answers warm — only once the stage that reads it (meaning) is promoted and live; in shadow
it reads Laya only when something else loaded it. Other warmers (text recognition) run either way.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import re
import threading
import time
import weakref
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from neo2.core import logger, runtime
from neo2.core.agent.registry import ToolContext, fence, registry
from neo2.core.config import settings
from neo2.core.providers.base import ImagePart, Message, ToolSpec

log = logger(__name__)

LOOK_S = 0.5  # the whole first look (elements and picture, side by side) is ready by then, or dropped
MAX_LINES = 150  # elements listed (the model's own ax_tree call can always show more)
NEEDS_SCREEN_P = 0.5  # Laya's needs_screen at or above this: the screen itself matters (a picture)
MARK = "[NEO's first look"  # opens the attached block (and nothing a user types)
STUB = "[NEO looked at the screen here, before the first step.]"  # what a finished run keeps of it
_SCREEN_CATEGORY = "computer"  # the tool family that reads and works the screen
_OBSERVE = frozenset({"ax_tree", "ax_find", "screenshot"})  # a newer one of these makes the first look stale
_CHANGED = "Changed: "  # NEO's own words, in an action's result, for what it changed on the screen
_SHOT = re.compile(r'shot="([^"]+)"')
SLOW_SKIP_S = 300.0  # an app whose look missed the deadline isn't waited for again for this long
_slow_until: dict[int, float] = {}  # pid -> time.monotonic() until which it's skipped


# ---- settings -----------------------------------------------------------------------------------


def _setting(name: str, app_default: Any, library_default: Any) -> Any:
    """A setting of neo2.core.config (``prefetch_look``, ``prefetch_look_s``, ``prewarm_resolver``;
    NEO_<NAME> in NEO.app's environment and .env): an embedded agent gets its host's, or the library
    default when the host's settings don't say."""
    v = getattr(settings(), name, None)
    if v is not None:
        return v
    return library_default if runtime.library_bound() else app_default


def enabled() -> bool:
    """The early first look is on (NEO_PREFETCH_LOOK=0 turns it off; an embedded agent's host opts in)."""
    return bool(_setting("prefetch_look", True, False))


def _deadline_s() -> float:
    return max(0.05, float(_setting("prefetch_look_s", LOOK_S, LOOK_S)))


# ---- when a look is worth taking ------------------------------------------------------------------


@dataclass(frozen=True)
class Want:
    """The evidence for a first look. ``image``: a picture too (the screen itself is needed)."""

    image: bool
    reason: str


def _screen_tool(name: str) -> bool:
    t = registry().get(name) if name else None
    return t is not None and not t.hidden and t.category == _SCREEN_CATEGORY


def wanted(
    d: Any, *, only: Sequence[str] | None = None, game: Any = None, images: Sequence[ImagePart] | None = None
) -> Want | None:
    """Whether the request's own evidence says its run will look at the screen first.

    Laya's needs_screen, a screen tool the reflex picked, or a game to play (its run looks at the
    board). Not the one-tool lane (`only`): that tool finds its target itself, and a list of
    elements would only slow its single call down. A picture only when the screen itself is needed
    and the caller didn't bring one (`images`: what the voice layer already saw)."""
    if only or d is None:
        return None
    screen_p = float(getattr(d, "needs_screen_p", 0.0) or 0.0)
    needs_screen = bool(getattr(d, "needs_screen", False)) or screen_p >= NEEDS_SCREEN_P
    why = [w for w, ok in (
        ("game", game is not None),
        ("needs_screen", needs_screen),
        ("screen_tool", _screen_tool(getattr(d, "tool", "") or "")),
    ) if ok]  # fmt: skip
    if not why:
        return None
    return Want(image=(needs_screen or game is not None) and not images, reason="+".join(why))


def uses_screen(tools: Iterable[ToolSpec | str]) -> bool:
    """The run was given the tools that read and work the screen: only then is a look any use to it."""
    return any(_screen_tool(t if isinstance(t, str) else t.name) for t in tools)


# ---- which app ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """The app in front, cleared for a first look. ``why``: "named" (the request names it), "game"
    (the game being played) or "last" (no app named; it's the one NEO last worked in)."""

    app: str
    pid: int
    why: str


def _gates() -> Any:
    """The frozen gate module (neo2.core.tools.computer.gates), looked up at call time like the
    handlers do (a test may swap a helper in it)."""
    from neo2.core.tools.computer import gates

    return gates


def _protected(name: str, pid: int) -> bool:
    """NEO's own window or an app NEO never touches (`protected_apps`, a host's approval dialog):
    never looked at early. Any doubt counts as protected."""
    from neo2.core.tools.computer import ax

    try:
        return bool(ax.is_neo(pid, name) or _gates()._protected(name, pid))
    except Exception:  # noqa: BLE001 - can't tell: fail closed
        return True


def _ids(name: str, pid: int) -> set[str]:
    """Every name the front app goes by, lower-case: its own, its bundle id, its executable's."""
    from neo2.core.tools.computer import ax

    return {n.strip().lower() for n in ({name} | set(ax.app_names(pid))) if n and n.strip()}


def _same_app(app: Any, name: str, pid: int) -> bool:
    """Is the inventory `app` (the one the request names, or the game) the front app `name`/`pid`?
    By bundle id wherever the front app's is known (a name alone can be another app's too), else
    by the app's own name, a name it goes by, or its bundle id among the front app's names."""
    from neo2.core.decide import inventory

    want = (getattr(app, "bundle_id", "") or "").strip().lower()
    try:
        front = inventory.app_for_pid(pid)
    except Exception:  # noqa: BLE001 - not in the last scan: the names decide
        front = None
    if want and front is not None and front.bundle_id:
        return front.bundle_id.strip().lower() == want
    ids = _ids(name, pid)
    names = {str(n).strip().lower() for n in (getattr(app, "name", ""), *getattr(app, "aliases", ()))}
    return bool(want and want in ids) or bool((names - {""}) & ids)


def named_apps(text: str) -> list[Any] | None:
    """The installed apps the request names (neo2.core.decide.inventory's own names for them), or None
    when the apps haven't been read yet — then nobody can tell, and nothing is looked at. Never
    starts a scan of the installed apps: the last completed one or nothing."""
    from neo2.core.decide import inventory

    apps = inventory.snapshot()
    if apps is None:
        return None
    found: list[Any] = []
    t = text or ""
    for _ in range(4):  # "compare Safari with Notes": each name, masked once found
        hit = inventory.named_span(t, apps)
        if hit is None:
            break
        app, start, end = hit
        if not any(a is app or (a.bundle_id, a.path) == (app.bundle_id, app.path) for a in found):
            found.append(app)
        t = t[:start] + " " * (end - start) + t[end:]
    return found


def target(text: str, *, game: Any = None) -> Target | None:
    """The app in front, if a first look at it is allowed for this request (see the module doc):
    it's the request's app — named in it, the game, or (none named) the one NEO last worked in —
    and not NEO's own or a protected app. Reads the front app fresh (for a game always: the game is
    the app the request is about); call it off the event loop."""
    from neo2.core.tools.computer import apps, ax

    # First what the request is about, from what NEO already knows (no AX, no process spawned):
    # most requests stop here without the front app ever being read.
    named: list[Any] = []
    last = ""
    if game is None:
        found = named_apps(text)
        if found is None or len(found) > 1:
            return None  # the apps aren't known yet, or the request is about more than one
        named = found
        if not named:
            last = (apps.target() or "").strip().lower()
            if not last:
                return None  # no app named, and NEO hasn't worked in one lately
    if not ax.is_trusted():
        return None  # the model's own look says why, and how to fix it
    name, pid = ax.working_app()
    if not name or not pid or _protected(name, pid):
        return None
    if _slow_until.get(pid, 0.0) > time.monotonic():
        return None  # its last look missed the deadline: the model looks for itself a while
    if game is not None:
        return Target(name, pid, "game") if _same_app(game, name, pid) else None
    if named:
        return Target(name, pid, "named") if _same_app(named[0], name, pid) else None
    return Target(name, pid, "last") if last in _ids(name, pid) else None


# ---- the look -------------------------------------------------------------------------------------


_ID = re.compile(r"^(\s*)([A-Za-z]{1,3}\d+)\s")  # an element line starts with its id (e12)
# A position or a size, however a line spells it: "@(120,48 30x22)", "(120, 48)", "x=120 y=48".
_COORDS = re.compile(
    r"\s*@\([^)]*\)"
    r"|\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*(?:[,\s]\s*\d+(?:\.\d+)?\s*[x×,]\s*\d+(?:\.\d+)?\s*)?\)"
    r"|\b[xy]\s*[=:]\s*-?\d+(?:\.\d+)?"
)


def _line(e: Any) -> str:
    """An element as the model can act on it by id — with no position or size in it."""
    bits = [str(e.id), str(e.role).removeprefix("AX")]
    if e.title:
        bits.append(f'"{str(e.title)[:80]}"')
    if e.value and e.value != e.title:
        bits.append(f"= {str(e.value)[:80]!r}")
    if not getattr(e, "enabled", True):
        bits.append("(disabled)")
    if getattr(e, "focused", False):
        bits.append("(focused)")
    return _COORDS.sub("", " ".join(bits))


def compact(raw: str) -> str:
    """ax.snapshot's text as ids, roles and labels only: every element rebuilt from what its id
    was issued for, anything else (the header, "… more elements") with any coordinates cut out."""
    from neo2.core.tools.computer import ax

    out = []
    for line in raw.splitlines():
        m = _ID.match(line)
        e = ax.get(m.group(2)) if m else None
        out.append(m.group(1) + _line(e) if (m and e is not None) else _COORDS.sub("", line))
    return "\n".join(out)


# The AX core's line for a large view with no accessible parts (ax._canvas_line): a board's evidence.
_CANVAS = re.compile(r"no accessible parts \(a board or canvas\):[^@\n]*@\((-?\d+),(-?\d+) (\d+)x(\d+)\)")


@dataclass(frozen=True)
class _Listed:
    text: str  # compact: ids, roles and labels
    canvas: tuple[float, float, float, float] | None = None  # a view drawn in one piece (screen points)


def _elements(app: str, pid: int) -> _Listed | None:
    """The app's window as a compact list, or None (another process answers to its name, or the
    app didn't answer)."""
    from neo2.core.tools.computer import ax

    if ax.pid_for_app(app) != pid:
        return None  # its name picks another process: never list a different app's window
    # (the walk ends by the first look's own deadline: a slow or hung app never holds the request up)
    raw = ax.snapshot(app, max_lines=MAX_LINES, deadline_s=_deadline_s())
    if not isinstance(raw, str) or not raw.strip() or raw.lstrip().startswith(("Error", "NEEDS_")):
        return None
    m = _CANVAS.search(raw)
    canvas = tuple(float(v) for v in m.groups()) if m else None
    return _Listed(compact(raw), canvas)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Picture:
    image: ImagePart
    shot: str  # the shot id its pixels are clicked by
    size: tuple[int, int]  # pixels
    covered: bool = False  # another window is in front of part of it: a click there lands on that window
    window: tuple[float, float, float, float] | None = None  # the window's frame on screen (points)


def _picture(pid: int) -> Picture | None:
    """A picture of the app's front window, with a shot id so a seeing model can click by its pixels:
    the window's own pixels, as the screenshot tool pictures that window (screen.window_shot) —
    nothing of any other app's, even where another window is in front of part of it. A click at such
    a part would land on that other window: the shot refuses it (its ``covered`` parts). None without
    Screen Recording (never asked for from here), when the window isn't the app's any more, or when
    the picture can't be mapped to the screen (the window server can't say where the window is or
    what is in front of it)."""
    from neo2.core.tools.computer import focus, pointer, screen

    if not screen.can_capture():
        return None
    wid = focus.front_window_id(pid)
    if not wid:
        return None
    cap = screen.window_shot(wid)
    if not cap.clickable or not cap.w or not cap.h or (cap.pid and cap.pid != pid):
        return None  # nothing to click by (or another app's window now): nothing sent
    shot = pointer.remember(cap.x0, cap.y0, cap.ppp, cap.w, cap.h, covered=cap.covered)
    frame = (cap.x0, cap.y0, cap.w * cap.ppp, cap.h * cap.ppp)
    return Picture(cap.part, shot.id, (cap.w, cap.h), covered=bool(cap.covered), window=frame)


def _board(pic: Picture, canvas: tuple[float, float, float, float] | None) -> str:
    """NEO's words about a board in the first look's picture — its squares by name, so the first
    model call can name them (click cell=…) without a screenshot of its own — or "". Only a board NEO
    finds: a flat one anywhere in the window; one seen at an angle only in the large view with no
    accessible parts the walk reported (``canvas``). Never a hint, never a coordinate."""
    from PIL import Image

    from neo2.core.tools.computer import board, pointer

    shot = pointer.get_shot(pic.shot)
    if shot is None:
        return ""
    img = Image.open(io.BytesIO(pic.image.data))
    img.load()
    s = board.survey(
        board.Picture.from_shot(img.convert("RGB"), shot), shot=shot.id, area=canvas or pic.window,
        canvas=canvas is not None, window=pic.window,
    )  # fmt: skip
    return board.note(s.board, s.since, blind=False) if s.board is not None else ""


_OURS: weakref.WeakValueDictionary[int, ImagePart] = weakref.WeakValueDictionary()  # pictures attached


@dataclass
class Look:
    """What NEO read of the working window before the first model call."""

    app: str
    pid: int
    why: str
    elements: str  # compact: ids, roles, labels — no coordinates
    picture: Picture | None = None
    ms: float = 0.0
    board: str = ""  # NEO's words about a board in the picture (its squares by name), or ""

    @property
    def images(self) -> list[ImagePart]:
        return [self.picture.image] if self.picture else []

    @property
    def taint(self) -> tuple[str, ...]:
        """The tools this look stands in for: what the request has read, as if they had run."""
        return ("ax_tree", "screenshot") if self.picture else ("ax_tree",)

    def block(self) -> str:
        """The look as it goes into the goal message: NEO's own words, then the screen's, fenced."""
        which = {
            "named": "the app in front, which the request names",
            "game": "the game in front, the one to play",
        }.get(self.why, "the app in front, the one NEO last worked in")
        head = (
            f"{MARK}, taken as the request came in: {which}. It isn't refreshed — look again after "
            "you act. Its element ids work with the tools that take an element id.]"
        )
        parts = [head, fence("ax_tree", self.elements)]
        if self.picture is not None:
            w, h = self.picture.size
            what = (
                "that window's own pixels (another window is in front of part of it: a click there lands on that window)"
                if self.picture.covered
                else "that window alone"
            )
            # Conditional: a chain that could see when this was taken may answer with a blind brain.
            parts.append(
                f"[A picture of {what} is attached (screenshot {self.picture.shot}; the image is {w}×{h} "
                "px). If you can see it: to act on something only the picture shows, give its pixel "
                f'position in this image with shot="{self.picture.shot}". If you can\'t see it, use the '
                "element ids above — never coordinates.]"
            )
            if self.board:
                parts.append(f"[{self.board}]")
        return "\n".join(parts)

    def attach(
        self, goal: str, images: Sequence[ImagePart] | None, ctx: ToolContext
    ) -> tuple[str, list[ImagePart]]:
        """The goal message's text and pictures with the look in them — and the request tainted by
        it, exactly as ax_tree (and screenshot) would have (neo2.core.agent.registry.invoke): call it in
        the run's task (neo2.core.agent.grants), and only for a look that really goes to the model."""
        from neo2.core.agent import grants

        for name in self.taint:
            if name not in ctx.taint:
                ctx.taint.append(name)
            grants.read_untrusted(name)
        for im in self.images:
            _OURS[id(im)] = im
        return f"{goal}\n\n{self.block()}", [*(images or []), *self.images]


def _still_front(pid: int) -> bool:
    from neo2.core.tools.computer import ax

    return ax.working_app()[1] == pid


def _web_held(pid: int) -> bool:
    """Is this app's web tree switched on (NEO holds it on for a while after a look at a Chromium or
    Electron app: neo2.core.tools.computer.ax)?"""
    from neo2.core.tools.computer import ax

    held = getattr(ax, "_web_held", None)
    return isinstance(held, dict) and pid in held


def _web_settle_s() -> float:
    """How long, at most, a web tree just switched on takes to fill in before a walk (ax)."""
    from neo2.core.tools.computer import ax

    try:
        return max(0.0, float(getattr(ax, "_WEB_SETTLE_S", 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


async def _look(text: str, want: Want, game: Any, sees: bool, seen: dict[str, int]) -> Look | None:
    """The first look (None whenever any condition fails), off the event loop, with the picture
    taken while the elements are walked. `seen`: "pid", the app it is looking at, once known; "web",
    1 when that app's web tree was already on before this walk."""
    t0 = time.perf_counter()
    tgt = await asyncio.to_thread(target, text, game=game)
    if tgt is None:
        return None
    seen["web"] = int(_web_held(tgt.pid))
    seen["pid"] = tgt.pid
    shot = asyncio.ensure_future(asyncio.to_thread(_picture, tgt.pid)) if want.image and sees else None
    els = await asyncio.gather(asyncio.to_thread(_elements, tgt.app, tgt.pid), return_exceptions=True)
    pic = (await asyncio.gather(shot, return_exceptions=True))[0] if shot is not None else None
    if isinstance(els[0], BaseException):
        raise els[0]
    if els[0] is None or not await asyncio.to_thread(_still_front, tgt.pid):
        return None  # the app didn't answer, or another came to the front meanwhile
    if isinstance(pic, BaseException):
        if not isinstance(pic, Exception):
            raise pic  # (effects off: a test that didn't fake the capture)
        pic = None  # no picture: the list still stands
    words = ""
    if pic is not None:
        try:
            words = await asyncio.to_thread(_board, pic, els[0].canvas)
        except Exception:  # noqa: BLE001 - the board's words are a help, never the look's failure
            words = ""
    return Look(tgt.app, tgt.pid, tgt.why, els[0].text, pic, (time.perf_counter() - t0) * 1000, words)


@dataclass
class Pending:
    """A first look being taken (:func:`start`); :func:`collect` gets it."""

    want: Want
    task: asyncio.Future
    deadline: float  # time.monotonic()
    t0: float = field(default_factory=time.monotonic)
    seen: dict[str, int] = field(default_factory=dict)  # "pid": the app being looked at, once known

    def cancel(self) -> None:
        self.task.cancel()


def _retrieve(fut: asyncio.Future) -> None:
    """A look nobody collected: its error (if any) is never reported as "never retrieved"."""
    if not fut.cancelled():
        with contextlib.suppress(BaseException):
            fut.exception()


def sees(chain: Any) -> bool:
    """Whether the run's brain can see a picture (a chain: any brain in it, the seeing ones first).
    Never raises: a brain that can't be asked counts as blind, and gets no picture."""
    try:
        return bool(getattr(chain, "supports_vision", False))
    except Exception:  # noqa: BLE001
        return False


def start(
    text: str,
    d: Any,
    *,
    only: Sequence[str] | None = None,
    game: Any = None,
    images: Sequence[ImagePart] | None = None,
    sees: bool = False,
) -> Pending | None:
    """Start the first look for this request, if its evidence wants one (:func:`wanted`), the
    setting is on and this agent has the screen tools — to run beside playbook recall and tool
    selection. `game`: the game the request plays (neo2.core.decide.games); `sees`: the run's brain can see
    pictures. Call it inside the running request; nothing is read if it returns None."""
    if not enabled():
        return None
    want = wanted(d, only=only, game=game, images=images)
    if want is None or registry().get("ax_tree") is None:
        return None  # nothing says it will look, or this agent has no screen tools at all
    seen: dict[str, int] = {}
    task = asyncio.ensure_future(_look(text, want, game, bool(sees), seen))
    task.add_done_callback(_retrieve)
    return Pending(want, task, time.monotonic() + _deadline_s(), seen=seen)


def _settling(pending: Pending) -> float:
    """Seconds more to give a look past its deadline: the moment its app's web tree needs to fill in
    when this very look switched it on (a Chromium or Electron app's first look in a while) — once,
    on top of the walk, and a look saves a whole model round trip. 0 otherwise."""
    pid = pending.seen.get("pid")
    if not pid or pending.seen.get("web", 1) or not _web_held(pid):
        return 0.0
    return _web_settle_s()


async def collect(pending: Pending | None, tools: Iterable[ToolSpec | str]) -> Look | None:
    """The first look, when it's ready by its deadline and the run's `tools` can use it; else None
    (a look still being taken is left to finish on its own and thrown away)."""
    if pending is None:
        return None
    if not uses_screen(tools):
        pending.cancel()  # the run can't act on the screen: nothing it read goes anywhere
        return None
    waited_out = False
    while True:
        left = pending.deadline - time.monotonic()
        try:
            got = await asyncio.wait_for(asyncio.shield(pending.task), max(0.0, left))
            break
        except asyncio.CancelledError:
            if not pending.task.cancelled():
                pending.cancel()
                raise  # the request itself was cancelled
            return None
        except TimeoutError:
            if not waited_out and (more := _settling(pending)) > 0:
                waited_out = True  # a web tree switched on by this look: its one-time settle isn't slowness
                pending.deadline += more
                continue
            if pid := pending.seen.get("pid"):
                _slow_until[pid] = time.monotonic() + SLOW_SKIP_S  # its next requests don't wait for it
            log.info(f"  ~ first look not ready in {int((time.monotonic() - pending.t0) * 1000)} ms: dropped")
            return None
        except Exception as e:  # noqa: BLE001 - the run looks for itself, as it always could
            log.info(f"  ~ first look failed: {str(e)[:80]}")
            return None
    if got is not None:
        log.info(
            f"  ~ first look at {got.app} ({got.why}): {got.elements.count(chr(10))} lines"
            f"{', a picture' if got.picture else ''}, {got.ms:.0f} ms"
        )
    return got


# ---- keeping the history lean ---------------------------------------------------------------------


def _let_go(shot: str) -> None:
    """Nothing is clicked by the pixels of a picture the model no longer has: its shot id is let go
    (a click that names it is told to take a new screenshot)."""
    from neo2.core.tools.computer import pointer

    shots = getattr(pointer, "_shots", None)
    if isinstance(shots, dict):
        shots.pop(shot, None)


_BOARD_LINE = re.compile(r"^\[(NEO (?:found|mapped) (?:a board|a grid) .*?)(?: Pieces now\b.*)?\]$", re.M)


def _stub(m: Message, *, keep_board: bool = False) -> bool:
    """Replace an attached first look in this goal message with a one-line stub (and drop its
    picture, and its shot id). ``keep_board``: NEO's words on how a board's squares are named stay
    (mid-run, the board is still there — its pieces as they stood are dropped). True when there was one."""
    head, sep, rest = m.text.partition("\n\n" + MARK)
    if not sep:
        return False
    tail = rest.rsplit("</untrusted>", 1)[-1]  # NEO's own notes, after the fenced list
    board = _BOARD_LINE.search(tail) if keep_board else None
    m.text = f"{head}\n\n{STUB}" + (f"\n[{board.group(1).strip()}]" if board else "")
    m.images = [im for im in m.images if _OURS.get(id(im)) is not im]
    if found := _SHOT.search(tail):
        _let_go(found.group(1))
    return True


def forget(messages: Iterable[Message]) -> None:
    """A finished run's turns, before they go into the conversation history: the first look was for
    that run only (the screen has moved on), so each goal message keeps just a stub of it."""
    for m in messages:
        if m.role == "user":
            _stub(m)


def _stale_after(r: Any) -> bool:
    """Does this result make the first look stale: a newer look at the screen, or an action that
    reports it changed something there?"""
    if not getattr(r, "ok", True) or r.content.startswith(("Error", "NEEDS_")):
        return False
    return r.name in _OBSERVE or (_screen_tool(r.name) and _CHANGED in r.content)


def supersede(messages: Sequence[Message], *, start: int = 0) -> None:
    """In a run, once the model has looked again itself (ax_tree, ax_find, a screenshot) or an action
    reports it changed the screen, the first look is stale: stub it, like the loop stubs every older
    observation. The loop calls it before each model call (required wiring: neo2.core.agent.loop)."""
    first = next((i for i in range(start, len(messages)) if messages[i].role == "user" and MARK in messages[i].text), None)
    if first is None:
        return
    for m in messages[first + 1 :]:
        if m.role == "tool" and any(_stale_after(r) for r in m.tool_results):
            _stub(messages[first], keep_board=True)
            return


# ---- prewarm --------------------------------------------------------------------------------------

PREWARM_EVERY_S = 20.0  # one prewarm per this long at most (a wake, then the words that follow it)
_WARM_Q = {"warm": {"type": "noul", "instructions": "Is this a warm-up?"}}
_warm_mu = threading.Lock()
_last_warm = float("-inf")
_warmers: list[Callable[[], object]] = []


def register_warmer(fn: Callable[[], object]) -> None:
    """Something else the next request's resolver will want warm (called in the prewarm thread,
    errors ignored): the commands workstream's label index, say."""
    if fn not in _warmers:
        _warmers.append(fn)


def prewarm(reason: str = "wake") -> bool:
    """The user started talking to NEO: load the resolver's Laya (neo2.core.reflex.zeroshot) now if it
    was let go, or keep it from being let go if it's loaded — only while a resolver stage that acts
    reads it (:func:`_resolver_reads_laya`) — and run the other warmers, in a thread of its own,
    never waited for. True when a prewarm was started. Not in an embedded agent (no voice there),
    not with Laya off, not more than once per :data:`PREWARM_EVERY_S`; ``prewarm_resolver`` turns
    it off."""
    global _last_warm
    try:
        if runtime.library_bound() or settings().reflex != "laya":
            return False
        if not _setting("prewarm_resolver", True, False):
            return False
        laya = _resolver_reads_laya()
    except Exception:  # noqa: BLE001 - never in the way of listening
        return False
    if not laya and not _warmers:
        return False
    now = time.monotonic()
    with _warm_mu:
        if now - _last_warm < PREWARM_EVERY_S:
            return False
        _last_warm = now
    threading.Thread(target=_warm, args=(reason, laya), name="neo-prewarm", daemon=True).start()
    return True


def _resolver_reads_laya() -> bool:
    """Does a resolver stage that acts read Laya — the meaning stage, promoted and live
    (resolve.live_stages)? In shadow it reads Laya only when Laya is loaded anyway (resolve's
    _ask_if_loaded): loading it for that, 1.7 GB at every wake and a pass that keeps it from ever
    being let go, would buy only shadow logs (decision 40)."""
    from neo2.core.tools.computer import resolve

    return "meaning" in resolve.live_stages()


def _warm(reason: str, laya: bool = True) -> None:
    t0 = time.perf_counter()
    if laya:
        try:
            from neo2.core.reflex import zeroshot

            was = zeroshot.loaded()
            # wait 0: a load starts in zeroshot's own thread and this returns at once; a loaded model
            # answers one tiny question (if Laya is free), which counts as use and keeps it loaded.
            zeroshot.ask({"text": "warm up"}, _WARM_Q, wait_s=0.0)
            log.info(f"  ~ prewarm ({reason}): resolver Laya {'kept warm' if was else 'loading'} ({(time.perf_counter() - t0) * 1000:.0f} ms)")
        except Exception as e:  # noqa: BLE001
            log.info(f"  ~ prewarm ({reason}) failed: {str(e)[:80]}")
    for fn in list(_warmers):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            log.info(f"  ~ prewarm ({reason}): {getattr(fn, '__name__', 'warmer')} failed: {str(e)[:80]}")


__all__ = [
    "LOOK_S",
    "Look",
    "Pending",
    "Picture",
    "Target",
    "Want",
    "collect",
    "compact",
    "enabled",
    "forget",
    "named_apps",
    "prewarm",
    "register_warmer",
    "sees",
    "start",
    "supersede",
    "target",
    "uses_screen",
    "wanted",
]
