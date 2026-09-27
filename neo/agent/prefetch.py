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
  the request names no app, the one NEO last worked in. Never NEO's own window, never a protected
  app (the gate helpers decide, fail closed), never without Accessibility;
* what: the window's elements as a compact list — ids, roles, labels, no coordinates, so a brain
  that can't see is never handed any — and a picture of that window alone (its own pixels, never
  what another app's window puts over it) only when Laya says the screen itself is needed, the
  brain can see, and Screen Recording is already allowed (it is never asked for from here);
* how long: a hard deadline (``prefetch_look_s``); a look that isn't ready by then is dropped.

What is attached is fenced as untrusted, and taints the request exactly as the tools would —
only when it is attached. The ``prefetch_look`` setting turns all of it off.

**Prewarm.** The resolver's zero-shot Laya checkpoint (neo.reflex.zeroshot, typed-decisions) is
let go after ten idle minutes, and a question that finds it unloaded pays a 0.4-0.7 s load. The
early actor (neo.agent.early) calls :func:`prewarm` when the user starts talking to NEO (wake word,
push-to-talk, the first words of an utterance), so the load happens while they speak and the
resolver answers warm.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import os
import re
import threading
import time
import weakref
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from neo import runtime
from neo.agent.registry import ToolContext, fence, registry
from neo.config import settings
from neo.providers.base import ImagePart, Message, ToolSpec

log = logging.getLogger(__name__)

LOOK_S = 0.5  # the whole first look (elements and picture, side by side) is ready by then, or dropped
MAX_LINES = 150  # elements listed (the model's own ax_tree call can always show more)
NEEDS_SCREEN_P = 0.5  # Laya's needs_screen at or above this: the screen itself matters (a picture)
MARK = "[NEO's first look"  # opens the attached block (and nothing a user types)
STUB = "[NEO looked at the screen here, before the first step.]"  # what a finished run keeps of it
_SCREEN_CATEGORY = "computer"  # the tool family that reads and works the screen
_OBSERVE = frozenset({"ax_tree", "screenshot"})  # a newer one of these makes the first look stale
SLOW_SKIP_S = 300.0  # an app whose look missed the deadline isn't waited for again for this long
_slow_until: dict[int, float] = {}  # pid -> time.monotonic() until which it's skipped


# ---- settings -----------------------------------------------------------------------------------


def _setting(name: str, app_default: Any, library_default: Any) -> Any:
    """A setting the integrator adds to neo.config (``prefetch_look``, ``prefetch_look_s``,
    ``prewarm_resolver``). Until it's there: NEO.app reads ``NEO_<NAME>`` itself; an embedded agent
    never reads the environment, and gets the library default."""
    v = getattr(settings(), name, None)
    if v is not None:
        return v
    if runtime.library_bound():
        return library_default
    raw = os.environ.get(f"NEO_{name.upper()}")
    if raw is None or not raw.strip():
        return app_default
    if isinstance(app_default, bool):
        return raw.strip().lower() not in ("0", "false", "off", "no")
    try:
        return type(app_default)(raw)
    except (TypeError, ValueError):
        return app_default


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
    """Where the gate helpers live: the frozen gate module once computer/__init__.py is split, the
    package until then. Looked up at call time, like the handlers do."""
    try:
        from neo.tools.computer import gates  # type: ignore[attr-defined]

        return gates
    except ImportError:
        from neo.tools import computer

        return computer


def _protected(name: str, pid: int) -> bool:
    """NEO's own window or an app NEO never touches (`protected_apps`, a host's approval dialog):
    never looked at early. Any doubt counts as protected."""
    from neo.tools.computer import ax

    try:
        return bool(ax.is_neo(pid, name) or _gates()._protected(name, pid))
    except Exception:  # noqa: BLE001 - can't tell: fail closed
        return True


def _ids(name: str, pid: int) -> set[str]:
    """Every name the front app goes by, lower-case: its own, its bundle id, its executable's."""
    from neo.tools.computer import ax

    return {n.strip().lower() for n in ({name} | set(ax.app_names(pid))) if n and n.strip()}


def _same_app(app: Any, name: str, pid: int) -> bool:
    """Is the inventory `app` (the one the request names, or the game) the front app `name`/`pid`?
    By bundle id wherever the front app's is known (a name alone can be another app's too), else
    by the app's own name, a name it goes by, or its bundle id among the front app's names."""
    from neo.decide import inventory

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
    """The installed apps the request names (neo.decide.inventory's own names for them), or None
    when the apps haven't been read yet — then nobody can tell, and nothing is looked at. Never
    starts a scan of the installed apps: the last completed one or nothing."""
    from neo.decide import inventory

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
    and not NEO's own or a protected app. Reads the front app fresh; call it off the event loop."""
    from neo.tools.computer import apps, ax

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
    from neo.tools.computer import ax

    out = []
    for line in raw.splitlines():
        m = _ID.match(line)
        e = ax.get(m.group(2)) if m else None
        out.append(m.group(1) + _line(e) if (m and e is not None) else _COORDS.sub("", line))
    return "\n".join(out)


def _elements(app: str, pid: int) -> str | None:
    """The app's window as a compact list, or None (another process answers to its name, or the
    app didn't answer)."""
    from neo.tools.computer import ax

    if ax.pid_for_app(app) != pid:
        return None  # its name picks another process: never list a different app's window
    raw = ax.snapshot(app, max_lines=MAX_LINES)
    if not isinstance(raw, str) or not raw.strip() or raw.lstrip().startswith(("Error", "NEEDS_")):
        return None
    return compact(raw)


def _window_bounds(window_id: int) -> tuple[float, float, float, float] | None:
    """(x, y, width, height) of a window, in screen points (None: unknown)."""
    import Quartz as Q

    info = Q.CGWindowListCopyWindowInfo(Q.kCGWindowListOptionIncludingWindow, int(window_id)) or []
    b = info[0].get("kCGWindowBounds") if len(info) else None
    if not b or float(b["Width"]) <= 0 or float(b["Height"]) <= 0:
        return None
    return float(b["X"]), float(b["Y"]), float(b["Width"]), float(b["Height"])


@dataclass(frozen=True)
class Picture:
    image: ImagePart
    shot: str  # the shot id its pixels are clicked by
    size: tuple[int, int]  # pixels


def _picture(pid: int) -> Picture | None:
    """A picture of the app's front window and nothing else (its own pixels: a window of another app
    over it is not in it), with a shot id so a seeing model can click by its pixels. None without
    Screen Recording — which is never asked for from here."""
    from PIL import Image

    from neo.tools.computer import focus, pointer, screen

    if not screen.can_capture():
        return None
    wid = focus.front_window_id(pid)
    bounds = _window_bounds(wid) if wid else None
    if bounds is None:
        return None
    img = screen.capture_window(wid)
    w, h = Image.open(io.BytesIO(img.data)).size
    if not w or not h:
        return None
    shot = pointer.remember(bounds[0], bounds[1], bounds[2] / w, w, h)
    return Picture(img, shot.id, (w, h))


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

    @property
    def images(self) -> list[ImagePart]:
        return [self.picture.image] if self.picture else []

    @property
    def taint(self) -> tuple[str, ...]:
        """The tools this look stands in for: what the request has read, as if they had run."""
        return ("ax_tree", "screenshot") if self.picture else ("ax_tree",)

    def block(self) -> str:
        """The look as it goes into the goal message: NEO's own words, then the screen's, fenced."""
        head = (
            f"{MARK}, taken as the request came in: the app in front, which is the app this request "
            "is about. It isn't refreshed — look again after you act. Its element ids work with the "
            "tools that take an element id.]"
        )
        parts = [head, fence("ax_tree", self.elements)]
        if self.picture is not None:
            w, h = self.picture.size
            parts.append(
                f"[A picture of that window alone is attached (screenshot {self.picture.shot}; the image "
                f'is {w}×{h} px). To act on something only the picture shows, give its pixel position in '
                f'this image with shot="{self.picture.shot}".]'
            )
        return "\n".join(parts)

    def attach(
        self, goal: str, images: Sequence[ImagePart] | None, ctx: ToolContext
    ) -> tuple[str, list[ImagePart]]:
        """The goal message's text and pictures with the look in them — and the request tainted by
        it, exactly as ax_tree (and screenshot) would have (neo.agent.registry.invoke): call it in
        the run's task (neo.agent.grants), and only for a look that really goes to the model."""
        from neo.agent import grants

        for name in self.taint:
            if name not in ctx.taint:
                ctx.taint.append(name)
            grants.read_untrusted(name)
        for im in self.images:
            _OURS[id(im)] = im
        return f"{goal}\n\n{self.block()}", [*(images or []), *self.images]


def _still_front(pid: int) -> bool:
    from neo.tools.computer import ax

    return ax.working_app()[1] == pid


def look(text: str, want: Want, *, game: Any = None, sees: bool = False) -> Look | None:
    """Take the first look now (blocking; see :func:`start`). None whenever any condition fails."""
    t0 = time.perf_counter()
    tgt = target(text, game=game)
    if tgt is None:
        return None
    pic: Picture | None = None
    if want.image and sees:
        with contextlib.suppress(Exception):  # no picture: the list still stands
            pic = _picture(tgt.pid)
    els = _elements(tgt.app, tgt.pid)
    if els is None or not _still_front(tgt.pid):
        return None  # the app didn't answer, or another came to the front meanwhile
    return Look(tgt.app, tgt.pid, tgt.why, els, pic, (time.perf_counter() - t0) * 1000)


async def _look(text: str, want: Want, game: Any, sees: bool, seen: dict[str, int]) -> Look | None:
    """:func:`look` off the event loop, with the picture taken while the elements are walked.
    `seen["pid"]`: the app it is looking at, once known."""
    t0 = time.perf_counter()
    tgt = await asyncio.to_thread(target, text, game=game)
    if tgt is None:
        return None
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
    return Look(tgt.app, tgt.pid, tgt.why, els[0], pic, (time.perf_counter() - t0) * 1000)


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
    selection. `game`: the game the request plays (neo.decide.games); `sees`: the run's brain can see
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


async def collect(pending: Pending | None, tools: Iterable[ToolSpec | str]) -> Look | None:
    """The first look, when it's ready by its deadline and the run's `tools` can use it; else None
    (a look still being taken is left to finish on its own and thrown away)."""
    if pending is None:
        return None
    if not uses_screen(tools):
        pending.cancel()  # the run can't act on the screen: nothing it read goes anywhere
        return None
    left = pending.deadline - time.monotonic()
    try:
        got = await asyncio.wait_for(asyncio.shield(pending.task), max(0.0, left))
    except asyncio.CancelledError:
        if not pending.task.cancelled():
            pending.cancel()
            raise  # the request itself was cancelled
        return None
    except TimeoutError:
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


def _stub(m: Message) -> bool:
    """Replace an attached first look in this goal message with a one-line stub (and drop its
    picture). True when there was one."""
    head, sep, _ = m.text.partition("\n\n" + MARK)
    if not sep:
        return False
    m.text = f"{head}\n\n{STUB}"
    m.images = [im for im in m.images if _OURS.get(id(im)) is not im]
    return True


def forget(messages: Iterable[Message]) -> None:
    """A finished run's turns, before they go into the conversation history: the first look was for
    that run only (the screen has moved on), so each goal message keeps just a stub of it."""
    for m in messages:
        if m.role == "user":
            _stub(m)


def supersede(messages: Sequence[Message], *, start: int = 0) -> None:
    """In a run, once the model has looked again itself (a newer ax_tree or screenshot result),
    the first look is stale: stub it, like the loop stubs every older observation."""
    first = next((i for i in range(start, len(messages)) if messages[i].role == "user" and MARK in messages[i].text), None)
    if first is None:
        return
    for m in messages[first + 1 :]:
        if m.role == "tool" and any(r.name in _OBSERVE and not r.content.startswith(("Error", "NEEDS_")) for r in m.tool_results):
            _stub(messages[first])
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
    """The user started talking to NEO: load the resolver's Laya (neo.reflex.zeroshot) now if it
    was let go, or keep it from being let go if it's loaded, in a thread of its own — never waited
    for. True when a prewarm was started. Not in an embedded agent (no voice there), not with Laya
    off, not more than once per :data:`PREWARM_EVERY_S`; ``prewarm_resolver`` turns it off."""
    global _last_warm
    try:
        if runtime.library_bound() or settings().reflex != "laya":
            return False
        if not _setting("prewarm_resolver", True, False):
            return False
    except Exception:  # noqa: BLE001 - never in the way of listening
        return False
    now = time.monotonic()
    with _warm_mu:
        if now - _last_warm < PREWARM_EVERY_S:
            return False
        _last_warm = now
    threading.Thread(target=_warm, args=(reason,), name="neo-prewarm", daemon=True).start()
    return True


def _warm(reason: str) -> None:
    t0 = time.perf_counter()
    try:
        from neo.reflex import zeroshot

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
    "look",
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
