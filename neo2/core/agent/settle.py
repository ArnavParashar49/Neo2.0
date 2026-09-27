"""Wait for an app that was just opened to be ready, instead of sleeping a fixed time.

"open Notes and type hello": the step after open_app used to wait a fixed 0.8 s
(session._quick_steps, early._run) — too long when the app was already running (most of the
time: it only came forward), too short for a cold launch (the first keystrokes went nowhere). Now
it waits until the app is in front with a focused window — and, when the next step types, a
field that takes text — checking every 50 ms, twice in a row, never longer than 2 s.

When nothing reports a field that takes text (an editor whose role Accessibility doesn't name),
or the app in front has no focused window (all its windows closed, one that never takes focus),
the app being in front is enough once the old fixed time (0.8 s) has passed: never slower than
before for an app that is up. Only reads: the front app, its focused window, the keyboard focus.
Nothing is pressed, raised or typed, and nothing keeps running after the wait.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from neo2.core import logger

log = logger(__name__)

CAP_S = 2.0  # never wait longer than this: the step goes ahead as it did before
POLL_S = 0.05
STABLE = 2  # ready this many looks in a row (a window that flashes up while the app loads isn't)
OLD_SETTLE_S = 0.8  # the fixed wait this replaces: a focused window is enough after it
# The next steps that type into the app: they wait for a field that takes text.
TYPING = frozenset({"type_text", "paste", "ax_set_value"})


@dataclass(frozen=True)
class Ready:
    front: bool  # the app acted on is the frontmost one
    window: bool  # it has a focused (or main) window
    editable: bool = False  # the keyboard focus is in something that takes text


def _probe(app: str | None, typing: bool) -> Ready:
    """What the Mac says now (Accessibility reads only)."""
    from neo2.core.tools.computer import apps, ax
    from neo2.core.tools.computer.focus import front_app

    want = (app or apps.target() or "").strip()
    name, pid = front_app()
    if not pid:
        return Ready(False, False)
    front = not want or name.lower() == want.lower() or (ax.pid_for_app(want) == pid)
    _, win = ax.focused_window()
    editable = bool(typing and win is not None and ax.focused_editable())
    return Ready(front, win is not None, editable)


# The look the wait uses: (app, typing) → Ready. A test replaces it (the suite never reads the Mac).
probe: Callable[[str | None, bool], Ready] = _probe


async def app_ready(
    *,
    typing: bool = False,
    app: str | None = None,
    cap: float = CAP_S,
    every: float = POLL_S,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
) -> float:
    """Wait until the app just opened (`app`, else the app NEO last opened) is ready for the next
    step: in front with a focused window, and with the keyboard in a field that takes text when
    `typing` — or, once OLD_SETTLE_S has passed, in front at all (no window or field may ever come:
    the old fixed wait went ahead then too). Returns the seconds waited; after `cap` it gives up
    waiting and the step goes ahead."""
    t0 = clock()
    streak = 0
    while True:
        waited = clock() - t0
        left = cap - waited
        if left <= 0:
            log.info(f"[settle] not ready after {cap:g}s; going ahead")
            return waited
        try:
            r = await asyncio.wait_for(asyncio.to_thread(probe, app, typing), timeout=left)
        except TimeoutError:
            return clock() - t0
        except Exception as e:  # noqa: BLE001 — can't tell: the old fixed wait, no more
            log.info(f"[settle] can't look ({type(e).__name__}); waiting the old {OLD_SETTLE_S:g}s")
            rest = OLD_SETTLE_S - (clock() - t0)
            if rest > 0:
                await sleep(rest)
            return clock() - t0
        waited = clock() - t0
        late = waited >= OLD_SETTLE_S
        ok = r.front and (r.window or late) and (not typing or r.editable or late)
        streak = streak + 1 if ok else 0
        if streak >= STABLE:
            return waited
        await sleep(every)
