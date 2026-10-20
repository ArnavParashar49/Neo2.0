"""Search Screen's pictures: the still of the display the overlay is on, and an area in live mode.

The still ("freeze") is taken the moment the hold arms, before the overlay shows anything: one
ScreenCaptureKit capture of the whole display, whose content filter leaves NEO.app's own windows
out — so the overlay is never in it, whatever it draws later. It is taken whether or not
``settings().sck_capture`` is on for the computer tools (that switch is theirs), and never through
the ``screencapture`` CLI, which would capture the glow. Screen Recording is checked first and never
asked for here: without it nothing is captured (the overlay shows how to allow it), and
``screen.grab`` — which would put up macOS's prompt — is never called.

No ScreenCaptureKit (an older macOS, the framework missing), or a capture that failed: live mode.
The overlay stays over the live screen, and when the user selects an area the page hides the
overlay for a moment and the core captures just that area (:func:`grab_live`).

Points and pixels: the shell says where the display is in global points (top left) and its scale;
the still is that display at its own pixels (2 per point on Retina), as a screen.Frame that knows
its points per pixel, so any area cut from it maps back exactly.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from neo2.core import logger
from neo2.core.tools.computer import screen

log = logger(__name__)

TIMEOUT_S = 2.0  # one freeze's whole wait for ScreenCaptureKit
_NEAR_PT = 1.0  # the shell's display and the core's are the same within this
MAX_SCALE = 8.0


@dataclass(frozen=True)
class Spot:
    """The display the shell put the overlay on, as it said: global points (top left), its
    CGDirectDisplayID, and backing pixels per point."""

    id: int
    x: float
    y: float
    w: float
    h: float
    scale: float

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x < self.x + self.w and self.y <= y < self.y + self.h

    def shown(self) -> dict[str, float]:
        """What the page is told of it (its own size: everything it sends is display-local)."""
        return {"w": self.w, "h": self.h, "scale": self.scale}


def spot(raw: Any) -> Spot | None:
    """The shell's display, checked: finite numbers, a size, a sane scale (None: not one)."""
    if not isinstance(raw, dict):
        return None
    try:
        did = raw.get("id", 0)
        if isinstance(did, bool) or not isinstance(did, int) or not 0 <= did < 2**32:
            return None
        x, y, w, h, scale = (float(raw[k]) for k in ("x", "y", "w", "h", "scale"))
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, w, h, scale)):
        return None
    if not (1 <= w <= 100_000 and 1 <= h <= 100_000 and 0 < scale <= MAX_SCALE and abs(x) < 1e6 and abs(y) < 1e6):
        return None
    return Spot(did, x, y, w, h, scale)


@dataclass
class Freeze:
    """A freeze's outcome: the still, or why there is none ("no_permission", "no_sck", "failed")."""

    frame: screen.Frame | None
    why: str = ""
    ms: float = 0.0


def display_for(sp: Spot) -> screen.Display | None:
    """The core's own display for the shell's: by its id, else the one at its middle — and only
    where the two agree on where it is (within a point)."""
    shown = screen.displays()
    d = next((d for d in shown if sp.id and d.did == sp.id), None) or screen.display_at(sp.x + sp.w / 2, sp.y + sp.h / 2)
    if d is None:
        return None
    if max(abs(d.x - sp.x), abs(d.y - sp.y), abs(d.w - sp.w), abs(d.h - sp.h)) > _NEAR_PT:
        return None
    return d


def freeze(sp: Spot, *, timeout: float = TIMEOUT_S) -> Freeze:
    """The still of the shell's display, NEO's own windows left out (see the module's doc)."""
    t0 = time.perf_counter()
    if not screen.can_capture():
        return Freeze(None, "no_permission")
    d = display_for(sp)
    if d is None:
        log.info("[lens] the shell's display isn't one the core sees: live mode")
        return Freeze(None, "failed")
    from neo2.core.tools.computer import sck

    if not sck.enabled():
        return Freeze(None, "no_sck")
    try:
        img = sck.capture_rect(d.x, d.y, d.w, d.h, d, scale=sp.scale, timeout=timeout)
    except sck.Failed as e:
        log.info(f"[lens] the still wasn't taken ({str(e)[:80]}): live mode")
        return Freeze(None, "failed", (time.perf_counter() - t0) * 1000)
    ms = (time.perf_counter() - t0) * 1000
    frame = screen.Frame(img, d.x, d.y, d.w / max(1, img.width), backend="sck", grab_ms=ms)
    return Freeze(frame, "", ms)


def clamp(sp: Spot, rect: Any) -> tuple[float, float, float, float] | None:
    """An area the page sent ([x, y, w, h], display-local points), cut to the display (None: not
    four finite numbers, or nothing of it is on the display)."""
    if not isinstance(rect, list | tuple) or len(rect) != 4 or any(isinstance(v, bool) for v in rect):
        return None
    try:
        x, y, w, h = (float(v) for v in rect)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return None
    if w < 0:
        x, w = x + w, -w
    if h < 0:
        y, h = y + h, -h
    left, top = max(0.0, x), max(0.0, y)
    right, bottom = min(sp.w, x + w), min(sp.h, y + h)
    if right - left < 1 or bottom - top < 1:
        return None
    return left, top, right - left, bottom - top


def cut(frame: screen.Frame, sp: Spot, rect: tuple[float, float, float, float]) -> screen.Frame:
    """The still's part showing a display-local area, at the display's own pixels."""
    x, y, w, h = rect
    return frame.cut(sp.x + x, sp.y + y, w, h)


def grab_live(sp: Spot, rect: tuple[float, float, float, float]) -> screen.Frame:
    """Live mode: the area captured now (the page has hidden the overlay first). Screen Recording is
    checked first: never macOS's prompt from here (PermissionError instead)."""
    if not screen.can_capture():
        raise PermissionError("no Screen Recording")
    x, y, w, h = rect
    gx, gy = sp.x + x, sp.y + y
    left, top = math.floor(gx), math.floor(gy)
    region = (left, top, max(1, math.ceil(gx + w) - left), max(1, math.ceil(gy + h) - top))
    return screen.grab(region)


__all__ = ["Freeze", "Spot", "clamp", "cut", "display_for", "freeze", "grab_live", "spot"]
