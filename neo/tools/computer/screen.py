"""Screenshots (the vision fallback). Downscaled to logical *points* so the pixel
coordinates the model reads off the image are the same ones the input tools take."""

from __future__ import annotations

import io
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import Quartz as Q
from PIL import Image

from neo.providers.base import ImagePart

_MAX_W = 1600  # keep token cost sane; 14"/16" MBP logical widths are 1512/1728


def can_capture() -> bool:
    return bool(Q.CGPreflightScreenCaptureAccess())


@dataclass(frozen=True)
class Display:
    """One screen, in global points: ``index`` is screencapture's -D number (1 = the main one)."""

    index: int
    x: float
    y: float
    w: float
    h: float

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def words(self) -> str:
        return f"display {self.index}: {self.w:g}×{self.h:g} pt at ({self.x:g},{self.y:g})"


def displays() -> list[Display]:
    """Every active display, the main one first (as screencapture numbers them)."""
    ids: list[int] = []
    try:
        err, got, n = Q.CGGetActiveDisplayList(16, None, None)
        if err == 0 and got:
            ids = [int(d) for d in list(got)[: int(n)]]
    except Exception:  # noqa: BLE001 - fall back to the main display alone
        ids = []
    main = int(Q.CGMainDisplayID())
    ids = [main] + [d for d in ids if d != main]
    out = []
    for i, d in enumerate(ids, 1):
        b = Q.CGDisplayBounds(d)
        out.append(Display(i, float(b.origin.x), float(b.origin.y), float(b.size.width), float(b.size.height)))
    return out


def display_at(x: float, y: float) -> Display | None:
    """The display a global point is on (None: off every screen)."""
    return next((d for d in displays() if d.contains(x, y)), None)


def display_points() -> tuple[int, int]:
    b = Q.CGDisplayBounds(Q.CGMainDisplayID())
    return int(b.size.width), int(b.size.height)


def _grab(cmd_extra: list[str]) -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = Path(f.name)
    try:
        subprocess.run(["screencapture", "-x", "-t", "png", *cmd_extra, str(path)], check=True, timeout=10)
        return Image.open(path).convert("RGB")
    finally:
        path.unlink(missing_ok=True)


def capture(region: tuple[int, int, int, int] | None = None, display: Display | None = None) -> tuple[ImagePart, float]:
    """Return (png image, scale) where scale = points-per-image-pixel. ``region``: a rectangle in
    global points; else the whole of ``display`` (the main one when None)."""
    if not can_capture():
        Q.CGRequestScreenCaptureAccess()
        raise PermissionError(
            "Screen Recording permission is off. Enable it for this app in System Settings → "
            "Privacy & Security → Screen Recording, then try again."
        )
    if region:
        x, y, w, h = region
        img = _grab(["-R", f"{x},{y},{w},{h}"])
        width_pt = float(w)
    else:
        d = display or displays()[0]
        img = _grab(["-D", str(d.index)] if d.index != 1 else [])
        width_pt = d.w
    # Retina captures are 2x; map to points, then cap width.
    target_w = int(min(round(width_pt), _MAX_W))
    if img.width > target_w * 1.01:
        img = img.resize((target_w, max(1, round(img.height * target_w / img.width))), Image.LANCZOS)
    points_per_px = width_pt / img.width
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return ImagePart(buf.getvalue(), "image/png"), points_per_px


# ---- did anything change? --------------------------------------------------------------------------
# A click reports what it did: a small picture of the screen around the point, taken just before
# and just after, compared at low resolution. The pointer isn't in screencapture's pictures, and
# NEO's own window never is (it hides from screen capture), so neither counts as a change.
_SAMPLE_R = 60  # points around each point
_SAMPLE_GRID = 24  # each sample is compared as this many × this many grey cells
_CELL_DIFF = 18  # a cell that moved this much (0-255) changed…
_CELLS_CHANGED = 3  # …and this many changed cells are a change


def sample_points(points: list[tuple[float, float]], r: float = _SAMPLE_R) -> list[Image.Image] | None:
    """A small grey picture of the screen around each point (None without Screen Recording, or
    when the capture fails: nothing can be said about change). Never asks for the permission."""
    try:
        if not points or not can_capture():
            return None
        left = min(p[0] for p in points) - r
        top = min(p[1] for p in points) - r
        right = max(p[0] for p in points) + r
        bottom = max(p[1] for p in points) + r
        img = _grab(["-R", f"{int(left)},{int(top)},{int(right - left)},{int(bottom - top)}"])
        k = img.width / max(1.0, right - left)  # pixels per point
        out = []
        for px, py in points:
            box = (round((px - r - left) * k), round((py - r - top) * k), round((px + r - left) * k), round((py + r - top) * k))
            out.append(img.crop(box).convert("L").resize((_SAMPLE_GRID, _SAMPLE_GRID), Image.BILINEAR))
        return out
    except Exception:  # noqa: BLE001 - no picture: unknown, not "unchanged"
        return None


def differs(a: Image.Image, b: Image.Image) -> bool:
    """Do two samples of the same place show something different?"""
    if a.size != b.size:
        return True
    pa, pb = a.convert("L").tobytes(), b.convert("L").tobytes()
    changed = sum(1 for u, v in zip(pa, pb, strict=True) if abs(int(u) - int(v)) > _CELL_DIFF)
    return changed >= _CELLS_CHANGED


_MIN_CUT = 50  # points: a part of a window smaller than this is no page to look at


def _window_bounds(window_id: int) -> tuple[float, float, float, float] | None:
    """(x, y, width, height) of a window on screen, in points from the top left (None: unknown)."""
    try:
        info = Q.CGWindowListCopyWindowInfo(Q.kCGWindowListOptionIncludingWindow, int(window_id)) or []
        b = info[0].get("kCGWindowBounds") if len(info) else None
        return (float(b["X"]), float(b["Y"]), float(b["Width"]), float(b["Height"])) if b else None
    except Exception:  # noqa: BLE001 - no bounds: the window is captured whole
        return None


def _cut(img: Image.Image, bounds: tuple[float, float, float, float] | None, region: tuple[float, ...]) -> Image.Image:
    """The part of a window's picture that shows ``region`` (screen points), or the whole picture
    when that part isn't in the window."""
    if bounds is None or bounds[2] <= 0:
        return img
    x, y, w, h = bounds
    left, top = max(region[0], x), max(region[1], y)
    right, bottom = min(region[0] + region[2], x + w), min(region[1] + region[3], y + h)
    if right - left < _MIN_CUT or bottom - top < _MIN_CUT:
        return img
    k = img.width / w  # pixels per point (2 on a Retina display)
    return img.crop((round((left - x) * k), round((top - y) * k), round((right - x) * k), round((bottom - y) * k)))


def capture_window(window_id: int, region: tuple[float, float, float, float] | None = None) -> ImagePart:
    """A png of one window (its own pixels, even where another window covers it; no shadow),
    at most ``_MAX_W`` wide. ``region`` (screen points, x y width height): only that part of it,
    a page without the browser's tabs, say. For looking at it, not for clicking: no coordinates
    come with it."""
    if not can_capture():
        Q.CGRequestScreenCaptureAccess()
        raise PermissionError(
            "Screen Recording permission is off. Enable it for this app in System Settings → "
            "Privacy & Security → Screen Recording, then try again."
        )
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = Path(f.name)
    try:
        cmd = ["screencapture", "-x", "-o", "-t", "png", "-l", str(int(window_id)), str(path)]
        subprocess.run(cmd, check=True, timeout=10)
        img = Image.open(path).convert("RGB")
    finally:
        path.unlink(missing_ok=True)
    if region is not None:
        img = _cut(img, _window_bounds(window_id), region)
    if img.width > _MAX_W:
        img = img.resize((_MAX_W, round(img.height * _MAX_W / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return ImagePart(buf.getvalue(), "image/png")
