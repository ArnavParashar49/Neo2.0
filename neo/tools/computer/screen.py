"""Screenshots (the vision fallback). Downscaled to logical *points* so the pixel
coordinates the model reads off the image are the same ones the input tools take."""

from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path

import Quartz as Q
from PIL import Image

from neo import effects
from neo.providers.base import ImagePart

_MAX_W = 1600  # keep token cost sane; 14"/16" MBP logical widths are 1512/1728


def can_capture() -> bool:
    return bool(Q.CGPreflightScreenCaptureAccess())


def display_points() -> tuple[int, int]:
    b = Q.CGDisplayBounds(Q.CGMainDisplayID())
    return int(b.size.width), int(b.size.height)


def _request_access() -> None:
    """macOS's Screen Recording question (once; after that it points to System Settings)."""
    effects.guard("permission", "ask macOS for Screen Recording", via=Q.CGRequestScreenCaptureAccess)
    Q.CGRequestScreenCaptureAccess()


def capture(region: tuple[int, int, int, int] | None = None) -> tuple[ImagePart, float]:
    """Return (png image, scale) where scale = points-per-image-pixel."""
    if not can_capture():
        _request_access()
        raise PermissionError(
            "Screen Recording permission is off. Enable it for this app in System Settings → "
            "Privacy & Security → Screen Recording, then try again."
        )
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = Path(f.name)
    cmd = ["screencapture", "-x", "-t", "png"]
    if region:
        x, y, w, h = region
        cmd += ["-R", f"{x},{y},{w},{h}"]
    cmd.append(str(path))
    subprocess.run(cmd, check=True, timeout=10)
    img = Image.open(path).convert("RGB")
    path.unlink(missing_ok=True)

    # Retina captures are 2x; map to points, then cap width.
    pw, _ = display_points()
    target_w = region[2] if region else pw
    target_w = min(target_w, _MAX_W)
    scale_img = img.width / target_w  # image px per output px
    if scale_img > 1.01:
        img = img.resize((target_w, round(img.height / scale_img)), Image.LANCZOS)
    points_per_px = (region[2] if region else pw) / img.width
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return ImagePart(buf.getvalue(), "image/png"), points_per_px


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
        _request_access()
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
