"""Screenshots (the vision fallback). Downscaled to logical *points* so the pixel
coordinates the model reads off the image are the same ones the input tools take."""

from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path

import Quartz as Q
from PIL import Image

from neo.providers.base import ImagePart

_MAX_W = 1600  # keep token cost sane; 14"/16" MBP logical widths are 1512/1728


def can_capture() -> bool:
    return bool(Q.CGPreflightScreenCaptureAccess())


def display_points() -> tuple[int, int]:
    b = Q.CGDisplayBounds(Q.CGMainDisplayID())
    return int(b.size.width), int(b.size.height)


def capture(region: tuple[int, int, int, int] | None = None) -> tuple[ImagePart, float]:
    """Return (png image, scale) where scale = points-per-image-pixel."""
    if not can_capture():
        Q.CGRequestScreenCaptureAccess()
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
