"""Screenshots (the vision fallback) and the small pictures NEO compares to tell whether an input
changed anything.

One capture, kept. Every picture is grabbed once at the display's own resolution (a Retina display:
2 pixels per point) and kept in memory as a :class:`Frame` (the pixels, where they are on the screen,
points per pixel, when). What the model gets is cut and scaled from it and encoded once; the board
finder, OCR and change checks read the same frame instead of capturing again.

What the model sees maps exactly onto the screen: a :class:`Capture` says how its pixels become
screen points (``screen = origin + pixel × ppp``), which is what a shot id remembers.

* A whole display: scaled to *points* (a Retina display halves exactly: ``Image.reduce(2)``, about
  3 ms) and at most ``_MAX_W`` wide, so on a 14" MacBook Pro pixels are points.
* A region or a zoom: the display's own pixels (0.5 pt per pixel on Retina) while they fit
  ``_ZOOM_MAX``; else points; a small crop is scaled up to ``_ZOOM_MIN`` (a model reads small text
  better big).
* A window: its own pixels (even where another window covers it, no shadow) with the same mapping,
  from the window server's bounds, so a window shot can be clicked through.
* Encoded per ``settings().shot_format``: JPEG q85 4:4:4 by default (4–5 ms, 215–366 KB), PNG
  (compress level 1) as the A/B switch back. Never PNG ``optimize=True`` (101–292 ms).

Where the pixels come from: ScreenCaptureKit in process (sck.py) when ``settings().sck_capture`` is
on and the framework is there, else the ``screencapture`` CLI writing an uncompressed TIFF (ImageIO
encodes a Retina frame in about 5 ms and PIL decodes it in about 6, against PNG's 38–125 and 22–66).
One capture at a time. Everything after the capture takes about 10 ms (it took 145–380 ms).
"""

from __future__ import annotations

import io
import logging
import math
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import Quartz as Q
from PIL import Image

from neo import effects
from neo.providers.base import ImagePart

log = logging.getLogger(__name__)

_MAX_W = 1600  # a whole display or a window: at most this wide (14"/16" MBP logical widths: 1512/1728)
_ZOOM_MAX = 1536  # a region or zoom keeps the display's own pixels up to this long edge…
_ZOOM_MIN = 768  # …and a smaller one is scaled up to at least this long edge
_AFTER_MAX = 1024  # a look after an action (medium detail): at most this long edge
ZOOM_W, ZOOM_H = 320, 200  # a zoom's default area, in points
JPEG_QUALITY = 85  # with 4:4:4 chroma: PSNR 38.5–41.5 dB against the lossless picture
_MAX_BYTES = 400_000  # the upload budget for one picture
# What each kind of picture asks the brain for (ImagePart.detail; a provider without such a knob ignores
# it). A retry after a miss may ask for "ultra" itself.
_DETAIL = {"full": "high", "region": "high", "zoom": "high", "window": "high", "after": "medium"}
_CAPTURE_WAIT_S = 12.0  # (the CLI's own timeout is 10 s)


def can_capture() -> bool:
    return bool(Q.CGPreflightScreenCaptureAccess())


@dataclass(frozen=True)
class Display:
    """One screen, in global points: ``index`` is screencapture's -D number (1 = the main one).
    ``did``: its CGDirectDisplayID (0: unknown); ``scale``: backing pixels per point (0: unknown)."""

    index: int
    x: float
    y: float
    w: float
    h: float
    did: int = 0
    scale: float = 0.0

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def words(self) -> str:
        return f"display {self.index}: {self.w:g}×{self.h:g} pt at ({self.x:g},{self.y:g})"


def _backing_scale(did: int) -> float:
    try:
        mode = Q.CGDisplayCopyDisplayMode(did)
        pts = Q.CGDisplayModeGetWidth(mode)
        return float(Q.CGDisplayModeGetPixelWidth(mode)) / pts if pts else 0.0
    except Exception:  # noqa: BLE001 - unknown: the picture itself tells
        return 0.0


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
        out.append(
            Display(i, float(b.origin.x), float(b.origin.y), float(b.size.width), float(b.size.height), d, _backing_scale(d))
        )
    return out


def display_at(x: float, y: float) -> Display | None:
    """The display a global point is on (None: off every screen)."""
    return next((d for d in displays() if d.contains(x, y)), None)


def display_points() -> tuple[int, int]:
    b = Q.CGDisplayBounds(Q.CGMainDisplayID())
    return int(b.size.width), int(b.size.height)


def _request_access() -> None:
    """macOS's Screen Recording question (once; after that it points to System Settings)."""
    effects.guard("permission", "ask macOS for Screen Recording", via=Q.CGRequestScreenCaptureAccess)
    Q.CGRequestScreenCaptureAccess()


def _need_permission() -> None:
    if not can_capture():
        _request_access()
        raise PermissionError(
            "Screen Recording permission is off. Enable it for this app in System Settings → "
            "Privacy & Security → Screen Recording, then try again."
        )


# ---- settings (read forgivingly: a typo or a missing key never stops a screenshot) ------------------


def _setting(name: str, default: object) -> object:
    try:
        from neo.config import settings

        v = getattr(settings(), name, None)
    except Exception:  # noqa: BLE001 - no settings here (a raw thread in library mode): the default
        return default
    return default if v is None else v


def shot_format() -> str:
    """"jpeg" (the default) or "png": settings().shot_format, the A/B switch back to lossless."""
    v = str(_setting("shot_format", "jpeg")).strip().lower()
    return "png" if v in ("png", "lossless") else "jpeg"


def _sck_on() -> bool:
    """In-process capture (ScreenCaptureKit) when settings().sck_capture says so and it is installed."""
    v = _setting("sck_capture", False)
    on = v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on", "auto")
    if not on:
        return False
    from neo.tools.computer import sck

    return sck.enabled()


def _image_limit() -> tuple[int, int] | None:
    """The largest picture the brain about to see it takes, as (width, height) in pixels, from
    neo.providers.image_limit() (an int: the longest edge; a pair: a box; None: NEO's own caps)."""
    try:
        from neo import providers

        fn = getattr(providers, "image_limit", None)
        lim = fn() if callable(fn) else None
    except Exception:  # noqa: BLE001 - unknown: NEO's own caps
        return None
    if isinstance(lim, bool) or lim is None:
        return None
    if isinstance(lim, int | float) and lim > 0:
        return int(lim), int(lim)
    if isinstance(lim, tuple | list) and len(lim) == 2 and all(isinstance(v, int | float) and v > 0 for v in lim):
        return int(lim[0]), int(lim[1])
    return None


# ---- the pixels ----------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Frame:
    """One capture as it came from the screen: ``image`` (RGB), whose pixel (0, 0) is at screen
    point (``x0``, ``y0``), ``ppp`` screen points per pixel (0.5 on Retina), taken at ``t``
    (time.monotonic). ``window_id``: a window's own pixels (0: whatever was on screen there)."""

    image: Image.Image
    x0: float
    y0: float
    ppp: float
    t: float = field(default_factory=time.monotonic)
    window_id: int = 0
    backend: str = "cli"  # "cli" (screencapture) or "sck" (in process)
    grab_ms: float = 0.0

    @property
    def rect(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0, self.image.width * self.ppp, self.image.height * self.ppp

    def age(self) -> float:
        return time.monotonic() - self.t

    def to_screen(self, ix: float, iy: float) -> tuple[float, float]:
        return self.x0 + ix * self.ppp, self.y0 + iy * self.ppp

    def box(self, x: float, y: float, w: float, h: float) -> tuple[int, int, int, int]:
        """The pixels of a rectangle in screen points (not clipped: PIL pads what's outside)."""
        k = 1.0 / self.ppp
        return (round((x - self.x0) * k), round((y - self.y0) * k), round((x + w - self.x0) * k), round((y + h - self.y0) * k))

    def covers(self, x: float, y: float, w: float, h: float) -> bool:
        fx, fy, fw, fh = self.rect
        return fx - 0.5 <= x and fy - 0.5 <= y and x + w <= fx + fw + 0.5 and y + h <= fy + fh + 0.5

    def cut(self, x: float, y: float, w: float, h: float) -> Frame:
        """The part of this frame showing a rectangle in screen points, clipped to the frame."""
        fx, fy, fw, fh = self.rect
        left, top = max(x, fx), max(y, fy)
        right, bottom = min(x + w, fx + fw), min(y + h, fy + fh)
        if right - left <= 0 or bottom - top <= 0:
            raise ValueError("that part isn't in the picture")
        l_, t_, r_, b_ = self.box(left, top, right - left, bottom - top)
        img = self.image.crop((l_, t_, max(r_, l_ + 1), max(b_, t_ + 1)))
        return Frame(img, self.x0 + l_ * self.ppp, self.y0 + t_ * self.ppp, self.ppp, self.t, self.window_id, self.backend)


@dataclass(frozen=True, eq=False)
class Capture:
    """A picture for a model and how its pixels map to the screen: screen = (x0, y0) + pixel × ppp.
    ``clickable`` False: the mapping isn't known (a window whose bounds didn't match its picture) —
    for looking at, not for clicking through."""

    part: ImagePart
    x0: float
    y0: float
    ppp: float
    w: int
    h: int
    frame: Frame
    kind: str = "full"
    clickable: bool = True
    post_ms: float = 0.0  # NEO's own work after the capture: cut, scale, encode

    def to_screen(self, ix: float, iy: float) -> tuple[float, float]:
        return self.x0 + ix * self.ppp, self.y0 + iy * self.ppp


_capture_lock = threading.Lock()  # one capture at a time (ScreenCaptureKit queues concurrent ones badly)
_last: Frame | None = None  # the newest picture taken for a model (not the small change samples)


def last_frame(max_age_s: float | None = None) -> Frame | None:
    """The full-resolution frame of the newest picture taken for a model — a screenshot, a zoom, a
    window, a look after an action (None: none yet, or older than ``max_age_s`` seconds)."""
    f = _last
    if f is None or (max_age_s is not None and f.age() > max_age_s):
        return None
    return f


def _keep(f: Frame) -> None:
    global _last
    _last = f


# screencapture's file format: an uncompressed TIFF. Measured on a synthetic 3024×1964 frame with the
# ImageIO encoder screencapture writes through, plus PIL's decode: TIFF 11–12 ms, BMP 14, JPEG 22–29
# (and lossy twice), PNG 60–183. Any failure with it switches to PNG for the rest of the run.
_CLI_FORMAT = "tiff"
_cli_format = _CLI_FORMAT


def _grab(cmd_extra: list[str]) -> Image.Image:
    """One ``screencapture`` run (``cmd_extra``: -R x,y,w,h · -D n · -o -l id) → its picture, RGB."""
    global _cli_format
    fmt = _cli_format
    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as f:
        path = Path(f.name)
    try:
        if not _capture_lock.acquire(timeout=_CAPTURE_WAIT_S):
            raise TimeoutError("another screen capture is still running")
        try:
            subprocess.run(["screencapture", "-x", "-t", fmt, *cmd_extra, str(path)], check=True, timeout=10)
        except subprocess.CalledProcessError as e:
            failed: Exception | None = e
        else:
            failed = None
        finally:
            _capture_lock.release()
        if failed is None:
            try:
                return Image.open(path).convert("RGB")  # (loads it, before the file goes)
            except OSError as e:  # (a file PIL can't read)
                failed = e
        if fmt == "png":
            raise failed
        log.warning("screencapture -t %s failed (%s): PNG from now on", fmt, failed)
        _cli_format = "png"
    finally:
        path.unlink(missing_ok=True)
    return _grab(cmd_extra)


def _grab_rect(x: int, y: int, w: int, h: int, *, fallback: bool = True) -> tuple[Image.Image, str]:
    """The pixels of a rectangle in global points (on one display), at the display's resolution.
    ``fallback`` False: when the in-process capture is on and fails, no picture (sck.Failed) rather
    than the CLI's — a before/after pair of change samples must come one way to be compared."""
    if _sck_on():
        from neo.tools.computer import sck

        d = display_at(x + w / 2, y + h / 2) or display_at(x, y)
        try:
            if d is None:
                raise sck.Failed("that rectangle isn't on a display")
            return sck.capture_rect(x, y, w, h, d), "sck"
        except sck.Failed as e:
            if not fallback:
                raise
            log.info("in-process capture failed (%s): the CLI instead", e)
    return _grab(["-R", f"{x},{y},{w},{h}"]), "cli"


def _grab_display(d: Display) -> tuple[Image.Image, str]:
    if _sck_on():
        from neo.tools.computer import sck

        try:
            return sck.capture_rect(d.x, d.y, d.w, d.h, d), "sck"
        except sck.Failed as e:
            log.info("in-process capture failed (%s): the CLI instead", e)
    return _grab(["-D", str(d.index)] if d.index != 1 else []), "cli"


def _grab_window(window_id: int) -> tuple[Image.Image, str]:
    if _sck_on():
        from neo.tools.computer import sck

        try:
            return sck.capture_window(window_id), "sck"
        except sck.Failed as e:
            log.info("in-process window capture failed (%s): the CLI instead", e)
    return _grab(["-o", "-l", str(int(window_id))]), "cli"


def grab(region: tuple[int, int, int, int] | None = None, display: Display | None = None) -> Frame:
    """A capture at the display's own resolution, as a Frame (nothing encoded). ``region``: a
    rectangle in global points (on one display); else the whole of ``display`` (the main one)."""
    _need_permission()
    t = time.perf_counter()
    if region:
        x, y, w, h = (int(v) for v in region)
        img, backend = _grab_rect(x, y, w, h)
        x0, y0, width = float(x), float(y), float(w)
    else:
        d = display or displays()[0]
        img, backend = _grab_display(d)
        x0, y0, width = d.x, d.y, d.w
    ms = (time.perf_counter() - t) * 1000
    return Frame(img, x0, y0, width / max(1, img.width), backend=backend, grab_ms=ms)


# ---- what the model gets -------------------------------------------------------------------------


def encode(img: Image.Image, fmt: str | None = None) -> tuple[bytes, str]:
    """(bytes, mime): JPEG q85 4:4:4, or PNG at compress level 1 — never PNG's optimize (100–290 ms).
    A JPEG past the upload budget (a photo, noise) is made again at q75 4:2:0: about half the bytes."""
    buf = io.BytesIO()
    if (fmt or shot_format()) == "png":
        img.save(buf, format="PNG", compress_level=1)
        return buf.getvalue(), "image/png"
    rgb = img if img.mode == "RGB" else img.convert("RGB")
    rgb.save(buf, format="JPEG", quality=JPEG_QUALITY, subsampling=0)
    if buf.tell() > _MAX_BYTES:
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=75, subsampling=2)
    return buf.getvalue(), "image/jpeg"


def _part(data: bytes, mime: str, detail: str) -> ImagePart:
    """An ImagePart asking for ``detail`` (ImagePart.detail: Step 0 adds the field; until then it
    rides along as an attribute, which providers read with getattr)."""
    try:
        return ImagePart(data, mime, detail=detail)  # type: ignore[call-arg]
    except TypeError:
        part = ImagePart(data, mime)
        part.detail = detail  # type: ignore[attr-defined]
        return part


def _size(frame: Frame, kind: str, limit: tuple[int, int] | None) -> tuple[int, int]:
    """The picture's size for a model. A whole display: points, at most _MAX_W wide. A region, a
    zoom, a window, an after-look: the display's own pixels while they fit the kind's cap, else
    points while those fit, else scaled to the cap; a small region or zoom grows to _ZOOM_MIN."""
    iw, ih = frame.image.size
    k = 1.0 / frame.ppp  # pixels per point (2 on Retina)
    if kind == "full":
        s = min(1.0, 1.0 / k if k > 1.0 else 1.0, _MAX_W / iw)
    else:
        cap = {"window": _MAX_W, "after": _AFTER_MAX}.get(kind, _ZOOM_MAX)
        long = max(iw, ih)
        if long <= cap:
            s = 1.0
        elif k > 1.0 and long / k <= cap:
            s = 1.0 / k
        else:
            s = cap / long
        if kind in ("region", "zoom") and long * s < _ZOOM_MIN:
            s = _ZOOM_MIN / long
    if limit:
        s = min(s, limit[0] / iw, limit[1] / ih)
    return max(1, round(iw * s)), max(1, round(ih * s))


def _resize(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Down by a whole factor with reduce() (a Retina halving: ~3 ms, against LANCZOS's ~50), the rest
    BILINEAR; up (a small crop) with LANCZOS."""
    if img.size == size:
        return img
    tw, th = size
    if tw > img.width:
        return img.resize(size, Image.LANCZOS)
    n = int(img.width / tw + 1e-6)
    if n >= 2 and img.width // n >= tw and img.height // n >= th:
        img = img.reduce(n)
    return img if img.size == size else img.resize(size, Image.BILINEAR)


def render(
    frame: Frame,
    kind: str = "full",
    *,
    rect: tuple[float, float, float, float] | None = None,
    detail: str | None = None,
    fmt: str | None = None,
    clickable: bool = True,
) -> Capture:
    """The picture a model gets from ``frame`` (or from its part showing ``rect``, screen points):
    scaled for ``kind`` ("full", "region", "zoom", "window", "after") and the brain's image limit,
    encoded per the shot format, with the mapping of its pixels to the screen."""
    t = time.perf_counter()
    src = frame.cut(*rect) if rect is not None else frame
    img = _resize(src.image, _size(src, kind, _image_limit()))
    data, mime = encode(img, fmt)
    ppp = src.image.width * src.ppp / img.width
    part = _part(data, mime, detail or _DETAIL.get(kind, "high"))
    return Capture(part, src.x0, src.y0, ppp, img.width, img.height, frame, kind, clickable, post_ms=(time.perf_counter() - t) * 1000)


def shot(
    region: tuple[int, int, int, int] | None = None,
    display: Display | None = None,
    *,
    kind: str | None = None,
    detail: str | None = None,
    fmt: str | None = None,
) -> Capture:
    """A screenshot: the whole of ``display`` (the main one when None), or ``region`` (global points,
    on one display). Its frame is kept (last_frame) for the board finder, OCR and change checks."""
    frame = grab(region, display)
    _keep(frame)
    return render(frame, kind or ("region" if region else "full"), detail=detail, fmt=fmt)


def capture(region: tuple[int, int, int, int] | None = None, display: Display | None = None) -> tuple[ImagePart, float]:
    """(the picture, points per image pixel): the screenshot tool's call. ``region``: a rectangle in
    global points; else the whole of ``display`` (the main one when None). The mapping's origin is
    the region's top-left corner, or the display's."""
    c = shot(region, display)
    return c.part, c.ppp


def zoom(
    x: float, y: float, w: float = ZOOM_W, h: float = ZOOM_H, *, display: Display | None = None, detail: str = "high"
) -> Capture:
    """A close look around a point (screen points): ``w``×``h`` points centred on it, moved to stay on
    its display, at the display's own resolution (a small one scaled up to _ZOOM_MIN)."""
    d = display or display_at(x, y) or displays()[0]
    w, h = max(8.0, min(float(w), d.w)), max(8.0, min(float(h), d.h))
    left = min(max(x - w / 2, d.x), d.x + d.w - w)
    top = min(max(y - h / 2, d.y), d.y + d.h - h)
    return shot((round(left), round(top), round(w), round(h)), kind="zoom", detail=detail)


def _whole(left: float, top: float, right: float, bottom: float) -> tuple[int, int, int, int]:
    """The rectangle of whole points (x, y, width, height) that covers these edges."""
    x, y = math.floor(left), math.floor(top)
    return x, y, max(1, math.ceil(right) - x), max(1, math.ceil(bottom) - y)


def after_look(
    rect: tuple[float, float, float, float],
    points: list[tuple[float, float]] | None = None,
    r: float | None = None,
    *,
    fmt: str | None = None,
    like: list[Image.Image] | None = None,
) -> tuple[Capture, list[Image.Image] | None]:
    """One capture for a look after an action: the picture of ``rect`` (medium detail) and the change
    samples around ``points`` (see sample_points), all cut from the same frame, the way sample_points
    cuts them (a sample of an unchanged screen is the same either way). ``like``: the samples taken
    before — no samples come back when this frame was taken another way (can't compare). A ``rect``
    off the points' display: ValueError."""
    _need_permission()
    x, y, w, h = rect
    pts = list(points or [])
    rr = _SAMPLE_R if r is None else r
    if pts:
        left = min([x] + [p[0] - rr for p in pts])
        top = min([y] + [p[1] - rr for p in pts])
        right = max([x + w] + [p[0] + rr for p in pts])
        bottom = max([y + h] + [p[1] + rr for p in pts])
    else:
        left, top, right, bottom = x, y, x + w, y + h
    home = display_at(*pts[0]) if pts else display_at(x + w / 2, y + h / 2) or display_at(x, y)
    if home is not None:  # (only what is on that screen is in the picture: the one sample_points uses)
        left, top = max(left, home.x), max(top, home.y)
        right, bottom = min(right, home.x + home.w), min(bottom, home.y + home.h)
    frame = grab(_whole(left, top, right, bottom))
    _keep(frame)
    cap = render(frame, "after", rect=rect, fmt=fmt)
    if not pts or _via(like) not in ("", frame.backend):
        return cap, None
    return cap, samples_from(frame, pts, rr)


# ---- windows -------------------------------------------------------------------------------------

_MIN_CUT = 50  # points: a part of a window smaller than this is no page to look at


def _window_bounds(window_id: int) -> tuple[float, float, float, float] | None:
    """(x, y, width, height) of a window on screen, in points from the top left (None: unknown)."""
    try:
        info = Q.CGWindowListCopyWindowInfo(Q.kCGWindowListOptionIncludingWindow, int(window_id)) or []
        b = info[0].get("kCGWindowBounds") if len(info) else None
        return (float(b["X"]), float(b["Y"]), float(b["Width"]), float(b["Height"])) if b else None
    except Exception:  # noqa: BLE001 - no bounds: the window is captured whole
        return None


def _main_scale() -> float:
    try:
        return _backing_scale(int(Q.CGMainDisplayID())) or 2.0
    except Exception:  # noqa: BLE001
        return 2.0


def window_shot(
    window_id: int, region: tuple[float, float, float, float] | None = None, *, detail: str = "high", fmt: str | None = None
) -> Capture:
    """One window's own pixels (even where another window covers it; no shadow), at most _MAX_W on
    its long edge, mapped to the screen through the window server's bounds for the window (clickable
    when the picture matches them). ``region`` (screen points, x y width height): only that part of it
    — a page without the browser's tabs — when at least _MIN_CUT points of it are in the window."""
    _need_permission()
    was = _window_bounds(window_id)
    t = time.perf_counter()
    img, backend = _grab_window(window_id)
    grab_ms = (time.perf_counter() - t) * 1000
    bounds = _window_bounds(window_id)
    got = {"window_id": int(window_id), "backend": backend, "grab_ms": grab_ms}
    clickable = False
    if bounds is not None and bounds[2] > 0 and bounds[3] > 0:
        kx, ky = img.width / bounds[2], img.height / bounds[3]
        # (a picture with a shadow, of another size, or of a window that moved meanwhile: no mapping)
        clickable = was == bounds and abs(kx - ky) <= 0.02 * max(kx, ky)
        frame = Frame(img, bounds[0], bounds[1], bounds[2] / img.width, **got)
    else:
        frame = Frame(img, 0.0, 0.0, 1.0 / _main_scale(), **got)
    cut = None
    if region is not None and bounds is not None and bounds[2] > 0:
        x, y, w, h = bounds
        left, top = max(region[0], x), max(region[1], y)
        right, bottom = min(region[0] + region[2], x + w), min(region[1] + region[3], y + h)
        if right - left >= _MIN_CUT and bottom - top >= _MIN_CUT:
            cut = (left, top, right - left, bottom - top)
    _keep(frame)
    return render(frame, "window", rect=cut, detail=detail, fmt=fmt, clickable=clickable)


def capture_window(window_id: int, region: tuple[float, float, float, float] | None = None) -> ImagePart:
    """A picture of one window (see window_shot), for looking at: no coordinates come with it."""
    return window_shot(window_id, region).part


# ---- did anything change? --------------------------------------------------------------------------
# A click reports what it did: a small picture of the screen around the point, taken just before
# and just after, compared at low resolution. The pointer isn't in the pictures, and NEO's own window
# never is (it hides from screen capture; the in-process capture leaves it out by name and pid), so
# neither counts as a change.
_SAMPLE_R = 60  # points around each point
_SAMPLE_GRID = 24  # each sample is compared as this many × this many grey cells
_CELL_DIFF = 18  # a cell that moved this much (0-255) changed…
_CELLS_CHANGED = 3  # …and this many changed cells are a change


def samples_from(frame: Frame, points: list[tuple[float, float]], r: float = _SAMPLE_R) -> list[Image.Image]:
    """The change samples around each point, cut from a frame that shows them (no capture). Each
    says which capture path it came from (``info["via"]``): only samples of one path compare."""
    out = []
    for px, py in points:
        box = frame.box(px - r, py - r, 2 * r, 2 * r)
        sample = frame.image.crop(box).convert("L").resize((_SAMPLE_GRID, _SAMPLE_GRID), Image.BILINEAR)
        sample.info["via"] = frame.backend
        out.append(sample)
    return out


def _via(like: list[Image.Image] | None) -> str:
    return str(like[0].info.get("via", "")) if like else ""


def sample_points(
    points: list[tuple[float, float]],
    r: float = _SAMPLE_R,
    frame: Frame | None = None,
    like: list[Image.Image] | None = None,
) -> list[Image.Image] | None:
    """A small grey picture of the screen around each point (None without Screen Recording, or
    when the capture fails: nothing can be said about change). Never asks for the permission. All
    the points come from one capture; with ``frame`` (one taken after the input, showing them all)
    from that frame, with no capture at all. ``like``: the samples these will be compared with —
    taken the same way (the in-process capture and the CLI see colours differently), or None."""
    try:
        if not points:
            return None
        via = _via(like)
        if frame is not None and all(frame.covers(p[0] - r, p[1] - r, 2 * r, 2 * r) for p in points):
            return samples_from(frame, points, r) if via in ("", frame.backend) else None
        if not can_capture():
            return None
        left = min(p[0] for p in points) - r
        top = min(p[1] for p in points) - r
        right = max(p[0] for p in points) + r
        bottom = max(p[1] for p in points) + r
        home = display_at(*points[0])
        if home is not None:  # only what's on that screen is in the picture
            left, top = max(left, home.x), max(top, home.y)
            right, bottom = min(right, home.x + home.w), min(bottom, home.y + home.h)
        if right - left < 2 or bottom - top < 2:
            return None
        x, y, w, h = _whole(left, top, right, bottom)
        if via == "cli":
            img, backend = _grab(["-R", f"{x},{y},{w},{h}"]), "cli"
        else:  # (the in-process path when it's on, with no CLI fallback: a pair never mixes the two)
            img, backend = _grab_rect(x, y, w, h, fallback=False)
            if via and backend != via:
                return None
        return samples_from(Frame(img, float(x), float(y), w / max(1, img.width), backend=backend), points, r)
    except Exception:  # noqa: BLE001 - no picture: unknown, not "unchanged"
        return None


def differs(a: Image.Image, b: Image.Image) -> bool:
    """Do two samples of the same place show something different? A text caret blinking near the
    point — one thin upright strip of cells, nothing else — is not a change (a click that moved the
    keyboard focus says so through Accessibility)."""
    if a.size != b.size:
        return True
    pa, pb = a.convert("L").tobytes(), b.convert("L").tobytes()
    width = a.size[0]
    cells = [i for i, (u, v) in enumerate(zip(pa, pb, strict=True)) if abs(int(u) - int(v)) > _CELL_DIFF]
    if len(cells) < _CELLS_CHANGED:
        return False
    cols = {i % width for i in cells}
    rows = {i // width for i in cells}
    wide, tall = max(cols) - min(cols) + 1, max(rows) - min(rows) + 1
    return not (wide <= 2 and tall >= 2 * wide)  # (a caret straddling two cells is two columns wide)
