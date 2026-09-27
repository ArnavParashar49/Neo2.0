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
  ``_ZOOM_MAX`` and ``_MAX_AREA`` (about a whole display's worth); else points; a small crop is
  scaled up to ``_ZOOM_MIN`` (a model reads small text better big).
* A window: its own pixels (even where another window covers it, no shadow) with the same mapping,
  from the window server's bounds, so a window shot can be clicked through — except where another
  window was over it on the screen (``Capture.covered``: a click there lands on that window).
* Encoded per ``settings().shot_format``: JPEG q85 4:4:4 by default (4–5 ms, 215–366 KB), PNG
  (compress level 1) as the A/B switch back. Never PNG ``optimize=True`` (101–292 ms). A JPEG still
  past the upload budget is made smaller, never blurrier: q85 4:4:4 always.

Where the pixels come from: ScreenCaptureKit in process (sck.py) when ``settings().sck_capture`` is
on and the framework is there, else the ``screencapture`` CLI writing an uncompressed TIFF (ImageIO
encodes a Retina frame in about 5 ms and PIL decodes it in about 6, against PNG's 38–125 and 22–66).
One capture at a time. Everything after the capture takes about 10 ms (it took 145–380 ms).
"""

from __future__ import annotations

import io
import math
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import Quartz as Q
from PIL import Image

from neo2.core import effects, logger
from neo2.core.providers.base import ImagePart

log = logger(__name__)

_MAX_W = 1600  # a whole display or a window: at most this wide (14"/16" MBP logical widths: 1512/1728)
_ZOOM_MAX = 1536  # a region or zoom keeps the display's own pixels up to this long edge…
_ZOOM_MIN = 768  # …and a smaller one is scaled up to at least this long edge
_MAX_AREA = 1_600_000  # a window, region or zoom: at most about a whole display's pixels (1512×982)
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
        from neo2.core.config import settings

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
    from neo2.core.tools.computer import sck

    return sck.enabled()


def _brains_limit() -> object:
    """neo2.core.providers.image_limit() — the strictest limit of the brains that may see the picture
    (an ImageLimit, or a pair / an int where a host or a test declares one that way) — or None."""
    try:
        from neo2.core import providers

        fn = getattr(providers, "image_limit", None)
        return fn() if callable(fn) else None
    except Exception:  # noqa: BLE001 - unknown: NEO's own caps
        return None


def _image_limit(lim: object = ...) -> tuple[int, int] | None:
    """The largest picture the brain about to see it takes as sent: (longest edge in pixels — 0: no
    cap on the edge, total pixels — 0: no cap on the area), from neo2.core.providers.image_limit()
    (an ImageLimit; a pair in that order; an int: the longest edge only; None: NEO's own caps)."""
    if lim is ...:
        lim = _brains_limit()
    if isinstance(lim, bool) or lim is None:
        return None
    edge = getattr(lim, "long_edge", ...)
    if edge is not ...:  # (an ImageLimit: its edge and pixel caps; its bytes are _byte_budget's)
        e, px = edge or 0, getattr(lim, "pixels", None) or 0
        ok = all(isinstance(v, int | float) and not isinstance(v, bool) and v >= 0 for v in (e, px))
        return (int(e), int(px)) if ok and (e or px) else None
    if isinstance(lim, int | float) and lim > 0:
        return int(lim), 0
    if isinstance(lim, tuple | list) and len(lim) == 2 and all(isinstance(v, int | float) and v > 0 for v in lim):
        return int(lim[0]), int(lim[1])
    return None


def _byte_budget(lim: object = ...) -> int:
    """The upload budget for one picture: NEO's own (_MAX_BYTES), or less when a brain that may see
    it takes less (ImageLimit.bytes)."""
    if lim is ...:
        lim = _brains_limit()
    b = getattr(lim, "bytes", None)
    if isinstance(b, int) and not isinstance(b, bool) and b > 0:
        return max(16_000, min(_MAX_BYTES, b))
    return _MAX_BYTES


# ---- the pixels ----------------------------------------------------------------------------------


Rect = tuple[float, float, float, float]  # x, y, width, height in screen points


def covered_at(covered: tuple[Rect, ...] | list[Rect], x: float, y: float) -> bool:
    """Is screen point (x, y) in one of these rectangles (the parts of a window others were over)?"""
    return any(cx <= x < cx + cw and cy <= y < cy + ch for cx, cy, cw, ch in covered)


def _overlaps(rects: list[Rect] | tuple[Rect, ...], area: Rect) -> tuple[Rect, ...]:
    """The parts of ``rects`` inside ``area``."""
    ax_, ay, aw, ah = area
    out = []
    for x, y, w, h in rects:
        left, top = max(x, ax_), max(y, ay)
        right, bottom = min(x + w, ax_ + aw), min(y + h, ay + ah)
        if right > left and bottom > top:
            out.append((left, top, right - left, bottom - top))
    return tuple(out)


@dataclass(frozen=True, eq=False)
class Frame:
    """One capture as it came from the screen: ``image`` (RGB), whose pixel (0, 0) is at screen
    point (``x0``, ``y0``), ``ppp`` screen points per pixel (0.5 on Retina), taken at ``t``
    (time.monotonic, when the capture began).

    ``window_id``: a window's own pixels (0: whatever was on screen there), owned by ``pid``. They
    include what other windows were over on the screen (``covered``, screen points): only where
    :meth:`shows` says so is a pixel what the screen showed. ``mapped`` False: where the pixels are
    on the screen isn't known (x0, y0, ppp are a guess) — never map a point through it."""

    image: Image.Image
    x0: float
    y0: float
    ppp: float
    t: float = field(default_factory=time.monotonic)
    window_id: int = 0
    backend: str = "cli"  # "cli" (screencapture) or "sck" (in process)
    grab_ms: float = 0.0
    pid: int = 0
    covered: tuple[Rect, ...] = ()
    mapped: bool = True

    @property
    def screen_like(self) -> bool:
        """What the screen showed, everywhere in it: mapped, and no window's own pixels."""
        return self.mapped and not self.window_id

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

    def shows(self, x: float, y: float) -> bool:
        """Is this frame's pixel at screen point (x, y) what the screen showed there when it was
        taken? In the frame, mapped, and (a window's own pixels) no other window over it."""
        fx, fy, fw, fh = self.rect
        return self.mapped and fx <= x < fx + fw and fy <= y < fy + fh and not covered_at(self.covered, x, y)

    def cut(self, x: float, y: float, w: float, h: float) -> Frame:
        """The part of this frame showing a rectangle in screen points, clipped to the frame."""
        fx, fy, fw, fh = self.rect
        left, top = max(x, fx), max(y, fy)
        right, bottom = min(x + w, fx + fw), min(y + h, fy + fh)
        if right - left <= 0 or bottom - top <= 0:
            raise ValueError("that part isn't in the picture")
        l_, t_, r_, b_ = self.box(left, top, right - left, bottom - top)
        img = self.image.crop((l_, t_, max(r_, l_ + 1), max(b_, t_ + 1)))
        return replace(self, image=img, x0=self.x0 + l_ * self.ppp, y0=self.y0 + t_ * self.ppp)


@dataclass(frozen=True, eq=False)
class Capture:
    """A picture for a model and how its pixels map to the screen: screen = (x0, y0) + pixel × ppp.
    ``clickable`` False: the mapping isn't known (a window whose bounds didn't match its picture, or
    that isn't on the screen) — for looking at, not for clicking through.

    A window shot (``window_id``, owned by ``pid``) shows the window's own pixels, also where other
    windows were over it on the screen (``covered``, screen points, cut to the picture): a click
    there would land on the window on top. :meth:`clear_at` says where a click lands on what the
    picture shows."""

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
    window_id: int = 0
    pid: int = 0
    covered: tuple[Rect, ...] = ()

    def to_screen(self, ix: float, iy: float) -> tuple[float, float]:
        return self.x0 + ix * self.ppp, self.y0 + iy * self.ppp

    def clear_at(self, sx: float, sy: float) -> bool:
        """Does the screen show at point (sx, sy) what this picture shows there — a click through
        it lands on what the model saw? Clickable, in the picture, and no other window over it."""
        inside = self.x0 <= sx < self.x0 + self.w * self.ppp and self.y0 <= sy < self.y0 + self.h * self.ppp
        return self.clickable and inside and not covered_at(self.covered, sx, sy)


_capture_lock = threading.Lock()  # one capture at a time (ScreenCaptureKit queues concurrent ones badly)
_last: Frame | None = None  # the newest picture taken for a model (not the small change samples)


def last_frame(max_age_s: float | None = None) -> Frame | None:
    """The full-resolution frame of the newest picture taken for a model — a screenshot, a zoom, a
    window, a look after an action (None: none yet, older than ``max_age_s`` seconds, or the newest
    picture has no known place on the screen: a window shot that isn't mapped is never kept).

    Before mapping a point through it to act there, a consumer checks ``window_id``: a window's own
    pixels are the screen only where ``frame.shows(x, y)``; change samples need ``screen_like``;
    ``backend`` says which capture path it came from (samples of different paths don't compare)."""
    f = _last
    if f is None or (max_age_s is not None and f.age() > max_age_s):
        return None
    return f


def _keep(f: Frame | None) -> None:
    """The newest picture's frame (None: the newest picture has none that maps)."""
    global _last
    _last = f if f is not None and f.mapped else None


# screencapture's file format: an uncompressed TIFF. Measured on a synthetic 3024×1964 frame with the
# ImageIO encoder screencapture writes through, plus PIL's decode: TIFF 11–12 ms, BMP 14, JPEG 22–29
# (and lossy twice), PNG 60–183. It switches to PNG for the rest of the run only once PNG is shown to
# work where TIFF didn't: a TIFF PIL can't read, or a whole display that failed as TIFF and not as
# PNG. A window that closed or a region off the screen fails either way: one run, and TIFF stays.
_CLI_FORMAT = "tiff"
_cli_format = _CLI_FORMAT


class _Unreadable(OSError):
    """screencapture wrote a file PIL can't read."""


def _cli_run(fmt: str, cmd_extra: list[str]) -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as f:
        path = Path(f.name)
    try:
        if not _capture_lock.acquire(timeout=_CAPTURE_WAIT_S):
            raise TimeoutError("another screen capture is still running")
        try:
            subprocess.run(["screencapture", "-x", "-t", fmt, *cmd_extra, str(path)], check=True, timeout=10)
        finally:
            _capture_lock.release()
        try:
            return Image.open(path).convert("RGB")  # (loads it, before the file goes)
        except OSError as e:
            raise _Unreadable(f"screencapture -t {fmt} wrote a picture that can't be read ({e})") from e
    finally:
        path.unlink(missing_ok=True)


def _grab(cmd_extra: list[str]) -> Image.Image:
    """One ``screencapture`` run (``cmd_extra``: -R x,y,w,h · -D n · -o -l id) → its picture, RGB."""
    global _cli_format
    fmt = _cli_format
    try:
        return _cli_run(fmt, cmd_extra)
    except _Unreadable as e:
        if fmt == "png":
            raise
        failed: Exception = e
    except subprocess.CalledProcessError as e:
        if fmt == "png" or "-l" in cmd_extra or "-R" in cmd_extra:
            raise  # (a window or a region: what failed is it, not the format)
        failed = e
    img = _cli_run("png", cmd_extra)  # (fails too: nothing to do with the format, which stays)
    log.warning("screencapture -t %s failed (%s) where PNG works: PNG from now on", fmt, failed)
    _cli_format = "png"
    return img


def _grab_rect(
    x: int, y: int, w: int, h: int, *, fallback: bool = True, display: Display | None = None
) -> tuple[Image.Image, str]:
    """The pixels of a rectangle in global points (on one display, ``display`` when known), at the
    display's resolution. ``fallback`` False: when the in-process capture is on and fails, no
    picture (sck.Failed) rather than the CLI's — a before/after pair of change samples must come
    one way to be compared."""
    if _sck_on():
        from neo2.core.tools.computer import sck

        d = display or display_at(x + w / 2, y + h / 2) or display_at(x, y)
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
        from neo2.core.tools.computer import sck

        try:
            return sck.capture_rect(d.x, d.y, d.w, d.h, d), "sck"
        except sck.Failed as e:
            log.info("in-process capture failed (%s): the CLI instead", e)
    return _grab(["-D", str(d.index)] if d.index != 1 else []), "cli"


def _grab_window(window_id: int) -> tuple[Image.Image, str]:
    if _sck_on():
        from neo2.core.tools.computer import sck

        try:
            return sck.capture_window(window_id), "sck"
        except sck.Failed as e:
            log.info("in-process window capture failed (%s): the CLI instead", e)
    return _grab(["-o", "-l", str(int(window_id))]), "cli"


def _on_display(region: tuple[int, int, int, int]) -> tuple[tuple[int, int, int, int], Display | None]:
    """(a region cut to the display it is on, that display): its centre's display, else its
    top-left corner's. Only what is on that screen is ever in a picture of it, whichever way it is
    captured. Off every display: as it is, and None."""
    x, y, w, h = (int(v) for v in region)
    d = display_at(x + w / 2, y + h / 2) or display_at(x, y)
    if d is None:
        return (x, y, w, h), None
    left, top = max(x, d.x), max(y, d.y)
    right, bottom = min(x + w, d.x + d.w), min(y + h, d.y + d.h)
    if right - left < 1 or bottom - top < 1:
        return (x, y, w, h), None
    return _whole(left, top, right, bottom), d


def grab(region: tuple[int, int, int, int] | None = None, display: Display | None = None) -> Frame:
    """A capture at the display's own resolution, as a Frame (nothing encoded). ``region``: a
    rectangle in global points, cut to its display; else the whole of ``display`` (the main one).
    The frame's time is when the capture began."""
    _need_permission()
    t0, t = time.monotonic(), time.perf_counter()
    if region:
        (x, y, w, h), home = _on_display(region)
        img, backend = _grab_rect(x, y, w, h, display=home)
        x0, y0, width = float(x), float(y), float(w)
    else:
        d = display or displays()[0]
        img, backend = _grab_display(d)
        x0, y0, width = d.x, d.y, d.w
    ms = (time.perf_counter() - t) * 1000
    return Frame(img, x0, y0, width / max(1, img.width), t0, backend=backend, grab_ms=ms)


# ---- what the model gets -------------------------------------------------------------------------


def encode(img: Image.Image, fmt: str | None = None) -> tuple[bytes, str]:
    """(bytes, mime): JPEG q85 4:4:4, or PNG at compress level 1 — never PNG's optimize (100–290 ms)."""
    buf = io.BytesIO()
    if (fmt or shot_format()) == "png":
        img.save(buf, format="PNG", compress_level=1)
        return buf.getvalue(), "image/png"
    rgb = img if img.mode == "RGB" else img.convert("RGB")
    rgb.save(buf, format="JPEG", quality=JPEG_QUALITY, subsampling=0)
    return buf.getvalue(), "image/jpeg"


def _fit(img: Image.Image, fmt: str | None, budget: int = _MAX_BYTES) -> tuple[Image.Image, bytes, str]:
    """(the picture sent, its bytes, mime). A JPEG past the upload budget (a photo, noise) is made
    smaller until it fits, at the same q85 4:4:4 (the quality the A/B test fixes): the caller maps
    from the picture that is sent. A PNG (the lossless arm of the A/B test) is sent as it is."""
    src, k = img, 1.0
    data, mime = encode(img, fmt)
    for _ in range(6):
        if mime != "image/jpeg" or len(data) <= budget or min(img.size) <= 16:
            break
        k *= math.sqrt(budget / len(data)) * 0.95  # (bytes go about with the area)
        # (always from the first picture: no blur on blur; BOX, an area average, is the quickest
        # clean way down by a little — 4 ms for 1.5 MP, against BILINEAR's 7)
        img = src.resize((max(1, round(src.width * k)), max(1, round(src.height * k))), Image.BOX)
        data, mime = encode(img, fmt)
    return img, data, mime


def _part(data: bytes, mime: str, detail: str) -> ImagePart:
    """An ImagePart asking for ``detail`` (ImagePart.detail)."""
    return ImagePart(data, mime, detail=detail)  # type: ignore[arg-type]


def _size(frame: Frame, kind: str, limit: tuple[int, int] | None) -> tuple[int, int]:
    """The picture's size for a model. A whole display: points, at most _MAX_W wide. A region, a
    zoom, a window, an after-look: the display's own pixels while they fit the kind's cap on the
    long edge and _MAX_AREA (so a q85 JPEG stays near the upload budget), else points while those
    fit, else scaled to fit; a small region or zoom grows to _ZOOM_MIN."""
    iw, ih = frame.image.size
    k = 1.0 / frame.ppp  # pixels per point (2 on Retina)
    if kind == "full":
        s = min(1.0, 1.0 / k if k > 1.0 else 1.0, _MAX_W / iw)
    else:
        cap = {"window": _MAX_W, "after": _AFTER_MAX}.get(kind, _ZOOM_MAX)
        long, area = max(iw, ih), iw * ih

        def fits(s: float) -> bool:
            return long * s <= cap and area * s * s <= _MAX_AREA

        if fits(1.0):
            s = 1.0
        elif k > 1.0 and fits(1.0 / k):
            s = 1.0 / k
        else:
            s = min(cap / long, math.sqrt(_MAX_AREA / area))
        if kind in ("region", "zoom") and long * s < _ZOOM_MIN:
            s = _ZOOM_MIN / long
    if limit:  # (longest edge, total pixels): the brain's own limit, so it never shrinks the picture itself
        if limit[0] > 0:
            s = min(s, limit[0] / max(iw, ih))
        if limit[1] > 0:
            s = min(s, math.sqrt(limit[1] / (iw * ih)))
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
    lim = _brains_limit()
    img, data, mime = _fit(_resize(src.image, _size(src, kind, _image_limit(lim))), fmt, _byte_budget(lim))
    ppp = src.image.width * src.ppp / img.width
    part = _part(data, mime, detail or _DETAIL.get(kind, "high"))
    return Capture(
        part, src.x0, src.y0, ppp, img.width, img.height, frame, kind, clickable and frame.mapped,
        post_ms=(time.perf_counter() - t) * 1000, window_id=frame.window_id, pid=frame.pid,
        covered=_overlaps(frame.covered, src.rect),
    )


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
    frame, samples = look_frame(rect, points, r, like=like)
    return render(frame, "after", rect=rect, fmt=fmt), samples


def look_frame(
    rect: tuple[float, float, float, float],
    points: list[tuple[float, float]] | None = None,
    r: float | None = None,
    *,
    like: list[Image.Image] | None = None,
    samples: bool = True,
) -> tuple[Frame, list[Image.Image] | None]:
    """after_look's capture without the picture for a model: (the frame showing ``rect`` and every
    sample's box, the change samples cut from it — None when ``like`` came another way, or
    ``samples`` is False: the caller cuts them, sample_points(points, frame=…)). The frame is kept
    (last_frame); render(frame, "after", rect=rect) makes the after-look from it when one is sent —
    one capture for the change check, the after-look and the board's look after a move."""
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
    if not pts or not samples or _via(like) not in ("", frame.backend):
        return frame, None
    return frame, samples_from(frame, pts, rr)


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


def _window_stack(window_id: int) -> tuple[int, list[Rect]] | None:
    """(the window's owner pid, the on-screen windows over it — any app's, NEO's overlay, the Dock —
    as rectangles in points), from the window server (about 0.5 ms). None: it couldn't say, or the
    window isn't on the screen (minimised, another Space). A window nothing is drawn in (alpha 0)
    covers nothing."""
    try:
        opts = Q.kCGWindowListOptionOnScreenAboveWindow | Q.kCGWindowListOptionIncludingWindow
        info = Q.CGWindowListCopyWindowInfo(opts, int(window_id))
    except Exception:  # noqa: BLE001 - unknown
        return None
    if info is None:
        return None
    pid, above, found = 0, [], False
    for w in info:
        try:
            if int(w.get("kCGWindowNumber", 0) or 0) == int(window_id):
                if not w.get("kCGWindowIsOnscreen"):
                    return None
                pid, found = int(w.get("kCGWindowOwnerPID", 0) or 0), True
                continue
            b = w.get("kCGWindowBounds") or {}
            x, y, ww, hh = (float(b.get(k, 0) or 0) for k in ("X", "Y", "Width", "Height"))
            alpha = float(w.get("kCGWindowAlpha", 1) or 0)
        except Exception:  # noqa: BLE001 - a window that won't say where it is: can't tell what it covers
            return None
        if ww > 0 and hh > 0 and alpha > 0:
            above.append((x, y, ww, hh))
    return (pid, above) if found else None


def window_shot(
    window_id: int, region: tuple[float, float, float, float] | None = None, *, detail: str = "high", fmt: str | None = None
) -> Capture:
    """One window's own pixels (even where another window covers it; no shadow), at most _MAX_W on
    its long edge and _MAX_AREA, mapped to the screen through the window server's bounds for the
    window: clickable when the picture matches them and the window is on the screen, except where
    another window was over it (``covered``; see Capture.clear_at). ``region`` (screen points, x y
    width height): only that part of it — a page without the browser's tabs — when at least
    _MIN_CUT points of it are in the window. Its frame is kept (last_frame) only when it maps."""
    _need_permission()
    was = _window_bounds(window_id)
    t0, t = time.monotonic(), time.perf_counter()
    img, backend = _grab_window(window_id)
    grab_ms = (time.perf_counter() - t) * 1000
    bounds = _window_bounds(window_id)
    stack = _window_stack(window_id)
    pid, above = stack if stack is not None else (0, [])
    got = {"t": t0, "window_id": int(window_id), "backend": backend, "grab_ms": grab_ms, "pid": pid}
    if bounds is not None and bounds[2] > 0 and bounds[3] > 0:
        kx, ky = img.width / bounds[2], img.height / bounds[3]
        # (a picture with a shadow, of another size, of a window that moved meanwhile, or of one not
        # on the screen: no mapping)
        mapped = was == bounds and abs(kx - ky) <= 0.02 * max(kx, ky) and stack is not None
        covered = _overlaps(above, bounds) if mapped else ()
        frame = Frame(img, bounds[0], bounds[1], bounds[2] / img.width, covered=covered, mapped=mapped, **got)
    else:
        frame = Frame(img, 0.0, 0.0, 1.0 / _main_scale(), mapped=False, **got)
    cut = None
    if region is not None and bounds is not None and bounds[2] > 0:
        x, y, w, h = bounds
        left, top = max(region[0], x), max(region[1], y)
        right, bottom = min(region[0] + region[2], x + w), min(region[1] + region[3], y + h)
        if right - left >= _MIN_CUT and bottom - top >= _MIN_CUT:
            cut = (left, top, right - left, bottom - top)
    _keep(frame)
    return render(frame, "window", rect=cut, detail=detail, fmt=fmt, clickable=frame.mapped)


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
    says which capture path it came from (``info["via"]``): only samples of one path compare. A
    window's own pixels, or a frame with no known place: ValueError (they aren't the screen)."""
    if not frame.screen_like:
        raise ValueError("change samples come only from a picture of the screen")
    out = []
    for px, py in points:
        box = frame.box(px - r, py - r, 2 * r, 2 * r)
        full = frame.image.crop(box)
        sample = full.convert("L").resize((_SAMPLE_GRID, _SAMPLE_GRID), Image.BILINEAR)
        sample.info["via"] = frame.backend
        # (the full-detail crop, for reading which words changed there: ocr.sample_change)
        sample.info["full"], sample.info["ppp"] = full, frame.ppp
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
        if (
            frame is not None
            and frame.screen_like  # (never a window's own pixels: a capture of the screen instead)
            and all(frame.covers(p[0] - r, p[1] - r, 2 * r, 2 * r) for p in points)
        ):
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
        t0 = time.monotonic()
        if via == "cli":
            img, backend = _grab(["-R", f"{x},{y},{w},{h}"]), "cli"
        else:  # (the in-process path when it's on, with no CLI fallback: a pair never mixes the two)
            img, backend = _grab_rect(x, y, w, h, fallback=False)
            if via and backend != via:
                return None
        return samples_from(Frame(img, float(x), float(y), w / max(1, img.width), t0, backend=backend), points, r)
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
