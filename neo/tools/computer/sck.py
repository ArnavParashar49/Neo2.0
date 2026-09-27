"""ScreenCaptureKit one-shot captures, in NEO's own process: no ``screencapture`` process, no file,
no PNG round trip (others measured about 0.09 s for a window and 0.13 s for a display on macOS 26,
against 0.2–0.3 s for the CLI).

screen.py uses it when ``settings().sck_capture`` is on and pyobjc-framework-ScreenCaptureKit is
installed (macOS 14 or later). Anything that goes wrong — no framework, an error, no answer within
``TIMEOUT_S`` — raises :class:`Failed` and screen.py falls back to the CLI. ``TIMEOUT_S`` is one
capture's whole wait (the lock, the list of windows, the picture), not each step's. Two unanswered
calls with no answered capture between them switch it off for the rest of the run (a completion
handler that never comes back costs its wait once or twice, not on every screenshot; the list of
windows answering is not a capture answering).

* NEO's own windows are left out of every display picture by a filter, app by app, exactly as
  ax.is_neo tells them (its overlay asks macOS to hide it from capture; ScreenCaptureKit need not
  honour that). A NEO window on screen now that wasn't when the cached list was fetched (an
  overlay that restarted) makes it fetch the list again first. The rect APIs
  (captureImageInRect, captureScreenshotWithRect) take no filter, so they are never used.
* One capture at a time: ScreenCaptureKit queues concurrent display captures badly.
* Every capture is a real effect (effects.guard, kind "capture"): refused while effects are off,
  unless the framework in use is a test's fake.
* No cursor. Pixels come back 8-bit BGRA in sRGB and become an RGB PIL image.
* Nothing runs in the background: no stream, no observer. The list of displays and apps
  (SCShareableContent) is fetched on demand and reused for ``CONTENT_TTL_S``.

Dependency: pyobjc-framework-ScreenCaptureKit (the mac extra; decision 42). Without it every
capture is the CLI's.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import Quartz as Q
from PIL import Image

from neo import effects
from neo.decide._objc import pooled

log = logging.getLogger(__name__)

TIMEOUT_S = 2.0  # one capture's whole wait for ScreenCaptureKit
CONTENT_TTL_S = 10.0  # the displays and apps list is reused this long
_MISSES_OFF = 2  # unanswered calls, with no answered capture between them, that switch it off
_BGRA = 0x42475241  # kCVPixelFormatType_32BGRA ('BGRA')

_SC: Any = None  # the ScreenCaptureKit module (a test puts its fake here)
_lock = threading.Lock()
_content: tuple[float, Any, frozenset[int]] | None = None  # (when, the list, NEO's pids on screen then)
_misses = 0
_off = ""  # why it is off for the rest of the run ("": it isn't)


class Failed(Exception):
    """No picture from ScreenCaptureKit: the caller uses the CLI."""


def _framework() -> Any:
    global _SC
    if _SC is None:
        try:
            import ScreenCaptureKit  # type: ignore[import-not-found]
        except Exception as e:  # noqa: BLE001 - not installed (the mac extra without it) or too old a macOS
            raise Failed(f"ScreenCaptureKit isn't available ({type(e).__name__})") from None
        _SC = ScreenCaptureKit
    return _SC


def available() -> bool:
    """The framework is installed and has the one-shot capture (macOS 14+)."""
    try:
        sc = _framework()
    except Failed:
        return False
    return hasattr(sc, "SCScreenshotManager") and hasattr(
        sc.SCScreenshotManager, "captureImageWithFilter_configuration_completionHandler_"
    )


def enabled() -> bool:
    """Usable now: available, and not switched off for this run."""
    return not _off and available()


def reset() -> None:
    """Forget the cached content and any switch-off (tests; a display change)."""
    global _content, _misses, _off
    _content, _misses, _off = None, 0, ""


# ---- waiting for a completion handler ------------------------------------------------------------


def _missed(what: str) -> Failed:
    """One more unanswered call; the second with no answered capture since switches it off."""
    global _misses, _off
    _misses += 1
    if _misses >= _MISSES_OFF and not _off:
        _off = f"no answer to {_MISSES_OFF} calls in a row"
        log.warning("ScreenCaptureKit: %s; the screencapture CLI for the rest of this run", _off)
    return Failed(f"no answer to {what} in time")


def _wait(
    start: Callable[[Callable[..., None]], None], take: Callable[..., Any], what: str, deadline: float, *, capture: bool = False
) -> Any:
    """Start an asynchronous ScreenCaptureKit call and wait for its completion handler, which runs
    ``take`` on what it was handed (on ScreenCaptureKit's own queue: its image is copied out there,
    while it is surely alive). No answer by ``deadline`` (time.monotonic): Failed (a late answer is
    dropped), and a miss. No time left to ask at all: Failed, not a miss (nothing went unanswered).
    Only an answered ``capture`` clears the misses."""
    global _misses
    left = deadline - time.monotonic()
    if left <= 0:
        raise Failed(f"no answer to {what} in time (no time left to ask)")
    done = threading.Event()
    box: dict[str, Any] = {}

    def handler(*args: Any) -> None:
        try:
            box["value"] = take(*args)
        except BaseException as e:  # noqa: BLE001 - carried to the waiting thread
            box["error"] = e
        finally:
            done.set()

    start(handler)
    if not done.wait(left):
        raise _missed(what)
    if capture:
        _misses = 0
    if "error" in box:
        err = box["error"]
        raise err if isinstance(err, Failed) else Failed(f"{what}: {err}") from err
    return box["value"]


def _ns_error(error: Any) -> str:
    try:
        return str(error.localizedDescription())
    except Exception:  # noqa: BLE001
        return str(error)


def _content_now(deadline: float) -> Any:
    sc = _framework()

    def take(content: Any, error: Any) -> Any:
        if error is not None or content is None:
            raise Failed(f"no shareable content ({_ns_error(error)})")
        return content

    get = sc.SCShareableContent.getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_
    effects.guard("capture", "list the screen's windows for capture (ScreenCaptureKit)", via=get)
    return _wait(lambda h: get(True, True, h), take, "the list of windows", deadline)


def _shareable(deadline: float, *, fresh: bool = False) -> Any:
    global _content
    now = time.monotonic()
    if not fresh and _content is not None and now - _content[0] < CONTENT_TTL_S:
        return _content[1]
    seen = _neo_on_screen()  # (before the fetch: every NEO window in this set is in the list)
    content = _content_now(deadline)
    _content = (time.monotonic(), content, seen)
    return content


def _on_screen_owners() -> set[tuple[int, str]]:
    """(pid, name) of the owner of every window on the screen, from the window server (well under a
    millisecond). Empty when it can't say."""
    try:
        wins = Q.CGWindowListCopyWindowInfo(
            Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements, Q.kCGNullWindowID
        )
    except Exception:  # noqa: BLE001
        return set()
    out = set()
    for w in wins or []:
        try:
            pid = int(w.get("kCGWindowOwnerPID", 0) or 0)
            name = str(w.get("kCGWindowOwnerName", "") or "")
        except Exception:  # noqa: BLE001
            continue
        if pid > 0:
            out.add((pid, name))
    return out


def _neo_on_screen() -> frozenset[int]:
    """The pids of NEO's own processes with a window on the screen now (ax.is_neo)."""
    from neo.tools.computer import ax

    return frozenset(pid for pid, name in _on_screen_owners() if ax.is_neo(pid, name))


def _neo_new_since(content_seen: frozenset[int]) -> bool:
    """A NEO window on the screen now whose process had none when the cached list was fetched (an
    overlay that restarted): that list may not know it, so its filter couldn't leave it out."""
    return bool(_neo_on_screen() - content_seen)


def _neo_apps(content: Any) -> list[Any]:
    """NEO's own apps (its overlay, whatever launched this core) among those ScreenCaptureKit lists."""
    from neo.tools.computer import ax

    out = []
    for app in content.applications() or []:
        try:
            pid, name = int(app.processID()), str(app.applicationName() or "")
        except Exception:  # noqa: BLE001 - an app that won't say: not NEO's
            continue
        if ax.is_neo(pid, name):
            out.append(app)
    return out


def _display(content: Any, display: Any) -> Any:
    """ScreenCaptureKit's display for a screen.Display: by its id, else by where it is."""
    shown = list(content.displays() or [])
    for d in shown:
        if display.did and int(d.displayID()) == int(display.did):
            return d
    for d in shown:
        f = d.frame()
        if abs(float(f.origin.x) - display.x) < 1 and abs(float(f.origin.y) - display.y) < 1:
            return d
    raise Failed(f"display {display.index} isn't shareable")


def _config(sc: Any, width: int, height: int, source: tuple[float, float, float, float] | None) -> Any:
    cfg = sc.SCStreamConfiguration.alloc().init()
    if source is not None:
        cfg.setSourceRect_(((source[0], source[1]), (source[2], source[3])))
    cfg.setWidth_(int(width))
    cfg.setHeight_(int(height))
    cfg.setShowsCursor_(False)
    cfg.setPixelFormat_(_BGRA)
    cfg.setColorSpaceName_(Q.kCGColorSpaceSRGB)
    return cfg


def _capture(sc: Any, flt: Any, cfg: Any, deadline: float) -> Image.Image:
    shoot = sc.SCScreenshotManager.captureImageWithFilter_configuration_completionHandler_
    effects.guard("capture", "capture the screen in-process (ScreenCaptureKit)", via=shoot)

    def take(image: Any, error: Any) -> Image.Image:
        if error is not None or image is None:
            raise Failed(f"no picture ({_ns_error(error)})")
        return to_pil(image)

    return _wait(lambda h: shoot(flt, cfg, h), take, "the screen capture", deadline, capture=True)


def _locked[T](fn: Callable[[], T], deadline: float) -> T:
    """``fn`` alone (one capture at a time), anything it raises but a refused effect as Failed: an
    API that moved or answered oddly leaves the CLI to do the capture."""
    if not _lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
        raise Failed("another capture is still running")
    try:
        return fn()
    except Failed:
        raise
    except Exception as e:  # noqa: BLE001 - (an effects refusal is a BaseException: it goes through)
        raise Failed(f"{type(e).__name__}: {e}") from e
    finally:
        _lock.release()


@pooled
def capture_rect(
    x: float, y: float, w: float, h: float, display: Any, *, scale: float | None = None, timeout: float = TIMEOUT_S
) -> Image.Image:
    """The pixels of a rectangle in global points on ``display`` (a screen.Display), NEO's own windows
    left out, at ``scale`` pixels per point (None: the display's own, 2 on Retina). Only the part on
    the display is captured (screen.grab cuts a region to its display first, so it maps). All of it
    within ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    if _off:
        raise Failed(_off)
    sc = _framework()
    left, top = max(x, display.x), max(y, display.y)
    right, bottom = min(x + w, display.x + display.w), min(y + h, display.y + display.h)
    if right - left < 1 or bottom - top < 1:
        raise Failed("that rectangle isn't on the display")

    def run() -> Image.Image:
        fresh = _content is not None and _neo_new_since(_content[2])  # (a NEO window newer than the list)
        content = _shareable(deadline, fresh=fresh)
        try:
            shown = _display(content, display)
        except Failed:  # (a display newer than the cached list: look again once)
            if fresh:
                raise
            content = _shareable(deadline, fresh=True)
            shown = _display(content, display)
        flt = sc.SCContentFilter.alloc().initWithDisplay_excludingApplications_exceptingWindows_(
            shown, _neo_apps(content), []
        )
        k = scale or display.scale or _filter_scale(sc, flt) or 1.0
        cfg = _config(sc, round((right - left) * k), round((bottom - top) * k), (left - display.x, top - display.y, right - left, bottom - top))
        return _capture(sc, flt, cfg, deadline)

    return _locked(run, deadline)


@pooled
def capture_window(window_id: int, *, timeout: float = TIMEOUT_S) -> Image.Image:
    """One window's own pixels (even where others cover it), no shadow, at its display's resolution,
    within ``timeout`` seconds in all."""
    deadline = time.monotonic() + timeout
    if _off:
        raise Failed(_off)
    sc = _framework()

    def run() -> Image.Image:
        win = None
        for fresh in (False, True):  # (a window newer than the cached list: look again once)
            content = _shareable(deadline, fresh=fresh)
            win = next((w for w in content.windows() or [] if int(w.windowID()) == int(window_id)), None)
            if win is not None:
                break
        if win is None:
            raise Failed(f"window {window_id} isn't shareable")
        flt = sc.SCContentFilter.alloc().initWithDesktopIndependentWindow_(win)
        f = win.frame()
        k = _filter_scale(sc, flt) or _scale_at(float(f.origin.x + f.size.width / 2), float(f.origin.y + f.size.height / 2))
        cfg = _config(sc, round(float(f.size.width) * k), round(float(f.size.height) * k), None)
        if hasattr(cfg, "setIgnoreShadowsSingleWindow_"):
            cfg.setIgnoreShadowsSingleWindow_(True)
        return _capture(sc, flt, cfg, deadline)

    return _locked(run, deadline)


def _scale_at(x: float, y: float) -> float:
    from neo.tools.computer import screen

    d = screen.display_at(x, y)
    return d.scale if d is not None and d.scale else 2.0


def _filter_scale(sc: Any, flt: Any) -> float:
    """Pixels per point of what a filter captures (SCShareableContentInfo, macOS 14); 0: unknown."""
    try:
        return float(sc.SCShareableContent.infoForFilter_(flt).pointPixelScale())
    except Exception:  # noqa: BLE001
        return 0.0


# ---- CGImage → PIL ---------------------------------------------------------------------------------


def _raw_mode(info: int) -> str | None:
    """PIL's raw mode for an 8-bit, 32-bit-per-pixel CGImage's bytes (None: another layout)."""
    order = info & Q.kCGBitmapByteOrderMask
    alpha = info & Q.kCGBitmapAlphaInfoMask
    little = order == Q.kCGBitmapByteOrder32Little
    big = order in (Q.kCGBitmapByteOrderDefault, Q.kCGBitmapByteOrder32Big)
    first = alpha in (Q.kCGImageAlphaPremultipliedFirst, Q.kCGImageAlphaFirst, Q.kCGImageAlphaNoneSkipFirst)
    last = alpha in (Q.kCGImageAlphaPremultipliedLast, Q.kCGImageAlphaLast, Q.kCGImageAlphaNoneSkipLast)
    if first:
        return "BGRX" if little else "XRGB" if big else None
    if last:
        return "XBGR" if little else "RGBX" if big else None
    return None


def _redraw(cg: Any, w: int, h: int) -> Any:
    """Any other layout (16-bit, HDR, float) drawn into an 8-bit BGRA sRGB bitmap."""
    space = Q.CGColorSpaceCreateWithName(Q.kCGColorSpaceSRGB)
    ctx = Q.CGBitmapContextCreate(
        None, w, h, 8, 0, space, Q.kCGImageAlphaPremultipliedFirst | Q.kCGBitmapByteOrder32Little
    )
    if ctx is None:
        raise Failed("couldn't convert the picture")
    Q.CGContextDrawImage(ctx, Q.CGRectMake(0, 0, w, h), cg)
    return Q.CGBitmapContextCreateImage(ctx)


def to_pil(cg: Any) -> Image.Image:
    """A CGImage as an RGB PIL image (its bytes copied: nothing points into ScreenCaptureKit's memory)."""
    w, h = int(Q.CGImageGetWidth(cg)), int(Q.CGImageGetHeight(cg))
    if w <= 0 or h <= 0:
        raise Failed("an empty picture")
    raw = None
    if Q.CGImageGetBitsPerComponent(cg) == 8 and Q.CGImageGetBitsPerPixel(cg) == 32:
        raw = _raw_mode(int(Q.CGImageGetBitmapInfo(cg)))
    if raw is None:
        cg, raw = _redraw(cg, w, h), "BGRX"
    data = bytes(Q.CGDataProviderCopyData(Q.CGImageGetDataProvider(cg)))
    return Image.frombytes("RGB", (w, h), data, "raw", raw, int(Q.CGImageGetBytesPerRow(cg)), 1)
