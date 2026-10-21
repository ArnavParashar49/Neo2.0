"""What a click on Search Screen can pick, worked out from the still once the overlay is open (never
for a hold that was let go of early): the lines of text and the blocks they make, the things in
the picture, and the windows on that display — all in display-local points.

* lines — Apple Vision's text recognition over the whole still (ocr.read with plan_for(""): fast
  where it reads this Mac's languages). Not ocr.read_area: its "whose words" filter would drop
  everything under NEO's own overlay, which is all of it now. Boxes only, unless VoiceOver is on
  (then each line's words, at most 120 characters, label the regions while tabbing).
* blocks — lines in one column with less than 1.3 lines' height between them.
* objects — Vision's objectness saliency (up to 3 boxes), on a 1024-px copy.
* instances — Vision's foreground instance mask (up to 8 things), each thing's box from its mask.
* windows — the window server's on-screen windows on that display (front first, NEO's own left
  out, at least 120 × 80 points).

Running Vision is a "model" effect (neo2.core.effects): a test sets a fake engine (:func:`set_engine`)
and ocr's own (ocr.set_engine). Each part that can't run is just missing: a click then falls back to
the next kind of region (the page's pick order), down to a box around the point.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Protocol

from PIL import Image

from neo2.core import effects, logger
from neo2.core.lens.capture import Spot

log = logger(__name__)

SMALL_EDGE = 1024  # objects and instances are found on a copy this long
MAX_OBJECTS, MAX_INSTANCES = 3, 8
MAX_LINES = 1500  # lines sent to the page (reading order): more than any screen shows
TEXT_CHARS = 120
MIN_WINDOW = (120.0, 80.0)
BLOCK_COLUMN_PT = 12.0
BLOCK_GAP = 1.3  # lines' heights between two lines of one block, at most
_FRAMEWORK = "/System/Library/Frameworks/Vision.framework"

Box01 = tuple[float, float, float, float]  # 0–1 of the picture, origin top left


class Engine(Protocol):
    def objects(self, img: Image.Image) -> list[Box01]: ...

    def instances(self, img: Image.Image) -> list[Box01]: ...


class _Vision:
    """Apple Vision's saliency and foreground-instance requests, loaded on first use."""

    def __init__(self) -> None:
        self._ns: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def _classes(self) -> dict[str, Any]:
        effects.guard("model", "load Apple Vision for Search Screen's smart pick")
        with self._lock:
            if self._ns is None:
                names = ("VNGenerateObjectnessBasedSaliencyImageRequest", "VNGenerateForegroundInstanceMaskRequest",
                         "VNImageRequestHandler")  # fmt: skip
                try:
                    import Vision

                    self._ns = {k: getattr(Vision, k, None) for k in names}
                except ImportError:
                    import objc

                    ns: dict[str, Any] = {}
                    objc.loadBundle("Vision", ns, bundle_path=_FRAMEWORK)
                    self._ns = {k: ns.get(k) for k in names}
            return self._ns

    @staticmethod
    def _perform(ns: dict[str, Any], img: Image.Image, req: Any) -> bool:
        from neo2.core.vision.apple import _cgimage

        cg, data = _cgimage(img)
        handler = ns["VNImageRequestHandler"].alloc().initWithCGImage_options_(cg, {})
        ok = handler.performRequests_error_([req], None)
        del data
        return bool(ok[0] if isinstance(ok, tuple) else ok)

    def objects(self, img: Image.Image) -> list[Box01]:
        effects.guard("model", "run Apple Vision's saliency for Search Screen")
        from neo2.core.decide._objc import pool

        ns = self._classes()
        cls = ns.get("VNGenerateObjectnessBasedSaliencyImageRequest")
        if cls is None:
            return []
        with pool():
            req = cls.alloc().init()
            if not self._perform(ns, img, req):
                return []
            out: list[Box01] = []
            for o in req.results() or []:
                for so in o.salientObjects() or []:
                    b = so.boundingBox()
                    x, y, w, h = float(b.origin.x), float(b.origin.y), float(b.size.width), float(b.size.height)
                    out.append((x, 1.0 - y - h, w, h))  # (Vision's origin is the bottom left)
            return out

    def instances(self, img: Image.Image) -> list[Box01]:
        effects.guard("model", "run Apple Vision's foreground instances for Search Screen")
        from neo2.core.decide._objc import pool

        ns = self._classes()
        cls = ns.get("VNGenerateForegroundInstanceMaskRequest")  # (macOS 14)
        if cls is None:
            return []
        import Quartz as Q

        with pool():
            req = cls.alloc().init()
            if not self._perform(ns, img, req):
                return []
            out: list[Box01] = []
            for o in (req.results() or [])[:1]:
                n = int(o.allInstances().count())
                pb = o.instanceMask()  # one byte per pixel: 0 the background, 1…n each thing
                Q.CVPixelBufferLockBaseAddress(pb, 1)
                try:
                    w, h = int(Q.CVPixelBufferGetWidth(pb)), int(Q.CVPixelBufferGetHeight(pb))
                    row = int(Q.CVPixelBufferGetBytesPerRow(pb))
                    raw = bytes(Q.CVPixelBufferGetBaseAddress(pb).as_buffer(row * h))
                finally:
                    Q.CVPixelBufferUnlockBaseAddress(pb, 1)
                mask = Image.frombuffer("L", (w, h), raw, "raw", "L", row, 1)
                out += instance_boxes(mask, n)
            return out


def instance_boxes(mask: Image.Image, n: int) -> list[Box01]:
    """Each thing's box (0–1, top left) from an instance mask whose pixel values are 1…n."""
    w, h = mask.size
    out: list[Box01] = []
    for i in range(1, min(n, 255) + 1):
        bb = mask.point(lambda v, i=i: 255 if v == i else 0).getbbox()
        if bb is not None:
            out.append((bb[0] / w, bb[1] / h, (bb[2] - bb[0]) / w, (bb[3] - bb[1]) / h))
    return out


_engine: Engine | None = None
_vision: _Vision | None = None


def set_engine(e: Engine | None) -> None:
    """Use ``e`` instead of Apple Vision (tests); None: Vision again."""
    global _engine
    _engine = e


def engine() -> Engine:
    global _vision
    if _engine is not None:
        return _engine
    if _vision is None:
        _vision = _Vision()
    return _vision


def _small(img: Image.Image) -> Image.Image:
    if max(img.size) <= SMALL_EDGE:
        return img if img.mode == "RGB" else img.convert("RGB")
    k = SMALL_EDGE / max(img.size)
    return img.convert("RGB").resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.BILINEAR)


def _r(v: float) -> float:
    return round(float(v), 1)


def group_blocks(boxes: list[tuple[float, float, float, float]]) -> tuple[list[int], list[list[float]]]:
    """(each line's block, the blocks' boxes): a line joins the block whose last line starts within
    BLOCK_COLUMN_PT of it, sideways, with less than BLOCK_GAP lines' height between them."""
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))
    of = [0] * len(boxes)
    blocks: list[list[float]] = []  # [x0, y0, x1, y1]
    last: list[tuple[float, float, float, float]] = []
    for i in order:
        x, y, w, h = boxes[i]
        hit = -1
        for b, (lx, ly, lw, lh) in enumerate(last):
            gap = y - (ly + lh)
            if abs(x - lx) <= BLOCK_COLUMN_PT and -0.5 * max(h, lh) <= gap < BLOCK_GAP * max(h, lh):
                hit = b
                break
        if hit < 0:
            blocks.append([x, y, x + w, y + h])
            last.append((x, y, w, h))
            hit = len(blocks) - 1
        else:
            bb = blocks[hit]
            bb[0], bb[1], bb[2], bb[3] = min(bb[0], x), min(bb[1], y), max(bb[2], x + w), max(bb[3], y + h)
            last[hit] = (x, y, w, h)
        of[i] = hit
    return of, [[_r(b[0]), _r(b[1]), _r(b[2] - b[0]), _r(b[3] - b[1])] for b in blocks]


def _lines(frame: Any, sp: Spot, texts: bool) -> tuple[list[list[Any]], list[list[float]], list[str]]:
    from neo2.core.tools.computer import ocr

    if not ocr.available():
        return [], [], []
    plan = ocr.plan_for("")
    if plan is None:
        return [], [], []
    try:
        reading = ocr.read(frame, plan=plan)
    except ocr.Failed as e:
        log.info(f"[lens] no lines for smart pick ({str(e)[:60]})")
        return [], [], []
    lines = list(reading.lines)[:MAX_LINES]
    boxes = [(ln.box[0] - sp.x, ln.box[1] - sp.y, ln.box[2], ln.box[3]) for ln in lines]
    of, blocks = group_blocks(boxes)
    out = [[_r(x), _r(y), _r(w), _r(h), of[i]] for i, (x, y, w, h) in enumerate(boxes)]
    said = [ln.text[:TEXT_CHARS] for ln in lines] if texts else []
    return out, blocks, said


def _boxes(raw: list[Box01], sp: Spot, n: int, *, most: float = 1.0) -> list[list[float]]:
    out = []
    for x, y, w, h in raw:
        if w <= 0 or h <= 0 or w * h > most:
            continue
        out.append([_r(x * sp.w), _r(y * sp.h), _r(w * sp.w), _r(h * sp.h)])
        if len(out) >= n:
            break
    return out


def windows(sp: Spot) -> list[list[float]]:
    """The on-screen windows on this display, front first: [x, y, w, h] display-local, cut to it.
    Layer 0 only (no menu bar, panels or NEO's overlay), never NEO's own, at least MIN_WINDOW."""
    try:
        import Quartz as Q

        from neo2.core.tools.computer import ax

        wins = Q.CGWindowListCopyWindowInfo(
            Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements, Q.kCGNullWindowID
        )
    except Exception:  # noqa: BLE001 — no window server to ask: no windows
        return []
    out: list[list[float]] = []
    for w in wins or []:
        try:
            if int(w.get("kCGWindowLayer", 0) or 0) != 0:
                continue
            pid, name = int(w.get("kCGWindowOwnerPID", 0) or 0), str(w.get("kCGWindowOwnerName", "") or "")
            if ax.is_neo(pid, name):
                continue
            b = w.get("kCGWindowBounds") or {}
            x, y, ww, hh = (float(b.get(k, 0) or 0) for k in ("X", "Y", "Width", "Height"))
        except Exception:  # noqa: BLE001 — a window that won't say where it is: not one to pick
            continue
        left, top = max(x, sp.x), max(y, sp.y)
        right, bottom = min(x + ww, sp.x + sp.w), min(y + hh, sp.y + sp.h)
        if right - left >= MIN_WINDOW[0] and bottom - top >= MIN_WINDOW[1]:
            out.append([_r(left - sp.x), _r(top - sp.y), _r(right - left), _r(bottom - top)])
    return out[:40]


def compute_text(frame: Any | None, sp: Spot, *, texts: bool = False) -> dict[str, Any]:
    """Smart pick's first part, the quick one: the lines of text, their paragraphs and the windows
    (``frame`` None, live mode: windows only); objects and instances empty."""
    t0 = time.perf_counter()
    out: dict[str, Any] = {"lines": [], "blocks": [], "objects": [], "instances": [], "windows": windows(sp)}
    if frame is not None:
        lines, blocks, said = _lines(frame, sp, texts)
        out["lines"], out["blocks"] = lines, blocks
        if texts:
            out["texts"] = said
    log.info(f"[lens] lines in {(time.perf_counter() - t0) * 1000:.0f} ms: {len(out['lines'])} lines, {len(out['windows'])} windows")
    return out


def compute_things(frame: Any, sp: Spot) -> dict[str, Any]:
    """Smart pick's second part: the things — salient objects and foreground instances (each part
    that can't run is empty)."""
    t0 = time.perf_counter()
    out: dict[str, Any] = {"objects": [], "instances": []}
    small = _small(frame.image)
    e = engine()
    try:
        out["objects"] = _boxes(e.objects(small), sp, MAX_OBJECTS)
    except Exception as ex:  # noqa: BLE001 — no things from saliency: lines and windows still pick
        log.info(f"[lens] no objects for smart pick ({type(ex).__name__})")
    try:
        out["instances"] = _boxes(e.instances(small), sp, MAX_INSTANCES, most=0.6)
    except Exception as ex:  # noqa: BLE001
        log.info(f"[lens] no instances for smart pick ({type(ex).__name__})")
    log.info(f"[lens] things in {(time.perf_counter() - t0) * 1000:.0f} ms: {len(out['objects'])} objects, {len(out['instances'])} things")
    return out


def compute(frame: Any | None, sp: Spot, *, texts: bool = False) -> dict[str, Any]:
    """Every kind of region the page's smart pick reads (``frame`` None, live mode: windows only).
    Never raises for a part that can't run: that part is empty."""
    out = compute_text(frame, sp, texts=texts)
    if frame is not None:
        out.update(compute_things(frame, sp))
    return out


__all__ = ["compute", "compute_text", "compute_things", "group_blocks", "instance_boxes", "set_engine", "windows"]
