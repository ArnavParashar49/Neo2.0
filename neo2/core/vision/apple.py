"""Apple Vision on a camera frame: on this Mac, nothing to download, a few tens of ms.

Three requests on one image handler, in one pass (:func:`analyse`):

* labels — ``VNClassifyImageRequest``: Apple's ~1,300-label taxonomy for the whole frame ("cup",
  "beverage", "keyboard"), top few with their confidence;
* text — ``VNRecognizeTextRequest`` in fast mode: the words printed on the thing held up (a label,
  a book cover, a screen);
* hands — ``VNDetectHumanHandPoseRequest``: up to two hands, each with its box and how many fingers
  are raised (from the joints: a finger is up when its tip is well past its middle joint, seen from
  the wrist; the thumb when its tip is clear of the palm).

The frame reaches Vision as a CGImage over the picture's own bytes (no PNG round trip), and is never
written anywhere. With pyobjc-framework-Vision installed its metadata is used; without it the system
framework is loaded through objc.loadBundle (the hand joints' names with loadBundleVariables). Off
macOS, or with neither: :class:`~neo2.core.vision.Unavailable`.

Loading and running Vision is a "model" effect (neo2.core.effects): refused in NEO's tests and
scratch scripts — a test sets a fake engine (:func:`set_engine`).
"""

from __future__ import annotations

import math
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from neo2.core import effects, logger
from neo2.core.vision import Unavailable

log = logger(__name__)

Box = tuple[float, float, float, float]  # x, y, width, height — 0..1 of the frame, origin top left

_FRAMEWORK = "/System/Library/Frameworks/Vision.framework"
MAX_EDGE = 1280  # a bigger frame is scaled down first: labels and hands don't need more, text rarely
TOP_K = 5
MIN_LABEL_CONF = 0.2
MIN_TEXT_CONF = 0.4
MAX_TEXT_LINES = 8
MAX_TEXT_CHARS = 80  # one line, at most this long
MAX_HANDS = 2
MIN_JOINT_CONF = 0.3
_RUN_WAIT_S = 5.0

# Our joint names → the suffix of Vision's VNHumanHandPoseObservationJointName… constants.
JOINTS: dict[str, str] = {
    "wrist": "Wrist",
    "thumb_cmc": "ThumbCMC", "thumb_mp": "ThumbMP", "thumb_ip": "ThumbIP", "thumb_tip": "ThumbTip",
    "index_mcp": "IndexMCP", "index_pip": "IndexPIP", "index_dip": "IndexDIP", "index_tip": "IndexTip",
    "middle_mcp": "MiddleMCP", "middle_pip": "MiddlePIP", "middle_dip": "MiddleDIP", "middle_tip": "MiddleTip",
    "ring_mcp": "RingMCP", "ring_pip": "RingPIP", "ring_dip": "RingDIP", "ring_tip": "RingTip",
    "little_mcp": "LittleMCP", "little_pip": "LittlePIP", "little_dip": "LittleDIP", "little_tip": "LittleTip",
}  # fmt: skip
_FINGERS = ("index", "middle", "ring", "little")


@dataclass
class Hand:
    box: Box  # 0..1 of the frame, origin top left
    fingers: int | None = None  # raised, thumb included; None: the joints weren't clear enough
    side: str = ""  # "left" / "right" as Vision tells it, "" when it can't
    conf: float = 0.0


@dataclass
class Seen:
    """What Apple Vision made of one frame."""

    labels: list[tuple[str, float]] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    hands: list[Hand] = field(default_factory=list)
    ms: float = 0.0


class Engine(Protocol):
    def analyse(self, img: Image.Image, *, labels: bool, text: bool, hands: bool) -> Seen: ...


# ---- hands: counting fingers from the joints ------------------------------------------------------


Point = tuple[float, float, float]  # x, y (0..1, any origin), confidence


def _dist(a: Point, b: Point, aspect: float) -> float:
    return math.hypot((a[0] - b[0]) * aspect, a[1] - b[1])


def count_fingers(points: Mapping[str, Point], aspect: float = 1.0) -> int | None:
    """Fingers raised (0–5) from a hand's joints (our names: :data:`JOINTS`). ``aspect``: the frame's
    width / height, so distances are the same in both directions. None: too few joints were clear."""
    p = {k: v for k, v in points.items() if v[2] >= MIN_JOINT_CONF}
    wrist = p.get("wrist")
    if wrist is None:
        return None
    up, known = 0, 0
    for f in _FINGERS:
        tip, pip = p.get(f"{f}_tip"), p.get(f"{f}_pip")
        if tip is None or pip is None:
            continue
        known += 1
        if _dist(wrist, tip, aspect) > 1.15 * _dist(wrist, pip, aspect):
            up += 1
    if known < 3:
        return None
    tip, ip, imcp, lmcp = p.get("thumb_tip"), p.get("thumb_ip"), p.get("index_mcp"), p.get("little_mcp")
    if tip is not None and ip is not None and imcp is not None:
        palm = _dist(imcp, lmcp, aspect) if lmcp is not None else _dist(wrist, imcp, aspect) * 0.8
        clear = _dist(tip, imcp, aspect) > 0.85 * palm
        if clear and _dist(wrist, tip, aspect) > _dist(wrist, ip, aspect):
            up += 1
    return up


def hand_box(points: Mapping[str, Point], *, flip_y: bool = True, pad: float = 0.1) -> Box | None:
    """The box around a hand's clear joints, 0..1 of the frame with the origin top left (Vision's
    points have it bottom left: ``flip_y``), padded by ``pad`` of its size on each side."""
    pts = [(x, 1.0 - y if flip_y else y) for x, y, c in points.values() if c >= MIN_JOINT_CONF]
    if len(pts) < 3:
        return None
    x0, x1 = min(x for x, _ in pts), max(x for x, _ in pts)
    y0, y1 = min(y for _, y in pts), max(y for _, y in pts)
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    x0, y0, x1, y1 = max(0.0, x0 - px), max(0.0, y0 - py), min(1.0, x1 + px), min(1.0, y1 + py)
    return (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))


# ---- the real engine: Apple Vision -------------------------------------------------------------------


def _fit(img: Image.Image) -> Image.Image:
    if max(img.size) <= MAX_EDGE:
        return img
    s = MAX_EDGE / max(img.size)
    return img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BILINEAR)


def _cgimage(img: Image.Image) -> tuple[Any, bytes]:
    """An RGB CGImage over the picture's own bytes. The bytes are returned too: they must outlive
    the request."""
    import Quartz as Q

    rgbx = (img if img.mode == "RGB" else img.convert("RGB")).convert("RGBX")
    data = rgbx.tobytes()
    provider = Q.CGDataProviderCreateWithData(None, data, len(data), None)
    cg = Q.CGImageCreate(
        rgbx.width, rgbx.height, 8, 32, rgbx.width * 4, Q.CGColorSpaceCreateDeviceRGB(),
        Q.kCGImageAlphaNoneSkipLast, provider, None, False, Q.kCGRenderingIntentDefault,
    )  # fmt: skip
    if cg is None:
        raise Unavailable("Apple Vision couldn't take this frame")
    return cg, data


def _ok(r: Any) -> tuple[bool, Any]:
    if isinstance(r, tuple):
        return bool(r[0]), (r[1] if len(r) > 1 else None)
    return bool(r), None


class _Vision:
    """Apple's Vision requests. Loaded on first use; one frame at a time."""

    _CLASSES = ("VNClassifyImageRequest", "VNRecognizeTextRequest", "VNDetectHumanHandPoseRequest",
                "VNImageRequestHandler")  # fmt: skip

    def __init__(self) -> None:
        self._ns: dict[str, Any] | None = None
        self._joints: dict[str, str] = {}  # Vision's joint name (its string) → ours
        self._group = None  # VNHumanHandPoseObservationJointsGroupNameAll
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()

    def _load(self) -> dict[str, Any]:
        with self._load_lock:
            if self._ns is not None:
                return self._ns
            consts = ["VNHumanHandPoseObservationJointsGroupNameAll"] + [
                f"VNHumanHandPoseObservationJointName{v}" for v in JOINTS.values()
            ]
            try:
                import Vision  # pyobjc-framework-Vision: its metadata

                ns = {k: getattr(Vision, k) for k in self._CLASSES}
                got = {k: getattr(Vision, k, None) for k in consts}
            except ImportError:
                import objc

                ns = {}
                bundle = objc.loadBundle("Vision", ns, bundle_path=_FRAMEWORK)
                got = {}
                try:
                    objc.loadBundleVariables(bundle, got, [(k, b"@") for k in consts])
                except Exception as e:  # noqa: BLE001 — no joint names: hands are left out, the rest works
                    log.info(f"[vision] hand joints unavailable: {type(e).__name__}")
            self._group = got.get("VNHumanHandPoseObservationJointsGroupNameAll")
            self._joints = {
                str(got[f"VNHumanHandPoseObservationJointName{v}"]): k
                for k, v in JOINTS.items()
                if got.get(f"VNHumanHandPoseObservationJointName{v}") is not None
            }
            self._ns = ns
            return ns

    def analyse(self, img: Image.Image, *, labels: bool, text: bool, hands: bool) -> Seen:
        effects.guard("model", "run Apple Vision on a camera frame")
        from neo2.core.decide._objc import pool

        ns = self._load()
        if not self._run_lock.acquire(timeout=_RUN_WAIT_S):
            raise Unavailable("Apple Vision is busy with another frame")
        try:
            with pool():
                img = _fit(img)
                cg, data = _cgimage(img)
                reqs: dict[str, Any] = {}
                if labels:
                    reqs["labels"] = ns["VNClassifyImageRequest"].alloc().init()
                if text:
                    r = ns["VNRecognizeTextRequest"].alloc().init()
                    r.setRecognitionLevel_(1)  # fast
                    r.setUsesLanguageCorrection_(True)  # words on things, not UI labels
                    reqs["text"] = r
                if hands and self._joints:
                    r = ns["VNDetectHumanHandPoseRequest"].alloc().init()
                    r.setMaximumHandCount_(MAX_HANDS)
                    reqs["hands"] = r
                handler = ns["VNImageRequestHandler"].alloc().initWithCGImage_options_(cg, {})
                ok, err = _ok(handler.performRequests_error_(list(reqs.values()), None))
                if not ok:  # one request failed (a model this Mac lacks): the others, one by one
                    log.info(f"[vision] Apple Vision: {str(err)[:80]}; retrying request by request")
                    for k, r in list(reqs.items()):
                        h = ns["VNImageRequestHandler"].alloc().initWithCGImage_options_(cg, {})
                        if not _ok(h.performRequests_error_([r], None))[0]:
                            reqs.pop(k)
                seen = Seen()
                if "labels" in reqs:
                    seen.labels = self._labels(reqs["labels"])
                if "text" in reqs:
                    seen.text = self._text(reqs["text"])
                if "hands" in reqs:
                    seen.hands = self._hands(reqs["hands"], img.width / max(1, img.height))
                del data  # (the bytes the CGImage reads from lived until here)
                return seen
        finally:
            self._run_lock.release()

    @staticmethod
    def _labels(req: Any) -> list[tuple[str, float]]:
        out = [(str(o.identifier()).replace("_", " "), float(o.confidence())) for o in (req.results() or [])]
        out.sort(key=lambda t: -t[1])
        return [(n, round(c, 3)) for n, c in out if c >= MIN_LABEL_CONF][:TOP_K]

    @staticmethod
    def _text(req: Any) -> list[str]:
        out: list[str] = []
        for o in req.results() or []:
            cands = o.topCandidates_(1)
            if not cands or float(cands[0].confidence()) < MIN_TEXT_CONF:
                continue
            s = " ".join(str(cands[0].string()).split())[:MAX_TEXT_CHARS]
            if s and s not in out:
                out.append(s)
            if len(out) >= MAX_TEXT_LINES:
                break
        return out

    def _hands(self, req: Any, aspect: float) -> list[Hand]:
        out: list[Hand] = []
        for o in (req.results() or [])[:MAX_HANDS]:
            got = o.recognizedPointsForJointsGroupName_error_(self._group, None)
            pts = got[0] if isinstance(got, tuple) else got
            points: dict[str, Point] = {}
            for name, pt in dict(pts or {}).items():
                ours = self._joints.get(str(name))
                if ours is not None:
                    loc = pt.location()
                    points[ours] = (float(loc.x), float(loc.y), float(pt.confidence()))
            box = hand_box(points)
            if box is None:
                continue
            side = {-1: "left", 1: "right"}.get(int(o.chirality()) if hasattr(o, "chirality") else 0, "")
            out.append(Hand(box=box, fingers=count_fingers(points, aspect), side=side, conf=round(float(o.confidence()), 3)))
        return out


_engine: Engine | None = None  # a test's engine (set_engine)
_vision: _Vision | None = None
_here: bool | None = None


def set_engine(e: Engine | None) -> None:
    """Use ``e`` instead of Apple Vision (tests); None: Vision again."""
    global _engine
    _engine = e


def _vision_here() -> bool:
    global _here
    if _here is None:
        try:
            import objc  # noqa: F401
            import Quartz  # noqa: F401

            _here = sys.platform == "darwin" and Path(_FRAMEWORK).exists()
        except ImportError:
            _here = False
    return _here


def available() -> bool:
    """Can Apple Vision look at a frame here (nothing is loaded to find out)?"""
    return _engine is not None or _vision_here()


def _get() -> Engine:
    global _vision
    if _engine is not None:
        return _engine
    if not _vision_here():
        raise Unavailable("Apple Vision isn't available here (it needs macOS and PyObjC)")
    if _vision is None:
        _vision = _Vision()
    return _vision


def analyse(img: Image.Image, *, labels: bool = True, text: bool = True, hands: bool = True) -> Seen:
    """Everything asked for in one pass over ``img``. Raises Unavailable (or EffectRefused with real
    effects off)."""
    t0 = time.perf_counter()
    seen = _get().analyse(img, labels=labels, text=text, hands=hands)
    seen.ms = round((time.perf_counter() - t0) * 1000, 1)
    return seen


def classify(img: Image.Image) -> list[tuple[str, float]]:
    """The frame's top labels: [(label, confidence 0..1)], best first."""
    return analyse(img, labels=True, text=False, hands=False).labels


def text(img: Image.Image) -> list[str]:
    """The words on the frame, a line each (fast mode, at most :data:`MAX_TEXT_LINES`)."""
    return analyse(img, labels=False, text=True, hands=False).text


def hands_summary(found: list[Hand]) -> dict[str, Any]:
    """{count, fingers (all hands' raised fingers, when every hand's are known), hands: [...]}."""
    counts = [h.fingers for h in found]
    return {
        "count": len(found),
        "fingers": sum(c for c in counts if c is not None) if found and None not in counts else None,
        "hands": [{"box": list(h.box), "fingers": h.fingers, "side": h.side} for h in found],
    }


def hands(img: Image.Image) -> dict[str, Any]:
    """The hands in the frame: {count, fingers, hands: [{box, fingers, side}]}."""
    return hands_summary(analyse(img, labels=False, text=False, hands=True).hands)


__all__ = [
    "Engine",
    "Hand",
    "Seen",
    "analyse",
    "available",
    "classify",
    "count_fingers",
    "hand_box",
    "hands",
    "hands_summary",
    "set_engine",
    "text",
]
