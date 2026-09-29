"""Object boxes on a camera frame: YOLOE-26s prompt-free (Ultralytics, AGPL-3.0: THIRD_PARTY_NOTICES.md),
through Core ML on the Mac's Neural Engine.

The user's choice: NEO 1's YOLO boxes were the ones they liked, and YOLOE-26s is that family's
open-vocabulary detector — 4,585 words (a mug, a guitar, a person...) where COCO's detectors know 80 —
in about 9 ms a frame (measured on an M3 Pro: a 1280x720 frame letterboxed to 640x384, the fp16
model with ~95% of its work on the Neural Engine, decoded: 8-9 ms from frame to boxes). No torch
runs here, so it never touches PyTorch's one MPS stream (neo2.core.gpu) and never waits on Laya.

The model: ``<models_dir>/yoloe-26s-pf.mlpackage`` (23 MB), made from Ultralytics' weights by
``scripts/get_yoloe.py`` (its own export: Ultralytics' Core ML export mislabels most of the 4,585
words in fp16). Compiled once into ``<models_dir>/yoloe-26s-pf.mlmodelc`` beside it — the Neural
Engine's first load of a new compiled model takes ~2 s, a known one ~0.06 s — and loaded on first
use (:func:`~neo2.core.vision.detect.warm` loads it ahead when the camera turns on). Never
downloaded here: missing, :class:`~neo2.core.vision.Unavailable` says how to make it.

What it gives: final boxes (YOLO26 is end-to-end: no NMS), one word each, x1 y1 x2 y2 in the
letterboxed input's pixels — :func:`letterbox` and :func:`decode` are Ultralytics' own pre- and
post-processing (its predict gives the same boxes, checked by get_yoloe.py).

Which words a box may take is set in the model itself (its ``classes`` input, a weight per word): a
hidden word never wins, so its box takes the best word allowed or drops out. The vocabulary is a
photo-tagging one (RAM++'s: "fairy", "share", "anniversary", "jazz artist"...), and on camera frames
its confident words were mostly wrong: on 55 sample frames (photos, a portrait with things pasted in,
rendered emoji) it named a baseball "fairy", a guitar "type, dagger", a chalkboard "share, energy", a
mouse "urinal" and a person "politician" or "jazz artist". So by default (the ``vision_words``
setting) a box may only take a word of :data:`EVERYDAY` — ~230 things held up to a camera or kept on
a desk, COCO's 80 among them, people as "person" — or of :data:`NOT_SHOWN`, body parts and clothes,
so a hand's box is called a hand (and left off the card) rather than the nearest everyday thing. On
the same frames most wrong words became no box, and every person "person". "all" is the whole
vocabulary but the words that are never a thing in front of the camera: the ones it gives frames
with nothing in them (``blank`` in its metadata: sky, darkness, still life), words whose first sense
in the Mac's dictionary isn't a noun (``pos``: "squeeze", "lick" — but not the objects among them,
:data:`OBJECTS_NOT_NOUNS`), and :data:`PHOTO_WORDS` about the picture rather than what's in it.
Which boxes the card draws — people as "person", no body parts, no duplicates — is
neo2.core.vision.pipeline's.

Bounded: a frame is at most :data:`MAX_PIXELS`, scaled into the model's fixed 640x384 input, and
never written anywhere; one frame at a time (a second waits :data:`_RUN_WAIT_S` at most, then is
skipped). A model that won't load (a copy cut short on its way into the models dir) is remembered:
not compiled again every frame, until its files change (:func:`broken`; "auto" then uses D-FINE).
Loading it is a "model" effect (neo2.core.effects): refused in NEO's tests.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
import threading
import time
from functools import cache
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from neo2.core import effects, logger
from neo2.core.vision import Unavailable

if TYPE_CHECKING:
    from neo2.core.vision.detect import Det

log = logger(__name__)

PKG = "yoloe-26s-pf.mlpackage"
COMPILED = "yoloe-26s-pf.mlmodelc"
SHAPE = (384, 640)  # (height, width) when the model doesn't say (its "imgsz" metadata does)
PAD = 114  # Ultralytics' letterbox grey
MAX_PIXELS = 40_000_000  # a bigger picture isn't a camera frame: refused
_STAMP = "neo-source"  # in the compiled folder: which package it was compiled from
_RUN_WAIT_S = 3.0

# Words that name things although the Mac's dictionary gives them as a verb or an adjective first
# ("tie", "watch", "can"): COCO's objects among them, and things held up to a camera.
OBJECTS_NOT_NOUNS = frozenset(
    {
        "tie", "train", "bear", "remote", "sink", "watch", "can", "goggles", "binocular", "drone", "tire", "stamp",
        "puzzle", "waffle", "comic", "cosmetic", "document", "print", "drink", "uniform", "succulent", "foil",
        "certificate", "husky", "siamese", "dog food", "suv", "atm",
    }
)  # fmt: skip
# Words about the picture, not what's in it: never an object's name.
PHOTO_WORDS = frozenset(
    {
        "3D CG rendering", "clip art", "image", "picture", "photo", "snapshot", "screenshot", "selfie",
        "portrait", "publicity portrait", "id photo", "file photo", "old photo", "couple photo",
        "family photo", "group photo", "team photo", "wedding photo", "photography", "street photography",
        "wedding photography", "close-up", "face close-up", "still life", "studio shot", "model",
    }
)  # fmt: skip
# The everyday vocabulary (the default): things held up to a camera or kept on a desk, COCO's 80
# objects among them (NEO 1's YOLO words), and people as one word, "person".
EVERYDAY = frozenset(
    {
        "person", "bicycle", "car", "motorbike", "motorcycle", "plane", "bus", "train", "truck", "boat", "bird",
        "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "parrot", "fish",
        "goldfish", "hamster", "rabbit", "turtle", "backpack", "umbrella", "handbag", "tie", "luggage",
        "frisbee", "sports ball", "baseball bat", "baseball glove", "skateboard", "tennis racket", "bottle",
        "wine glass", "beer glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "plant", "bed",
        "table", "television", "laptop", "mouse", "remote", "keyboard", "phone", "smartphone", "microwave",
        "oven", "toaster", "sink", "fridge", "book", "clock", "vase", "scissors", "teddy", "stuffed toy",
        "hair drier", "toothbrush", "pen", "pencil", "fountain pen", "crayon", "eraser", "ruler", "stapler",
        "tape", "notebook", "notepad", "paper", "envelope", "card", "credit card", "business card", "ticket",
        "receipt", "passport", "coin", "money", "calculator", "tablet", "tablet computer", "computer", "monitor",
        "computer monitor", "computer screen", "speaker", "microphone", "earphone", "camera", "webcam",
        "game controller", "joystick", "charger", "cable", "battery", "portable battery", "lamp", "candle", "mug",
        "coffee cup", "teacup", "paper cup", "glass bottle", "beer bottle", "wine bottle", "baby bottle",
        "thermos bottle", "jar", "can", "beer can", "box", "cardboard box", "gift box", "bag", "paper bag",
        "shopping bag", "key", "keycard", "watch", "alarm clock", "glasses", "sunglasses", "ring", "necklace",
        "bracelet", "earring", "hat", "cap", "baseball hat", "glove", "sock", "shoe", "toy", "doll", "figurine",
        "puppet", "ball", "baseball", "basketball", "football", "rugby ball", "balloon", "guitar", "violin",
        "ukulele", "flute", "drum", "magazine", "comic book", "document", "printed page", "blackboard", "chalk",
        "whiteboard", "flower", "bouquet", "rose", "cactus", "succulent", "fruit", "lemon", "lime", "grape",
        "strawberry", "pear", "peach", "avocado", "tomato", "potato", "onion", "egg", "bread", "chocolate",
        "chocolate bar", "candy", "candy bar", "chocolate chip cookie", "biscuit", "snack", "potato chip",
        "coffee", "tea", "juice", "drink", "soda", "beer", "wine", "toothpaste", "comb", "hairbrush", "brush",
        "lipstick", "perfume", "cosmetic", "soap", "towel", "tissue", "napkin", "plate", "chopstick", "spatula",
        "screwdriver", "hammer", "wrench", "drill", "tool", "medicine", "thermometer",
    }
)  # fmt: skip
# Allowed beside them but never drawn (neo2.core.vision.pipeline leaves these out): a hand's box is
# called a hand and a shirt's a shirt, instead of taking the nearest everyday word.
NOT_SHOWN = frozenset(
    {
        "hand", "arm", "shoulder", "face", "eye", "nose", "mouth", "ear", "hair", "long hair", "short hair", "skin",
        "beard", "head", "neck", "leg", "foot", "body", "forehead", "chest", "suit", "business suit", "shirt",
        "dress shirt", "t-shirt", "jacket", "coat", "sweater", "hoodie", "dress", "uniform", "military uniform",
        "blouse", "vest", "polo shirt", "cardigan", "sweatshirt", "sleeve", "neckband", "sports coat",
    }
)  # fmt: skip
VOCABULARIES = ("everyday", "all")


def _fit(frame_size: tuple[int, int], shape: tuple[int, int]) -> tuple[float, int, int, int, int]:
    """(scale, new width, new height, left pad, top pad) for a (w, h) frame into a (h, w) input:
    Ultralytics' LetterBox rounding, which :func:`decode` mirrors."""
    (w0, h0), (h, w) = frame_size, shape
    r = min(h / h0, w / w0)
    nw, nh = max(1, round(w0 * r)), max(1, round(h0 * r))
    return r, nw, nh, round((w - nw) / 2 - 0.1), round((h - nh) / 2 - 0.1)


def letterbox(img: Image.Image, shape: tuple[int, int] = SHAPE) -> Image.Image:
    """``img`` fitted into the model's (height, width), proportions kept, centred on grey 114. A frame
    exactly twice the size (1280x720) is halved by averaging 2x2 pixels, as OpenCV's resize in
    Ultralytics' predict does; any other size is scaled bilinearly."""
    if img.width < 1 or img.height < 1 or img.width * img.height > MAX_PIXELS:
        raise ValueError(f"not a camera frame ({img.width}x{img.height})")
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    _, nw, nh, left, top = _fit(img.size, shape)
    if img.size == (nw * 2, nh * 2):
        img = img.reduce(2)
    elif img.size != (nw, nh):
        img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (shape[1], shape[0]), (PAD, PAD, PAD))
    canvas.paste(img.convert("RGB"), (left, top))
    return canvas


def decode(
    out: dict[str, Any], frame_size: tuple[int, int], names: list[str], *, conf: float,
    shape: tuple[int, int] = SHAPE,
) -> list[tuple[str, float, float, float, float, float]]:  # fmt: skip
    """The model's outputs → [(word, score, x, y, w, h)], best first: x, y, w, h are 0..1 of the
    original (w, h) ``frame_size``, origin top left, clipped to it; only scores from ``conf`` up."""
    w0, h0 = frame_size
    r, _, _, left, top = _fit(frame_size, shape)
    boxes = np.asarray(out["boxes"], np.float32).reshape(-1, 4)
    scores = np.asarray(out["scores"], np.float32).reshape(-1)
    labels = np.asarray(out["labels"]).reshape(-1).astype(np.int64)
    dets = []
    for (x1, y1, x2, y2), s, c in zip(boxes.tolist(), scores.tolist(), labels.tolist(), strict=True):
        if not s >= conf or not 0 <= c < len(names):  # (NaN too)
            continue
        x1, x2 = (min(max((v - left) / r, 0.0), w0) for v in (x1, x2))
        y1, y2 = (min(max((v - top) / r, 0.0), h0) for v in (y1, y2))
        if x2 - x1 >= 1 and y2 - y1 >= 1:
            dets.append((names[c], float(s), x1 / w0, y1 / h0, (x2 - x1) / w0, (y2 - y1) / h0))
    return sorted(dets, key=lambda d: -d[1])


def weights(
    names: list[str], pos: list[str], blank: list[int], *, words: frozenset[str] | None = None
) -> np.ndarray:
    """The ``classes`` input, (1, words) float32, 1 for a word a box may take and 0 for a hidden one:
    with ``words``, those of the model's words only; else every word but those that are never a thing
    in front of the camera (see the module doc)."""
    if words is not None:
        return np.array([[1.0 if n in words else 0.0 for n in names]], np.float32).reshape(1, len(names))
    w = np.ones((1, len(names)), np.float32)
    for i, n in enumerate(names):
        p = pos[i] if i < len(pos) else ""
        if n in PHOTO_WORDS or (p not in ("noun", "phrase", "") and n not in OBJECTS_NOT_NOUNS):
            w[0, i] = 0.0
    w[0, [i for i in blank if 0 <= i < len(names)]] = 0.0
    return w


def vocabulary() -> str:
    """The ``vision_words`` setting: "everyday" (the default, and for anything it doesn't know) or "all"."""
    try:
        from neo2.core.config import settings

        v = str(getattr(settings(), "vision_words", "") or "").strip().lower()
    except Exception:  # noqa: BLE001 — no agent bound (library mode)
        v = ""
    return v if v in VOCABULARIES else "everyday"


def _models_dir() -> Path | None:
    try:
        from neo2.core.config import settings

        return Path(settings().models_dir)
    except Exception:  # noqa: BLE001 — no agent bound (library mode)
        return None


def package() -> Path | None:
    """Where the model is: ``<models_dir>/yoloe-26s-pf.mlpackage`` when it's there, else None."""
    md = _models_dir()
    p = md / PKG if md is not None else None
    return p if p is not None and (p / "Manifest.json").is_file() else None


@cache
def runnable() -> str:
    """Why Core ML can't run it on this Mac ("" when it can): macOS 14+, pyobjc's Core ML. (Asked
    every preview frame; neither changes while NEO runs.)"""
    if sys.platform != "darwin":
        return "The object detector runs on macOS only"
    try:
        major = int((platform.mac_ver()[0] or "0").split(".")[0])
    except ValueError:
        major = 0
    if major < 14:
        return "The object detector (YOLOE, Core ML) needs macOS 14 or later"
    if find_spec("CoreML") is None or find_spec("Quartz") is None:
        return "The object detector needs pyobjc-framework-CoreML (NEO's vision extra)"
    return ""


def missing_words() -> str:
    md = _models_dir()
    where = f" in {md}" if md is not None else ""
    return (
        f"The object detector (YOLOE-26s, 23 MB) isn't set up yet{where}: scripts/get_yoloe.py makes it "
        "from Ultralytics' weights (40 MB, downloaded only with --yes)."
    )


def available() -> tuple[bool, str]:
    """(can it run here, why not). Looks for the files and at an earlier failed load; loads nothing."""
    if why := runnable():
        return False, why
    pkg = package()
    if pkg is None:
        return False, missing_words()
    return (False, why) if (why := broken(pkg)) else (True, "")


def _stamp(pkg: Path) -> str:
    """The package's files, by size and time: a rebuilt package is compiled afresh."""
    files = sorted(f for f in pkg.rglob("*") if f.is_file())
    return json.dumps([[str(f.relative_to(pkg)), f.stat().st_size, f.stat().st_mtime_ns] for f in files])


_broken: dict[str, tuple[str, str]] = {}  # package path -> (its stamp, why it didn't load)


def broken(pkg: Path | None = None) -> str:
    """Why the package (the models dir's when None) failed to load, when it did and hasn't changed
    since; "" otherwise. Such a package isn't tried again until its files change: a copy cut short
    would otherwise be compiled afresh (~2.5 s of CPU) every preview frame."""
    pkg = pkg if pkg is not None else package()
    got = _broken.get(str(pkg)) if pkg is not None else None
    if got is None:
        return ""
    try:
        return got[1] if _stamp(pkg) == got[0] else ""
    except OSError:
        return got[1]


def _metadata(model: Any, CoreML: Any) -> dict[str, str]:
    meta = model.modelDescription().metadata() or {}
    return dict(meta.get(CoreML.MLModelCreatorDefinedKey) or {})


def _to_numpy(arr: Any, CoreML: Any) -> np.ndarray:
    """An MLMultiArray → numpy (a copy), whatever its element type and strides."""
    dt = {
        CoreML.MLMultiArrayDataTypeFloat16: np.float16, CoreML.MLMultiArrayDataTypeFloat32: np.float32,
        CoreML.MLMultiArrayDataTypeInt32: np.int32, CoreML.MLMultiArrayDataTypeDouble: np.float64,
    }[arr.dataType()]  # fmt: skip
    shape, strides = [int(s) for s in arr.shape()], [int(s) for s in arr.strides()]
    got: dict[str, bytes] = {}

    def grab(ptr: Any, size: int) -> None:
        got["b"] = bytes(ptr) if isinstance(ptr, bytes | bytearray | memoryview) else bytes(ptr[:size])

    arr.getBytesWithHandler_(grab)
    flat = np.frombuffer(got["b"], dt)
    item = np.dtype(dt).itemsize
    return np.lib.stride_tricks.as_strided(flat, shape, [s * item for s in strides]).copy()


class YOLOE:
    """YOLOE-26s through Core ML (pyobjc), on the CPU and Neural Engine. Loaded once; one frame at a time."""

    floor = 0.25  # boxes from here up come back (Ultralytics' own default)...
    sure = 0.45  # ...and from here up are drawn on their own (NEO 1 drew YOLOv8n's from 0.42)

    def __init__(self) -> None:
        self._model: Any = None
        self._names: list[str] = []
        self._tags: tuple[list[str], list[int]] = ([], [])  # (pos, blank): the model's word tags
        self._shape = SHAPE
        self._classes: dict[str, Any] = {}  # vocabulary -> its word weights as an MLMultiArray, made once
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _compiled(self, pkg: Path, CoreML: Any, NSURL: Any) -> Path:
        """The compiled model beside the package (compiled now if it isn't, or is of an older one)."""
        dest = pkg.parent / COMPILED
        stamp = _stamp(pkg)
        try:
            if (dest / _STAMP).read_text() == stamp:
                return dest
        except OSError:
            pass
        url, err = CoreML.MLModel.compileModelAtURL_error_(NSURL.fileURLWithPath_(str(pkg)), None)
        if url is None:
            raise Unavailable(f"The object detector's model couldn't be compiled ({str(err)[:80]})")
        tmp = Path(str(url.path()))
        part = dest.with_name(dest.name + ".part")
        try:
            shutil.rmtree(part, ignore_errors=True)
            shutil.move(str(tmp), str(part))
            (part / _STAMP).write_text(stamp)
            shutil.rmtree(dest, ignore_errors=True)
            part.rename(dest)
            return dest
        except OSError:  # the models folder isn't writable: this once, from where Core ML put it
            shutil.rmtree(part, ignore_errors=True)
            return tmp if tmp.is_dir() else dest

    def _load(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            if why := runnable():
                raise Unavailable(why)
            pkg = package()
            if pkg is None:
                raise Unavailable(missing_words())
            if why := broken(pkg):
                raise Unavailable(why)  # (the frames that waited on the failed load: at once)
            effects.guard("model", "load the YOLOE object detector (Core ML) into memory")
            stamp = _stamp(pkg)
            try:
                self._open(pkg)
            except Exception as e:  # noqa: BLE001 — Core ML's own errors too: remembered, not retried
                why = str(e) if isinstance(e, Unavailable) else f"The object detector's model couldn't be loaded ({type(e).__name__})"
                _broken[str(pkg)] = (stamp, why)
                log.info(f"[vision] YOLOE not loaded: {why[:160]}")
                raise Unavailable(why) from e
            _broken.pop(str(pkg), None)

    def _open(self, pkg: Path) -> None:
        import CoreML
        from Foundation import NSURL

        from neo2.core.decide._objc import pool

        t0 = time.perf_counter()
        cfg = CoreML.MLModelConfiguration.alloc().init()
        cfg.setComputeUnits_(CoreML.MLComputeUnitsCPUAndNeuralEngine)  # the GPU is Laya's

        def load(path: Path) -> tuple[Any, Any]:
            return CoreML.MLModel.modelWithContentsOfURL_configuration_error_(NSURL.fileURLWithPath_(str(path)), cfg, None)

        with pool():
            model, err = load(path := self._compiled(pkg, CoreML, NSURL))
            if model is None and path == pkg.parent / COMPILED:  # a kept copy this macOS can't read
                shutil.rmtree(path, ignore_errors=True)
                model, err = load(self._compiled(pkg, CoreML, NSURL))
            if model is None:
                raise Unavailable(f"The object detector's model couldn't be loaded ({str(err)[:80]})")
            meta = _metadata(model, CoreML)
            try:
                names = [str(n) for n in json.loads(meta["names"])]
                h, w = (int(v) for v in json.loads(meta.get("imgsz") or json.dumps(SHAPE)))
                pos = [str(p) for p in json.loads(meta.get("pos") or "[]")]
                blank = [int(i) for i in json.loads(meta.get("blank") or "[]")]
            except (KeyError, TypeError, ValueError) as e:
                raise Unavailable("The object detector's model isn't the one scripts/get_yoloe.py makes") from e
        self._names, self._tags, self._shape, self._classes = names, (pos, blank), (h, w), {}
        self._model = model
        shown = int(self._weights(vocabulary()).sum())
        log.info(
            f"[vision] YOLOE loaded (Core ML, Neural Engine) in {(time.perf_counter() - t0) * 1000:.0f} ms; "
            f"{len(names)} words, {shown} of them allowed ({vocabulary()})"
        )

    def _weights(self, vocab: str) -> np.ndarray:
        return weights(self._names, *self._tags, words=EVERYDAY | NOT_SHOWN if vocab == "everyday" else None)

    def _classes_for(self, vocab: str) -> Any:
        """The ``classes`` input for a vocabulary, as the MLMultiArray Core ML reads: made once each."""
        if (arr := self._classes.get(vocab)) is None:
            import CoreML

            arr, _ = CoreML.MLMultiArray.alloc().initWithShape_dataType_error_(
                [1, len(self._names)], CoreML.MLMultiArrayDataTypeFloat32, None
            )
            data = self._weights(vocab).tobytes()
            arr.getMutableBytesWithHandler_(lambda ptr, size, strides: data)  # (the block's return: the contents)
            self._classes[vocab] = arr
        return arr

    def _pixels(self, img: Image.Image) -> Any:
        """The letterboxed frame as the 32BGRA pixel buffer Core ML reads (it makes the model's RGB)."""
        import Quartz

        h, w = img.height, img.width
        err, pb = Quartz.CVPixelBufferCreate(None, w, h, Quartz.kCVPixelFormatType_32BGRA, None, None)
        if err != 0 or pb is None:
            raise RuntimeError(f"CVPixelBufferCreate: {err}")
        Quartz.CVPixelBufferLockBaseAddress(pb, 0)
        try:
            bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
            buf = Quartz.CVPixelBufferGetBaseAddress(pb).as_buffer(bpr * h)
            bgra = np.frombuffer(img.convert("RGBA").tobytes("raw", "BGRA"), np.uint8).reshape(h, w * 4)
            np.frombuffer(buf, np.uint8).reshape(h, bpr)[:, : w * 4] = bgra  # (rows may be padded)
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(pb, 0)
        return pb

    def _run(self, img: Image.Image) -> dict[str, np.ndarray]:
        import CoreML

        feats = {
            "image": CoreML.MLFeatureValue.featureValueWithPixelBuffer_(self._pixels(letterbox(img, self._shape))),
            "classes": CoreML.MLFeatureValue.featureValueWithMultiArray_(self._classes_for(vocabulary())),
        }
        provider, err = CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(feats, None)
        if provider is None:
            raise RuntimeError(f"Core ML input: {str(err)[:80]}")
        res, err = self._model.predictionFromFeatures_error_(provider, None)
        if res is None:
            raise RuntimeError(f"Core ML prediction: {str(err)[:80]}")
        return {k: _to_numpy(res.featureValueForName_(k).multiArrayValue(), CoreML) for k in ("boxes", "scores", "labels")}

    def detect(self, img: Image.Image, *, min_score: float, limit: int) -> list[Det]:
        from neo2.core.decide._objc import pool

        self._load()
        if not self._run_lock.acquire(timeout=_RUN_WAIT_S):
            return []  # the previous frame is still being read: the preview skips this one
        try:
            with pool():
                out = self._run(img)
            dets = decode(out, img.size, self._names, conf=min_score, shape=self._shape)[: max(0, limit)]
        finally:
            self._run_lock.release()
        box = ("x", "y", "w", "h")
        return [{"label": n, "score": round(s, 3), "box": dict(zip(box, (round(v, 4) for v in b), strict=True))} for n, s, *b in dets]


__all__ = [
    "COMPILED", "EVERYDAY", "MAX_PIXELS", "NOT_SHOWN", "OBJECTS_NOT_NOUNS", "PHOTO_WORDS", "PKG", "VOCABULARIES", "YOLOE",
    "available", "broken", "decode", "letterbox", "missing_words", "package", "runnable", "vocabulary", "weights",
]  # fmt: skip
