"""Object boxes on a camera frame, from one of two detectors (the ``vision_detector`` setting):

* YOLOE-26s (:mod:`.yolo`, Ultralytics, AGPL-3.0): the user's choice — NEO 1's YOLO family, open
  vocabulary (4,585 words), Core ML on the Neural Engine, ~9 ms a frame; its model is made by
  ``scripts/get_yoloe.py`` into NEO's models dir. "auto" (the default) uses it whenever its model is
  there, "yoloe" only it.
* D-FINE small (ustc-community/dfine-small-coco, Apache-2.0, 40 MB, the 80 COCO objects) through
  transformers on the CPU: the fallback — "auto" without YOLOE's model, or the setting is its repo
  id or a folder.

D-FINE never runs on the GPU: PyTorch's MPS stream is the process's one (neo2.core.gpu), and a
detector frame there beside a Laya pass aborted NEO. Taking turns under the GPU lock instead cost both
sides — measured on an M3 Pro, 640 px, a preview frame every 0.4 s: D-FINE 33 ms alone on MPS but
69 ms (p90 94) waiting for Laya, and Laya's passes p90 36 → 38 ms, worst 40 → 81 ms; a Laya load or
retrain would hold the boxes up for seconds. On the CPU a frame takes ~150 ms (p90 160, ~180 while
Laya works; the first ~280) and Laya's passes don't notice it. (YOLOE has no torch at all.)

Loaded on first use (:func:`warm` loads it ahead, in a thread, when the camera turns on) and kept:
both are small. Each engine says from which score its boxes come back and from which they are sure
enough to show on their own (:func:`scores`: their scales differ); at most :data:`MAX_BOXES` come
back. Boxes are 0..1 of the frame, origin top left — what the live preview draws. Which of them the
card draws and the brains hear of is neo2.core.vision.pipeline's.

Never downloads on its own: YOLOE's model is only ever made by get_yoloe.py (its download needs
``--yes``); D-FINE's weights are looked for in NEO's models dir (``<models_dir>/dfine-small-coco``,
or a Hugging Face cache there), and only fetched from Hugging Face when the user said yes
(``vision_download_ok``). Missing: :class:`~neo2.core.vision.Unavailable` with a message saying so.

Loading either is a "model" effect (neo2.core.effects): refused in NEO's tests — a test sets a fake
detector (:func:`set_engine`).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Protocol, TypedDict

from PIL import Image

from neo2.core import effects, logger
from neo2.core.vision import Unavailable, yolo

log = logger(__name__)

REPO = "ustc-community/dfine-small-coco"
MAX_EDGE = 640
MIN_SCORE = 0.5  # a detector that doesn't say its own (a test's)
MAX_BOXES = 30
_RUN_WAIT_S = 3.0


class Det(TypedDict):
    label: str
    score: float
    box: dict[str, float]  # x, y, w, h — 0..1 of the frame, origin top left


class Detector(Protocol):
    """A detector: boxes from ``min_score`` up, best first, at most ``limit``. It may say its scale
    with ``floor`` (its boxes come back from there up) and ``sure`` (shown on their own from there)."""

    def detect(self, img: Image.Image, *, min_score: float, limit: int) -> list[Det]: ...


def _setting(name: str, default: Any) -> Any:
    try:
        from neo2.core.config import settings

        return getattr(settings(), name, default)
    except Exception:  # noqa: BLE001 — no agent bound (library mode): the default
        return default


def _models_dir() -> Path | None:
    try:
        from neo2.core.config import settings

        return Path(settings().models_dir)
    except Exception:  # noqa: BLE001
        return None


_MODES = ("", "auto", "yoloe")  # the setting's words that aren't D-FINE's repo or folder


def mode() -> str:
    """Which detector the setting asks for: "auto" (YOLOE when its model is here, else D-FINE),
    "yoloe", or "dfine" (a repo id or a folder)."""
    v = str(_setting("vision_detector", "auto") or "auto").strip().lower()
    return v if v in ("auto", "yoloe") else "auto" if not v else "dfine"


def source() -> tuple[str, bool]:
    """D-FINE's (where to load the weights from, may it download). A folder when one holds them: the setting
    itself when it is a folder, else ``<models_dir>/<name>`` (the orchestrator's fetch, e.g.
    ``hf download ustc-community/dfine-small-coco --local-dir ~/.neo/models/dfine-small-coco``);
    else the repo id, looked up only in the Hugging Face cache unless downloads are allowed."""
    repo = str(_setting("vision_detector", "") or "").strip()
    if repo.lower() in _MODES:
        repo = REPO
    if Path(repo).expanduser().is_dir():
        return str(Path(repo).expanduser()), False
    md = _models_dir()
    if md is not None:
        tail = repo.rstrip("/").split("/")[-1]
        for d in (md / tail, md / repo.replace("/", "--"), md / repo):
            if (d / "config.json").is_file():
                return str(d), False
    return repo, bool(_setting("vision_download_ok", False))


def _missing_words() -> str:
    md = _models_dir()
    where = f" in {md}" if md is not None else ""
    return (
        "The object detector (D-FINE small, 40 MB) isn't downloaded yet"
        f"{where}. NEO only downloads it once you say yes (the vision_download_ok setting)."
    )


def _fit(img: Image.Image) -> Image.Image:
    img = img if img.mode == "RGB" else img.convert("RGB")
    if max(img.size) <= MAX_EDGE:
        return img
    s = MAX_EDGE / max(img.size)
    return img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BILINEAR)


def to_dets(
    scores: list[float], labels: list[str], boxes: list[list[float]], w: int, h: int, *,
    min_score: float = MIN_SCORE, limit: int = MAX_BOXES,
) -> list[Det]:  # fmt: skip
    """Pixel boxes (x0, y0, x1, y1) → :class:`Det`, 0..1 of a w × h frame, best first, bounded."""
    out: list[Det] = []
    for s, lab, b in sorted(zip(scores, labels, boxes, strict=True), key=lambda t: -t[0]):
        if s < min_score or len(out) >= limit:
            continue
        x0, y0 = max(0.0, min(1.0, b[0] / w)), max(0.0, min(1.0, b[1] / h))
        x1, y1 = max(0.0, min(1.0, b[2] / w)), max(0.0, min(1.0, b[3] / h))
        if x1 <= x0 or y1 <= y0:
            continue
        box = {"x": round(x0, 4), "y": round(y0, 4), "w": round(x1 - x0, 4), "h": round(y1 - y0, 4)}
        out.append(Det(label=str(lab), score=round(float(s), 3), box=box))
    return out


class _DFine:
    """D-FINE small through transformers, on the CPU. Loaded once; one frame at a time."""

    floor = 0.4  # its boxes from here up come back...
    sure = 0.5  # ...and are shown on their own from here (COCO's 80 words: NEO 1's YOLOv8n drew from 0.42)

    def __init__(self) -> None:
        self._model: Any = None
        self._proc: Any = None
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()

    def _load(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            src, download = source()
            effects.guard("model", f"load the D-FINE object detector ({Path(src).name}) into memory")
            try:
                import torch  # noqa: F401 — the models run on it: missing, the message below says so
                from transformers import AutoImageProcessor, DFineForObjectDetection
            except ImportError as e:
                raise Unavailable("The object detector needs transformers and torch (NEO's vision extra)") from e
            md = _models_dir()
            kw: dict[str, Any] = {"local_files_only": not download}
            if md is not None and not Path(src).is_dir():
                kw["cache_dir"] = str(md / "hf")
            t0 = time.perf_counter()
            try:
                proc = AutoImageProcessor.from_pretrained(src, **kw)
                model = DFineForObjectDetection.from_pretrained(src, **kw)
            except (OSError, ValueError) as e:  # not in the folder or the cache (and no download allowed)
                if download:
                    raise Unavailable(f"The object detector couldn't be downloaded ({type(e).__name__})") from e
                raise Unavailable(_missing_words()) from e
            self._model, self._proc = model.eval(), proc  # from_pretrained leaves it on the CPU
            log.info(f"[vision] D-FINE loaded on the CPU in {(time.perf_counter() - t0) * 1000:.0f} ms")

    def _run(self, img: Image.Image, min_score: float) -> tuple[list[float], list[str], list[list[float]]]:
        import torch

        inputs = self._proc(images=img, return_tensors="pt")
        with torch.inference_mode():
            outputs = self._model(**inputs)
        res = self._proc.post_process_object_detection(
            outputs, target_sizes=torch.tensor([(img.height, img.width)]), threshold=min_score
        )[0]
        names = self._model.config.id2label
        scores = [float(s) for s in res["scores"].tolist()]
        labels = [str(names.get(int(i), int(i))) for i in res["labels"].tolist()]
        return scores, labels, [list(map(float, b)) for b in res["boxes"].tolist()]

    def detect(self, img: Image.Image, *, min_score: float, limit: int) -> list[Det]:
        self._load()
        if not self._run_lock.acquire(timeout=_RUN_WAIT_S):
            return []  # the previous frame is still being read: the preview skips this one
        try:
            img = _fit(img)
            return to_dets(*self._run(img, min_score), img.width, img.height, min_score=min_score, limit=limit)
        finally:
            self._run_lock.release()

    @property
    def loaded(self) -> bool:
        return self._model is not None


_engine: Detector | None = None  # a test's detector (set_engine)
_dfine: _DFine | None = None
_yolo: yolo.YOLOE | None = None
_made = threading.Lock()  # one instance of each, whichever thread (the warm-up, a frame) asks first
_warming: threading.Thread | None = None


def set_engine(e: Detector | None) -> None:
    """Use ``e`` instead of the real detectors (tests); None: the real ones again."""
    global _engine
    _engine = e


def _dfine_available() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False, "The object detector needs transformers and torch (NEO's vision extra)"
    src, download = source()
    if Path(src).is_dir() or download:
        return True, ""
    md = _models_dir()
    if md is not None and any((md / "hf").glob(f"models--{src.replace('/', '--')}/snapshots/*/config.json")):
        return True, ""
    return False, _missing_words()


def engine() -> str:
    """The detector a frame goes to now: "test", "yoloe" or "dfine" (see :func:`mode`)."""
    if _engine is not None:
        return "test"
    m = mode()
    return "dfine" if m == "dfine" or (m == "auto" and not yolo.available()[0]) else "yoloe"


def _get() -> Detector:
    global _dfine, _yolo
    if _engine is not None:
        return _engine
    choice = engine()
    with _made:
        if choice == "yoloe":
            if _yolo is None:
                _yolo = yolo.YOLOE()
            return _yolo
        if _dfine is None:
            _dfine = _DFine()
        return _dfine


def available() -> tuple[bool, str]:
    """(can boxes be drawn here, why not). Looks for the model files; loads nothing. "auto" with
    neither detector here says what YOLOE needs (the one NEO ships with)."""
    if _engine is not None:
        return True, ""
    m = mode()
    if m == "dfine":
        return _dfine_available()
    ok, why = yolo.available()
    if ok or m == "yoloe":
        return ok, why
    return (True, "") if _dfine_available()[0] else (False, why)


def loaded() -> bool:
    return _engine is not None or bool(getattr(_get(), "loaded", False))


def scores() -> tuple[float, float]:
    """(floor, sure) of the detector a frame goes to now: :func:`detect` returns its boxes from
    ``floor`` up; from ``sure`` up a box is shown on its own (below it, only when Apple Vision names
    the same thing: neo2.core.vision.pipeline)."""
    d = _get()
    floor = float(getattr(d, "floor", MIN_SCORE))
    return floor, max(floor, float(getattr(d, "sure", floor)))


def detect(img: Image.Image, *, min_score: float | None = None, limit: int = MAX_BOXES) -> list[Det]:
    """The objects in the frame: [{label, score, box: {x, y, w, h}}], best first, at most ``limit``,
    from ``min_score`` up (the detector's own floor when None). Raises Unavailable (not installed,
    not there) — or EffectRefused with real effects off."""
    d = _get()
    floor = scores()[0] if min_score is None else float(min_score)
    return d.detect(img, min_score=floor, limit=max(0, min(int(limit), MAX_BOXES)))


def warm() -> bool:
    """Load the detector in the background (the camera just turned on), so the first frame's boxes
    don't wait on it (YOLOE's first load of a new model compiles it for the Neural Engine: ~2.5 s,
    then ~0.05 s). Never raises; True when a load was started."""
    global _warming
    if (_warming is not None and _warming.is_alive()) or not available()[0] or loaded():
        return False

    def load() -> None:
        try:
            d = _get()
            if isinstance(d, _DFine | yolo.YOLOE):
                d._load()
        except (Exception, effects.EffectRefused) as e:  # noqa: BLE001 — a warm-up never fails anything
            log.info(f"[vision] detector warm-up skipped: {str(e)[:120]}")

    _warming = threading.Thread(target=load, name="neo-vision-warm", daemon=True)
    _warming.start()
    return True


__all__ = [
    "MAX_BOXES", "REPO", "Det", "Detector", "available", "detect", "engine", "loaded", "mode", "scores",
    "set_engine", "source", "to_dets", "warm",
]  # fmt: skip
