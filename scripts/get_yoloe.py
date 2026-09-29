#!/usr/bin/env python3
"""Fetch YOLOE-26s prompt-free (Ultralytics, AGPL-3.0, ~4,585 open-vocabulary classes) and turn it into
the Core ML model NEO's camera draws boxes with: ``<out>/yoloe-26s-pf.mlpackage``.

Why not Ultralytics' own ``export(format="coreml")``:

* its fp16 model carries the class index as a float, and fp16 can't hold odd whole numbers past 2048 —
  with 4,585 classes most labels come out as a neighbour ("underwater" → "underclothes"). Here the
  label is its own int32 output.
* NEO wants boxes and a word, not masks: the mask coefficients and 32x160x160 prototypes are left out.
* its export skips the proposal filter (LRPC) that Ultralytics' ``predict`` applies to the prompt-free
  head; here it is kept (as a multiply, so shapes stay fixed), so the Core ML boxes are ``predict``'s.
* the word of each box (an argmax over 4,585) is worked out for the 300 kept boxes only: over every
  anchor it ran on the CPU and took 4x the rest; now ~95% of the work is on the Neural Engine (~9 ms).
* an optional ``classes`` input — one weight per class, 0..1, default 1 — lets the caller hide words
  of the vocabulary at run time (a box then takes its best allowed word) without exporting again.

The model: input ``image`` RGB, height x width from the ``imgsz`` metadata — 384x640 by default (a 16:9
camera frame, as Ultralytics' predict letterboxes it at imgsz=640), ``--imgsz 640`` for any shape —
letterboxed: scaled to fit, centred on grey 114; the model scales 0..255 to 0..1 itself. Optional
``classes`` float32 (1, 4585). Outputs ``boxes`` (1, 300, 4) x1 y1 x2 y2 in the input's pixels, ``scores``
(1, 300) 0..1, ``labels`` (1, 300) int32 index into ``names``. YOLO26 is end-to-end: these are final
boxes, no NMS; keep the ones over a score. :func:`letterbox` and :func:`decode` are the reference
pre/post-processing (numpy + PIL). Metadata (JSON strings): ``names``, ``imgsz`` [h, w], ``pos`` (each
word's part of speech in the Mac's dictionary) and ``blank`` (the words the model gives empty frames:
its scene vocabulary) — data for the caller's word filter.

Nothing is downloaded without ``--yes`` (the .pt is 40.0 MB from Ultralytics' GitHub release, checked
by size and SHA-256); nothing else is ever fetched — the conversion runs with the network blocked.

These are build tools, never NEO's own environment (coremltools 9.0's converter needs numpy <= 2.3.5;
NEO runs 2.5, and runs the result with what it already has: pyobjc's Core ML, neo2.core.vision.yolo).
One command makes a throwaway environment with the versions this was tested with and runs it there::

    uv run --no-project --python 3.12 --with ultralytics==8.4.165 --with coremltools==9.0 \\
        --with numpy==2.3.5 --with torch==2.14.0 --with torchvision==0.29.0 \\
        scripts/get_yoloe.py --out ~/.neo/models --yes

(``--check some.jpg ...`` also compares its boxes with Ultralytics' own predict.) A minute or two and 1 GB
of memory on an M3 Pro; the result is ``~/.neo/models/yoloe-26s-pf.mlpackage`` (23 MB), which NEO's
camera uses the next time it turns on. The weights and the model are Ultralytics', AGPL-3.0
(THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
import tempfile
import time
import types
import urllib.request
from pathlib import Path

URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg-pf.pt"
PT = "yoloe-26s-seg-pf.pt"
PT_BYTES = 40_003_064
PT_SHA256 = "01605da3f2e0e5325df45d5f5eef8e40b5b5c7d085399be3c33ac638258b150a"
PKG = "yoloe-26s-pf.mlpackage"
NEEDS = {"ultralytics": "8.4.165", "coremltools": "9.0"}
IMGSZ = (384, 640)  # (height, width): what Ultralytics' predict at imgsz=640 feeds for a 16:9 frame
PAD = 114
MAX_DET = 300
PF_CONF = 0.001  # the prompt-free proposal filter's cut in Ultralytics' predict (YOLOEDetect.conf default)
FETCH_DEADLINE_S = 600


def log(msg: str) -> None:
    print(f"[get_yoloe] {msg}", flush=True)


# -- download --------------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _good(path: Path) -> bool:
    return path.is_file() and path.stat().st_size == PT_BYTES and _sha256(path) == PT_SHA256


def fetch(dest: Path) -> None:
    """Download the .pt to ``dest`` — capped at the expected size and FETCH_DEADLINE_S, checked by SHA-256,
    written to a .part file first so a broken download never looks like the real thing."""
    part = dest.with_name(dest.name + ".part")
    h, n, t0 = hashlib.sha256(), 0, time.monotonic()
    log(f"downloading {URL} ({PT_BYTES / 1e6:.1f} MB) ...")
    try:
        with urllib.request.urlopen(URL, timeout=60) as r, open(part, "wb") as f:
            size = int(r.headers.get("Content-Length") or 0)
            if size and size != PT_BYTES:
                raise SystemExit(f"unexpected size {size} bytes (want {PT_BYTES}); not downloading")
            while chunk := r.read(1 << 20):
                n += len(chunk)
                if n > PT_BYTES or time.monotonic() - t0 > FETCH_DEADLINE_S:
                    raise SystemExit("download too big or too slow; stopped")
                h.update(chunk)
                f.write(chunk)
        if n != PT_BYTES or h.hexdigest() != PT_SHA256:
            raise SystemExit(f"download check failed ({n} bytes, sha256 {h.hexdigest()})")
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)
    log(f"got {dest} ({n} bytes, sha256 ok) in {time.monotonic() - t0:.1f}s")


def offline() -> None:
    """Block every non-loopback connection from here on, and tell Ultralytics it is offline and may
    not pip-install: the conversion must not fetch anything (fonts, text encoders, updates)."""
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    local = {"localhost", "127.0.0.1", "::1", None}
    connect, gai = socket.socket.connect, socket.getaddrinfo

    def _connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if self.family == socket.AF_UNIX or host in local:
            return connect(self, address)
        raise OSError(f"get_yoloe: network blocked ({address!r})")

    def _gai(host, *a, **k):
        if host in local:
            return gai(host, *a, **k)
        raise OSError(f"get_yoloe: network blocked ({host!r})")

    socket.socket.connect, socket.getaddrinfo = _connect, _gai


def check_versions() -> None:
    from importlib.metadata import PackageNotFoundError, version

    bad = []
    for pkg, want in NEEDS.items():
        try:
            got = version(pkg)
        except PackageNotFoundError:
            got = None
        if got != want:
            bad.append(f"{pkg}=={want} (have {got})")
    np_version = tuple(int(v) for v in version("numpy").split(".")[:2])
    if np_version > (2, 3):  # coremltools 9.0's torch frontend breaks on numpy 2.4+ (apple/coremltools#2633)
        bad.append(f"numpy<=2.3.5 (have {version('numpy')})")
    if bad:
        raise SystemExit("needs " + ", ".join(bad) + " — see this script's docstring for the venv")


# -- export ----------------------------------------------------------------------------------------


def _boxes_only(self, x):
    """Replaces the YOLOE seg head's forward for export: the one-to-one prompt-free branch, gated by the
    proposal filter like ``predict``, weighted per class by the ``classes`` input, one word per box (YOLOE's
    default, agnostic), top MAX_DET boxes. Masks are never computed.

    Same result as ``predict``, in two passes, so the anchors x 4,585 score grid never leaves the Neural
    Engine: each anchor's best score is taken there; the word (an argmax, which the ANE can't do) is worked
    out after the top-k, from the kept anchors' features only. A full argmax on the CPU took 4x the model.
    Batch 1 (a static export): plain 1-D gathers — an N-D gather of ints pushes the whole model off the ANE."""
    import torch
    import torch.nn.functional as F
    from ultralytics.utils.tal import dist2bbox, make_anchors

    x = x[: self.nl]  # YOLOEModel appends class embeddings; the prompt-free head doesn't use them
    weights = self.class_weights  # (1, nc)
    vocab = [(v.weight.view(v.weight.shape[0], -1), v.bias) for v in (lr.vocab for lr in self.one2one_lrpc)]
    feats, dist, best = [], [], []
    for i, lrpc in enumerate(self.one2one_lrpc):  # each level has its own vocabulary (a Linear or a 1x1 conv)
        cls_feat, loc_feat = self.one2one_cv3[i](x[i]), self.one2one_cv2[i](x[i])
        dist.append(lrpc.loc(loc_feat).flatten(2))
        feats.append(cls_feat.flatten(2).transpose(1, 2))  # (1, n, C)
        scores = F.linear(feats[-1], *vocab[i]).sigmoid() * weights.unsqueeze(1)  # (1, n, nc)
        if lrpc.enabled:  # the proposal filter that drops unlikely anchors (not on the coarsest level)
            scores = scores * (lrpc.pf(cls_feat).sigmoid() > PF_CONF).flatten(2).transpose(1, 2).float()
        best.append(scores.amax(2))  # (1, n)
    anchors, strides = (a.transpose(0, 1) for a in make_anchors(x, self.stride, 0.5))
    boxes = dist2bbox(torch.cat(dist, 2), anchors.unsqueeze(0), xywh=False, dim=1) * strides  # (1, 4, N)
    best = torch.cat(best, 1)[0]  # (N,)
    idx = best.topk(self.max_det)[1]
    top = best.index_select(0, idx)  # not topk's own values: the GPU returns those wrong
    feat = torch.cat(feats, 1)[0].index_select(0, idx)  # (k, C): the kept anchors' class features
    logits, start = 0, 0
    for (w, b), f in zip(vocab, feats, strict=True):
        inside = ((idx >= start) & (idx < start + f.shape[1])).float().unsqueeze(1)
        logits = logits + F.linear(feat, w, b) * inside
        start += f.shape[1]
    label = (logits.sigmoid() * weights).argmax(1)
    return boxes[0].index_select(1, idx).t()[None], top[None], label.int()[None]


def build(pt: Path):
    """The prompt-free model as a torch module ``(image 0..1 BCHW, classes (1, nc)) -> (boxes, scores,
    labels)`` with :func:`_boxes_only` as its head, and its class names."""
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.modules import C2f, YOLOEDetect
    from ultralytics.nn.modules.block import Attention

    yolo = YOLO(str(pt))
    net = yolo.model.float().eval()
    head = net.model[-1]
    if not (isinstance(head, YOLOEDetect) and hasattr(head, "one2one_lrpc") and head.reg_max == 1):
        raise SystemExit(f"{pt} is not a YOLOE-26 prompt-free checkpoint")
    names = [yolo.names[i] for i in range(len(yolo.names))]
    net.end2end = True  # the one-to-one (NMS-free) branch
    for p in net.parameters():
        p.requires_grad = False
    net.fuse(verbose=False)  # folds BatchNorm, drops the one-to-many branch
    for m in net.modules():  # what Ultralytics' exporter does for coreml
        if isinstance(m, Attention):
            m.format = "coreml"
        elif isinstance(m, C2f):
            m.forward = m.forward_split
    head.export, head.format, head.dynamic, head.max_det = True, "coreml", False, MAX_DET
    head.forward = types.MethodType(_boxes_only, head)

    class Wrap(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, image, classes):
            self.net.model[-1].class_weights = classes
            return self.net(image)

    return Wrap(net).eval(), names


def export(pt: Path, dest: Path, fp16: bool = True, imgsz: tuple[int, int] = IMGSZ) -> Path:
    """Trace :func:`build`'s model and convert it to an ML program saved at ``dest`` (an .mlpackage)."""
    import coremltools as ct
    import numpy as np
    import torch
    from ultralytics import __version__

    model, names = build(pt)
    im, w = torch.zeros(1, 3, *imgsz), torch.ones(1, len(names))
    with torch.no_grad():
        model(im, w)  # warm the anchors
        ts = torch.jit.trace(model, (im, w), strict=False, check_trace=False)
    mlm = ct.convert(
        ts,
        inputs=[
            ct.ImageType(
                "image", shape=im.shape, scale=1 / 255, bias=[0.0, 0.0, 0.0], color_layout=ct.colorlayout.RGB
            ),
            ct.TensorType(
                "classes", shape=w.shape, dtype=np.float32, default_value=np.ones(w.shape, np.float32)
            ),
        ],
        outputs=[ct.TensorType("boxes"), ct.TensorType("scores"), ct.TensorType("labels")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS14,  # optional inputs with a default need macOS 14
        skip_model_load=True,
    )
    mlm.short_description = (
        f"YOLOE-26s prompt-free (Ultralytics), boxes only. image: {imgsz[1]}x{imgsz[0]} RGB letterbox (pad 114). "
        "classes: optional per-class weight 0..1. boxes: x1 y1 x2 y2 in input px; scores; labels: index into names."
    )
    mlm.author = "Ultralytics (weights); converted for NEO by scripts/get_yoloe.py"
    mlm.license = "AGPL-3.0 (https://ultralytics.com/license)"
    mlm.version = __version__
    mlm.user_defined_metadata.update(
        {
            "names": json.dumps(names, ensure_ascii=False),
            "imgsz": json.dumps(list(imgsz)),  # [height, width]
            "pad": str(PAD),
            "max_det": str(MAX_DET),
            "end2end": "True",
            "agnostic": "True",
            "pf_conf": str(PF_CONF),
            "precision": "fp16" if fp16 else "fp32",
            "source": f"{URL} sha256={PT_SHA256}",
            "ultralytics": __version__,
            "torch": torch.__version__,
        }
    )
    mlm.save(str(dest))
    return dest


# -- reference pre/post-processing (numpy + PIL only; what NEO does at run time) ---------------------


def _fit(frame_size: tuple[int, int], shape: tuple[int, int]) -> tuple[float, int, int, int, int]:
    """(scale, new width, new height, left pad, top pad) for a (w, h) frame into a (h, w) input — Ultralytics'
    LetterBox rounding, which :func:`decode` must mirror."""
    (w0, h0), (h, w) = frame_size, shape
    r = min(h / h0, w / w0)
    nw, nh = round(w0 * r), round(h0 * r)
    return r, nw, nh, round((w - nw) / 2 - 0.1), round((h - nh) / 2 - 0.1)


def letterbox(img, shape: tuple[int, int] = IMGSZ):
    """Fit ``img`` (PIL) into the model's (height, width) keeping its proportions, centred on grey 114.
    Halving (a 1280x720 frame) averages 2x2 pixels, as OpenCV's resize in Ultralytics' predict does."""
    from PIL import Image

    img = img.convert("RGB")
    _, nw, nh, left, top = _fit(img.size, shape)
    if img.size == (nw * 2, nh * 2):
        img = img.reduce(2)
    elif img.size != (nw, nh):
        img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (shape[1], shape[0]), (PAD, PAD, PAD))
    canvas.paste(img, (left, top))
    return canvas


def decode(
    out: dict,
    frame_size: tuple[int, int],
    names: list[str],
    conf: float = 0.25,
    shape: tuple[int, int] = IMGSZ,
):
    """Model outputs → [(label, score, x, y, w, h)], best first; x, y, w, h are 0..1 of the ORIGINAL frame
    (``frame_size`` = its (width, height)), origin top left, clipped to it."""
    import numpy as np

    w0, h0 = frame_size
    r, _, _, left, top = _fit(frame_size, shape)
    boxes = np.asarray(out["boxes"], np.float32).reshape(-1, 4)
    scores = np.asarray(out["scores"], np.float32).reshape(-1)
    labels = np.asarray(out["labels"]).reshape(-1).astype(np.int64)
    dets = []
    for (x1, y1, x2, y2), s, c in zip(boxes.tolist(), scores.tolist(), labels.tolist(), strict=True):
        if not s >= conf or not 0 <= c < len(names):  # also drops NaN
            continue
        x1, x2 = (min(max((v - left) / r, 0.0), w0) for v in (x1, x2))
        y1, y2 = (min(max((v - top) / r, 0.0), h0) for v in (y1, y2))
        if x2 - x1 >= 1 and y2 - y1 >= 1:
            dets.append((names[c], s, x1 / w0, y1 / h0, (x2 - x1) / w0, (y2 - y1) / h0))
    return sorted(dets, key=lambda d: -d[1])


def names_of(mlmodel) -> list[str]:
    return json.loads(mlmodel.user_defined_metadata["names"])


def shape_of(mlmodel) -> tuple[int, int]:
    h, w = json.loads(mlmodel.user_defined_metadata["imgsz"])
    return h, w


# -- vocabulary tags: data for the caller's word filter, from the Mac's dictionary and the model itself ----


def dictionary_pos(words: list[str]) -> list[str]:
    """Each word's first part of speech in the Mac's own dictionary (CoreServices, offline): "noun", "verb",
    "adjective", ...; "phrase" for a phrase it doesn't know (nearly always a noun: "beer can"); "" unknown.
    Lets the caller hide the verbs and adjectives of the vocabulary ("squeeze", "extrude", "beautiful"), which
    win boxes on real photos."""
    import ctypes
    import ctypes.util
    import re

    cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
    cs = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreServices"))

    class CFRange(ctypes.Structure):
        _fields_ = [("location", ctypes.c_long), ("length", ctypes.c_long)]

    cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
    cf.CFStringGetLength.restype, cf.CFStringGetLength.argtypes = ctypes.c_long, [ctypes.c_void_p]
    cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    cs.DCSCopyTextDefinition.restype = ctypes.c_void_p
    cs.DCSCopyTextDefinition.argtypes = [ctypes.c_void_p, ctypes.c_void_p, CFRange]
    utf8 = 0x08000100
    kind = re.compile(
        r"\b(noun|verb|adjective|adverb|exclamation|preposition|pronoun|abbreviation|conjunction|determiner"
        r"|prefix|suffix|symbol)\b"
    )

    def first(word: str) -> str:
        s = cf.CFStringCreateWithCString(None, word.encode(), utf8)
        if not s:
            return ""
        d = cs.DCSCopyTextDefinition(None, s, CFRange(0, cf.CFStringGetLength(s)))
        cf.CFRelease(s)
        if not d:
            return ""
        size = cf.CFStringGetLength(d) * 4 + 1
        buf = ctypes.create_string_buffer(size)
        cf.CFStringGetCString(d, buf, size, utf8)
        cf.CFRelease(d)
        m = kind.search(buf.value.decode("utf-8", "replace")[:400])  # the headword's first sense
        return m.group(1) if m else ""

    return [first(w) or ("phrase" if " " in w else "") for w in words]


def blank_words(mlmodel, names: list[str], shape: tuple[int, int], conf: float = 0.2, rounds: int = 24):
    """Class indices the model gives to frames with nothing in them (plain colours, gradients, blur): its
    own scene/background vocabulary ("sky", "darkness", "close-up", "evening light", "studio shot", ...).
    Round by round, each round's words hidden for the next, until a round finds none or ``rounds`` ran."""
    import numpy as np
    from PIL import Image, ImageFilter

    size, rng = (1280, 720), np.random.default_rng(0)
    colours = [(0, 0, 0), (255, 255, 255), (128, 128, 128), (200, 180, 160), (60, 40, 30), (30, 60, 120)]
    colours += [(90, 140, 60), (220, 200, 170), (170, 120, 90), (240, 230, 220)]
    frames = [Image.new("RGB", size, c) for c in colours]
    ramp = np.linspace(0, 1, size[1])[:, None, None]
    for a, b in (
        ((0, 0, 0), (255, 255, 255)),
        ((40, 40, 60), (200, 190, 180)),
        ((20, 30, 80), (150, 180, 220)),
    ):
        rows = (np.array(a) * (1 - ramp) + np.array(b) * ramp).astype(np.uint8)
        frames.append(Image.fromarray(np.repeat(rows, size[0], 1)))
    for r in (3, 8, 20):
        noise = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        frames.append(Image.fromarray(noise).filter(ImageFilter.GaussianBlur(r)))
    inputs = [letterbox(f, shape) for f in frames]
    weights, found = np.ones((1, len(names)), np.float32), []
    for _ in range(rounds):
        new = set()
        for x in inputs:
            out = mlmodel.predict({"image": x, "classes": weights})
            new |= set(
                np.asarray(out["labels"]).reshape(-1)[np.asarray(out["scores"]).reshape(-1) >= conf].tolist()
            )
        new -= set(found)
        if not new:
            break
        found += sorted(new)
        weights[0, sorted(new)] = 0
    return sorted(found)


def tag(src: Path, dest: Path) -> tuple[int, int]:
    """Save ``src`` at ``dest`` with ``pos`` (JSON list, one per class) and ``blank`` (JSON list of class
    indices) added to its metadata. Returns (how many words aren't nouns, how many are blank-frame words)."""
    import coremltools as ct

    mlm = ct.models.MLModel(str(src), compute_units=ct.ComputeUnit.CPU_AND_NE)
    try:
        names, shape = names_of(mlm), shape_of(mlm)
        pos, blank = dictionary_pos(names), blank_words(mlm, names, shape)
        mlm.user_defined_metadata.update({"pos": json.dumps(pos), "blank": json.dumps(blank)})
        mlm.save(str(dest))
    finally:
        _forget(mlm)
    return sum(p not in ("noun", "phrase", "") for p in pos), len(blank)


def _forget(mlm) -> None:
    """Delete the compiled copy Core ML made in the temp dir to load ``mlm`` (22 MB each, else left behind)."""
    shutil.rmtree(mlm.get_compiled_model_path(), ignore_errors=True)


# -- check against Ultralytics ---------------------------------------------------------------------


def _iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0.0, min(ax2, bx2) - max(a[0], b[0])), max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def check(pkg: Path, pt: Path, images: list[Path], strong: float = 0.5) -> bool:
    """Ultralytics' predict on the .pt (same letterbox: ``rect=False``) vs the Core ML model (Neural Engine,
    :func:`letterbox`) on the same images. Every .pt box scoring >= ``strong`` needs a Core ML box with
    IoU >= 0.7; of those, >= 80% must carry the same word — fp16 can swap near-tied words of the 4,585
    (coffee / espresso), which is not a conversion fault (fp32 export: identical)."""
    import coremltools as ct
    from PIL import Image
    from ultralytics import YOLO

    yolo = YOLO(str(pt))
    mlm = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_AND_NE)
    names, shape = names_of(mlm), shape_of(mlm)
    n = boxed = worded = 0
    for p in images:
        r = yolo.predict(
            str(p), imgsz=list(shape), rect=False, conf=0.25, nms=False, device="cpu", verbose=False
        )[0]
        ref = [
            (r.names[int(c)], float(s), *(float(v) for v in b))
            for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xywhn, strict=True)
        ]
        ref = [(k, s, x - w / 2, y - h / 2, w, h) for k, s, x, y, w, h in ref]
        img = Image.open(p)
        got = decode(mlm.predict({"image": letterbox(img, shape)}), img.size, names, 0.25, shape)
        for d in (d for d in ref if d[1] >= strong):
            best = max(got, key=lambda g: _iou(g[2:], d[2:]), default=None)
            hit = best is not None and _iou(best[2:], d[2:]) >= 0.7
            n, boxed, worded = n + 1, boxed + hit, worded + (hit and best[0] == d[0])
        log(
            f"{p.name}: pt {[(d[0], round(d[1], 2)) for d in ref[:5]]} | coreml {[(d[0], round(d[1], 2)) for d in got[:5]]}"
        )
    log(f"strong .pt boxes: {n}, found by Core ML: {boxed}, same word: {worded}")
    _forget(mlm)
    return n > 0 and boxed >= 0.9 * n and worded >= 0.8 * boxed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True, type=Path, help=f"folder for {PKG} (and the downloaded {PT})")
    ap.add_argument(
        "--yes",
        action="store_true",
        help=f"allow downloading {PT} ({PT_BYTES / 1e6:.1f} MB) if it isn't there",
    )
    ap.add_argument("--weights", type=Path, help=f"use this {PT} instead of <out>/{PT}")
    ap.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=list(IMGSZ),
        metavar="N",
        help="input size: N (square) or H W (default 384 640: 16:9 frames; 640 for any shape)",
    )
    ap.add_argument(
        "--fp32", action="store_true", help="float32 model (default fp16: half the size, same boxes)"
    )
    ap.add_argument(
        "--check",
        nargs="*",
        type=Path,
        default=[],
        metavar="IMG",
        help="compare with Ultralytics' predict on these images",
    )
    a = ap.parse_args(argv)
    imgsz = (a.imgsz * 2)[:2] if len(a.imgsz) in (1, 2) else None
    if not imgsz or any(v % 32 or not 64 <= v <= 1280 for v in imgsz):
        raise SystemExit("--imgsz: one or two multiples of 32 between 64 and 1280")

    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    pt = (a.weights or out / PT).expanduser()
    if not _good(pt):
        if a.weights:
            raise SystemExit(f"{pt}: not the expected {PT} ({PT_BYTES} bytes, sha256 {PT_SHA256[:12]}…)")
        if not a.yes:
            log(f"{pt} is missing; re-run with --yes to download it ({PT_BYTES / 1e6:.1f} MB from {URL})")
            return 2
        fetch(pt)
    offline()
    check_versions()
    with (
        tempfile.TemporaryDirectory(prefix="yoloe-cfg-") as cfg,
        tempfile.TemporaryDirectory(dir=out) as stage,
    ):
        os.environ["YOLO_CONFIG_DIR"] = cfg  # Ultralytics writes its settings here, not ~/Library
        os.environ.setdefault("MPLCONFIGDIR", cfg)
        t0 = time.monotonic()
        raw = export(pt, Path(stage) / "raw.mlpackage", fp16=not a.fp32, imgsz=(imgsz[0], imgsz[1]))
        not_nouns, blank = tag(raw, Path(stage) / PKG)
        pkg = out / PKG
        if pkg.exists():
            shutil.rmtree(pkg)
        (Path(stage) / PKG).replace(pkg)
        mb = sum(f.stat().st_size for f in pkg.rglob("*") if f.is_file()) / 1e6
        log(
            f"wrote {pkg} ({mb:.1f} MB) in {time.monotonic() - t0:.0f}s; {not_nouns} words aren't nouns, {blank} blank-frame words"
        )
        if a.check and not check(pkg, pt, a.check):
            log("Core ML and Ultralytics disagree")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
