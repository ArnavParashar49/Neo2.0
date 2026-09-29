"""The camera's YOLOE detector (neo2/core/vision/yolo.py) and which detector a frame goes to
(neo2/core/vision/detect.py), and the boxes drawn the way NEO 1 drew its YOLO's (neo2/core/vision/
pipeline.py): people as "person", one box per thing, no body parts or scenery, at most ten.

Hermetic: Core ML, Quartz and Foundation are fakes (a stand-in model that returns the outputs a test
gives it and keeps what it was given); the real model is refused under pytest. Nothing is loaded
onto the Neural Engine, no camera, nothing leaves the Mac."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from neo2.core import config, effects
from neo2.core.vision import Unavailable, apple, detect, pipeline, yolo

# ---- a fake Core ML ------------------------------------------------------------------------------------

F16, F32, I32, F64 = 65552, 65568, 131104, 65600  # (Core ML's MLMultiArrayDataType values)
NAMES = ["cup", "squeeze", "sky", "watch", "model", "beer can", "person", "man", "hand", "table", "ipad"]
POS = ["noun", "verb", "noun", "verb", "noun", "phrase", "noun", "noun", "noun", "noun", ""]
BLANK = [2]


class Arr:
    """An MLMultiArray Core ML hands back: its type, shape, strides and bytes."""

    def __init__(self, a: np.ndarray, kind: int) -> None:
        self.a, self.kind = np.ascontiguousarray(a), kind

    def dataType(self) -> int:
        return self.kind

    def shape(self) -> list[int]:
        return list(self.a.shape)

    def strides(self) -> list[int]:
        return [s // self.a.itemsize for s in self.a.strides]

    def getBytesWithHandler_(self, fn) -> None:
        fn(self.a.tobytes(), self.a.nbytes)


class Weights:
    """The MLMultiArray NEO fills with its word weights."""

    def __init__(self) -> None:
        self.shape = self.data = None

    def initWithShape_dataType_error_(self, shape, kind, err):
        assert kind == F32
        self.shape = list(shape)
        return self, None

    def getMutableBytesWithHandler_(self, fn) -> None:
        self.data = np.frombuffer(fn(None, 0, None), np.float32).reshape(self.shape)


class CoreMLFake:
    """What NEO uses of Core ML, and a record of it: compiles, loads, what each prediction got."""

    def __init__(self, tmp: Path, outputs: dict[str, np.ndarray], meta: dict[str, str]) -> None:
        self.tmp, self.outputs, self.meta = tmp, outputs, meta
        self.compiled = self.loaded = self.refuse = 0
        self.units: list[int] = []
        self.inputs: list[dict] = []

    def module(self) -> types.ModuleType:
        fake, m = self, types.ModuleType("CoreML")
        m.MLMultiArrayDataTypeFloat16, m.MLMultiArrayDataTypeFloat32 = F16, F32
        m.MLMultiArrayDataTypeInt32, m.MLMultiArrayDataTypeDouble = I32, F64
        m.MLComputeUnitsCPUAndNeuralEngine, m.MLModelCreatorDefinedKey = 3, "MLModelCreatorDefinedKey"

        class Model:
            def modelDescription(self):
                return types.SimpleNamespace(metadata=lambda: {"MLModelCreatorDefinedKey": fake.meta})

            def predictionFromFeatures_error_(self, provider, err):
                fake.inputs.append(provider.d)
                out = {
                    "boxes": Arr(fake.outputs["boxes"].astype(np.float16), F16),
                    "scores": Arr(fake.outputs["scores"].astype(np.float16), F16),
                    "labels": Arr(fake.outputs["labels"].astype(np.int32), I32),
                }
                return types.SimpleNamespace(
                    featureValueForName_=lambda k: types.SimpleNamespace(multiArrayValue=lambda: out[k])
                ), None

        class MLModel:
            @staticmethod
            def compileModelAtURL_error_(url, err):
                assert Path(url.path()).name == yolo.PKG
                fake.compiled += 1
                out = fake.tmp / f"compiled-{fake.compiled}" / "yoloe-26s-pf.mlmodelc"
                out.mkdir(parents=True)
                (out / "model.mil").write_text("compiled")
                return URL(str(out)), None

            @staticmethod
            def modelWithContentsOfURL_configuration_error_(url, cfg, err):
                assert (Path(url.path()) / "model.mil").is_file()
                if fake.refuse:  # (a compiled copy an older macOS made)
                    fake.refuse -= 1
                    return None, "unreadable"
                fake.loaded += 1
                fake.units.append(cfg.units)
                return Model(), None

        class Config:
            units = None

            @classmethod
            def alloc(cls):
                return cls()

            def init(self):
                return self

            def setComputeUnits_(self, u):
                self.units = u

        class Provider:
            @classmethod
            def alloc(cls):
                return cls()

            def initWithDictionary_error_(self, d, err):
                self.d = dict(d)
                return self, None

        m.MLModel, m.MLModelConfiguration, m.MLDictionaryFeatureProvider = MLModel, Config, Provider
        m.MLMultiArray = types.SimpleNamespace(alloc=Weights)
        m.MLFeatureValue = types.SimpleNamespace(
            featureValueWithPixelBuffer_=lambda pb: ("pixels", pb),
            featureValueWithMultiArray_=lambda a: ("array", a),
        )
        return m


class URL:
    def __init__(self, p: str) -> None:
        self.p = p

    def path(self) -> str:
        return self.p


class PixelBuffer:
    def __init__(self, w: int, h: int) -> None:
        self.w, self.h, self.bpr = w, h, w * 4 + 64  # Core Video pads its rows
        self.buf = bytearray(self.bpr * h)

    def rgb(self, x: int, y: int) -> tuple[int, int, int]:
        b, g, r, _ = self.buf[y * self.bpr + x * 4 : y * self.bpr + x * 4 + 4]
        return r, g, b


def quartz() -> types.ModuleType:
    q = types.ModuleType("Quartz")
    q.kCVPixelFormatType_32BGRA = 1111970369
    q.CVPixelBufferCreate = lambda a, w, h, fmt, attrs, out: (0, PixelBuffer(w, h))
    q.CVPixelBufferLockBaseAddress = q.CVPixelBufferUnlockBaseAddress = lambda pb, flags: 0
    q.CVPixelBufferGetBytesPerRow = lambda pb: pb.bpr
    q.CVPixelBufferGetBaseAddress = lambda pb: types.SimpleNamespace(as_buffer=lambda n: memoryview(pb.buf)[:n])
    return q


def outputs(rows: list[tuple[list[float], float, int]]) -> dict[str, np.ndarray]:
    """The model's three outputs: 300 rows, these first (box in the 640x384 input's pixels, score,
    word), the rest nothing."""
    boxes, scores, labels = np.zeros((1, 300, 4)), np.zeros((1, 300)), np.zeros((1, 300), np.int32)
    for i, (b, s, c) in enumerate(rows):
        boxes[0, i], scores[0, i], labels[0, i] = b, s, c
    return {"boxes": boxes, "scores": scores, "labels": labels}


def meta() -> dict[str, str]:
    import json

    return {"names": json.dumps(NAMES), "imgsz": "[384, 640]", "pos": json.dumps(POS), "blank": json.dumps(BLANK)}


@pytest.fixture
def s(monkeypatch):
    """NEO's settings (the test's own data dir), the real detectors, Core ML runnable here."""
    st = config.Settings()
    st.ensure_dirs()
    monkeypatch.setattr(config, "_settings", st)
    monkeypatch.setattr(detect, "_yolo", None)
    monkeypatch.setattr(detect, "_dfine", None)
    monkeypatch.setattr(detect, "_warming", None)
    monkeypatch.setattr(yolo, "_broken", {})
    monkeypatch.setattr(yolo, "runnable", lambda: "")
    detect.set_engine(None)
    yield st
    detect.set_engine(None)
    apple.set_engine(None)


def put_model(models: Path) -> Path:
    """The files scripts/get_yoloe.py leaves in the models dir (their contents don't matter here)."""
    pkg = models / yolo.PKG
    (pkg / "Data" / "com.apple.CoreML").mkdir(parents=True)
    (pkg / "Manifest.json").write_text("{}")
    (pkg / "Data" / "com.apple.CoreML" / "model.mlmodel").write_bytes(b"\0" * 64)
    return pkg


@pytest.fixture
def coreml(s, tmp_path, monkeypatch):
    """The model in the models dir and a fake Core ML under it; ``fake.outputs`` is what it returns."""
    put_model(s.models_dir)
    fake = CoreMLFake(tmp_path / "coreml-tmp", outputs([]), meta())
    monkeypatch.setitem(sys.modules, "CoreML", fake.module())
    monkeypatch.setitem(sys.modules, "Quartz", quartz())
    monkeypatch.setitem(sys.modules, "Foundation", types.SimpleNamespace(NSURL=types.SimpleNamespace(fileURLWithPath_=URL)))
    return fake


@pytest.fixture
def told(monkeypatch):
    """The brains stubbed: what each camera question told them (its hints)."""
    from neo2.core.vision import answer

    got: list[dict] = []

    async def ask(img, question, hints=None):
        got.append(hints)
        return answer.Reply("(stub)", "stub", False)

    monkeypatch.setattr(pipeline._answer, "ask", ask)
    monkeypatch.setattr(pipeline._answer, "prewarm", lambda: False)
    return got


def frame(w: int = 1280, h: int = 720, color=(200, 40, 40)) -> Image.Image:
    return Image.new("RGB", (w, h), color)


def det(label: str, score: float, x: float, y: float, w: float, h: float) -> detect.Det:
    return detect.Det(label=label, score=score, box={"x": x, "y": y, "w": w, "h": h})


# ---- pre- and post-processing (Ultralytics' own) -------------------------------------------------------------


def test_a_720p_frame_is_halved_onto_the_grey_letterbox(monkeypatch):
    img = Image.new("RGB", (1280, 720), (10, 20, 30))
    img.putpixel((0, 0), (250, 0, 0))
    x = yolo.letterbox(img, (384, 640))
    assert x.size == (640, 384) and x.mode == "RGB"
    assert x.getpixel((5, 11)) == (114, 114, 114) and x.getpixel((5, 372)) == (114, 114, 114)  # pad: 12 rows each
    assert x.getpixel((5, 12)) == (10, 20, 30) and x.getpixel((5, 371)) == (10, 20, 30)
    assert x.getpixel((0, 12)) == (70, 15, 23)  # 2x2 averaged, as OpenCV's halving
    square = yolo.letterbox(Image.new("L", (1000, 1000), 0), (384, 640))
    assert square.getpixel((127, 0)) == (114, 114, 114) and square.getpixel((128, 0)) == (0, 0, 0)  # 384 wide, centred
    monkeypatch.setattr(yolo, "MAX_PIXELS", 640 * 480)
    with pytest.raises(ValueError):
        yolo.letterbox(Image.new("L", (641, 480)), (384, 640))  # not a camera frame: never scaled


def test_boxes_are_decoded_back_onto_the_frame_best_first():
    out = outputs([
        ([100, 112, 300, 212], 0.6, 0),  # letterbox px → frame px: (v - pad) / 0.5
        ([0, 0, 640, 384], 0.9, 6),  # over the pad: clipped to the frame
        ([10, 20, 30, 40], 0.2, 0),  # under the score
        ([10, 20, 30, 40], float("nan"), 0),
        ([10, 20, 30, 40], 0.8, 99),  # not a word of the model's
        ([50, 50, 50.2, 90], 0.8, 0),  # under a pixel wide
    ])  # fmt: skip
    got = yolo.decode(out, (1280, 720), NAMES, conf=0.25, shape=(384, 640))
    assert [(g[0], round(g[1], 2)) for g in got] == [("person", 0.9), ("cup", 0.6)]
    assert [round(v, 4) for v in got[0][2:]] == [0.0, 0.0, 1.0, 1.0]
    assert [round(v, 4) for v in got[1][2:]] == [0.1562, 0.2778, 0.3125, 0.2778]


def test_words_that_are_never_a_thing_are_hidden_in_the_model():
    w = yolo.weights(NAMES, POS, BLANK)
    assert w.shape == (1, len(NAMES)) and w.dtype == np.float32
    hidden = {n for n, v in zip(NAMES, w[0], strict=True) if v == 0}
    # a verb ("squeeze"), what empty frames get ("sky"), a word about the picture ("model")
    assert hidden == {"squeeze", "sky", "model"}  # "watch" is a verb first, and a thing: shown


# ---- which detector ------------------------------------------------------------------------------------------


def test_yoloe_is_the_detector_whenever_its_model_is_there(s):
    assert s.vision_detector == "auto" and detect.mode() == "auto"
    assert detect.engine() == "yoloe"  # nothing there yet: YOLOE, the one NEO is set up with...
    ok, why = detect.available()
    assert not ok and "get_yoloe.py" in why and str(s.models_dir) in why  # ...says what's missing
    put_model(s.models_dir)
    assert detect.engine() == "yoloe" and detect.available() == (True, "")
    assert detect.scores() == (0.25, 0.45)


def test_the_setting_picks_one(s, monkeypatch, tmp_path):
    monkeypatch.setattr(s, "vision_detector", "yoloe")
    assert detect.engine() == "yoloe" and not detect.available()[0]  # only YOLOE: missing is missing
    folder = tmp_path / "dfine"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    put_model(s.models_dir)
    monkeypatch.setattr(s, "vision_detector", str(folder))  # D-FINE's folder: D-FINE, YOLOE there or not
    assert detect.mode() == "dfine" and detect.engine() == "dfine" and detect.source() == (str(folder), False)
    assert detect.scores() == (0.4, 0.5)
    monkeypatch.setattr(s, "vision_detector", "auto")
    monkeypatch.setattr(yolo, "runnable", lambda: "The object detector (YOLOE, Core ML) needs macOS 14 or later")
    assert detect.engine() == "yoloe" and "macOS 14" in detect.available()[1]  # a Mac that can't run it...
    (s.models_dir / "dfine-small-coco").mkdir()
    (s.models_dir / "dfine-small-coco" / "config.json").write_text("{}")
    assert detect.engine() == "dfine" and detect.available() == (True, "")  # ...uses D-FINE when it's there


def test_a_test_s_detector_says_its_own_scale(s):
    class Fake:
        floor, sure = 0.1, 0.3

        def detect(self, img, *, min_score, limit):
            return [det("cup", min_score, 0, 0, 0.5, 0.5)]

    detect.set_engine(Fake())
    assert detect.engine() == "test" and detect.scores() == (0.1, 0.3)
    assert detect.detect(frame())[0]["score"] == 0.1  # the detector's floor when none is asked for
    detect.set_engine(types.SimpleNamespace(detect=lambda img, **kw: []))
    assert detect.scores() == (detect.MIN_SCORE, detect.MIN_SCORE)


# ---- YOLOE through (a fake) Core ML ----------------------------------------------------------------------


def test_the_real_model_never_loads_in_the_tests(coreml):
    with pytest.raises(effects.EffectRefused):
        detect.detect(frame())
    assert coreml.compiled == coreml.loaded == 0 and not detect.loaded()
    assert pipeline.preview(frame()) == []  # the live card: no boxes, no error


def test_a_frame_goes_through_core_ml_and_back_as_boxes(coreml, s):
    coreml.outputs = outputs([([100, 112, 300, 212], 0.9, 0), ([0, 12, 640, 372], 0.7, 7), ([1, 13, 2, 14], 0.3, 10)])
    img = frame(color=(200, 40, 40))
    with effects.allow("model"):
        got = detect.detect(img)
    assert [(d["label"], d["score"]) for d in got] == [("cup", 0.9), ("man", 0.7), ("ipad", 0.3)]
    assert got[0]["box"] == {"x": 0.1562, "y": 0.2778, "w": 0.3125, "h": 0.2778}
    (sent,) = coreml.inputs
    kind, pb = sent["image"]
    assert kind == "pixels" and (pb.w, pb.h) == (640, 384)  # the model's input, never the frame
    assert pb.rgb(10, 0) == (114, 114, 114) and pb.rgb(10, 200) == (200, 40, 40)  # letterboxed, BGRA as Core Video
    kind, weights = sent["classes"]
    assert kind == "array" and weights.data.tolist() == yolo.weights(NAMES, POS, BLANK, words=yolo.EVERYDAY | yolo.NOT_SHOWN).tolist()
    assert coreml.units == [3]  # the CPU and Neural Engine: the GPU is Laya's
    assert detect.loaded() and not detect.warm()  # loaded once, kept


def test_the_compiled_model_is_kept_and_made_again_for_a_new_one(coreml, s):
    with effects.allow("model"):
        detect.detect(frame())
        yolo.YOLOE()._load()  # (a new process, the same model: nothing compiled)
    assert (coreml.compiled, coreml.loaded) == (1, 2)
    compiled = s.models_dir / yolo.COMPILED
    assert (compiled / "model.mil").is_file() and sorted(p.name for p in s.models_dir.iterdir()) == [yolo.COMPILED, yolo.PKG]
    weights = s.models_dir / yolo.PKG / "Data" / "com.apple.CoreML" / "model.mlmodel"
    weights.write_bytes(b"\1" * 80)  # get_yoloe.py made it again
    os.utime(weights, ns=(1, 1))
    with effects.allow("model"):
        yolo.YOLOE()._load()
    assert coreml.compiled == 2 and (compiled / "model.mil").is_file()
    assert not list(s.models_dir.glob("*.part"))
    coreml.refuse = 1  # the kept copy can't be read (macOS was updated): made again, once
    with effects.allow("model"):
        yolo.YOLOE()._load()
    assert (coreml.compiled, coreml.loaded) == (3, 4)


def test_a_model_that_isn_t_get_yoloe_s_says_so(coreml):
    coreml.meta = {"imgsz": "[384, 640]"}  # no words
    with effects.allow("model"), pytest.raises(Unavailable, match="get_yoloe.py"):
        detect.detect(frame())


def test_the_camera_turning_on_loads_it_ahead(coreml):
    with effects.allow("model"):
        assert detect.warm()
        detect._warming.join(5)
    assert detect.loaded() and coreml.loaded == 1 and not detect.warm()


def test_one_frame_at_a_time(coreml, monkeypatch):
    coreml.outputs = outputs([([100, 112, 300, 212], 0.9, 0)])
    monkeypatch.setattr(yolo, "_RUN_WAIT_S", 0.05)
    with effects.allow("model"):
        detect.detect(frame())
        y = detect._get()
        with y._run_lock:  # the previous frame is still being read
            got: list = []
            t = threading.Thread(target=lambda: got.append(detect.detect(frame())))
            t.start()
            t.join(5)
    assert got == [[]]  # skipped, not queued


# ---- the boxes on the card: NEO 1's way ----------------------------------------------------------------------

SURE = (0.25, 0.45)


def test_people_are_drawn_as_person_and_things_as_themselves():
    dets = [det("man", 0.8, 0.1, 0.0, 0.8, 1.0), det("cup", 0.9, 0.45, 0.35, 0.15, 0.25), det("guitar", 0.5, 0.0, 0.5, 0.3, 0.4)]
    assert [(d["label"], d["score"]) for d in pipeline.objects(dets, scores=SURE)] == [
        ("cup", 0.9), ("person", 0.8), ("guitar", 0.5),
    ]  # fmt: skip


def test_one_box_per_thing():
    dets = [
        det("person", 0.9, 0.1, 0.1, 0.4, 0.8), det("woman", 0.85, 0.11, 0.1, 0.4, 0.8),  # the same person twice
        det("person", 0.8, 0.55, 0.1, 0.4, 0.8),  # someone beside them
        det("cup", 0.9, 0.45, 0.35, 0.15, 0.25), det("coffee cup", 0.7, 0.46, 0.36, 0.14, 0.24),  # one mug, two words
        det("coffee", 0.6, 0.48, 0.38, 0.06, 0.05),  # what's in it: a thing of its own
    ]  # fmt: skip
    got = [(d["label"], d["score"]) for d in pipeline.objects(dets, scores=SURE)]
    assert got == [("person", 0.9), ("cup", 0.9), ("person", 0.8), ("coffee", 0.6)]


def test_no_body_parts_groups_scenery_or_specks():
    dets = [
        det("hand", 0.9, 0.4, 0.5, 0.2, 0.3), det("face", 0.8, 0.3, 0.1, 0.2, 0.2), det("crowd", 0.8, 0, 0, 1, 1),
        det("football team", 0.7, 0, 0, 0.5, 0.5), det("table", 0.9, 0.0, 0.05, 1.0, 0.95),  # the whole frame
        det("resin", 0.9, 0.5, 0.5, 0.02, 0.02),  # a speck
        det("person", 0.9, 0.0, 0.0, 1.0, 1.0),  # the user, right at the camera: always drawn
        det("book", 0.8, 0.1, 0.2, 0.6, 0.6),
    ]  # fmt: skip
    assert [d["label"] for d in pipeline.objects(dets, scores=SURE)] == ["person", "book"]


def test_apple_vision_adds_what_it_names_and_never_hides_a_sure_box():
    dets = [det("guitar", 0.3, 0.1, 0.1, 0.5, 0.8), det("dagger", 0.3, 0.7, 0.1, 0.1, 0.3), det("bus", 0.9, 0.0, 0.5, 0.3, 0.3)]
    labels = [("musical instrument", 0.8), ("guitar", 0.7)]
    assert [d["label"] for d in pipeline.objects(dets, labels, scores=SURE)] == ["bus", "guitar"]
    assert [d["label"] for d in pipeline.objects(dets, None, scores=SURE)] == ["bus"]
    assert [d["label"] for d in pipeline.objects([det("guitar", 0.2, 0.1, 0.1, 0.5, 0.8)], labels, scores=SURE)] == []


def test_at_most_ten():
    dets = [det(f"thing {i}", 0.9 - i / 100, (i % 5) / 5, (i // 5) / 4, 0.15, 0.2) for i in range(15)]
    got = pipeline.objects(dets, scores=SURE)
    assert len(got) == pipeline.MAX_SHOWN == 10 and got[0]["label"] == "thing 0"


def test_the_live_card_draws_people_too():
    class Fake:
        floor, sure = 0.25, 0.45

        def detect(self, img, *, min_score, limit):
            assert min_score == 0.25
            return [det("man", 0.9, 0.1, 0.0, 0.8, 1.0), det("cup", 0.6, 0.45, 0.35, 0.15, 0.25), det("sky", 0.3, 0, 0, 1, 0.3)]

    detect.set_engine(Fake())
    try:
        boxes = pipeline.preview(frame())
    finally:
        detect.set_engine(None)
    assert [(b["label"], b["score"]) for b in boxes] == [("person", 0.9), ("cup", 0.6)]
    assert boxes[1] == {"x": 0.45, "y": 0.35, "w": 0.15, "h": 0.25, "label": "cup", "score": 0.6}


# ---- what's in the hand, and what the brains are told ---------------------------------------------------------


def test_what_am_i_holding_is_never_a_person_a_hand_or_the_table():
    hand = apple.Hand(box=(0.4, 0.5, 0.2, 0.3), fingers=3)
    person = det("woman", 0.95, 0.1, 0.0, 0.8, 1.0)
    hand_box = det("hand", 0.9, 0.4, 0.5, 0.2, 0.3)
    table = det("table", 0.9, 0.0, 0.3, 1.0, 0.7)  # under it all, 70% of the frame
    phone = det("phone", 0.6, 0.42, 0.45, 0.1, 0.2)
    assert pipeline.holding([hand], [person, hand_box, table, phone]) == "phone"
    assert pipeline.holding([hand], [person, hand_box, table]) is None
    assert pipeline.holding([], [phone]) is None


def test_the_brains_hear_of_a_few_things_not_people():
    dets = [det("man", 0.9, 0, 0, 1, 1), det("cup", 0.9, 0, 0, 0.1, 0.1), det("hand", 0.8, 0, 0, 0.1, 0.1),
            det("cup", 0.7, 0.5, 0.5, 0.1, 0.1), *(det(f"thing {i}", 0.6, 0, 0, 0.1, 0.1) for i in range(5))]  # fmt: skip
    assert pipeline.hint_objects(dets) == ["cup", "thing 0", "thing 1", "thing 2"]


# ---- the everyday vocabulary, a model that won't load, library agents -------------------------------------


def test_the_everyday_vocabulary_is_the_default_and_all_of_it_a_choice(coreml, s, monkeypatch):
    assert s.vision_words == "everyday" and yolo.vocabulary() == "everyday"
    ev = yolo.weights(NAMES, POS, BLANK, words=yolo.EVERYDAY | yolo.NOT_SHOWN)
    assert {n for n, v in zip(NAMES, ev[0], strict=True) if v} == {"cup", "watch", "beer can", "person", "table", "hand"}
    with effects.allow("model"):
        detect.detect(frame())
        monkeypatch.setattr(s, "vision_words", "all")  # the whole open vocabulary, as the user may choose
        detect.detect(frame())
    (_, a), (_, b) = (sent["classes"] for sent in coreml.inputs)
    assert a.data.tolist() == ev.tolist() and b.data.tolist() == yolo.weights(NAMES, POS, BLANK).tolist()
    monkeypatch.setattr(s, "vision_words", "nonsense")
    assert yolo.vocabulary() == "everyday"  # a typo never opens the whole vocabulary


def test_a_model_that_won_t_load_isn_t_compiled_again_every_frame(coreml, s, tmp_path, monkeypatch):
    coreml.refuse = 10**6  # a copy cut short on its way into the models dir: Core ML can't read it
    with effects.allow("model"):
        with pytest.raises(Unavailable, match="couldn't be loaded"):
            detect.detect(frame())
        tried = coreml.compiled
        with pytest.raises(Unavailable, match="couldn't be loaded"):
            detect.detect(frame())
        assert pipeline.preview(frame()) == []
    assert coreml.compiled == tried  # remembered: not compiled again
    ok, why = detect.available()
    assert not ok and "couldn't be loaded" in why and detect.engine() == "yoloe"  # no D-FINE here: YOLOE's reason
    folder = s.models_dir / "dfine-small-coco"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    assert detect.engine() == "dfine" and detect.available() == (True, "")  # "auto" falls back to D-FINE
    weights = s.models_dir / yolo.PKG / "Data" / "com.apple.CoreML" / "model.mlmodel"
    weights.write_bytes(b"\1" * 80)  # a good copy this time
    os.utime(weights, ns=(1, 1))
    coreml.refuse = 0
    assert detect.engine() == "yoloe"
    with effects.allow("model"):
        detect.detect(frame())
    assert coreml.compiled == tried + 1 and detect.loaded()


def test_auto_with_neither_detector_names_yoloe(s, monkeypatch, told):
    monkeypatch.setattr(s, "vision_download_ok", False)
    assert detect.engine() == "yoloe"
    with pytest.raises(Unavailable, match="get_yoloe.py"):
        detect.detect(frame())  # the detector the user chose, not D-FINE's download
    got = asyncio.run(pipeline.look("what's this?", frame()))
    assert any("get_yoloe.py" in n for n in got["notes"]) and not any("D-FINE" in n for n in got["notes"])


def test_the_warm_up_loads_the_bound_agent_s_detector(tmp_path, monkeypatch):
    from neo2.core import runtime
    from neo2.core.config import LibrarySettings

    agent = LibrarySettings(data_dir=tmp_path / "agent", models_path=tmp_path / "agent-models", camera=True)
    agent.ensure_dirs()
    put_model(agent.models_dir)
    monkeypatch.setattr(detect, "_yolo", None)
    monkeypatch.setattr(detect, "_dfine", None)
    monkeypatch.setattr(detect, "_warming", None)
    monkeypatch.setattr(yolo, "runnable", lambda: "")
    seen: list = []

    def load(self) -> None:
        seen.append((type(self).__name__, str(yolo._models_dir())))

    monkeypatch.setattr(yolo.YOLOE, "_load", load)
    monkeypatch.setattr(detect._DFine, "_load", load)
    monkeypatch.setattr(runtime, "_library", True)
    rt = runtime.Runtime("cam", agent, None)
    with runtime.bind(rt.scope()):
        assert detect.engine() == "yoloe" and detect.warm()
        detect._warming.join(5)
    assert seen == [("YOLOE", str(agent.models_dir))] and detect._dfine is None


# ---- people, what's worn, what's held ----------------------------------------------------------------------


def test_every_everyday_word_is_drawn_as_itself_and_the_hands_and_clothes_never():
    assert not yolo.EVERYDAY & yolo.NOT_SHOWN
    assert {w for w in yolo.EVERYDAY if pipeline.name(w) != w} == set()  # ("person" is itself)
    assert {pipeline.name(w) for w in yolo.NOT_SHOWN} == {None}


def test_the_open_vocabulary_s_people_are_person():
    for word in ("preacher", "jazz artist", "rock artist", "teacher", "mother", "tourist", "politician", "twin", "brunette"):
        assert pipeline.name(word) == "person", word
    for word in ("afro", "bangs", "ponytail", "hairstyle", "tattoo", "army"):
        assert pipeline.name(word) is None, word
    assert pipeline.name("speaker") == "speaker" and pipeline.name("CD player") == "CD player"  # things, though


def test_a_role_word_is_a_person_on_the_card_and_never_a_hint():
    dets = [
        det("preacher", 0.929, 0.05, 0.08, 0.45, 0.92), det("preacher", 0.855, 0.52, 0.05, 0.45, 0.95),
        det("tie", 0.472, 0.3, 0.5, 0.05, 0.3),
    ]  # fmt: skip
    got = pipeline.objects(dets, scores=SURE)
    assert [d["label"] for d in got] == ["person", "person", "tie"]
    assert pipeline.hint_objects(got) == ["tie"]
    hand = apple.Hand(box=(0.60, 0.30, 0.05, 0.08), fingers=5)
    assert pipeline.holding([hand], got) is None
    same = [det("jazz artist", 0.9, 0.2, 0.0, 0.6, 1.0), det("rock artist", 0.8, 0.35, 0.1, 0.6, 0.9)]
    assert pipeline.SAME_WORD < round(pipeline._iou(same[0]["box"], same[1]["box"]), 2) == 0.55 < pipeline.SAME_THING
    assert [d["label"] for d in pipeline.objects(same, scores=SURE)] == ["person"]  # one person, one box


def webcam() -> list[detect.Det]:
    """What YOLOE gives a person at a webcam with glasses, a hat and a tie (a portrait photo's boxes)."""
    return [
        det("person", 0.866, 0.27, 0.05, 0.47, 0.95), det("glasses", 0.884, 0.42, 0.28, 0.16, 0.09),
        det("hat", 0.837, 0.40, 0.05, 0.21, 0.2), det("tie", 0.746, 0.47, 0.62, 0.07, 0.12),
        det("keycard", 0.889, 0.34, 0.8, 0.08, 0.05),
    ]  # fmt: skip


def test_an_empty_hand_beside_the_face_holds_nothing():
    dets = webcam()
    for box in ((0.60, 0.25, 0.12, 0.34), (0.24, 0.22, 0.12, 0.34), (0.45, 0.62, 0.12, 0.30)):
        assert pipeline.holding([apple.Hand(box=box, fingers=5)], dets) is None, box
    cup = det("cup", 0.85, 0.63, 0.30, 0.08, 0.14)
    assert pipeline.holding([apple.Hand(box=(0.60, 0.25, 0.12, 0.34), fingers=3)], [*dets, cup]) == "cup"


def test_the_brains_hear_first_of_what_s_held_then_what_isn_t_worn():
    cup = det("cup", 0.85, 0.67, 0.64, 0.19, 0.28)  # on the desk, beside the person
    dets = pipeline.objects([*webcam(), cup], scores=SURE)
    assert pipeline.hint_objects(dets)[0] == "cup" and "cup" in pipeline.hint_objects(dets)[:1]
    pen = det("pen", 0.6, 0.5, 0.5, 0.04, 0.1)
    assert pipeline.hint_objects([*dets, pen], held="pen")[:2] == ["pen", "cup"]


def test_a_finger_count_isn_t_told_what_the_hand_holds(s, told):
    hand = apple.Hand(box=(0.60, 0.25, 0.12, 0.34), fingers=5)
    cup = det("cup", 0.85, 0.63, 0.30, 0.08, 0.14)

    class Eyes:
        def analyse(self, img, *, labels, text, hands):
            return apple.Seen(labels=[], text=[], hands=[hand])

    class Boxes:
        floor, sure = 0.25, 0.45

        def detect(self, img, *, min_score, limit):
            return [cup]

    apple.set_engine(Eyes())
    detect.set_engine(Boxes())
    asyncio.run(pipeline.look("how many fingers am I holding up?", frame()))
    asyncio.run(pipeline.look("what am I holding?", frame()))
    assert told[0]["holding"] is None and told[1]["holding"] == "cup"


def test_the_on_device_answer_says_its_things_in_plain_english():
    from neo2.core.vision import answer

    said = answer.local_answer("what's this?", {"objects": ["glasses", "apple", "cup", "scissors"]}, "off")
    assert said.endswith("I can see glasses, an apple and a cup.")
    assert answer.local_answer("what am I holding?", {"holding": "scissors"}, "off").endswith("you're holding scissors.")
    assert answer.local_answer("what am I holding?", {"holding": "umbrella"}, "off").endswith("you're holding an umbrella.")
    assert "a glass bottle" in answer.local_answer("?", {"objects": ["glass bottle"]}, "off")
    assert "a bus" in answer.local_answer("?", {"objects": ["bus"]}, "off")


def test_the_installer_makes_the_model_with_the_tools_it_was_tested_with():
    """install-app.sh makes YOLOE's model when ~/.neo/models has none, with the same pinned build
    tools as get_yoloe.py's own check and the README's command (a download only with --yes)."""
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("get_yoloe", root / "scripts" / "get_yoloe.py")
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)  # (its imports are the standard library's: nothing runs)
    sh, readme = (root / "scripts" / "install-app.sh").read_text(), (root / "README.md").read_text()
    import re

    pins = {text: dict(re.findall(r"--with ([a-z]+)==([0-9.]+)", text)) for text in (sh, readme, g.__doc__)}
    assert len({tuple(sorted(p.items())) for p in pins.values()}) == 1  # the same build tools everywhere
    (got,) = {tuple(sorted(p.items())) for p in pins.values()}
    assert dict(got).items() >= g.NEEDS.items() and {"numpy", "torch", "torchvision"} <= dict(got).keys()
    assert f'$HOME/.neo/models/{yolo.PKG}' in sh and '--out "$HOME/.neo/models" --yes' in sh
    assert g.PKG == yolo.PKG


def test_a_hand_the_detector_misnames_isn_t_what_it_holds():
    """Apple Vision's hand and a detector box in the same place are the hand, whatever word the box
    got (a rendered emoji hand is a "banana" to YOLOE): not held, not a hint. A phone in it still is."""
    hand = apple.Hand(box=(0.31, 0.06, 0.43, 0.86), fingers=5)
    itself = det("banana", 0.72, 0.30, 0.10, 0.40, 0.80)
    phone = det("remote", 0.61, 0.45, 0.40, 0.12, 0.20)
    assert pipeline.holding([hand], [itself]) is None and pipeline.hint_objects([itself], hands=[hand]) == []
    assert pipeline.holding([hand], [itself, phone]) == "remote"
    assert pipeline.hint_objects([itself, phone], held="remote", hands=[hand]) == ["remote"]
