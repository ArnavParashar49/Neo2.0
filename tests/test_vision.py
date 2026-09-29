"""The camera's eyes (neo2/core/vision): Apple Vision's labels, words and fingers, D-FINE's boxes, and
the answer from the cloud brains that see — Gemini, then Cerebras's vision model, the on-device
labels alone when those are off or down.

Hermetic: every layer is a fake (Apple Vision's engine, the detector, the brains); the real ones are
refused under pytest (no model load, no download, no camera, nothing leaves the Mac)."""

from __future__ import annotations

import asyncio
import io
import math
import sys
import threading
import types
import weakref

import pytest
from PIL import Image

from neo2.core import config, effects, gpu, health
from neo2.core import providers as prov
from neo2.core.providers import cerebras_hooks as hooks
from neo2.core.providers.base import Overloaded, QuotaExhausted, Turn
from neo2.core.vision import Unavailable, apple, detect, pipeline
from neo2.core.vision import answer as ans

# ---- fakes -------------------------------------------------------------------------------------


def frame(w: int = 1920, h: int = 1080, color=(200, 40, 40)) -> Image.Image:
    return Image.new("RGB", (w, h), color)


class FakeVision:
    """Apple Vision: a mug with words on it, one hand holding up three fingers."""

    def __init__(self, *, hands: list[apple.Hand] | None = None, slow: float = 0.0) -> None:
        self.calls: list[dict] = []
        self._hands = [apple.Hand(box=(0.4, 0.5, 0.2, 0.3), fingers=3, side="right")] if hands is None else hands
        self.slow = slow

    def analyse(self, img, *, labels, text, hands) -> apple.Seen:
        self.calls.append({"size": img.size, "labels": labels, "text": text, "hands": hands})
        if self.slow:
            threading.Event().wait(self.slow)
        return apple.Seen(
            labels=[("cup", 0.81), ("beverage", 0.44)] if labels else [],
            text=["HYDRO FLASK"] if text else [],
            hands=list(self._hands) if hands else [],
        )


class FakeDetector:
    def __init__(self, dets: list[detect.Det] | None = None, *, slow: float = 0.0, fail: BaseException | None = None):
        self.dets = dets if dets is not None else [
            detect.Det(label="cup", score=0.91, box={"x": 0.45, "y": 0.35, "w": 0.15, "h": 0.25}),
            detect.Det(label="person", score=0.95, box={"x": 0.1, "y": 0.0, "w": 0.8, "h": 1.0}),
            detect.Det(label="laptop", score=0.7, box={"x": 0.0, "y": 0.8, "w": 0.2, "h": 0.2}),
        ]  # fmt: skip
        self.slow, self.fail = slow, fail
        self.calls = 0

    def detect(self, img, *, min_score, limit):
        self.calls += 1
        if self.slow:
            threading.Event().wait(self.slow)
        if self.fail is not None:
            raise self.fail
        return [d for d in self.dets if d["score"] >= min_score][:limit]


class Brain:
    """A cloud brain: answers, is overloaded / out of quota, is slow, or says nothing. Keeps what it
    was sent."""

    supports_tools = True

    def __init__(self, name: str, *, vision: bool = True, mode: str = "ok") -> None:
        self.name, self.model, self.supports_vision, self.mode = name, f"{name}-model", vision, mode
        self.efforts = frozenset({"minimal", "low", "medium", "high"}) if name == "gemini" else None
        self.sent: list[dict] = []

    def cooling_until(self) -> float:
        return 0.0

    async def complete(self, messages, **kw) -> Turn:
        self.sent.append({"messages": messages, **kw})
        if self.mode == "busy":
            raise Overloaded(f"{self.name}: 503")
        if self.mode == "quota":
            raise QuotaExhausted(f"{self.name}: 429 quota")
        if self.mode == "slow":
            await asyncio.sleep(30)
        if self.mode == "late":  # answers, a little after the other brain
            await asyncio.sleep(0.3)
        if self.mode == "empty":
            return Turn(text="")
        return Turn(text=f"**A blue mug** from {self.name}.")


@pytest.fixture(autouse=True)
def _fresh_layers():
    ans._late_until.clear()
    yield
    apple.set_engine(None)
    detect.set_engine(None)
    ans._late_until.clear()


@pytest.fixture
def mac(monkeypatch):
    """NEO.app's settings with a Gemini key (a Cerebras key unless ``cerebras=False``), fake seeing
    brains, a fresh pool; Cerebras's warm-up recorded, never made."""
    warmed: list[int] = []
    monkeypatch.setattr(hooks, "prewarm_soon", lambda: warmed.append(1))
    monkeypatch.setattr(hooks, "_accounts", weakref.WeakKeyDictionary())
    monkeypatch.setattr(prov, "_cache", {})
    monkeypatch.setattr(prov, "_demoted", {})
    monkeypatch.setattr(health, "_live", {})

    def make(*, cerebras: bool = True, modes: dict[str, str] | None = None, **env):
        monkeypatch.setenv("GEMINI_API_KEY", "g-test")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.setenv("CEREBRAS_API_KEY", "csk-test-not-a-real-key" if cerebras else "")
        for k, v in env.items():
            monkeypatch.setenv(k, str(v))
        s = config.Settings()
        s.ensure_dirs()
        monkeypatch.setattr(config, "_settings", s)
        modes = modes or {}
        fakes = {
            n: Brain(n, vision=n not in ("groq", "cerebras"), mode=modes.get(n, "ok"))
            for n in ("gemini", "gemini_lite", "cerebras_vision", "cerebras", "groq", "groq_vision", "claude")
        }
        monkeypatch.setattr(prov, "_build", lambda n: fakes[n])
        fakes["_warmed"] = warmed  # type: ignore[assignment]
        return fakes

    return make


def _image_of(sent: dict) -> Image.Image:
    (msg,) = sent["messages"]
    (part,) = msg.images
    assert part.mime == "image/jpeg"
    return Image.open(io.BytesIO(part.data))


# ---- Apple Vision -------------------------------------------------------------------------------------


def _hand(up: set[str], *, thumb: bool) -> dict[str, apple.Point]:
    """A hand pointing up (Vision's y grows upward): the wrist at the bottom, each finger's joints above."""
    pts: dict[str, apple.Point] = {"wrist": (0.5, 0.1, 0.9)}
    for i, f in enumerate(("index", "middle", "ring", "little")):
        x = 0.4 + i * 0.05
        pts[f"{f}_mcp"] = (x, 0.3, 0.9)
        pts[f"{f}_pip"] = (x, 0.4, 0.9)
        pts[f"{f}_tip"] = (x, 0.6 if f in up else 0.33, 0.9)
    pts["thumb_ip"] = (0.35, 0.22, 0.9)
    pts["thumb_tip"] = (0.2 if thumb else 0.4, 0.3 if thumb else 0.25, 0.9)
    return pts


@pytest.mark.parametrize(
    ("up", "thumb", "want"),
    [
        (set(), False, 0),
        ({"index"}, False, 1),
        ({"index", "middle"}, False, 2),
        ({"index", "middle", "ring"}, False, 3),
        ({"index", "middle", "ring", "little"}, True, 5),
    ],
)
def test_fingers_are_counted_from_the_hand_s_joints(up, thumb, want):
    assert apple.count_fingers(_hand(up, thumb=thumb)) == want


def test_a_hand_whose_joints_are_unclear_has_no_finger_count():
    pts = {k: (x, y, 0.1) for k, (x, y, _) in _hand({"index"}, thumb=False).items()}
    assert apple.count_fingers(pts) is None
    assert apple.count_fingers({"index_tip": (0.5, 0.5, 0.9)}) is None


# A right hand, palm to the camera (y up; the palm, wrist to middle knuckle, 1 long), as Vision's joints.
_MCP = {"index": (-0.35, 0.95), "middle": (-0.1, 1.0), "ring": (0.15, 0.95), "little": (0.37, 0.85)}
_BONES = {"index": (0.45, 0.27, 0.2), "middle": (0.5, 0.3, 0.22), "ring": (0.47, 0.29, 0.21), "little": (0.37, 0.2, 0.18)}
_SPLAY = {"index": -0.12, "middle": -0.03, "ring": 0.06, "little": 0.16}
_THUMB = {  # cmc, mp, ip, tip
    "out": [(-0.2, 0.2), (-0.45, 0.42), (-0.65, 0.62), (-0.8, 0.8)],
    "alongside": [(-0.2, 0.2), (-0.4, 0.4), (-0.52, 0.62), (-0.6, 0.8)],
    "on_little": [(-0.2, 0.2), (-0.3, 0.45), (-0.05, 0.6), (0.2, 0.68)],  # holding the little finger down
    "on_ring": [(-0.2, 0.2), (-0.3, 0.45), (-0.1, 0.62), (0.08, 0.72)],
    "over_fist": [(-0.2, 0.2), (-0.35, 0.5), (-0.28, 0.78), (-0.05, 0.9)],
}


def _real_hand(up, thumb, *, turn=0.0, size=0.3, mirror=False, aspect=16 / 9):
    pts = {"wrist": (0.0, 0.0)}
    for f, (x, y) in _MCP.items():
        a, b, c = _BONES[f]
        if f in up:
            dx, dy = math.sin(_SPLAY[f]), math.cos(_SPLAY[f])
            chain = [(x + dx * t, y + dy * t) for t in (0, a, a + b, a + b + c)]
        else:  # curled into the palm: the first bone toward the camera, the rest folded down
            chain = [(x, y), (x, y + 0.12), (x, y - 0.08), (x + 0.02, y - 0.25)]
        pts.update(zip((f"{f}_mcp", f"{f}_pip", f"{f}_dip", f"{f}_tip"), chain, strict=True))
    pts.update(zip(("thumb_cmc", "thumb_mp", "thumb_ip", "thumb_tip"), _THUMB[thumb], strict=True))
    out = {}
    for k, (x, y) in pts.items():
        x = -x if mirror else x
        xr, yr = x * math.cos(turn) - y * math.sin(turn), x * math.sin(turn) + y * math.cos(turn)
        out[k] = (0.5 + xr * size / aspect, 0.4 + yr * size, 0.9)
    return out


@pytest.mark.parametrize(
    ("up", "thumb", "want"),
    [
        ((), "over_fist", 0),
        (("index",), "over_fist", 1),
        (("index", "middle"), "on_ring", 2),
        (("index", "middle", "ring"), "on_little", 3),  # the thumb holding the little finger down
        (("index", "middle"), "out", 3),
        (("index", "little"), "out", 3),
        (("index", "middle", "ring", "little"), "on_ring", 4),
        (("index", "middle", "ring", "little"), "alongside", 5),  # the thumb up beside the index
        (("index", "middle", "ring", "little"), "out", 5),
    ],
)
@pytest.mark.parametrize("turn", [0.0, 0.6, -1.4, math.pi])
@pytest.mark.parametrize("mirror", [False, True])
def test_fingers_are_counted_however_the_hand_is_turned(up, thumb, want, turn, mirror):
    pts = _real_hand(set(up), thumb, turn=turn, mirror=mirror)
    assert apple.count_fingers(pts, 16 / 9) == want
    far = _real_hand(set(up), thumb, turn=turn, mirror=mirror, size=0.1)  # a hand far from the camera
    assert apple.count_fingers(far, 16 / 9) == want


def test_an_unclear_finger_makes_the_count_a_range_never_a_wrong_number():
    pts = _real_hand({"index", "middle", "ring"}, "on_little")
    pts["ring_tip"] = (*pts["ring_tip"][:2], 0.1)  # the ring finger's tip unclear
    states = apple.finger_states(pts, 16 / 9)
    assert states == {"thumb": False, "index": True, "middle": True, "ring": None, "little": False}
    assert apple.count_fingers(pts, 16 / 9) is None and apple.finger_range(states) == (2, 3)
    hand = apple.Hand((0.4, 0.3, 0.2, 0.3), None, "right", 0.9, states)
    got = apple.hands_summary([hand])
    assert got["fingers"] is None and got["range"] == [2, 3] and got["raised"] == ["index", "middle"]
    for f in ("index", "middle", "ring"):  # three fingers unclear: no count at all
        pts[f"{f}_tip"] = (*pts[f"{f}_tip"][:2], 0.1)
    assert apple.finger_range(apple.finger_states(pts, 16 / 9)) is None


def test_a_half_bent_finger_is_unsure():
    pts = _real_hand({"index", "middle", "ring", "little"}, "out", size=1.0, aspect=1.0)
    x, y = 0.5 - 0.35, 0.4 + 0.95  # the index knuckle; its first bone up, the next two bent 60° each
    bent = {"index_pip": (x, y + 0.45), "index_dip": (x + 0.234, y + 0.585), "index_tip": (x + 0.407, y + 0.485)}
    pts.update({k: (*v, 0.9) for k, v in bent.items()})
    states = apple.finger_states(pts, 1.0)
    assert states["index"] is None and states["middle"] is True
    assert apple.count_fingers(pts, 1.0) is None and apple.finger_range(states) == (4, 5)


def test_a_hand_s_box_is_top_left_based_and_padded():
    x, y, w, h = apple.hand_box({"a": (0.2, 0.8, 0.9), "b": (0.4, 0.6, 0.9), "c": (0.3, 0.7, 0.9)}, pad=0.0)
    assert (round(x, 3), round(y, 3), round(w, 3), round(h, 3)) == (0.2, 0.2, 0.2, 0.2)
    assert apple.hand_box({"a": (0.2, 0.8, 0.9)}) is None  # too few clear joints


def test_labels_words_and_hands_through_the_engine():
    v = FakeVision()
    apple.set_engine(v)
    assert apple.classify(frame()) == [("cup", 0.81), ("beverage", 0.44)]
    assert apple.text(frame()) == ["HYDRO FLASK"]
    got = apple.hands(frame())
    assert got["count"] == 1 and got["fingers"] == 3 and got["hands"][0]["side"] == "right"
    assert [c["labels"] for c in v.calls] == [True, False, False]  # each asks only for its own
    two = apple.hands_summary([apple.Hand((0, 0, 0.1, 0.1), 2), apple.Hand((0.5, 0, 0.1, 0.1), None)])
    assert two["count"] == 2 and two["fingers"] is None  # one hand unclear: no total


def test_the_real_apple_vision_never_runs_in_the_tests():
    img = frame(64, 48)
    with pytest.raises((effects.EffectRefused, Unavailable)):
        apple.analyse(img)


# ---- D-FINE -------------------------------------------------------------------------------------------


def test_pixel_boxes_become_bounded_frame_boxes_best_first():
    dets = detect.to_dets(
        [0.4, 0.9, 0.6, 0.95], ["dog", "cup", "book", "person"],
        [[0, 0, 10, 10], [100, 50, 300, 250], [0, 0, 50, 50], [-20, 0, 800, 500]], 640, 480, limit=2,
    )  # fmt: skip
    assert [d["label"] for d in dets] == ["person", "cup"]  # below 0.5 dropped, best first, at most 2
    assert dets[0]["box"] == {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}  # clamped to the frame
    assert dets[1]["box"] == {"x": 0.1562, "y": 0.1042, "w": 0.3125, "h": 0.4167}


def test_the_detector_is_never_downloaded_unless_the_user_said_yes(tmp_path, monkeypatch):
    s = config.Settings()
    monkeypatch.setattr(config, "_settings", s)
    assert detect.source() == (detect.REPO, False)
    ok, why = detect.available()
    assert not ok and "vision_download_ok" in why and "40 MB" in why
    monkeypatch.setattr(s, "vision_download_ok", True)
    assert detect.source() == (detect.REPO, True)
    folder = s.models_dir / "dfine-small-coco"  # the orchestrator's fetch
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    monkeypatch.setattr(s, "vision_download_ok", False)
    assert detect.source() == (str(folder), False)
    assert detect.available() == (True, "")


def test_the_real_detector_never_loads_in_the_tests(monkeypatch):
    s = config.Settings()
    monkeypatch.setattr(config, "_settings", s)
    monkeypatch.setattr(s, "vision_download_ok", True)
    monkeypatch.setattr(detect, "_dfine", None)
    with pytest.raises((effects.EffectRefused, Unavailable)):
        detect.detect(frame(64, 48))
    assert not detect.loaded()
    assert pipeline.preview(frame(64, 48)) == []  # the live card: no boxes, no error


def test_the_detector_runs_on_the_cpu_and_never_waits_for_the_gpu(tmp_path, monkeypatch):
    """D-FINE off MPS: a frame there beside a Laya pass aborted NEO (neo2.core.gpu). So nothing it
    loads or reads is moved to a device, and a frame is read while Laya holds the GPU lock."""
    torch = pytest.importorskip("torch")
    moved: list[object] = []

    class Inputs(dict):
        def to(self, *a, **k):
            moved.append(("inputs", a, k))
            return self

    class Proc:
        @classmethod
        def from_pretrained(cls, src, **kw):
            assert kw.get("local_files_only") is True  # the folder on this Mac, never a download
            return cls()

        def __call__(self, images, return_tensors):
            return Inputs(pixel_values=torch.zeros(1, 3, 8, 8))

        def post_process_object_detection(self, outputs, target_sizes, threshold):
            h, w = (float(v) for v in target_sizes[0])
            return [{"scores": torch.tensor([0.9]), "labels": torch.tensor([0]),
                     "boxes": torch.tensor([[0.1 * w, 0.2 * h, 0.5 * w, 0.6 * h]])}]  # fmt: skip

    class Model:
        config = types.SimpleNamespace(id2label={0: "cup"})

        @classmethod
        def from_pretrained(cls, src, **kw):
            return cls()

        def to(self, *a, **k):
            moved.append(("model", a, k))
            return self

        def eval(self):
            return self

        def __call__(self, **inputs):
            return types.SimpleNamespace(**inputs)

    fake = types.ModuleType("transformers")
    fake.AutoImageProcessor, fake.DFineForObjectDetection = Proc, Model
    monkeypatch.setitem(sys.modules, "transformers", fake)
    folder = tmp_path / "dfine-small-coco"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    s = config.Settings()
    monkeypatch.setattr(config, "_settings", s)
    monkeypatch.setattr(s, "vision_detector", str(folder))
    monkeypatch.setattr(detect, "_dfine", None)
    got: list[list[detect.Det]] = []
    held, done = threading.Event(), threading.Event()

    def laya_busy():  # a Laya pass (or its load) holding the GPU the whole time
        with gpu.lock():
            held.set()
            done.wait(5)

    busy = threading.Thread(target=laya_busy)
    busy.start()
    held.wait(5)
    try:
        with effects.allow("model"):
            reader = threading.Thread(target=lambda: got.append(detect.detect(frame(640, 480))))
            reader.start()
            reader.join(5)
    finally:
        done.set()
        busy.join(5)
    assert got and got[0][0]["label"] == "cup"  # read while the GPU lock was held
    assert got[0][0]["box"] == {"x": 0.1, "y": 0.2, "w": 0.4, "h": 0.4}
    assert moved == []  # neither the weights nor a frame went to a device


def test_detect_is_bounded_and_the_card_gets_its_boxes():
    d = FakeDetector()
    detect.set_engine(d)
    assert len(detect.detect(frame(), limit=100)) <= detect.MAX_BOXES
    boxes = pipeline.preview(frame())
    assert boxes[0] == {"x": 0.45, "y": 0.35, "w": 0.15, "h": 0.25, "label": "cup", "score": 0.91}


# ---- what's in the hand ----------------------------------------------------------------------------


def test_what_am_i_holding_is_the_object_by_the_hand_not_the_person():
    hand = apple.Hand(box=(0.4, 0.5, 0.2, 0.3), fingers=3)
    assert pipeline.holding([hand], FakeDetector().dets) == "cup"
    far = [detect.Det(label="laptop", score=0.9, box={"x": 0.0, "y": 0.0, "w": 0.05, "h": 0.05})]
    assert pipeline.holding([hand], far) is None
    assert pipeline.holding([], FakeDetector().dets) is None


# ---- the answer ------------------------------------------------------------------------------------------

HINTS = {
    "objects": ["cup", "person"],
    "labels": [["cup", 0.81]],
    "text": ["HYDRO FLASK"],
    "hands": {"count": 1, "fingers": 3},
    "holding": "cup",
}


def test_the_still_is_one_small_jpeg():
    img = Image.open(io.BytesIO(ans.still(frame(4032, 3024))))
    assert img.format == "JPEG" and max(img.size) == ans.MAX_EDGE and img.size == (1024, 768)
    small = Image.open(io.BytesIO(ans.still(frame(640, 480).convert("RGBA"))))
    assert small.size == (640, 480)


def test_the_hints_are_one_short_line():
    line = ans.format_hints(HINTS)
    assert line == (
        'likely objects: cup, person; scene: cup; text seen: "HYDRO FLASK"; '
        "hand: 1 hand in view, holding something near it — likely a cup, 3 fingers up (counted from the "
        "hand's joints)"
    )
    sure = ans.format_hints({"hands": {"count": 1, "fingers": 3, "raised": ["thumb", "index", "little"]}})
    assert sure == "hand: 1 hand in view, 3 fingers up (counted from the hand's joints: thumb, index, little)"
    unsure = ans.format_hints({"hands": {"count": 1, "fingers": None, "range": [3, 4], "raised": ["index"]}})
    assert unsure == "hand: 1 hand in view, 3 or 4 fingers up (counted from the hand's joints)"
    assert ans.format_hints({"hands": {"count": 1, "fingers": None, "range": None}}) == "hand: 1 hand in view"
    assert ans.format_hints(None) == "" and len(ans.format_hints({"text": ["x" * 999] * 9})) <= 360


async def test_gemini_answers_first_with_one_small_still_and_the_hints(mac):
    b = mac()
    r = await ans.ask(frame(4032, 3024), "what am I holding?", HINTS)
    assert (r.by, r.cloud, r.text) == ("gemini", True, "A blue mug from gemini.")  # markdown gone
    (sent,) = b["gemini"].sent
    assert max(_image_of(sent).size) <= ans.MAX_EDGE
    words = sent["messages"][0].text
    assert words.startswith("what am I holding?") and "likely a cup" in words and "HYDRO FLASK" in words
    assert sent["effort"] == "minimal" and sent["max_tokens"] <= 384 and "language" in sent["system"]
    assert "finger count" in sent["system"]
    (other,) = b["cerebras_vision"].sent  # asked at the same time, with the same still and words
    assert other["effort"] == "low" and other["messages"][0].text == words
    assert _image_of(other).size == _image_of(sent).size
    assert not b["groq"].sent and not b["cerebras"].sent


async def test_gemini_s_answer_is_waited_for_a_little_even_when_cerebras_s_came_first(mac):
    mac(modes={"gemini": "late"})
    r = await ans.ask(frame(), "what is this?", HINTS)
    assert (r.by, r.text) == ("gemini", "A blue mug from gemini.")


async def test_past_gemini_s_time_cerebras_s_ready_answer_is_taken_at_once(mac, monkeypatch):
    monkeypatch.setattr(ans, "GEMINI_S", 0.05)
    b = mac(modes={"gemini": "slow"})
    r = await ans.ask(frame(), "what is this?", HINTS)
    assert r.by == "cerebras_vision" and r.ms < 1000  # (not the 30 s Gemini would take)
    assert len(b["gemini"].sent) == 1
    r = await ans.ask(frame(), "and now?", HINTS)  # Gemini rests: not asked, no wait
    assert r.by == "cerebras_vision" and len(b["gemini"].sent) == 1 and len(b["cerebras_vision"].sent) == 2
    monkeypatch.setattr(ans, "_late_until", {"gemini": 0.0})  # rested: asked again
    b["gemini"].mode = "ok"
    assert (await ans.ask(frame(), "and now?", HINTS)).by == "gemini"


@pytest.mark.parametrize("mode", ["busy", "quota", "slow", "empty"])
async def test_cerebras_vision_answers_when_gemini_can_t(mac, monkeypatch, mode):
    monkeypatch.setattr(ans, "GEMINI_S", 0.05)
    b = mac(modes={"gemini": mode})
    r = await ans.ask(frame(), "what is this?", HINTS)
    assert (r.by, r.cloud) == ("cerebras_vision", True)
    (sent,) = b["cerebras_vision"].sent
    assert _image_of(sent).size == (1024, 576)
    assert b["_warmed"]  # its connection was opened as the question came in
    assert not b["groq"].sent and not b["cerebras"].sent and not b["groq_vision"].sent  # only the two


async def test_both_brains_down_the_answer_is_the_labels_and_says_so(mac):
    b = mac(modes={"gemini": "busy", "cerebras_vision": "quota"})
    r = await ans.ask(frame(), "how many fingers am I holding up?", HINTS)
    assert (r.by, r.cloud, r.why) == ("local", False, "failed")
    assert r.text.startswith("I couldn't reach Gemini or Cerebras, so this is from the on-device labels only")
    assert "holding a cup" in r.text and "HYDRO FLASK" in r.text and "I count 3 fingers up." in r.text
    assert len(b["gemini"].sent) == len(b["cerebras_vision"].sent) == 1


async def test_cloud_answers_off_nothing_leaves_the_mac(mac):
    b = mac(NEO_VISION_CLOUD_OK="false")
    assert ans.prewarm() is False
    r = await ans.ask(frame(), "what am I holding?", HINTS)
    assert (r.by, r.why) == ("local", "off") and r.text.startswith("Cloud answers are off")
    assert not any(x.sent for n, x in b.items() if n != "_warmed") and not b["_warmed"]


async def test_without_a_cerebras_key_only_gemini_is_asked(mac):
    b = mac(cerebras=False, modes={"gemini": "busy"})
    assert ans.brains() == ["gemini"] and ans.prewarm() is False
    r = await ans.ask(frame(), "what is this?", "a cup")
    assert r.by == "local" and b["gemini"].sent and not b["cerebras_vision"].sent


async def test_a_brain_that_can_t_see_never_gets_the_picture(mac):
    b = mac()
    b["gemini"].supports_vision = False
    b["cerebras_vision"].supports_vision = False
    r = await ans.ask(frame(), "what is this?", HINTS)
    assert (r.by, r.why) == ("local", "none") and not b["gemini"].sent and not b["cerebras_vision"].sent


def test_cloud_answers_are_neo_app_s_default_and_off_for_an_embedded_agent():
    from neo2.core.config import LibrarySettings, Settings

    assert Settings.model_fields["vision_cloud_ok"].default is True  # the user's choice for NEO.app
    assert LibrarySettings.model_fields["vision_cloud_ok"].default is False  # a host opts in
    assert Settings.model_fields["vision_download_ok"].default is False
    assert LibrarySettings.model_fields["vision_download_ok"].default is False


def test_the_local_answer_without_anything_seen():
    assert ans.local_answer("what?", None, "none").endswith("I can't make out what it is.")
    assert "I can see a laptop" in ans.local_answer("what?", {"objects": ["person", "laptop"]})


# ---- the pipeline -------------------------------------------------------------------------------------------


async def test_look_runs_the_layers_then_the_answer(mac):
    b = mac()
    apple.set_engine(FakeVision())
    detect.set_engine(FakeDetector())
    shown: list[dict] = []
    out = await pipeline.look("what am I holding?", frame(), on_local=shown.append)
    assert out["answer"] == "A blue mug from gemini." and out["by"] == "gemini" and out["cloud"]
    assert out["labels"] == [["cup", 0.81], ["beverage", 0.44]] and out["text"] == ["HYDRO FLASK"]
    assert out["holding"] == "cup" and out["hands"]["fingers"] == 3 and out["notes"] == []
    assert {"label", "score", "x", "y", "w", "h"} <= out["boxes"][0].keys()
    assert shown and shown[0]["boxes"] == out["boxes"] and "answer" not in shown[0]  # boxes before the answer
    assert set(out["ms"]) == {"apple", "detect", "answer", "total"}
    assert "likely objects: cup;" in b["gemini"].sent[0]["messages"][0].text


async def test_look_goes_on_without_a_slow_or_missing_layer(mac):
    mac()
    apple.set_engine(FakeVision())
    detect.set_engine(FakeDetector(slow=1.0))
    out = await pipeline.look("what is this?", frame(), local_wait_s=0.1)
    assert out["answer"] and out["boxes"] == [] and "the object detector is still loading" in out["notes"]
    detect.set_engine(FakeDetector(fail=Unavailable("The object detector isn't downloaded yet")))
    out = await pipeline.look("what is this?", frame())
    assert out["notes"] == ["The object detector isn't downloaded yet"] and out["labels"]


async def test_look_with_every_layer_missing_still_answers(mac):
    mac(NEO_VISION_CLOUD_OK="0")
    apple.set_engine(FakeVision())
    detect.set_engine(FakeDetector(fail=effects.EffectRefused("model", "load", "test")))
    out = await pipeline.look("what am I holding?", frame())
    assert out["by"] == "local" and out["answer"].startswith("Cloud answers are off")
    assert out["notes"] == ["the object detector failed (EffectRefused)"]
