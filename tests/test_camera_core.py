"""The camera core: capture (a fake camera), the chat's live card, the websocket wiring and camera_look's
privacy gate. The real camera is never turned on here (effects are off: AVFoundation is refused)."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import types

import pytest
from PIL import Image

from neo2.core import effects, runtime
from neo2.core.agent.registry import USERS_WORDS, ToolContext, catalog
from neo2.core.config import settings
from neo2.core.tools.browser_live import unpack_frame
from neo2.core.tools.camera import capture, live
from neo2.core.tools.camera import tool as cam_tool
from neo2.core.tools.camera.live import KIND_CAMERA, camera_live


class FakeCamera:
    """Pushes a small image from its own thread (like AVFoundation's dispatch queue) until stopped."""

    instances: list[FakeCamera] = []

    def __init__(self, fps: float = 60.0) -> None:
        self.fps = fps
        self.started = self.stopped = False
        self._stop = threading.Event()
        FakeCamera.instances.append(self)

    def start(self, push) -> None:
        self.started = True

        def run() -> None:
            n = 0
            while not self._stop.is_set():
                n += 1
                push(Image.new("RGB", (64, 36), (n * 10 % 255, 80, 120)))
                self._stop.wait(1.0 / self.fps)

        threading.Thread(target=run, daemon=True).start()

    def stop(self) -> None:
        self.stopped = True
        self._stop.set()


class Viewer:
    pass


class FakeChannel:
    def __init__(self) -> None:
        self.sent: list[tuple[str, dict, bool]] = []
        self.frames: list[bytes] = []
        self.clients: list[object] = [Viewer()]

    def camera_send(self, type_: str, data: dict, *, recent: bool = False) -> None:
        self.sent.append((type_, json.loads(json.dumps(data)), recent))

    def camera_frame(self, payload: bytes, who) -> None:
        if any(who(c) for c in self.clients):
            self.frames.append(payload)

    def types(self) -> list[str]:
        return [t for t, _, _ in self.sent]

    def states(self) -> list[dict]:
        return [d for t, d, _ in self.sent if t == "camera_state"]


@pytest.fixture
def cam(monkeypatch):
    live.reset()
    ch = FakeChannel()
    FakeCamera.instances.clear()
    monkeypatch.setattr(camera_live, "channel", ch)
    monkeypatch.setattr(camera_live, "backend_factory", FakeCamera)
    monkeypatch.setattr(cam_tool, "SETTLE_S", 0.0)
    monkeypatch.setattr(cam_tool, "STILL_WINDOW_S", 0.0)
    monkeypatch.setattr(cam_tool, "DARK_WAIT_S", 0.0)
    yield ch
    live.reset()


def _pipeline(monkeypatch, fn) -> None:
    """A stand-in for neo2.core.vision.pipeline (the vision branch's module)."""
    pkg = types.ModuleType("neo2.core.vision")
    mod = types.ModuleType("neo2.core.vision.pipeline")
    mod.look = fn
    pkg.pipeline = mod
    monkeypatch.setitem(sys.modules, "neo2.core.vision", pkg)
    monkeypatch.setitem(sys.modules, "neo2.core.vision.pipeline", mod)


def _user(words: str, origin: str = "agent") -> ToolContext:
    return ToolContext(user_text=words, extras={USERS_WORDS: words}, origin=origin)  # type: ignore[arg-type]


# ---- capture ------------------------------------------------------------------------------------


async def test_session_latest_frames_and_stop():
    cam = FakeCamera()
    s = capture.CameraSession(backend=cam, fps=30)
    await s.start()
    assert await s.wait_frame(2.0)
    img = s.latest()
    assert img is not None and img.size == (64, 36)
    got = []
    async for jpeg, w, h, ts in s.frames():
        assert jpeg[:2] == b"\xff\xd8" and (w, h) == (64, 36) and ts > 0
        got.append(jpeg)
        if len(got) == 2:
            break
    s.stop()
    assert cam.stopped and s.latest() is None  # nothing seen stays in memory
    s.stop()  # idempotent


async def test_preview_frames_end_on_stop():
    s = capture.CameraSession(backend=FakeCamera(), fps=30)
    await s.start()

    async def consume() -> int:
        n = 0
        async for _ in s.frames():
            n += 1
        return n

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.15)
    s.stop()
    assert await asyncio.wait_for(task, 2.0) >= 1


def test_missing_avfoundation_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "AVFoundation", None)
    why = capture.available()
    assert "vision extra" in why and "AVFoundation" in why


async def test_missing_avfoundation_refuses_to_start(monkeypatch):
    monkeypatch.setitem(sys.modules, "AVFoundation", None)
    with pytest.raises(capture.CameraUnavailable, match="vision extra"):
        await capture.CameraSession().start()


async def test_the_real_camera_is_refused_under_tests():
    pytest.importorskip("AVFoundation")
    with pytest.raises(effects.EffectRefused) as e:
        await capture.CameraSession().start()
    assert e.value.kind == effects.CAMERA
    with pytest.raises(effects.EffectRefused):
        capture.AVBackend().start(lambda img: None)


# ---- the live card ------------------------------------------------------------------------------


async def test_card_opens_streams_and_closes(cam):
    sid = await camera_live.open("what am I holding?")
    assert camera_live.on and sid
    assert cam.sent[0] == ("camera_open", {"sid": sid, "reason": "what am I holding?"}, True)
    assert [d["status"] for d in cam.states()][:2] == ["starting", "live"]
    for _ in range(40):
        if cam.frames:
            break
        await asyncio.sleep(0.05)
    meta, img = unpack_frame(cam.frames[0])
    assert meta["kind"] == KIND_CAMERA and meta["sid"] == sid and img[:2] == b"\xff\xd8"
    snap = camera_live.hello_snapshot()["live"]
    assert snap and snap["sid"] == sid and snap["status"] == "live"

    camera_live.update(
        "answered",
        labels=[("mug", 0.9), "cup"],
        boxes=[{"x": 0.1, "y": 0.2, "w": 0.3, "h": 1.4, "label": "cup", "score": 0.8}],
        answer="A mug.",
    )
    st = cam.states()[-1]
    assert st["labels"] == ["mug", "cup"] and st["answer"] == "A mug." and st["boxes"][0]["h"] == 1.0

    assert camera_live.close("done")
    assert FakeCamera.instances[-1].stopped and not camera_live.on
    assert cam.states()[-1]["status"] == "off"
    assert cam.sent[-1] == ("camera_close", {"sid": sid, "reason": "done"}, True)
    assert camera_live.hello_snapshot() == {"live": None}
    assert not camera_live.close("done")


async def test_follow_up_reuses_then_turns_off(cam):
    sid = await camera_live.open("q1")
    camera_live.schedule_off(0.05)
    assert await camera_live.open("q2") == sid  # a follow-up in the window: the same camera
    await asyncio.sleep(0.1)
    assert camera_live.on  # the follow-up cancelled the turn-off
    camera_live.schedule_off(0.05)
    await asyncio.sleep(0.15)
    assert not camera_live.on and len(FakeCamera.instances) == 1


async def test_card_commands(cam):
    sid = await camera_live.open("q")
    viewer = cam.clients[0]
    camera_live.command(viewer, {"cmd": "camera_view", "visible": False})
    assert not camera_live.watching(viewer)
    camera_live.command(viewer, {"cmd": "camera_view", "visible": True})
    assert camera_live.watching(viewer)
    camera_live.command(viewer, {"cmd": "camera_popout", "sid": sid, "popped": True})
    assert cam.states()[-1]["popout"] is True
    camera_live.command(viewer, {"cmd": "camera_popout", "sid": sid, "popped": False})
    assert cam.states()[-1]["popout"] is False
    camera_live.command(viewer, {"cmd": "camera_stop", "sid": "an-old-card"})
    assert camera_live.on  # a stale card's stop is for a camera that's already off
    camera_live.command(viewer, {"cmd": "camera_stop", "sid": sid})
    assert not camera_live.on and cam.sent[-1][2] and cam.sent[-1][1]["reason"] == "stopped"


async def test_erase_turns_the_camera_off(cam):
    from neo2.core.events import app_bus

    await camera_live.open("q")
    await app_bus().publish("erased", counts={})
    assert not camera_live.on


# ---- the websocket server --------------------------------------------------------------------------


async def test_ws_camera_lane():
    from neo2.core.server.ws import UIServer, _Client

    class Conn:
        pass

    srv = UIServer(on_command=lambda cmd: asyncio.sleep(0))
    a, b = _Client(Conn()), _Client(Conn())  # type: ignore[arg-type]
    srv._clients = {a.conn: a, b.conn: b}
    srv.camera_send("camera_state", {"sid": "s", "status": "live"})
    assert json.loads(a.q[-1])["type"] == "camera_state" and b.q
    assert not srv._recent  # state isn't replayed
    srv.camera_send("camera_open", {"sid": "s", "reason": "r"}, recent=True)
    assert srv._recent[-1]["type"] == "camera_open"
    srv.camera_frame(b"f1", lambda cl: cl is a)
    srv.camera_frame(b"f2", lambda cl: cl is a)
    assert a.cam == b"f2" and b.cam is None  # one latest-frame slot, only for who shows the card
    a.offer_frame(b"page")
    assert a.frame == b"page" and a.cam == b"f2"  # the browser's slot is its own


# ---- camera_look ----------------------------------------------------------------------------------


def test_registered_as_a_camera_tool():
    from neo2.core.tools import PRESETS

    t = catalog().get("camera_look")
    assert t and t.category == "camera" and "macos" in t.requires and t.untrusted
    assert all("camera_look" not in names for names in PRESETS.values())


@pytest.mark.parametrize(
    "ctx",
    [
        ToolContext(user_text="what am I holding?", origin="agent"),  # a model's call, no words of the user's
        _user("summarize this page"),  # the user asked something else: the page said "look"
        _user("open safari", origin="live"),
    ],
)
async def test_camera_only_on_the_users_own_request(cam, monkeypatch, ctx):
    _pipeline(monkeypatch, lambda q, img: {"answer": "x"})
    out = await catalog().invoke("camera_look", {"question": "what am I holding?"}, ctx)
    assert not out.ok and "ask" in out.text
    assert not FakeCamera.instances and not cam.sent  # never turned on, no card


async def test_refused_in_library_mode_and_when_off(cam, monkeypatch):
    monkeypatch.setattr(runtime, "_library", True)
    out = await cam_tool.camera_look(
        {"question": "what am I holding"}, _user("what am I holding", origin="user")
    )
    assert not out.ok
    monkeypatch.setattr(runtime, "_library", False)
    monkeypatch.setenv("NEO_CAMERA", "false")
    from neo2.core import config

    monkeypatch.setattr(config, "_settings", None)
    out = await cam_tool.camera_look(
        {"question": "what am I holding"}, _user("what am I holding", origin="user")
    )
    assert not out.ok and "settings" in out.text
    assert not FakeCamera.instances


async def test_look_answers_and_keeps_the_camera_for_a_follow_up(cam, monkeypatch):
    seen = []

    async def look(question, img):
        seen.append((question, img.size))
        return {
            "answer": "You're holding a red mug.",
            "labels": [("mug", 0.93)],
            "boxes": [{"x": 0.2, "y": 0.3, "w": 0.4, "h": 0.4, "label": "cup", "score": 0.88}],
            "text": [],
            "ms": 40,
        }

    _pipeline(monkeypatch, look)
    monkeypatch.setenv("NEO_CAMERA_FOLLOW_S", "0.2")
    from neo2.core import config

    monkeypatch.setattr(config, "_settings", None)
    assert settings().camera_follow_s == 0.2
    out = await catalog().invoke(
        "camera_look", {"question": "what am I holding?"}, _user("what am I holding?")
    )
    assert out.ok and out.text.endswith("red mug.") and out.say is True
    assert seen == [("what am I holding?", (64, 36))]
    statuses = [d["status"] for d in cam.states()]
    assert statuses[:4] == ["starting", "live", "analysing", "answered"]
    assert cam.states()[-1]["labels"] == ["mug"] and cam.states()[-1]["boxes"][0]["label"] == "cup"
    assert camera_live.on  # the follow-up window

    # "and now?" while it's on: the same camera, no new words about looking needed
    out = await catalog().invoke("camera_look", {"question": "and now?"}, _user("and now?"))
    assert out.ok and len(FakeCamera.instances) == 1
    await asyncio.sleep(0.35)
    assert not camera_live.on and FakeCamera.instances[0].stopped


async def test_sync_pipeline_and_missing_vision(cam, monkeypatch):
    monkeypatch.setitem(sys.modules, "neo2.core.vision", None)
    out = await cam_tool.camera_look({"question": "what am I holding"}, _user("what am I holding"))
    assert "isn't installed" in out.text
    camera_live.close("done")
    _pipeline(monkeypatch, lambda q, img: {"answer": "Two fingers."})
    out = await cam_tool.camera_look(
        {"question": "how many fingers"}, _user("how many fingers am I holding up")
    )
    assert out.text == "Two fingers."


async def test_failure_and_cancel_turn_the_camera_off(cam, monkeypatch):
    def boom(q, img):
        raise RuntimeError("no")

    _pipeline(monkeypatch, boom)
    out = await cam_tool.camera_look({"question": "q"}, _user("what am I holding"))
    assert not out.ok and not camera_live.on

    async def slow(q, img):
        await asyncio.sleep(10)

    _pipeline(monkeypatch, slow)
    task = asyncio.create_task(cam_tool.camera_look({"question": "q"}, _user("look at me")))
    await asyncio.sleep(0.2)
    assert camera_live.on
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not camera_live.on and cam.sent[-1][1]["reason"] == "stopped"


def test_asks_to_look():
    for s in (
        "what am I holding?",
        "turn on the camera",
        "how many fingers am I holding up",
        "can you see me",
        "what's in my hand",
    ):
        assert cam_tool.asks_to_look(s), s
    for s in ("open safari", "what's the weather", "summarize this"):
        assert not cam_tool.asks_to_look(s), s


async def test_local_hints_reach_the_card_before_the_answer(cam, monkeypatch):
    async def look(question, img, *, on_local=None):
        on_local(
            {
                "labels": [["bottle", 0.8]],
                "boxes": [{"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.5, "label": "bottle", "score": 0.7}],
            }
        )
        return {"answer": "A water bottle.", "labels": [["bottle", 0.8]], "boxes": []}

    _pipeline(monkeypatch, look)
    out = await cam_tool.camera_look({"question": "what am I holding"}, _user("what am I holding"))
    assert out.text == "A water bottle."
    early = [d for d in cam.states() if d["status"] == "analysing" and d["labels"]]
    assert early and early[0]["boxes"][0]["label"] == "bottle"  # drawn while the brain answered


async def test_live_preview_gets_boxes(cam, monkeypatch):
    calls = []

    def preview(img):
        calls.append(img.size)
        return [{"label": "cup", "score": 0.9, "box": [0.1, 0.2, 0.3, 0.4]}]

    _pipeline(monkeypatch, lambda q, img: {"answer": "x"})
    monkeypatch.setattr(sys.modules["neo2.core.vision.pipeline"], "preview", preview, raising=False)
    monkeypatch.setattr(live, "PREVIEW_BOXES_S", 0.02)
    await camera_live.open("q")
    for _ in range(50):
        if any(d["boxes"] for d in cam.states()):
            break
        await asyncio.sleep(0.02)
    box = next(d for d in cam.states() if d["boxes"])["boxes"][0]
    assert box == {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4, "label": "cup", "score": 0.9, "id": box["id"]}
    camera_live.update("analysing")
    n = len(calls)
    await asyncio.sleep(0.1)
    assert len(calls) <= n + 1  # no preview detection while a still is analysed
    camera_live.close("done")


def test_a_thing_keeps_its_box_from_frame_to_frame(cam):
    """Each thing keeps its id and its place in the list while it stays in view: the card moves its box
    with it, and never slides one thing's box over to another (the UI keys boxes by id)."""
    camera_live.sid = "s1"

    def b(label, x, score):
        return {"x": x, "y": 0.3, "w": 0.2, "h": 0.3, "label": label, "score": score}

    camera_live.update("live", boxes=[b("cup", 0.1, 0.9), b("glasses", 0.5, 0.8)])
    first = {x["label"]: x["id"] for x in camera_live.boxes}
    camera_live.update("live", boxes=[b("glasses", 0.51, 0.85), b("pen", 0.75, 0.84), b("cup", 0.12, 0.7)])  # a new order
    assert [x["label"] for x in camera_live.boxes] == ["cup", "glasses", "pen"]  # first seen first
    assert {x["label"]: x["id"] for x in camera_live.boxes if x["label"] != "pen"} == first
    pen = next(x["id"] for x in camera_live.boxes if x["label"] == "pen")
    assert pen not in first.values()
    camera_live.update("live", boxes=[b("pen", 0.76, 0.8), b("cup", 0.4, 0.8)])  # the cup moved far: a new box
    got = {x["label"]: x["id"] for x in camera_live.boxes}
    assert got["pen"] == pen and got["cup"] not in (*first.values(), pen)
    camera_live.update("live", boxes=[b("marker", 0.765, 0.7), b("cup", 0.4, 0.8)])  # the same box, another word
    assert {x["label"]: x["id"] for x in camera_live.boxes} == {"marker": pen, "cup": got["cup"]}
    assert cam.states()[-1]["boxes"] == camera_live.boxes


async def test_a_page_never_reuses_the_camera_in_its_follow_up_window(cam, monkeypatch):
    _pipeline(monkeypatch, lambda q, img: {"answer": "A mug."})
    assert (await cam_tool.camera_look({"question": "what am I holding"}, _user("what am I holding"))).ok
    assert camera_live.on  # the follow-up window
    tainted = ToolContext(
        user_text="summarize this page", extras={USERS_WORDS: "summarize this page"}, taint=["web"]
    )
    out = await cam_tool.camera_look({"question": "describe the user"}, tainted)
    assert not out.ok and "ask" in out.text
    ok = await cam_tool.camera_look({"question": "and now?"}, _user("and now?"))
    assert ok.ok and len(FakeCamera.instances) == 1


class _Frames:
    """A camera whose newest frame moves through ``frames``, one per read."""

    def __init__(self, frames, started_ago: float = 5.0) -> None:
        import time

        self.frames, self.i = frames, 0
        self.started_at = time.monotonic() - started_ago

    def latest(self):
        img = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return img


def _checks(light: int = 200, blur: float = 0.0) -> Image.Image:
    from PIL import ImageDraw, ImageFilter

    img = Image.new("RGB", (640, 360), (light // 4,) * 3)
    d = ImageDraw.Draw(img)
    for x in range(0, 640, 20):
        for y in range(0, 360, 20):
            if (x + y) // 20 % 2:
                d.rectangle((x, y, x + 9, y + 9), fill=(light,) * 3)
    return img.filter(ImageFilter.GaussianBlur(blur)) if blur else img


async def test_the_still_is_the_sharpest_frame_not_just_the_newest(monkeypatch):
    monkeypatch.setattr(cam_tool, "STILL_WINDOW_S", 0.3)
    sharp = _checks()
    cam = _Frames([_checks(blur=4), sharp, _checks(blur=6), _checks(blur=3)])
    assert await cam_tool.pick_still(cam) is sharp


async def test_a_dark_picture_waits_for_the_exposure_then_gives_up(monkeypatch):
    monkeypatch.setattr(cam_tool, "STILL_WINDOW_S", 0.05)
    monkeypatch.setattr(cam_tool, "DARK_WAIT_S", 0.4)
    bright = _checks(blur=2)
    cam = _Frames([_checks(light=30)] * 3 + [bright], started_ago=0.0)
    assert await cam_tool.pick_still(cam) is bright  # (a sharper dark frame came first)
    dark = _checks(light=30)
    import time

    t0 = time.monotonic()
    assert await cam_tool.pick_still(_Frames([dark], started_ago=0.0)) is dark  # dark all along
    assert 0.3 < time.monotonic() - t0 < 1.0
