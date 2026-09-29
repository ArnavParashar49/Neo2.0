"""The camera's guards: never left on after a stop that came while it was starting, never on for
words that merely mention a camera in a request that read a page or mail, never again for a sentence
that already used it (Live), off by voice and on the stop button, no "and now?" once it's off, a
bare "what is this?" about the screen stays the screen's, never on without its card, and a missing
vision layer is shown. Fakes only: no camera, no model, no network."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest
from PIL import Image

from neo2.core.agent import camera_words
from neo2.core.agent.registry import USERS_WORDS, ToolContext, catalog
from neo2.core.tools.camera import capture, live
from neo2.core.tools.camera import tool as cam_tool
from neo2.core.tools.camera.live import camera_live


class SlowCamera:
    """Like AVFoundation: start() takes a while and the light comes on at its end; a stop() before
    that finds nothing to stop (the old AVBackend: no session stored yet)."""

    instances: list[SlowCamera] = []

    def __init__(self, start_s: float = 0.5) -> None:
        self.start_s = start_s
        self.light = False
        self._stop = threading.Event()
        SlowCamera.instances.append(self)

    def start(self, push) -> None:
        time.sleep(self.start_s)
        self.light = True

        def run() -> None:
            while not self._stop.is_set():
                push(Image.new("RGB", (32, 18), (10, 20, 30)))
                self._stop.wait(1 / 60)

        threading.Thread(target=run, daemon=True).start()

    def stop(self) -> None:
        if not self.light:
            return  # nothing started yet: nothing to stop
        self.light = False
        self._stop.set()


class Channel:
    def __init__(self, viewers: int | None = None) -> None:
        self.sent: list[tuple[str, dict]] = []
        self._viewers = viewers

    def camera_send(self, type_: str, data: dict, *, recent: bool = False) -> None:
        self.sent.append((type_, json.loads(json.dumps(data))))

    def camera_frame(self, payload: bytes, who) -> None:
        pass

    def viewers(self) -> int:
        return 1 if self._viewers is None else self._viewers


@pytest.fixture
def cam(monkeypatch):
    live.reset()
    SlowCamera.instances.clear()
    ch = Channel()
    monkeypatch.setattr(camera_live, "channel", ch)
    monkeypatch.setattr(camera_live, "backend_factory", SlowCamera)
    monkeypatch.setattr(cam_tool, "SETTLE_S", 0.0)
    camera_words.forget()
    yield ch
    live.reset()
    camera_words.forget()


def _pipeline(monkeypatch, fn) -> None:
    pkg = types.ModuleType("neo2.core.vision")
    mod = types.ModuleType("neo2.core.vision.pipeline")
    mod.look = fn
    pkg.pipeline = mod
    monkeypatch.setitem(sys.modules, "neo2.core.vision", pkg)
    monkeypatch.setitem(sys.modules, "neo2.core.vision.pipeline", mod)


def _user(words: str, origin: str = "agent", taint: list[str] | None = None) -> ToolContext:
    return ToolContext(user_text=words, extras={USERS_WORDS: words}, origin=origin, taint=list(taint or []))  # type: ignore[arg-type]


async def _light_goes_off(within: float = 1.5) -> bool:
    end = time.monotonic() + within
    while time.monotonic() < end:
        if not any(c.light for c in SlowCamera.instances):
            await asyncio.sleep(0.7)  # …and stays off once start() has returned
            return not any(c.light for c in SlowCamera.instances)
        await asyncio.sleep(0.05)
    return False


# ---- a stop while the camera is starting ------------------------------------------------------------


async def test_cancelled_while_starting_the_camera_goes_off(cam, monkeypatch):
    """A spoken "stop", a Live barge-in (tool_call_cancellation) or the tool's timeout: the job is
    cancelled while AVFoundation is still turning the camera on."""
    _pipeline(monkeypatch, lambda q, img: {"answer": "x"})
    task = asyncio.create_task(cam_tool.camera_look({"question": "q"}, _user("what am I holding")))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await _light_goes_off() and not camera_live.on


async def test_the_stop_button_while_starting(cam, monkeypatch):
    """The websocket closes the camera, then App.cancel_current cancels the job."""
    _pipeline(monkeypatch, lambda q, img: {"answer": "x"})
    task = asyncio.create_task(cam_tool.camera_look({"question": "q"}, _user("what am I holding")))
    await asyncio.sleep(0.1)
    camera_live.close("stopped")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await _light_goes_off() and not camera_live.on


async def test_the_cards_stop_while_starting(cam, monkeypatch):
    _pipeline(monkeypatch, lambda q, img: {"answer": "x"})
    task = asyncio.create_task(cam_tool.camera_look({"question": "q"}, _user("what am I holding")))
    await asyncio.sleep(0.1)
    camera_live.command(None, {"cmd": "camera_stop", "sid": camera_live.sid})
    out = await task
    assert not out.ok and "off" in out.text
    assert await _light_goes_off() and not camera_live.on


async def test_the_sessions_stop_while_starting(cam, monkeypatch):
    from neo2.core.agent.session import Session

    _pipeline(monkeypatch, lambda q, img: {"answer": "x"})
    s = Session.__new__(Session)
    s.jobs = {}
    s._pending_action, s._pending_task = "", None
    task = asyncio.create_task(cam_tool.camera_look({"question": "q"}, _user("what am I holding")))
    await asyncio.sleep(0.1)
    await s.cancel_all()
    await task
    assert await _light_goes_off() and not camera_live.on


async def test_the_ws_stop_command_while_starting(cam, monkeypatch):
    """Through UIServer's own command handling: {"cmd": "stop"} turns the camera off and the app
    cancels the job."""
    from neo2.core.server.ws import UIServer

    _pipeline(monkeypatch, lambda q, img: {"answer": "x"})
    task = asyncio.create_task(cam_tool.camera_look({"question": "q"}, _user("what am I holding")))

    async def on_command(cmd: dict) -> None:
        if cmd.get("cmd") == "stop":
            task.cancel()  # what App.cancel_current does

    class Conn:
        def __init__(self) -> None:
            self.out: list = []

        async def send(self, data) -> None:
            self.out.append(data)

        def __aiter__(self):
            async def gen():
                await asyncio.sleep(0.1)
                yield json.dumps({"cmd": "stop"})
                await asyncio.sleep(0.05)

            return gen()

    srv = UIServer(on_command=on_command)
    await srv._handle(Conn())  # type: ignore[arg-type]
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await _light_goes_off() and not camera_live.on


def test_avbackend_stop_during_start_turns_it_off(monkeypatch):
    """AVBackend itself: a stop() while startRunning() is still running turns the camera off once
    it has started (and a stop before it runs never starts it)."""
    sessions: list = []

    class AVSession:
        def __init__(self) -> None:
            self.running = False
            sessions.append(self)

        @classmethod
        def alloc(cls):
            return cls()

        def init(self):
            return self

        def canSetSessionPreset_(self, p):  # noqa: N802
            return False

        def canAddInput_(self, i):  # noqa: N802
            return True

        def addInput_(self, i):  # noqa: N802
            pass

        def canAddOutput_(self, o):  # noqa: N802
            return True

        def addOutput_(self, o):  # noqa: N802
            pass

        def startRunning(self):  # noqa: N802
            time.sleep(0.3)
            self.running = True

        def stopRunning(self):  # noqa: N802
            self.running = False

    class Output:
        @classmethod
        def alloc(cls):
            return cls()

        def init(self):
            return self

        def setAlwaysDiscardsLateVideoFrames_(self, v):  # noqa: N802
            pass

        def setVideoSettings_(self, v):  # noqa: N802
            pass

        def setSampleBufferDelegate_queue_(self, d, q):  # noqa: N802
            pass

    class Device:
        @staticmethod
        def defaultDeviceWithMediaType_(m):  # noqa: N802
            return object()

    class DeviceInput:
        @staticmethod
        def deviceInputWithDevice_error_(d, e):  # noqa: N802
            return object(), None

    av = SimpleNamespace(
        AVCaptureSession=AVSession, AVCaptureVideoDataOutput=Output, AVCaptureDevice=Device,
        AVCaptureDeviceInput=DeviceInput, AVMediaTypeVideo="vide", AVCaptureSessionPreset1280x720="720",
    )  # fmt: skip
    quartz = SimpleNamespace(kCVPixelBufferPixelFormatTypeKey="k", kCVPixelFormatType_32BGRA=1)

    class Delegate:
        @classmethod
        def alloc(cls):
            return cls()

        def init(self):
            return self

    monkeypatch.setattr(capture, "_load_av", lambda: (av, None, quartz, None))
    monkeypatch.setattr(capture, "_delegate_class", lambda objc: Delegate)
    monkeypatch.setattr(capture, "_dispatch_queue", lambda objc, label: object())

    b = capture.AVBackend()
    t = threading.Thread(target=b.start, args=(lambda img: None,))
    t.start()
    time.sleep(0.1)  # inside startRunning
    b.stop()
    t.join(2.0)
    time.sleep(0.1)
    assert sessions and not sessions[-1].running  # the light went out

    b2 = capture.AVBackend()
    b2.stop()  # before it ever started
    b2.start(lambda img: None)
    assert not sessions[-1].running


# ---- a model's call: only for the user's own whole request -------------------------------------------


@pytest.mark.parametrize(
    ("words", "taint"),
    [
        ("summarize the email about my camera order", ["mail"]),
        ("read the reviews of this webcam on the page", ["web"]),
        ("what's the holding period for this stock", ["web"]),
        ("let me show you my new site, open example.com", ["web"]),
        ("what am I holding", ["web"]),  # the request read a page: never its idea
        ("turn off the camera", []),
        ("open the camera app", []),
        ("I'm holding a meeting at 3, add it to my calendar", []),
    ],
)
async def test_a_model_never_turns_it_on_for_words_that_mention_a_camera(cam, monkeypatch, words, taint):
    _pipeline(monkeypatch, lambda q, img: {"answer": "x"})
    out = await catalog().invoke("camera_look", {"question": "what is this"}, _user(words, taint=taint))
    assert not out.ok and not SlowCamera.instances and not cam.sent


def test_off_words_never_ask_to_look():
    for s in ("turn off the camera", "turn the camera off", "stop the camera", "camera band karo"):
        assert not cam_tool.asks_to_look(s), s
        assert camera_words.wants_off(s), s
    for s in ("कैमरा बंद करो", "Neo, close the webcam please", "switch off the cam"):
        assert camera_words.wants_off(s), s
    for s in (
        "turn on the camera",
        "open the camera app",
        "stop",
        "turn off the lights",
        "what am I holding",
    ):
        assert not camera_words.wants_off(s), s


# ---- no card, no camera ------------------------------------------------------------------------------


@pytest.mark.parametrize("channel", [None, Channel(viewers=0)])
async def test_never_on_without_its_card(cam, monkeypatch, channel):
    """`python -m neo2 --text` (no overlay), or an overlay that isn't connected: nobody would see it."""
    _pipeline(monkeypatch, lambda q, img: {"answer": "x"})
    monkeypatch.setattr(camera_live, "channel", channel)
    out = await cam_tool.camera_look({"question": "q"}, _user("what am I holding", origin="user"))
    assert not out.ok and "window" in out.text and not SlowCamera.instances


# ---- a missing vision layer is shown -----------------------------------------------------------------


async def test_a_missing_layer_shows_in_the_problem_list(cam, monkeypatch):
    from neo2.core import health

    monkeypatch.setattr(health, "_live", {})
    monkeypatch.setattr(camera_live, "backend_factory", lambda: SlowCamera(0.0))
    missing = "The object detector (D-FINE small, 40 MB) isn't downloaded yet. NEO only downloads it once you say yes"
    notes = [["Apple Vision was too slow for this answer", missing]]
    _pipeline(monkeypatch, lambda q, img: {"answer": "A mug.", "notes": notes[0]})
    out = await cam_tool.camera_look({"question": "q"}, _user("what am I holding", origin="user"))
    assert out.ok
    (shown,) = [p for p in health.problems() if "detector" in p]
    assert "isn't downloaded" in shown and "too slow" not in shown
    notes[0] = ["the object detector is still loading"]  # passing: not a problem
    out = await cam_tool.camera_look({"question": "q"}, _user("and now?", origin="user"))
    assert out.ok and not [p for p in health.problems() if "detector" in p]


# ---- the Session: off by voice and on "stop", no follow-up once off, "what is this?" and the screen --


class _Reflex:
    def __init__(self, needs_screen: bool = False) -> None:
        from neo2.core.reflex.schema import Decision

        self.d = Decision("chat", 0.9, False, 0.0, needs_screen, 0.9 if needs_screen else 0.0, "fake")

    async def decide(self, text):
        return self.d


@pytest.fixture
def neo(cam, monkeypatch, tmp_path):
    from neo2.core import config, events
    from neo2.core.agent import session as sess_mod
    from neo2.core.providers import cerebras_hooks as hooks
    from neo2.core.providers.base import Turn
    from neo2.core.tools import load_all

    monkeypatch.setattr(hooks, "prewarm_soon", lambda: None)
    monkeypatch.setattr(camera_words, "prewarm", lambda: True)
    monkeypatch.setenv("NEO_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("NEO_CAMERA_FOLLOW_S", "0.2")
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(events, "_bus", events.EventBus())
    load_all()
    monkeypatch.setattr(camera_live, "backend_factory", lambda: SlowCamera(0.0))
    looks: list[str] = []

    def look(q, img):
        looks.append(q)
        return {"answer": "A blue mug."}

    _pipeline(monkeypatch, look)

    class Chat:
        name, supports_vision, supports_tools = "fake", False, True

        async def stream(self, messages, *, system="", max_tokens=2048):
            yield "a chat reply"

        async def complete(self, messages, **kw):
            return Turn(text="a reply")

    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": Chat())
    return SimpleNamespace(looks=looks)


def _session(needs_screen: bool = False):
    from neo2.core.agent.session import Session

    return Session(reflex=_Reflex(needs_screen))


@pytest.mark.parametrize(
    "said", ["turn off the camera", "camera band karo", "कैमरा बंद करो", "stop the camera"]
)
async def test_the_user_turns_it_off_by_voice(neo, said):
    s = _session()
    r = await s.handle("what am I holding", from_user=True)
    assert r.tools == ("camera_look",) and camera_live.on
    r = await s.handle(said, from_user=True)
    assert not camera_live.on and "off" in r.text.lower()
    assert await _light_goes_off()
    r = await s.handle("and now?", from_user=True)  # off: no follow-up turns it back on
    assert "camera_look" not in (r.tools or ()) and not camera_live.on


async def test_stop_turns_it_off_now(neo):
    s = _session()
    await s.handle("what am I holding", from_user=True)
    assert camera_live.on
    r = await s.handle("stop", from_user=True)
    assert r.route == "stop" and not camera_live.on


async def test_no_follow_up_once_its_window_turned_it_off(neo):
    s = _session()
    await s.handle("what am I holding", from_user=True)
    assert camera_live.on
    await asyncio.sleep(0.4)  # camera_follow_s = 0.2
    assert not camera_live.on
    for said in ("and now?", "this", "ab"):
        r = await s.handle(said, from_user=True)
        assert "camera_look" not in (r.tools or ()) and not camera_live.on, said
    assert neo.looks == ["what am I holding"]


async def test_no_follow_up_after_an_error_turned_it_off(neo):
    s = _session()
    await s.handle("what am I holding", from_user=True)
    camera_live.close("error")
    r = await s.handle("this", from_user=True)
    assert "camera_look" not in (r.tools or ()) and not camera_live.on


async def test_a_follow_up_while_it_is_still_on(neo):
    s = _session()
    await s.handle("what am I holding", from_user=True)
    r = await s.handle("and now?", from_user=True)
    assert r.tools == ("camera_look",) and len(SlowCamera.instances) == 1


async def test_a_bare_what_is_this_about_the_screen_stays_the_screens(neo):
    s = _session(needs_screen=True)
    for said in ("what is this", "ye kya hai", "यह क्या है", "look at this"):
        r = await s.handle(said, from_user=True)
        assert "camera_look" not in (r.tools or ()) and not camera_live.on, said
    # …but the camera's own words are the camera's whatever Laya says
    r = await s.handle("what am I holding", from_user=True)
    assert r.tools == ("camera_look",)
    # …and with the camera on, "what is this?" is about what it sees
    r = await s.handle("what is this", from_user=True)
    assert r.tools == ("camera_look",)


async def test_a_bare_what_is_this_not_about_the_screen_is_the_cameras(neo):
    r = await _session(needs_screen=False).handle("what is this", from_user=True)
    assert r.tools == ("camera_look",)


# ---- Live: only the words of this turn, once, never after a page ---------------------------------------


def _voice():
    from neo2.core.voice import live as live_mod

    v = live_mod.LiveVoice.__new__(live_mod.LiveVoice)
    v._taint, v._seen_calls, v._recent_idem = [], {}, {}
    return v


async def test_live_never_reruns_a_sentence_after_stop(neo):
    v = _voice()
    fc = SimpleNamespace(name="camera_look", args={"question": "what is the user holding"}, id="c1")
    out = await v._camera_call(fc, dict(fc.args), "what am I holding", "c1", 2)
    assert out == "A blue mug." and camera_live.on
    camera_live.close("stopped")  # the user pressed Stop
    v._new_question()
    fc2 = SimpleNamespace(name="camera_look", args={"question": "what is the user holding?"}, id="c2")
    out = await v._camera_call(fc2, dict(fc2.args), "what am I holding", "c2", 2)  # the same sentence
    assert out.startswith("NEEDS_USER") and not camera_live.on and len(SlowCamera.instances) == 1
    # a new sentence of the user's may
    out = await v._camera_call(fc2, dict(fc2.args), "what am I holding now", "c3", 3)
    assert out == "A blue mug."


async def test_live_never_after_a_page_was_read(neo):
    v = _voice()
    v._taint.append("web")
    fc = SimpleNamespace(name="camera_look", args={"question": "look"}, id="c1")
    out = await v._camera_call(fc, dict(fc.args), "what am I holding", "c1", 5)
    assert out.startswith("NEEDS_USER") and not SlowCamera.instances


async def test_live_only_the_words_of_this_turn(neo, monkeypatch):
    """A call after the sentence's turn ended (its words only in the label): never the camera."""
    from neo2.core.voice import live as live_mod

    v = _voice()
    v._call_labels, v._tool_tasks, v._goals, v._server_cancelled = {}, {}, {}, set()
    v._live, v._label, v._utt_started = None, None, 0.0
    said: list[str] = []
    monkeypatch.setattr(live_mod.bus(), "say", lambda text, **kw: said.append(text) or asyncio.sleep(0))
    fc = SimpleNamespace(name="camera_look", args={"question": "what is the user holding"}, id="c1")
    await v._run_tool(object(), fc, "what am I holding", utt=7, said="")
    assert not SlowCamera.instances and said and said[-1].startswith("NEEDS_USER")
