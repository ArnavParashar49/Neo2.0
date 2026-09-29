"""The camera end to end: the user's words → the router (fastpath.camera) → camera_look → the camera
(a fake) → Apple Vision and the object detector (fakes) → the answer from Gemini, else Cerebras's vision model (fake
brains), else the on-device labels → said, with the chat's card updated, and the camera off after
the follow-up window, on the card's stop, and on the stop button. Nothing real: no camera, no model,
no network."""

from __future__ import annotations

import asyncio
import json
import threading
import weakref

import pytest
from PIL import Image

from neo2.core import config, health
from neo2.core import providers as prov
from neo2.core.agent import camera_words
from neo2.core.agent.registry import catalog
from neo2.core.providers import cerebras_hooks as hooks
from neo2.core.providers.base import Overloaded, Turn
from neo2.core.tools.camera import live
from neo2.core.tools.camera import tool as cam_tool
from neo2.core.tools.camera.live import camera_live
from neo2.core.vision import apple, detect


class FakeCamera:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self.stopped = False

    def start(self, push) -> None:
        def run() -> None:
            while not self._stop.is_set():
                push(Image.new("RGB", (160, 90), (30, 90, 200)))
                self._stop.wait(1 / 60)

        threading.Thread(target=run, daemon=True).start()

    def stop(self) -> None:
        self.stopped = True
        self._stop.set()


class Channel:
    def __init__(self) -> None:
        self.sent: list[tuple[str, dict]] = []
        self.frames = 0

    def camera_send(self, type_: str, data: dict, *, recent: bool = False) -> None:
        self.sent.append((type_, json.loads(json.dumps(data))))

    def camera_frame(self, payload: bytes, who) -> None:
        self.frames += 1

    def states(self) -> list[dict]:
        return [d for t, d in self.sent if t == "camera_state"]


class FakeVision:
    def analyse(self, img, *, labels, text, hands) -> apple.Seen:
        return apple.Seen(
            labels=[("cup", 0.8)],
            text=["HYDRO FLASK"],
            hands=[apple.Hand(box=(0.4, 0.5, 0.2, 0.3), fingers=2, side="right")],
        )


class FakeDetector:
    def detect(self, img, *, min_score, limit):
        return [detect.Det(label="cup", score=0.9, box={"x": 0.45, "y": 0.35, "w": 0.15, "h": 0.25})]


class Brain:
    supports_tools = True

    def __init__(self, name: str, *, busy: bool = False) -> None:
        self.name, self.model, self.busy = name, f"{name}-model", busy
        self.supports_vision = name not in ("groq", "cerebras")
        self.efforts = frozenset({"minimal", "low", "medium", "high"}) if name == "gemini" else None
        self.sent: list[dict] = []

    def cooling_until(self) -> float:
        return 0.0

    async def complete(self, messages, **kw) -> Turn:
        self.sent.append({"messages": messages, **kw})
        if self.busy:
            raise Overloaded(f"{self.name}: 503")
        return Turn(text=f"You're holding a blue mug ({self.name}).")


class _Reflex:
    async def decide(self, text):
        from neo2.core.reflex.schema import Decision

        return Decision("chat", 0.9, False, 0.0, False, 0.0, "fake")


@pytest.fixture
def neo(monkeypatch):
    """NEO.app's settings (Gemini and Cerebras keys, a short follow-up window), fake brains, a fake
    camera and fake on-device layers; the session's own chat/agent brain must never be asked."""
    from neo2.core import events
    from neo2.core.agent import session as sess_mod
    from neo2.core.tools import load_all

    warmed: list[int] = []
    monkeypatch.setattr(hooks, "prewarm_soon", lambda: warmed.append(1))
    monkeypatch.setattr(hooks, "_accounts", weakref.WeakKeyDictionary())
    monkeypatch.setattr(prov, "_cache", {})
    monkeypatch.setattr(prov, "_demoted", {})
    monkeypatch.setattr(health, "_live", {})
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-test-not-a-real-key")
    monkeypatch.setenv("NEO_CAMERA_FOLLOW_S", "0.3")
    s = config.Settings()
    s.ensure_dirs()
    monkeypatch.setattr(config, "_settings", s)
    monkeypatch.setattr(events, "_bus", events.EventBus())
    load_all()
    assert catalog().get("camera_look") is not None

    brains = {n: Brain(n) for n in ("gemini", "gemini_lite", "cerebras_vision", "cerebras", "groq", "groq_vision")}
    monkeypatch.setattr(prov, "_build", lambda n: brains[n])

    chats: list[str] = []

    class Chat:  # the session's own chat/agent brain: never asked for a camera answer
        name, supports_vision, supports_tools = "fake", False, True

        async def stream(self, messages, *, system="", max_tokens=2048):
            chats.append("stream")
            yield "a chat reply"

        async def complete(self, messages, **kw):
            chats.append("complete")
            return Turn(text="a reply")

    monkeypatch.setattr(sess_mod, "brain", lambda purpose="agent": Chat())
    apple.set_engine(FakeVision())
    detect.set_engine(FakeDetector())
    live.reset()
    ch = Channel()
    monkeypatch.setattr(camera_live, "channel", ch)
    monkeypatch.setattr(camera_live, "backend_factory", FakeCamera)
    monkeypatch.setattr(cam_tool, "SETTLE_S", 0.0)
    monkeypatch.setattr(cam_tool, "STILL_WINDOW_S", 0.0)
    monkeypatch.setattr(cam_tool, "DARK_WAIT_S", 0.0)
    camera_words.forget()
    yield type("Neo", (), {"brains": brains, "channel": ch, "warmed": warmed, "settings": s, "chats": chats})
    live.reset()
    apple.set_engine(None)
    detect.set_engine(None)
    camera_words.forget()


def _session():
    from neo2.core.agent.session import Session

    return Session(reflex=_Reflex())


async def test_what_am_i_holding_end_to_end(neo):
    r = await _session().handle("Neo, what am I holding?", from_user=True)
    assert r.route == "quick" and r.tools == ("camera_look",)
    assert r.text == "You're holding a blue mug (gemini)."
    # One small still went to Gemini (and Cerebras, asked at once), with the on-device hints; nobody else.
    (sent,) = neo.brains["gemini"].sent
    words = sent["messages"][0].text
    assert "HYDRO FLASK" in words and "cup" in words and len(sent["messages"][0].images) == 1
    assert all(not b.sent for n, b in neo.brains.items() if n not in ("gemini", "cerebras_vision"))
    assert neo.warmed  # Cerebras's connection warmed when the request was recognised
    # The card: open, starting → live → analysing (the local boxes first) → answered, still on.
    types_ = [t for t, _ in neo.channel.sent]
    assert types_[0] == "camera_open" and "camera_close" not in types_
    statuses = [d["status"] for d in neo.channel.states()]
    assert statuses[:3] == ["starting", "live", "analysing"] and statuses[-1] == "answered"
    boxes_first = next(d for d in neo.channel.states() if d["status"] == "analysing" and d["boxes"])
    assert boxes_first["boxes"][0]["label"] == "cup" and "answer" not in boxes_first
    last = neo.channel.states()[-1]
    assert last["answer"] == r.text and "cup" in last["labels"]
    assert not neo.chats  # no text brain was asked
    await asyncio.sleep(0.1)
    assert camera_live.on and neo.channel.frames >= 1  # the live preview streams meanwhile
    # No picture ever rode the card's JSON.
    assert all("jpeg" not in json.dumps(d).lower() for _, d in neo.channel.sent)
    # …and off by itself after the follow-up window.
    await asyncio.sleep(0.6)
    assert not camera_live.on and neo.channel.sent[-1][0] == "camera_close"


async def test_a_follow_up_reuses_the_camera_then_the_cards_stop_ends_it(neo):
    s = _session()
    await s.handle("what is this", from_user=True)
    sid = camera_live.sid
    r = await s.handle("and now?", from_user=True)
    assert r.tools == ("camera_look",) and camera_live.sid == sid  # the same session, still on
    assert len(neo.brains["gemini"].sent) == 2
    camera_live.command(None, {"cmd": "camera_stop", "sid": sid})
    assert not camera_live.on
    r = await s.handle("and now?", from_user=True)  # stopped: no follow-up turns it back on
    assert "camera_look" not in (r.tools or ()) and not camera_live.on


async def test_gemini_busy_cerebras_answers(neo):
    neo.brains["gemini"].busy = True
    r = await _session().handle("what am I holding", from_user=True)
    assert "blue mug" in r.text and "cerebras" in r.text
    assert neo.brains["gemini"].sent and len(neo.brains["cerebras_vision"].sent) == 1


async def test_both_cloud_brains_down_the_local_labels_answer(neo):
    neo.brains["gemini"].busy = neo.brains["cerebras_vision"].busy = True
    r = await _session().handle("what am I holding", from_user=True)
    assert "cup" in r.text.lower() and r.tools == ("camera_look",)
    assert not neo.brains["groq"].sent and not neo.brains["cerebras"].sent  # never a blind brain


async def test_cloud_answers_off_nothing_leaves_the_mac(neo):
    neo.settings.vision_cloud_ok = False
    r = await _session().handle("what am I holding", from_user=True)
    assert "cup" in r.text.lower()
    assert all(not b.sent for b in neo.brains.values())


async def test_the_stop_button_turns_the_camera_off(neo):
    s = _session()
    await s.handle("what am I holding", from_user=True)
    assert camera_live.on
    await s.cancel_all()  # a spoken "stop", the stop button (App), an SDK host's stop: off now
    assert not camera_live.on
    r = await s.handle("and now?", from_user=True)
    assert "camera_look" not in (r.tools or ())


async def test_a_model_or_a_page_never_turns_it_on(neo):
    from neo2.core.agent.registry import ToolContext, registry

    r = await _session().handle("what am I holding", origin="live")  # Live's own goal, not the user's words
    assert "camera_look" not in (r.tools or ()) and not camera_live.on
    # A model calling it with no words of the user's (a page said so, a relay, a replay): refused.
    out = await registry().invoke("camera_look", {"question": "what is this"}, ToolContext(origin="agent"))
    assert not out.ok and not camera_live.on
    assert not any(t == "camera_open" for t, _ in neo.channel.sent)


def test_the_card_draws_what_the_detector_is_sure_of_people_too_like_neo_1():
    from neo2.core.vision.pipeline import objects

    box = {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}
    dets = [
        {"label": "guitar", "score": 0.3, "box": {**box, "x": 0.5}},  # unsure, but Apple Vision says "guitar"
        {"label": "sports ball", "score": 0.35, "box": {**box, "y": 0.5}},  # unsure, and nobody else says so
        {"label": "phone", "score": 0.9, "box": box},  # sure on its own
        {"label": "man", "score": 0.99, "box": {"x": 0.0, "y": 0.0, "w": 0.9, "h": 1.0}},  # the user: drawn, as "person"
    ]
    kept = [d["label"] for d in objects(dets, [("guitar", 0.79), ("musical_instrument", 0.79)], scores=(0.25, 0.45))]
    assert kept == ["person", "phone", "guitar"]
    assert [d["label"] for d in objects(dets, None, scores=(0.25, 0.45))] == ["person", "phone"]  # the live preview
