"""The camera, live in the chat: a card that shows what the camera sees while it is on.

While the camera is on, its preview goes to the overlay the way NEO's browser card's page does:
binary NBF1 frames (neo2.core.tools.browser_live.pack_frame, kind ``KIND_CAMERA``) through one
latest-frame slot per client that never queues, and JSON beside them:

* ``camera_open {sid, reason}`` — the card appears (kept in the transcript replay);
* ``camera_state {sid, status, labels, boxes, answer?, popout}`` — status is
  starting | live | analysing | answered | off; boxes are ``{x, y, w, h, label, score}`` in 0..1 of
  the frame;
* ``camera_close {sid, reason}`` — the camera is off (kept in the replay: the card shows it ended).

UI → core: ``camera_stop`` (off now), ``camera_popout {popped}`` (or ``camera_popin``),
``camera_view {visible}`` (a hidden view gets no frames), ``camera_hello`` (the card's state again).
The camera also goes off on the stop button, on an erase, and ``follow_s`` after an answer (a
follow-up question reuses it until then). The hello's ``camera`` is ``{"live": state | None}``.

While the camera is live, the object detector's boxes for the newest preview frame are drawn on the
card a few times a second (neo2.core.vision.pipeline.preview: YOLOE-26s on the Neural Engine, ~11 ms
a frame, on this Mac: nothing leaves it) — people too, as NEO 1 drew them.

Frames and stills never reach the event bus, the transcript replay, a log or the disk.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import time
import weakref
from collections.abc import Callable
from typing import Any

from neo2.core import logger
from neo2.core.events import app_bus
from neo2.core.tools.browser_live import new_sid, pack_frame
from neo2.core.tools.camera.capture import Backend, CameraSession

log = logger(__name__)

KIND_CAMERA = 3  # NBF1 kind of a camera preview frame (the browser's are 1 live, 2 still)
MAX_LABELS, MAX_BOXES = 8, 20
STATUSES = ("starting", "live", "analysing", "answered", "off")
PREVIEW_BOXES_S = 0.4  # how often the live preview's boxes are refreshed (one detection at a time)


def _boxes(raw: Any) -> list[dict]:
    out: list[dict] = []
    for b in raw or ():
        if not isinstance(b, dict):
            continue
        box = b.get("box")  # detect()'s [x, y, w, h] (0..1), when not spelled out
        if isinstance(box, list | tuple) and len(box) == 4 and "x" not in b:
            b = {**b, **dict(zip(("x", "y", "w", "h"), box, strict=True))}
        try:
            x, y, w, h = (min(1.0, max(0.0, float(b.get(k, 0.0)))) for k in ("x", "y", "w", "h"))
            score = float(b.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        out.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "label": str(b.get("label") or "")[:40],
                "score": round(score, 3),
            }
        )
        if len(out) >= MAX_BOXES:
            break
    return out


def _labels(raw: Any) -> list[str]:
    out: list[str] = []
    for item in raw or ():
        label = item[0] if isinstance(item, tuple | list) and item else item
        label = str(label or "").strip()[:40]
        if label and label not in out:
            out.append(label)
        if len(out) >= MAX_LABELS:
            break
    return out


class CameraLive:
    def __init__(self) -> None:
        self.channel: Any = (
            None  # the websocket server: camera_send(type, data, recent=…), camera_frame(payload, who)
        )
        self.backend_factory: Callable[[], Backend] | None = None  # tests: a fake camera
        self.session: CameraSession | None = None
        self.sid, self.sid_bytes = "", b""
        self.reason = ""
        self.status = "off"
        self.labels: list[str] = []
        self.boxes: list[dict] = []
        self.answer: str | None = None
        self.popout = False
        self._fid = 0
        self._pump: asyncio.Task | None = None
        self._boxer: asyncio.Task | None = None
        self._off: asyncio.TimerHandle | None = None
        self._hidden: weakref.WeakSet[Any] = weakref.WeakSet()  # views that said they aren't showing

    # ---- on / off ------------------------------------------------------------------------------

    @property
    def on(self) -> bool:
        return self.session is not None

    async def open(self, reason: str = "") -> str:
        """Turn the camera on and show the card (or keep it on: a follow-up). Returns its sid.
        Raises capture.CameraUnavailable (the user's words) when it can't."""
        self._cancel_off()
        if self.session is not None:
            return self.sid
        self.sid, self.sid_bytes = new_sid()
        self.reason = str(reason or "")[:120]
        self.labels, self.boxes, self.answer, self.popout = [], [], None, False
        session = CameraSession(backend=self.backend_factory() if self.backend_factory else None)
        self.session = session
        self._send("camera_open", {"sid": self.sid, "reason": self.reason}, recent=True)
        self.update("starting")
        try:
            await session.start()
        except BaseException:
            self.close("error")
            raise
        if self.session is not session:  # stopped while it started
            session.stop()
            return self.sid
        log.info("[camera] on (the user asked)")  # that it was used — never what it saw
        self._pump = asyncio.create_task(self._stream(self.sid, session))
        self._boxer = asyncio.create_task(self._preview_boxes(session))
        self.update("live")
        return self.sid

    def close(self, reason: str = "done") -> bool:
        """The camera off, the card told (False: it was off already)."""
        self._cancel_off()
        # Off, whyever (its window ran out, an error, the user's stop, an erase): "and now?" no
        # longer reuses it, and never turns it back on (it asks afresh).
        with contextlib.suppress(Exception):
            from neo2.core.agent import camera_words

            camera_words.forget()
        session, self.session = self.session, None
        if session is None:
            return False
        for task in (self._pump, self._boxer):
            if task is not None and not task.done():
                task.cancel()
        self._pump = self._boxer = None
        session.stop()
        sid = self.sid
        self.update("off")
        self._send("camera_close", {"sid": sid, "reason": reason}, recent=True)
        log.info(f"[camera] off ({reason})")
        self.labels, self.boxes, self.answer = [], [], None
        return True

    def schedule_off(self, delay: float) -> None:
        """Off in ``delay`` seconds unless a follow-up question keeps it on."""
        self._cancel_off()
        if self.session is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.close("done")
            return
        self._off = loop.call_later(max(0.0, float(delay)), self.close, "done")

    def _cancel_off(self) -> None:
        if self._off is not None:
            self._off.cancel()
            self._off = None

    # ---- the card ------------------------------------------------------------------------------

    def update(
        self, status: str, *, labels: Any = None, boxes: Any = None, answer: str | None = None
    ) -> None:
        if status in STATUSES:
            self.status = status
        if labels is not None:
            self.labels = _labels(labels)
        if boxes is not None:
            self.boxes = _boxes(boxes)
        if answer is not None:
            self.answer = str(answer)
        if self.sid:
            self._send("camera_state", self._state())

    def _state(self) -> dict:
        data: dict[str, Any] = {
            "sid": self.sid, "status": self.status, "labels": self.labels, "boxes": self.boxes, "popout": self.popout,
        }  # fmt: skip
        if self.answer is not None:
            data["answer"] = self.answer
        return data

    def live_state(self) -> dict | None:
        """The card's state with its reason: None while the camera is off."""
        if self.session is None:
            return None
        return {**self._state(), "reason": self.reason}

    def hello_snapshot(self) -> dict:
        """The hello's ``camera``: what a (re)connecting overlay needs to show the card."""
        return {"live": self.live_state()}

    def _send(self, type_: str, data: dict, *, recent: bool = False) -> None:
        ch = self.channel
        if ch is not None:
            with contextlib.suppress(Exception):
                ch.camera_send(type_, data, recent=recent)

    async def _stream(self, sid: str, session: CameraSession) -> None:
        async for jpeg, w, h, ts in session.frames():
            if self.sid != sid or self.session is not session:
                break
            ch = self.channel
            if ch is None:
                continue
            self._fid += 1
            payload = pack_frame(
                self.sid_bytes,
                self._fid,
                jpeg,
                kind=KIND_CAMERA,
                ts_ms=ts * 1000.0,
                vw=w,
                vh=h,
                iw=w,
                ih=h,
                dpr=1.0,
            )
            with contextlib.suppress(Exception):
                ch.camera_frame(payload, self.watching)

    def watching(self, conn: Any) -> bool:
        return conn not in self._hidden

    async def _preview_boxes(self, session: CameraSession) -> None:
        """The detector's boxes on the live preview (never while a still is analysed or answered: its
        own boxes stand then). Nothing when NEO's vision isn't installed or its detector isn't there."""
        try:
            pipeline = await asyncio.to_thread(importlib.import_module, "neo2.core.vision.pipeline")
        except Exception:  # noqa: BLE001 — no vision yet: a preview without boxes
            return
        preview = getattr(pipeline, "preview", None)
        if not callable(preview):
            return
        while self.session is session:
            await asyncio.sleep(PREVIEW_BOXES_S)
            img = session.latest() if self.status == "live" else None
            if img is None:
                continue
            try:
                boxes = _boxes(await asyncio.to_thread(preview, img))
            except Exception:  # noqa: BLE001
                return
            del img
            if self.session is session and self.status == "live" and boxes != self.boxes:
                self.update("live", boxes=boxes)

    # ---- the overlay's commands (never the bus, never a log) -----------------------------------

    def command(self, conn: Any, m: dict) -> None:
        kind = str(m.get("cmd") or "")
        sid = str(m.get("sid") or "")
        if sid and sid != self.sid:
            return  # a card from before: nothing to do with the camera now
        if kind == "camera_stop":
            self.close("stopped")
        elif kind in ("camera_popout", "camera_popin"):
            self.popout = kind == "camera_popout" and bool(m.get("popped", True))
            if self.session is not None:
                self.update(self.status)
        elif kind == "camera_view":
            if m.get("visible", True):
                self._hidden.discard(conn)
            else:
                self._hidden.add(conn)
        elif kind == "camera_hello" and conn is not None:
            snap = self.live_state()
            if snap is not None and hasattr(conn, "send_json"):
                conn.send_json(json.dumps({"type": "camera_state", "data": snap, "ts": time.time()}))

    def viewer_gone(self, conn: Any) -> None:
        self._hidden.discard(conn)


camera_live = CameraLive()


def _on_event(ev) -> None:
    if ev.type == "erased":
        camera_live.close("erased")


app_bus().subscribe(_on_event)


def reset() -> None:
    """The camera off and a fresh card (tests)."""
    camera_live.close("reset")
    channel, factory = camera_live.channel, camera_live.backend_factory
    camera_live.__dict__.update(CameraLive().__dict__)
    camera_live.channel, camera_live.backend_factory = channel, factory
