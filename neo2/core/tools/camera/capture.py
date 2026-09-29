"""The Mac's camera, in NEO's own process (PyObjC AVFoundation).

An AVCaptureSession with an AVCaptureVideoDataOutput at ~720p hands every frame to a delegate on a
dispatch queue of its own. The newest frame is kept in memory (``latest()``: the still a question is
answered from); while someone iterates ``frames()`` (the chat's live preview) each kept frame is also
encoded as a JPEG, at most ``fps`` a second. Frames never touch the disk, and nothing here logs one.

Turning the camera on is a real effect: ``effects.guard("camera", …)`` right before the session starts
(refused under NEO's tests and NEO_NO_EFFECTS=1 unless a fake backend stands in for AVFoundation).
Without pyobjc-framework-AVFoundation, :class:`CameraUnavailable` says what to install.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import threading
import time
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

from neo2.core import effects

PREVIEW_FPS = 12.0
JPEG_QUALITY = 70
WIDTH, HEIGHT = 1280, 720
_INSTALL = "install NEO's vision extra (pyobjc-framework-AVFoundation)"


class CameraUnavailable(RuntimeError):
    """The camera can't be used; the message is for the user."""


class Backend(Protocol):
    """What turns frames on: ``start(push)`` calls ``push(image)`` with a PIL image per frame (from
    any thread) until ``stop()``."""

    def start(self, push: Callable[[Any], None]) -> None: ...

    def stop(self) -> None: ...


def _load_av() -> tuple[Any, Any, Any, Any]:
    try:
        import AVFoundation  # type: ignore[import-not-found]
        import CoreMedia  # type: ignore[import-not-found]
        import objc
        import Quartz  # CoreVideo's pixel buffers
    except ImportError as e:
        raise CameraUnavailable(f"The camera isn't set up: {_INSTALL}.") from e
    return AVFoundation, CoreMedia, Quartz, objc


def available() -> str:
    """ "" when the camera's code is installed, else why not (for the user)."""
    try:
        _load_av()
    except CameraUnavailable as e:
        return str(e)
    return ""


def _dispatch_queue(objc: Any, label: bytes) -> Any:
    """A serial dispatch queue for the frame delegate (libdispatch's module when PyObjC has it)."""
    try:
        import dispatch  # type: ignore[import-not-found]  # pyobjc-framework-libdispatch

        return dispatch.dispatch_queue_create(label, None)
    except ImportError:
        pass
    import ctypes

    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    lib.dispatch_queue_create.restype = ctypes.c_void_p
    lib.dispatch_queue_create.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    return objc.objc_object(c_void_p=lib.dispatch_queue_create(label, None))


_delegate_cls: Any = None


def _delegate_class(objc: Any) -> Any:
    """The AVCaptureVideoDataOutputSampleBufferDelegate (one ObjC class per process)."""
    global _delegate_cls
    if _delegate_cls is None:
        from Foundation import NSObject  # type: ignore[import-not-found]

        protocols = []
        with contextlib.suppress(Exception):  # the delegate method's exact signature, from the protocol
            protocols = [objc.protocolNamed("AVCaptureVideoDataOutputSampleBufferDelegate")]

        class NEOCameraDelegate(NSObject, protocols=protocols):  # type: ignore[misc,valid-type,call-arg]
            def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, buf, conn):  # noqa: N802
                with objc.autorelease_pool():
                    fn = getattr(self, "_neo_on_frame", None)
                    if fn is not None:
                        with contextlib.suppress(Exception):  # a bad frame never kills the queue
                            fn(buf)

        _delegate_cls = NEOCameraDelegate
    return _delegate_cls


class AVBackend:
    """The built-in camera (or the first video device) through AVFoundation."""

    def __init__(self) -> None:
        self._session: Any = None
        self._delegate: Any = None
        self._queue: Any = None
        # stop() may come while start() is still turning the camera on (on its own thread: a
        # cancelled request can't interrupt it): start() then turns it straight off again.
        self._mu = threading.Lock()
        self._stopped = False

    @staticmethod
    async def authorize() -> None:
        """Ask macOS once; refuse with the user's words when the camera is blocked for NEO."""
        av, _, _, _ = _load_av()
        effects.guard(effects.CAMERA, "use the camera", via=av.AVCaptureDevice)  # before any camera call
        media = av.AVMediaTypeVideo
        status = int(av.AVCaptureDevice.authorizationStatusForMediaType_(media))
        if status == 0:  # not determined: macOS asks the user now (NEO.app's NSCameraUsageDescription)
            loop = asyncio.get_running_loop()
            fut: asyncio.Future[bool] = loop.create_future()

            def done(granted: bool) -> None:
                loop.call_soon_threadsafe(lambda: fut.done() or fut.set_result(bool(granted)))

            av.AVCaptureDevice.requestAccessForMediaType_completionHandler_(media, done)
            try:
                granted = await asyncio.wait_for(fut, 60.0)
            except TimeoutError:
                granted = False
            if not granted:
                raise CameraUnavailable("Camera access wasn't allowed for NEO.")
        elif status != 3:  # restricted / denied
            raise CameraUnavailable(
                "Camera access is off for NEO: turn it on in System Settings > Privacy & Security > Camera."
            )

    def start(self, push: Callable[[Any], None]) -> None:
        av, _cm, quartz, objc = _load_av()
        effects.guard(effects.CAMERA, "turn on the camera", via=av.AVCaptureSession)  # before any camera call
        from PIL import Image

        device = av.AVCaptureDevice.defaultDeviceWithMediaType_(av.AVMediaTypeVideo)
        if device is None:
            raise CameraUnavailable("No camera was found on this Mac.")
        dev_in, err = av.AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
        if dev_in is None:
            raise CameraUnavailable(
                f"The camera couldn't be opened ({err.localizedDescription() if err else 'busy?'})."
            )
        session = av.AVCaptureSession.alloc().init()
        if session.canSetSessionPreset_(av.AVCaptureSessionPreset1280x720):
            session.setSessionPreset_(av.AVCaptureSessionPreset1280x720)
        if not session.canAddInput_(dev_in):
            raise CameraUnavailable("The camera is in use by another app.")
        session.addInput_(dev_in)
        out = av.AVCaptureVideoDataOutput.alloc().init()
        out.setAlwaysDiscardsLateVideoFrames_(True)
        out.setVideoSettings_({quartz.kCVPixelBufferPixelFormatTypeKey: quartz.kCVPixelFormatType_32BGRA})

        def on_frame(buf: Any) -> None:
            pb = _cm.CMSampleBufferGetImageBuffer(buf)
            if pb is None:
                return
            quartz.CVPixelBufferLockBaseAddress(pb, 1)  # read-only
            try:
                w, h = quartz.CVPixelBufferGetWidth(pb), quartz.CVPixelBufferGetHeight(pb)
                row = quartz.CVPixelBufferGetBytesPerRow(pb)
                base = quartz.CVPixelBufferGetBaseAddress(pb)
                raw = bytes(base.as_buffer(row * h))
            finally:
                quartz.CVPixelBufferUnlockBaseAddress(pb, 1)
            push(Image.frombytes("RGB", (w, h), raw, "raw", "BGRX", row))

        delegate = _delegate_class(objc).alloc().init()
        delegate._neo_on_frame = on_frame
        self._queue = _dispatch_queue(objc, b"neo.camera.frames")
        out.setSampleBufferDelegate_queue_(delegate, self._queue)
        if not session.canAddOutput_(out):
            raise CameraUnavailable("The camera's video couldn't be read.")
        session.addOutput_(out)
        with self._mu:
            if self._stopped:  # stopped while it was being set up: never turned on
                delegate._neo_on_frame = None
                return
            self._session, self._delegate = session, delegate  # before it runs: stop() can reach it
        session.startRunning()
        with self._mu:
            stopped = self._stopped
        if stopped:  # stopped while it was starting: off at once (this is start's own thread)
            with contextlib.suppress(Exception):
                session.stopRunning()

    def stop(self) -> None:
        with self._mu:
            self._stopped = True
            session, self._session = self._session, None
            delegate, self._delegate = self._delegate, None
        if delegate is not None:
            delegate._neo_on_frame = None
        if session is not None:

            def off() -> None:
                with contextlib.suppress(Exception):
                    session.stopRunning()  # the camera light goes out

            # stopRunning blocks until the camera is off: never on the event loop's thread
            threading.Thread(target=off, name="neo-camera-off", daemon=True).start()


def _stop_quietly(backend: Backend, started: asyncio.Future | None = None) -> None:
    """The camera off after a start that was cancelled, failed or stopped meanwhile."""
    if started is not None and not started.cancelled():
        started.exception()  # (retrieved: a failed start has nothing on to turn off)
    with contextlib.suppress(Exception):
        backend.stop()


def encode_jpeg(img: Any, quality: int = JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


Frame = tuple[bytes, int, int, float]  # (jpeg, width, height, time.time())


class CameraSession:
    """One use of the camera: ``await start()`` (the light comes on), ``latest()`` for a still,
    ``async for jpeg, w, h, ts in frames()`` for the preview, ``stop()`` (idempotent)."""

    def __init__(
        self, *, backend: Backend | None = None, fps: float = PREVIEW_FPS, quality: int = JPEG_QUALITY
    ) -> None:
        self._backend: Backend | None = backend
        self.fps, self.quality = fps, quality
        self._latest: Any = None
        self._latest_ts = 0.0
        self._mu = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._first: asyncio.Event | None = None
        self._q: asyncio.Queue[Frame | None] | None = None  # the preview's latest-frame slot
        self._last_jpeg = 0.0
        self.running = False
        self.started_at = 0.0

    async def start(self) -> None:
        if self.running:
            return
        self._loop = asyncio.get_running_loop()
        self._first = asyncio.Event()
        if self._backend is None:
            _load_av()  # a clear "install" message before anything else
            await AVBackend.authorize()
            self._backend = AVBackend()
        backend = self._backend
        self.running = True
        self.started_at = time.monotonic()
        # The backend starts on a thread nothing can cancel: a cancel meanwhile (the user's stop, a
        # timeout) waits for it in the background and turns the camera off once it has started.
        started = asyncio.ensure_future(asyncio.to_thread(backend.start, self._push))
        try:
            await asyncio.shield(started)
        except BaseException:
            self.running = False
            if started.done():
                _stop_quietly(backend, started)
            else:
                started.add_done_callback(lambda f: _stop_quietly(backend, f))
            raise
        if not self.running:  # stop() came while it started (a backend may miss an early stop)
            _stop_quietly(backend, started)

    def _push(self, img: Any) -> None:
        """A frame from the backend (any thread)."""
        if not self.running:
            return
        now = time.time()
        with self._mu:
            self._latest, self._latest_ts = img, now
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        frame: Frame | None = None
        if self._q is not None and now - self._last_jpeg >= 1.0 / max(self.fps, 0.1):
            self._last_jpeg = now
            with contextlib.suppress(Exception):
                frame = (encode_jpeg(img, self.quality), img.width, img.height, now)
        with contextlib.suppress(RuntimeError):  # the loop is shutting down
            loop.call_soon_threadsafe(self._arrived, frame)

    def _arrived(self, frame: Frame | None) -> None:
        if self._first is not None:
            self._first.set()
        q = self._q
        if frame is not None and q is not None:
            with contextlib.suppress(asyncio.QueueEmpty):
                q.get_nowait()  # never queues pixels: the newest replaces an unsent one
            q.put_nowait(frame)

    async def wait_frame(self, timeout: float = 3.0) -> bool:
        """Wait for the first frame (True when one came)."""
        if self._latest is not None:
            return True
        if self._first is None:
            return False
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._first.wait(), timeout)
        return self._latest is not None

    def latest(self) -> Any:
        """The newest frame as a PIL image (None before the first)."""
        with self._mu:
            return self._latest

    async def frames(self) -> AsyncIterator[Frame]:
        """The preview: the newest frame as a JPEG, at most ``fps`` a second, until stop()."""
        self._q = asyncio.Queue(maxsize=1)
        try:
            while self.running:
                frame = await self._q.get()
                if frame is None:
                    break
                yield frame
        finally:
            self._q = None

    def stop(self) -> None:
        if not self.running and self._backend is None:
            return
        self.running = False
        backend = self._backend
        if backend is not None:
            with contextlib.suppress(Exception):
                backend.stop()
        q = self._q
        if q is not None:
            with contextlib.suppress(asyncio.QueueEmpty):
                q.get_nowait()
            q.put_nowait(None)
        with self._mu:
            self._latest = None  # nothing seen stays in memory after the camera is off
