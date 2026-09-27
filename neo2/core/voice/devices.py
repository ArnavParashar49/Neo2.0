"""Which audio devices macOS is using right now, and whether NEO may use the microphone.

PortAudio (under sounddevice) reads the device list once and never notices when AirPods connect
or the default input changes, so NEO asks CoreAudio directly — a cheap property read, fine to
poll every second.
"""

from __future__ import annotations

import ctypes
import struct
import sys

_ca = None


class _Addr(ctypes.Structure):
    _fields_ = [("mSelector", ctypes.c_uint32), ("mScope", ctypes.c_uint32), ("mElement", ctypes.c_uint32)]


def _fourcc(s: str) -> int:
    return struct.unpack(">I", s.encode())[0]


def _default(selector: str) -> int | None:
    global _ca
    if sys.platform != "darwin":
        return None
    try:
        if _ca is None:
            _ca = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
        addr = _Addr(_fourcc(selector), _fourcc("glob"), 0)
        dev, size = ctypes.c_uint32(0), ctypes.c_uint32(4)
        err = _ca.AudioObjectGetPropertyData(
            ctypes.c_uint32(1), ctypes.byref(addr), ctypes.c_uint32(0), None, ctypes.byref(size), ctypes.byref(dev)
        )
        return int(dev.value) if err == 0 else None
    except Exception:  # noqa: BLE001
        return None


def default_input() -> int | None:
    return _default("dIn ")


def default_output() -> int | None:
    return _default("dOut")


# AVAuthorizationStatus
NOT_DETERMINED, RESTRICTED, DENIED, AUTHORIZED = 0, 1, 2, 3


def mic_status() -> int:
    """macOS's answer to "may NEO use the microphone?" (AUTHORIZED when it can't tell)."""
    if sys.platform != "darwin":
        return AUTHORIZED
    try:
        import objc

        av: dict = {}
        objc.loadBundle("AVFoundation", av, bundle_path="/System/Library/Frameworks/AVFoundation.framework")
        return int(av["AVCaptureDevice"].authorizationStatusForMediaType_("soun"))  # AVMediaTypeAudio
    except Exception:  # noqa: BLE001
        return AUTHORIZED
