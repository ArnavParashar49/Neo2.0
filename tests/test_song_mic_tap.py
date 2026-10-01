"""Naming a song never opens the microphone a second time: while the voice's Mic runs, the song's
``mic`` source taps its frames (a second PortAudio input stream on the same device fails on macOS and
knocks the first one out); and the Mac's own sound arrives as PyObjC hands it over (a plain tuple for
the stream description)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from neo2.core.tools.music import capture

sd = pytest.importorskip("sounddevice")
audio = pytest.importorskip("neo2.core.voice.audio")


@pytest.fixture
def no_streams(monkeypatch):
    """No real device: Mic's stream is a stand-in whose frames a test pushes itself."""

    class Stream:
        def __init__(self, **kw):
            self.cb = kw["callback"]

        def start(self):
            pass

        def abort(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(sd, "RawInputStream", Stream)
    monkeypatch.setattr(audio, "_taps", ())
    yield Stream


def test_the_song_taps_the_running_voice_mic_instead_of_opening_the_device_again(no_streams):
    with pytest.MonkeyPatch.context() as m:
        from neo2.core import effects

        m.setattr(effects, "guard", lambda *a, **k: None)
        mic = audio.Mic()
        mic.start()
        assert audio.mic_running()
        got: list[np.ndarray] = []
        src = capture.MicSource()
        src.start(got.append)
        assert src._stream is None and src._untap is not None  # no second stream
        frame = (np.arange(320, dtype=np.int16) * 7).tobytes()
        mic._stream.cb(frame, 320, None, None)
        assert mic.q.qsize() == 1 and len(got) == 1 and got[0].tolist() == list(np.frombuffer(frame, np.int16))
        src.stop()
        mic._stream.cb(frame, 320, None, None)
        assert len(got) == 1  # untapped: nothing more
        assert mic.q.qsize() == 2  # the voice's queue never missed a frame
        mic.stop()


def test_without_a_voice_mic_the_song_opens_its_own_stream(no_streams):
    with pytest.MonkeyPatch.context() as m:
        from neo2.core import effects

        m.setattr(effects, "guard", lambda *a, **k: None)
        assert not audio.mic_running()
        src = capture.MicSource()
        src.start(lambda pcm: None)
        assert src._stream is not None and src._untap is None
        src.stop()


def test_a_tap_s_trouble_never_breaks_the_voice(no_streams):
    with pytest.MonkeyPatch.context() as m:
        from neo2.core import effects

        m.setattr(effects, "guard", lambda *a, **k: None)
        mic = audio.Mic()
        mic.start()

        def bad(frame: bytes) -> None:
            raise RuntimeError("boom")

        untap = audio.tap(bad)
        mic._stream.cb(b"\x00" * 640, 320, None, None)
        assert mic.q.qsize() == 1
        untap()
        mic.stop()


class _Block:
    def __init__(self, data: bytes) -> None:
        self.data = data


def _cm(asbd, frames: int, data: bytes):
    """CoreMedia, faked: the format description is whatever ``asbd`` is."""
    return SimpleNamespace(
        CMSampleBufferGetFormatDescription=lambda buf: "fd",
        CMAudioFormatDescriptionGetStreamBasicDescription=lambda fd: asbd,
        CMSampleBufferGetDataBuffer=lambda buf: _Block(data),
        CMSampleBufferGetNumSamples=lambda buf: frames,
        CMBlockBufferGetDataLength=lambda block: len(block.data),
        CMBlockBufferCopyDataBytes=lambda block, off, size, _: (0, block.data),
    )


def test_the_mac_s_sound_is_decoded_from_pyobjc_s_tuple_description():
    tone = np.sin(2 * math.pi * 440 * np.arange(320) / 16000).astype(np.float32) * 0.5
    data = tone.tobytes()
    # (mSampleRate, mFormatID, mFormatFlags, mBytesPerPacket, mFramesPerPacket, mBytesPerFrame,
    #  mChannelsPerFrame, mBitsPerChannel, mReserved): float, packed, non-interleaved, mono, 32-bit
    as_tuple = (16000.0, 1819304813, 41, 4, 1, 4, 1, 32, 0)
    pcm = capture._pcm_of(_cm(as_tuple, 320, data), "buf")
    assert pcm is not None and pcm.dtype == np.int16 and pcm.size == 320
    assert abs(float(pcm.max()) / 32767 - 0.5) < 0.01
    named = SimpleNamespace(mSampleRate=16000.0, mFormatFlags=41, mChannelsPerFrame=1, mBitsPerChannel=32)
    assert capture._pcm_of(_cm(named, 320, data), "buf").tolist() == pcm.tolist()
