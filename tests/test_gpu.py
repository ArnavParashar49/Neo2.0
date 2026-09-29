"""The one GPU lock (neo2/core/gpu.py): PyTorch's MPS stream is the process's one and not
thread-safe, so two passes in it at once abort NEO. Every torch-on-MPS user holds the same lock.

Fakes only: a stand-in "stream" that counts two passes inside it at once (the crash), a fake Laya
reflex and a fake zero-shot checkpoint. Nothing loads a model or touches the GPU."""

from __future__ import annotations

import threading
import time

import pytest

import neo2.core.reflex as reflex
from neo2.core import gpu
from neo2.core.reflex import zeroshot
from neo2.core.reflex.schema import Decision


class Stream:
    """MPS's command stream, as far as a test can tell: a pass inside while another one is is the
    Metal assertion that killed NEO."""

    def __init__(self) -> None:
        self._mu = threading.Lock()
        self.inside = self.overlaps = self.passes = 0

    def run(self) -> None:
        with self._mu:
            self.inside += 1
            self.overlaps += self.inside > 1
        time.sleep(0.002)  # long enough for another thread to walk in, were it let
        with self._mu:
            self.inside -= 1
            self.passes += 1


class FakeLaya:
    """The reflex's LayaReflex: one encoder pass per decision (no head: no embedding kept)."""

    last_embedding = center = last_kind = None

    def __init__(self, stream: Stream) -> None:
        self.stream = stream

    def decide(self, text: str) -> Decision:
        self.stream.run()
        return Decision("chat", 0.9, False, 0.0, False, 0.0, source="laya+head")


class FakeZeroShot:
    def __init__(self, stream: Stream) -> None:
        self.stream = stream

    def system_one(self, state, questions):
        self.stream.run()
        return {"answers": {q: {"choice": "A"} for q in questions}}


@pytest.fixture
def stream(monkeypatch):
    s = Stream()
    monkeypatch.setattr(reflex, "_laya", FakeLaya(s))
    monkeypatch.setattr(zeroshot, "_agent", FakeZeroShot(s))
    monkeypatch.setattr(zeroshot, "_used", 0.0)
    return s


def test_every_torch_gpu_user_holds_one_and_the_same_lock():
    assert reflex.laya_lock() is gpu.lock()  # the reflex, its learning, its memories
    assert zeroshot._lock() is gpu.lock()  # the zero-shot checkpoint's loads and passes
    assert not gpu.lock().locked()


def test_passes_from_every_thread_take_turns_on_the_gpu(stream):
    """The reflex deciding, a zero-shot question and a background re-embedding, all at once and
    over and over: never two passes in the stream together."""
    laya, n = reflex._laya, 25
    go = threading.Barrier(3)

    def decisions():
        go.wait()
        for _ in range(n):
            assert reflex.Reflex()._laya_decide("what am I holding").intent == "chat"

    def questions():
        go.wait()
        for _ in range(n):
            assert zeroshot.ask({"text": "open my notes"}, {"app": {}}, wait_s=5.0) is not None

    def learning():  # neo2.core.reflex.memory / uimemory's rebuild: a decide under the lock
        go.wait()
        for _ in range(n):
            with reflex.laya_lock():
                laya.decide("play some music")

    threads = [threading.Thread(target=f) for f in (decisions, questions, learning)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert stream.passes == 3 * n
    assert stream.overlaps == 0
    assert not gpu.lock().locked()
