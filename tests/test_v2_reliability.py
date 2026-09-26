"""Reliability (production audit, batch C): audio device changes, safe model downloads, timers
that survive sleep and restarts, a thread-safe memory store, bounded history, learning that keeps
learning, and a voice layer that recovers."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import threading
import time
import zipfile

import pytest

from neo.events import EventBus


@pytest.fixture(autouse=True)
def _bus(monkeypatch):
    from neo import events

    monkeypatch.setattr(events, "_bus", EventBus())


# ---- audio ------------------------------------------------------------------------------------


class _FakeStream:
    opened: list = []

    def __init__(self, *a, callback=None, **kw):
        self.cb, self.started, self.closed = callback, False, False
        _FakeStream.opened.append(self)

    def start(self):
        self.started = True

    def abort(self):
        self.started = False

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True

    def write(self, chunk):
        pass


def test_audio_reopens_on_the_current_devices(monkeypatch):
    from neo.voice import audio

    _FakeStream.opened = []
    calls = []
    monkeypatch.setattr(audio.sd, "RawInputStream", _FakeStream)
    monkeypatch.setattr(audio.sd, "OutputStream", _FakeStream)
    monkeypatch.setattr(audio.sd, "_terminate", lambda: calls.append("terminate"))
    monkeypatch.setattr(audio.sd, "_initialize", lambda: calls.append("initialize"))
    monkeypatch.setattr(audio, "_register", lambda o: audio._streams.add(o))
    mic, spk = audio.Mic(), audio.Speaker()
    mic.start()
    spk.start()
    try:
        first_in, first_out = _FakeStream.opened[0], _FakeStream.opened[1]
        audio.reopen_all("test")
        assert calls == ["terminate", "initialize"]
        assert first_in.closed and first_out.closed
        assert mic._stream is not first_in and mic._stream.started and spk._stream.started
        # a stream that stops delivering frames is noticed
        mic.last_frame = time.monotonic() - 10
        assert mic.stalled()
        mic._stream.cb(b"\x00\x00" * 320, 320, None, None)
        assert not mic.stalled() and mic.q.qsize() == 1
    finally:
        mic.stop()
        spk.close()
        audio._streams.discard(mic)
        audio._streams.discard(spk)


# ---- the wake-word model download -------------------------------------------------------------


def _model_zip(complete: bool = True) -> bytes:
    from neo.voice import wake

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"{wake._MODEL}/am/final.mdl", "m")
        if complete:
            z.writestr(f"{wake._MODEL}/conf/model.conf", "c")
    return buf.getvalue()


class _Resp:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def iter_bytes(self, n):
        yield self.data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_wake_model_download_is_checked_and_atomic(tmp_path, monkeypatch):
    import httpx

    from neo.config import settings
    from neo.voice import wake

    good = _model_zip()
    monkeypatch.setattr(wake, "_SHA256", hashlib.sha256(good).hexdigest())
    monkeypatch.setattr(httpx, "stream", lambda *a, **kw: _Resp(good))
    d = wake.ensure_model()
    assert (d / "am/final.mdl").is_file() and not list(settings().models_dir.glob(".download-*"))
    # a half-extracted folder from an interrupted first run is replaced, not trusted
    (d / "conf/model.conf").unlink()
    assert wake.ensure_model() == d and (d / "conf/model.conf").is_file()
    # a download that doesn't match the pinned hash is refused and leaves nothing behind
    import shutil

    shutil.rmtree(d)
    monkeypatch.setattr(httpx, "stream", lambda *a, **kw: _Resp(_model_zip(complete=False)))
    with pytest.raises(RuntimeError, match="checksum"):
        wake.ensure_model()
    assert not d.exists()


# ---- timers -----------------------------------------------------------------------------------


def test_timers_use_the_wall_clock_and_survive_a_restart(monkeypatch):
    from neo.agent.registry import ToolContext, registry
    from neo.tools import load_all, system

    load_all()
    notes = []

    async def fake_osa(script, timeout=10):
        notes.append(script)
        return ""

    monkeypatch.setattr(system, "osascript", fake_osa)

    async def run():
        out = await registry().get("timer").handler({"minutes": 0.05, "label": 'x" & (do shell script "id") & "'}, ToolContext())
        assert out.startswith("Timer set")
        saved = system._load_timers()
        assert len(saved) == 1 and saved[0]["due"] > time.time()
        # "restart": forget the running task, pretend the Mac slept past the deadline, restore
        for t in list(system._timer_tasks.values()):
            t.cancel()
        await asyncio.sleep(0)
        system._timer_tasks.clear()
        system._save_timers([{**saved[0], "due": time.time() - 120}])
        assert system.restore_timers() == 1
        for _ in range(50):
            if notes:
                break
            await asyncio.sleep(0.05)

    asyncio.run(run())
    assert notes and "late" in notes[0] and '\\"' in notes[0]  # the label is quoted, never code
    assert system._load_timers() == []


# ---- memory store -----------------------------------------------------------------------------


def test_memory_store_survives_concurrent_use(tmp_path):
    from neo.memory.store import Store

    st = Store(tmp_path / "m.sqlite3")
    errors = []

    def work(i):
        try:
            for j in range(40):
                st.remember(f"fact {i}-{j} about coffee", "note")
                st.recall("coffee fact", 4)
                st.save_playbook(f"goal {i} {j}", [{"tool": "x", "args": {}}], "ok")
                st.prompt_context("coffee")
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and st.count() == 240
    gone = st.forget_all()
    assert gone["memory"] == 240 and st.count() == 0 and st.recall("coffee") == []


# ---- history ----------------------------------------------------------------------------------


def test_history_is_bounded_and_drops_old_screenshots(monkeypatch):
    from neo.agent import session as sess_mod
    from neo.providers.base import ImagePart, Message, ToolResult

    summarised = []

    async def fake_summary(msgs, started, budget_s=8.0):
        summarised.append(len(msgs))
        return "s"

    monkeypatch.setattr(sess_mod.Session, "_summarize", staticmethod(fake_summary))
    s = sess_mod.Session()

    async def run():
        for i in range(120):
            await s._remember(
                Message.user(f"q{i}"),
                Message.tool([ToolResult("c", "screenshot", "shot", images=[ImagePart(b"x" * 1000, "image/png")])]),
            )
        await asyncio.sleep(0)

    asyncio.run(run())
    assert len(s.history) <= sess_mod._HISTORY_CAP
    assert s.history[0].role == "user"
    assert all(not r.images for m in s.history[:-1] for r in m.tool_results)  # only the newest keeps pixels
    assert summarised and sum(summarised) > 0  # what was let go got summarised first


# ---- learning ---------------------------------------------------------------------------------


def test_learning_counts_new_examples_by_time_and_backs_off_after_a_failure(monkeypatch):
    from neo.reflex import learn

    for i in range(40):
        learn.record(f"open the thing number {i} please", "quick_action", "open_app")
    assert learn.pending() == 40
    learn._write_state({"trained_at": time.time(), "trained_rows": 2000})  # the file is "full"
    time.sleep(0.01)
    learn.record("a brand new request", "chat")
    assert learn.pending() == 1  # still counting even though trained_rows is at the cap

    class Laya:
        _agent = object()

        def reload_head(self):
            pass

    import neo.reflex as reflex_mod
    from neo.reflex import finetune

    monkeypatch.setattr(reflex_mod, "get_laya", lambda: Laya())
    monkeypatch.setattr(finetune, "train", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("MPS out of memory")))
    assert learn.retrain() is None
    st = json.loads(learn._path(learn.STATE).read_text())
    assert "MPS" in st["last_error"] and not learn.due()  # waits the usual gap, no retry storm


def test_forget_also_clears_cached_features(monkeypatch):
    import numpy as np

    from neo.reflex import finetune, learn

    learn.record("remind me about the dentist", "quick_action", "reminder_add")
    finetune.save_cache(
        {"remind me about the dentist": np.ones(4, np.float32), "what time is it": np.ones(4, np.float32)}, {}
    )
    learn.forget_examples()
    emb, _ = finetune.load_cache()
    assert list(emb) == ["what time is it"]
    finetune.feature_cache().write_bytes(b"not a zip")  # a cut-off write from an old version
    assert finetune.load_cache() == ({}, {})


# ---- voice ------------------------------------------------------------------------------------


def test_voice_failures_are_classified():
    from neo.voice.live import _failure_kind
    from neo.voice.pipeline import live_refused

    assert _failure_kind(RuntimeError("429 RESOURCE_EXHAUSTED: quota")) == "quota"
    assert _failure_kind(TimeoutError()) == "network"
    assert _failure_kind(OSError("getaddrinfo failed")) == "network"
    assert live_refused(RuntimeError("404 models/x is not found for API version")) is True
    assert live_refused(TimeoutError()) is False and live_refused(RuntimeError("429 quota")) is False


def test_a_network_blip_at_startup_keeps_live_on(monkeypatch):
    from neo import config
    from neo.voice import live as live_mod
    from neo.voice import pipeline

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("NEO_VOICE", "auto")
    monkeypatch.setattr(config, "_settings", None)
    started = []

    async def no_local(*a):
        raise AssertionError("must not fall back to the on-device voice")

    monkeypatch.setattr(pipeline, "make_local", no_local)

    class V:
        def __init__(self, *a):
            pass

        async def probe(self):
            raise OSError("getaddrinfo failed")

        async def start(self):
            started.append("live")

    monkeypatch.setattr(live_mod, "LiveVoice", V)
    v = asyncio.run(pipeline.make_voice(lambda t: None, None))
    assert isinstance(v, V) and started == ["live"]
