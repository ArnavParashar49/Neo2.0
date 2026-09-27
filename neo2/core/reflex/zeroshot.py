"""Zero-shot Laya on its own checkpoint: loaded when neo/decide asks, let go when idle.

neo/decide asks Laya choices no head was trained for ("which of these apps should do the
request?", and "what kind of task is this?" while the reflex has no kind head). The reflex's
encoder is picked for its trained heads, and the multilingual one it moves to is the weakest of
the pinned repo's three at zero-shot, so these run on ``typed-decisions``. Measured on decide's
own questions (M3 Pro, MPS), typed-decisions / English root / multilingual:

* app choice among 2-6 fixture apps described by engine._describe (n=128): right 0.85 / 0.88 /
  0.81, sure and wrong 0.05 / 0.10 / 0.16; combined with the prior, typed-decisions never
  overturned a right prior (the English root did 5 times);
* task kind (390 mapped real requests): right 0.57 / 0.48 / 0.41, calibration error 0.09 / 0.38 /
  0.42.

Sharing the GPU with the reflex:

* NEO's start makes sure the checkpoint is on this Mac (:func:`prefetch`: a download on a fresh
  install, after the reflex's own; a few ms from the cache) without loading it. While it
  downloads, questions get None at once rather than waiting for it.
* Loaded on the first question, in a worker thread; a caller waits for it at most ``wait_s``,
  then gets None (its own fallback decides). The weights are read once with no lock held, so the
  load itself holds the shared Laya lock ~0.2-0.3 s.
* Every forward pass holds :func:`neo.reflex.laya_lock` (PyTorch on MPS is not thread-safe).
* :data:`IDLE_S` after the last question the model is dropped, giving its ~1.7 GB back. The first
  question after that pays the reload (see :data:`IDLE_S`).
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
from typing import Any

from neo import effects
from neo.config import settings

SUBFOLDER = "typed-decisions"
# This checkpoint's files only (the repo root and its siblings are other checkpoints).
FILES = [f"{SUBFOLDER}/{f}" for f in ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")]
# Unloaded this long after the last question. On an M3 Pro whose GPU NEO itself kept 20-86% busy,
# a question took 33-35 ms (p50) while loaded, 0.40-0.48 s when it found the model unloaded (read
# 0.10 s, load under the Laya lock, a warm-up pass, its own pass) and 0.57-0.70 s as the first of
# a process with the reflex loaded: requests minutes apart share one load, and the 1.7 GB comes
# back in between.
IDLE_S = 600.0
RETRY_S = 600.0  # a failed load or download is not retried sooner (each try reads ~0.8 GB)
BUSY_RETRY_S = 30.0  # an unload that finds Laya busy tries again after this
_WARM = {"warm": {"type": "noul", "instructions": "Is this a warm-up?"}}

_mu = threading.Lock()  # guards the fields below; never held while waiting for the Laya lock
_agent: Any = None
_loading: threading.Event | None = None  # set when the load in flight has finished
_fetching: threading.Thread | None = None  # a download in flight
_failed_at = float("-inf")
_used = 0.0
_timer: threading.Timer | None = None


def checkpoint_dir(download: bool = True) -> str:
    """The checkpoint at NEO's pinned Laya revision (downloaded once; afterwards from the cache).
    ``download=False`` only looks in the cache, whose folder may lack this checkpoint's files."""
    from huggingface_hub import snapshot_download

    from neo import models

    snap = snapshot_download(
        models.LAYA, revision=models.LAYA_REVISION, allow_patterns=FILES, local_files_only=not download
    )
    return os.path.join(snap, SUBFOLDER)


def _local() -> str | None:
    """The checkpoint's folder when all of it is on this Mac, else None (never downloads)."""
    try:
        path = checkpoint_dir(download=False)
    except OSError:  # not in the cache, or only partly (the hub checks its cached file list)
        return None
    # the revision's folder also holds the reflex's checkpoint: without a cached file list the hub
    # returns it whether or not this one is there
    ok = all(os.path.isfile(os.path.join(path, f)) for f in ("rl_agent_config.json", "model.safetensors"))
    return path if ok else None


def _find() -> str | None:
    """:func:`_local`, its weights read once so the load under the lock comes from memory."""
    path = _local()
    if path is not None:
        with open(os.path.join(path, "model.safetensors"), "rb") as f:
            while f.read(16 << 20):
                pass
    return path


def _open(path: str) -> Any:
    """Load the checkpoint onto NEO's Laya device (the caller holds the Laya lock)."""
    import laya

    effects.guard("model", "load Laya's zero-shot checkpoint into memory (1.7 GB)", via=laya.load)
    return laya.load(path, device=settings().laya_device)


def _close(agent: Any) -> None:
    """Free the model now (Laya's own context-manager exit) and hand MPS's cache back."""
    with contextlib.suppress(Exception):
        agent.__exit__(None, None, None)
    torch = sys.modules.get("torch")  # not imported: nothing was ever on the GPU
    if torch is not None:
        with contextlib.suppress(Exception):
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()


def _lock() -> threading.Lock:
    from neo.reflex import laya_lock

    return laya_lock()


def prefetch() -> None:
    """Download the checkpoint in the background if this Mac doesn't have it yet, without loading it."""
    global _fetching
    with _mu:
        if settings().reflex != "laya" or _fetching is not None or time.monotonic() - _failed_at < RETRY_S:
            return
        _fetching = threading.Thread(target=_download, name="neo-zeroshot-fetch", daemon=True)
        _fetching.start()


def _download() -> None:
    global _fetching, _failed_at
    try:
        if _local() is None:
            with _lock():  # the reflex loads (and on a fresh Mac downloads) its own checkpoint first
                pass
            checkpoint_dir()
    except Exception as e:  # noqa: BLE001 - offline, disk full: the prior decides meanwhile
        print(f"[decide] couldn't fetch zero-shot Laya ({str(e)[:80]})")
        with _mu:
            _failed_at = time.monotonic()
    finally:
        with _mu:
            _fetching = None


def available() -> bool:
    """Whether a question can be answered now or after a load: not while Laya is off, the
    checkpoint is still downloading, or it failed lately."""
    with _mu:
        if _agent is not None:
            return True
        return settings().reflex == "laya" and _fetching is None and time.monotonic() - _failed_at >= RETRY_S


def loaded() -> bool:
    """Whether the model is in memory now: a question costs no load."""
    with _mu:
        return _agent is not None


def ask(state: Any, questions: dict, wait_s: float, stale: threading.Event | None = None) -> dict | None:
    """One zero-shot pass (call it from a worker thread). None when Laya is off, failed to load or
    is still downloading, still loading or busy after ``wait_s``, or the caller gave up (``stale``)
    meanwhile."""
    global _used
    agent = _ready(wait_s)
    if agent is None:
        return None
    lock = _lock()
    if not lock.acquire(timeout=wait_s):
        return None  # the reflex, a retrain or another question has it
    try:
        with _mu:
            if agent is not _agent:
                return None  # let go while this waited
            _used = time.monotonic()
        if stale is not None and stale.is_set():
            return None  # nobody is waiting for this answer: don't hold the GPU for it
        return agent.system_one(state, questions)
    finally:
        lock.release()


def _ready(wait_s: float) -> Any:
    """The loaded model; otherwise start loading it and wait up to ``wait_s``."""
    global _loading
    with _mu:
        if _agent is not None:
            return _agent
        done = _loading
        if done is None:
            if settings().reflex != "laya" or _fetching is not None or time.monotonic() - _failed_at < RETRY_S:
                return None
            done = _loading = threading.Event()
            threading.Thread(target=_load, name="neo-zeroshot-load", daemon=True).start()
    done.wait(wait_s)
    with _mu:
        return _agent


def _load() -> None:
    global _agent, _loading, _failed_at, _used
    agent = None
    failed = True  # until the load is through: one refused (effects off: a BaseException) failed too
    try:
        path = _find()
        if path is None:
            prefetch()  # not on this Mac yet: its download answers no question, so nobody waits for it
        else:
            with _lock():
                agent = _open(path)
                # a new model's first pass is slow (Metal kernels): take it now, not in a question
                agent.system_one({"text": "warm up"}, _WARM)
        failed = False
    except Exception as e:  # noqa: BLE001 - no torch, no model, out of memory: the prior decides
        print(f"[decide] zero-shot Laya unavailable ({str(e)[:80]})")
        agent = None
    finally:
        with _mu:
            done, _loading = _loading, None
            if failed:
                _failed_at = time.monotonic()
            elif agent is not None:
                _agent, _used = agent, time.monotonic()
                _schedule(IDLE_S)
        if done is not None:
            done.set()


def _schedule(delay: float) -> None:
    """Check for idleness after ``delay`` seconds (the caller holds ``_mu``)."""
    global _timer
    _timer = threading.Timer(delay, _unload_if_idle)
    _timer.daemon = True
    _timer.start()


def _unload_if_idle() -> None:
    global _agent
    with _mu:
        if _agent is None:
            return
        idle = time.monotonic() - _used
        if idle < IDLE_S:
            _schedule(IDLE_S - idle)
            return
    lock = _lock()
    if not lock.acquire(blocking=False):  # Laya is working: never free a model mid-pass
        with _mu:
            _schedule(BUSY_RETRY_S)
        return
    try:
        with _mu:
            idle = time.monotonic() - _used
            if _agent is None or idle < IDLE_S:  # a question came in meanwhile
                if _agent is not None:
                    _schedule(IDLE_S - idle)
                return
            agent, _agent = _agent, None
        _close(agent)
        print("[decide] zero-shot Laya unloaded (idle)")
    finally:
        lock.release()


__all__ = ["SUBFOLDER", "IDLE_S", "ask", "available", "checkpoint_dir", "prefetch"]
