"""The one lock for PyTorch on the Mac's GPU (MPS), for the whole process.

PyTorch drives MPS through one command stream per process, and that stream is not thread-safe:
two threads putting work on it at once abort NEO on the spot — a Metal assertion no ``except``
catches (``failed assertion _status < MTLCommandBufferStatusCommitted … setCurrentCommandEncoder``,
seen the moment the camera's detector ran beside Laya). So every torch-on-MPS user holds
:func:`lock` around its GPU work: a forward pass, a training step, weights moved with
``.to(device)``, ``torch.mps.empty_cache()``. Only that: tokenising, numpy, reading files stay
outside, so a holder keeps it for about one pass (the reflex's ~15-30 ms) and voice doesn't wait.

Today that is Laya and nothing else — the reflex, its zero-shot checkpoint and its learning, whose
``neo2.core.reflex.laya_lock()`` is this very lock. Core ML, MLX and Apple Vision have their own
Metal queues and torch on the CPU has no stream: they need no lock. A new model on MPS takes it
too, or runs on the CPU (the camera's D-FINE does: ~150 ms a frame, measured, and it never waits
on Laya).
"""

from __future__ import annotations

import threading

_lock = threading.Lock()  # not reentrant: a holder never calls back into another holder


def lock() -> threading.Lock:
    """The process's lock for torch work on MPS (see the module doc). Hold it for GPU work only."""
    return _lock


__all__ = ["lock"]
