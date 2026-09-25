"""Reflex facade: rules → Laya (local) → Flash-Lite (cloud) with confidence-based escalation.

The Laya model is a process-wide singleton (1.7 GB on the GPU — load it once) shared with the
memory embedder. All access is serialised behind one lock: PyTorch on MPS is not thread-safe,
and two concurrent forward passes (or a load racing a call) trip a Metal command-buffer assertion.
"""

from __future__ import annotations

import asyncio
import re
import threading

from neo.config import settings
from neo.reflex.schema import Decision, Intent

_STOP_RE = re.compile(r"^\s*(stop|cancel|never ?mind|shut ?up|be quiet|go to sleep|quit|exit)\b", re.I)

_lock = threading.Lock()
_laya = None
_laya_failed = False


def laya_lock() -> threading.Lock:
    return _lock


def get_laya():
    """The shared LayaReflex (loads on first call), or None if Laya is unavailable/disabled."""
    global _laya, _laya_failed
    with _lock:
        if _laya is None and not _laya_failed and settings().reflex == "laya":
            try:
                from neo.reflex.laya_reflex import LayaReflex

                _laya = LayaReflex()
            except Exception as e:  # noqa: BLE001 — missing torch/model → cloud fallback
                print(f"[reflex] Laya unavailable ({e}); falling back to Flash-Lite")
                _laya_failed = True
        return _laya


def _rules(text: str) -> Decision | None:
    if _STOP_RE.match(text) and len(text.split()) <= 4:
        return Decision("stop", 1.0, False, 0.0, False, 0.0, source="rules")
    return None


class Reflex:
    def __init__(self) -> None:
        self._lite = None

    def _laya_decide(self, text: str) -> Decision | None:
        laya = get_laya()
        if laya is None:
            return None
        with _lock:
            return laya.decide(text)

    def _get_lite(self):
        if self._lite is None and settings().gemini_api_key:
            from neo.reflex.lite import LiteReflex

            self._lite = LiteReflex()
        return self._lite

    async def decide(self, text: str) -> Decision:
        if d := _rules(text):
            return d
        s = settings()
        if s.reflex == "off":
            return Decision("agent_task", 0.0, False, 0.0, False, 0.0, source="rules")
        if s.reflex == "laya":
            d = await asyncio.to_thread(self._laya_decide, text)
            if d is not None and d.intent_confidence >= s.reflex_confidence_floor:
                return d
        lite = self._get_lite()
        if lite:
            try:
                return await lite.decide(text)
            except Exception as e:  # noqa: BLE001
                print(f"[reflex] lite failed: {e}")
        # Nothing available — be safe: treat as an agent task, confirm anything risky.
        return Decision("agent_task", 0.0, True, 0.5, False, 0.0, source="rules")


def warmup(reflex: Reflex | None = None) -> None:
    """Load Laya, then the memory embedder, in ONE background thread.

    Both lazily import `transformers`; two threads doing that at once trip its lazy-module
    loader ("cannot import name 'AutoTokenizer'"), which silently disabled the reflex."""

    def _load() -> None:
        get_laya()
        try:
            from neo.memory import embed

            embed._get()
        except Exception as e:  # noqa: BLE001
            print(f"[memory] embedder warmup failed: {str(e)[:80]}")

    threading.Thread(target=_load, daemon=True).start()


__all__ = ["Reflex", "Decision", "Intent", "warmup", "get_laya", "laya_lock"]
