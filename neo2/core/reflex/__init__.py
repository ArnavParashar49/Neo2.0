"""Reflex facade: rules → Laya (local) → Flash-Lite (cloud) with confidence-based escalation.

The Laya model is a process-wide singleton (1.7 GB on the GPU — load it once) shared with the
memory embedder. All access is serialised behind one lock: PyTorch on MPS is not thread-safe,
and two concurrent forward passes (or a load racing a call) trip a Metal command-buffer assertion.
"""

from __future__ import annotations

import asyncio
import re
import threading

from neo2.core import logger
from neo2.core.config import settings
from neo2.core.reflex.schema import Decision, Intent, wants_reply
from neo2.core.selftest.phrases import is_request as asks_for_selftest

log = logger(__name__)

# Only the bare words cancel everything; "stop the timer" / "quit the app" are commands. Hindi and
# Hinglish too ("bas karo", "ruk jao", "चुप"): every trained head gets those wrong, the rule can't.
_STOP_RE = re.compile(
    r"^\s*(?:(?:hey\s+)?neo[\s,]+)?(?:okay\s+|ok\s+|arre\s+|arey\s+)?(?:stop|cancel|never\s*mind|shut\s*up|be\s+quiet|"
    r"go\s+to\s+sleep|quit|exit|stop\s+talking|that'?s\s+enough|forget\s+it|cancel\s+that|stop\s+it|"
    r"ruko|ruk\s+ja(?:o)?|bas(?:\s+kar(?:o|\s+do)?)?|chup(?:\s+(?:ho\s+ja(?:o)?|raho|kar(?:o)?))?|"
    r"reh?ne\s+do|rahne\s+do|chhod\s+do|chhodo|ja+ne\s+do|so\s+ja(?:o)?|"
    r"रुको|रुक\s+जाओ|बस(?:\s+करो)?|चुप(?:\s+(?:हो\s+जाओ|रहो|करो))?|रहने\s+दो|छोड़ो|छोड़\s+दो|जाने\s+दो)"
    r"(?:[\s,]+(?:it|that|now|please|neo|talking|already|na|yaar|abhi|ab|bhai|प्लीज़|अभी))*\s*[.!।]*\s*$",
    re.I,
)


def is_bare_stop(text: str) -> bool:
    return bool(_STOP_RE.match(text))

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
                from neo2.core.reflex.laya_reflex import LayaReflex

                _laya = LayaReflex()
            except Exception as e:  # noqa: BLE001 — missing torch/model → cloud fallback
                log.warning(f"[reflex] Laya unavailable ({e}); falling back to Flash-Lite")
                _laya_failed = True
        return _laya


def _rules(text: str) -> Decision | None:
    if is_bare_stop(text):
        return Decision("stop", 1.0, False, 0.0, False, 0.0, source="rules")
    if asks_for_selftest(text):
        # "check your computer controls": NEO's own self-test (neo2.core.selftest), run by its fast path.
        # Never conversation — a head that called it chat would answer instead of testing.
        return Decision("quick_action", 1.0, False, 0.0, False, 0.0, source="rules")
    return None


class Reflex:
    def __init__(self) -> None:
        self._lite = None

    def _laya_decide(self, text: str) -> Decision | None:
        laya = get_laya()
        if laya is None:
            return None
        with _lock:
            d = laya.decide(text)
            emb, center = getattr(laya, "last_embedding", None), getattr(laya, "center", None)
            kind = getattr(laya, "last_kind", None)
        if emb is None or center is None:
            return d
        from neo2.core.reflex import memory, uimemory

        try:  # (kept a moment, in RAM: a UI flow that settles as verified is keyed by it)
            uimemory.memory().seen(text, emb)
        except Exception:  # noqa: BLE001 - learning never breaks a decision
            pass
        # A request the user corrected a moment ago gets the fix now, not after the next retrain.
        return memory.recall(d, text, emb, center, kind)

    def _get_lite(self):
        if self._lite is None and settings().gemini_api_key:
            from neo2.core.reflex.lite import LiteReflex

            self._lite = LiteReflex()
        return self._lite

    async def decide(self, text: str) -> Decision:
        return _with_reply(await self._decide(text), text)

    async def _decide(self, text: str) -> Decision:
        if d := _rules(text):
            return d
        s = settings()
        if s.reflex == "off":
            return Decision("agent_task", 0.0, False, 0.0, False, 0.0, source="rules")
        if s.reflex == "laya":
            d = await asyncio.to_thread(self._laya_decide, text)
            # A v2 head's pick stands even when unsure — it beats a second-long cloud round trip
            # (and what runs teaches it). Only the old head hands low-confidence calls to lite.
            if d is not None and (d.source == "laya+unsure" or d.intent_confidence >= s.reflex_confidence_floor):
                return d
        lite = self._get_lite()
        if lite:
            try:
                return await lite.decide(text)
            except Exception as e:  # noqa: BLE001
                log.warning(f"[reflex] lite failed: {e}")
        # Nothing available — be safe: treat as an agent task, confirm anything risky.
        return Decision("agent_task", 0.0, True, 0.5, False, 0.0, source="rules")


def _with_reply(d: Decision, text: str) -> Decision:
    """Conversation always gets an answer; commands only when the head/regex says they ask for one."""
    if d.intent in ("chat", "stop"):
        d.reply, d.reply_p = True, 1.0
    elif d.source not in ("laya+head", "laya+unsure", "laya+memory") or d.reply_p == 1.0:  # no reply head: fall back
        d.reply = wants_reply(text)
        d.reply_p = 0.5
    return d


def warmup(reflex: Reflex | None = None) -> None:
    """Load Laya, then the memory embedder, in ONE background thread.

    Both lazily import `transformers`; two threads doing that at once trip its lazy-module
    loader ("cannot import name 'AutoTokenizer'"), which silently disabled the reflex."""

    def _load() -> None:
        get_laya()
        try:
            from neo2.core.memory import embed

            embed._get()
        except Exception as e:  # noqa: BLE001
            log.warning(f"[memory] embedder warmup failed: {str(e)[:80]}")

    threading.Thread(target=_load, daemon=True).start()


__all__ = ["Reflex", "Decision", "Intent", "warmup", "get_laya", "laya_lock", "is_bare_stop"]
