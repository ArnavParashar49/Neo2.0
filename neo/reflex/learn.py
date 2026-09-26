"""Laya learns from use.

Every request the user makes ends up handled somehow, and *how* is a label for what they said:

* Gemini Live answered without a tool → chat; it called one direct tool (open_app, weather…)
  → quick_action with that tool; it called agent_task → agent_task.
* NEO's own agent was sent a request and finished it without any tool → it was really chat;
  with a single fast tool → it was a quick action.

Those rows go to ``~/.neo/reflex_extra.jsonl`` (the file ``dataset.build`` already reads — on
this Mac only). When enough new ones have piled up and NEO has been idle for a while, the heads
are retrained inside the running core, reusing the Laya model already in memory, and swapped in
— only if they score at least as well as the current ones.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from pathlib import Path

from neo.config import settings

EXTRA = "reflex_extra.jsonl"
STATE = "reflex_learn.json"
_MIN_NEW = 30  # new examples before a retrain is worth it
_MIN_GAP_S = 6 * 3600  # at most this often…
_BURST = 150  # …unless this many have piled up
_IDLE_S = 180.0  # NEO must be idle this long before training starts
_lock = threading.Lock()
_training = False

_SKIP = re.compile(r"^\s*(?:(?:hey\s+)?neo|stop|cancel|never\s*mind|ok(?:ay)?|yes|yeah|no|thanks?(?:\s+you)?)\s*[.!?]*\s*$", re.I)


def _path(name: str) -> Path:
    return settings().data_dir / name


def record(text: str, intent: str, tool: str = "", source: str = "") -> bool:
    """Remember what this request turned out to be. Returns whether it was kept."""
    text = " ".join(str(text).split())
    if len(text.split()) < 2 or len(text) > 300 or _SKIP.match(text):
        return False
    if intent not in ("chat", "quick_action", "agent_task"):
        return False
    row = {"text": text, "intent": intent, "tool": tool if intent == "quick_action" else "", "source": source, "ts": time.time()}
    with _lock:
        path = _path(EXTRA)
        rows = _read(path)
        key = text.lower()
        rows = [r for r in rows if r.get("text", "").lower() != key]  # the latest verdict wins
        rows.append(row)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return True


def _read(path: Path) -> list[dict]:
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _state() -> dict:
    try:
        return json.loads(_path(STATE).read_text())
    except Exception:  # noqa: BLE001
        return {"trained_at": 0.0, "trained_rows": 0}


def pending() -> int:
    """Examples recorded since the last retrain."""
    return max(0, len(_read(_path(EXTRA))) - int(_state().get("trained_rows", 0)))


def due(now: float | None = None) -> bool:
    now = now or time.time()
    n = pending()
    since = now - float(_state().get("trained_at", 0.0))
    return n >= _BURST or (n >= _MIN_NEW and since >= _MIN_GAP_S)


def retrain() -> dict | None:
    """Retrain the heads in this process on the already-loaded Laya, then hot-swap them."""
    global _training
    from neo.reflex import get_laya, laya_lock
    from neo.reflex import finetune

    laya = get_laya()
    if laya is None or _training:
        return None
    _training = True
    try:
        n_rows = len(_read(_path(EXTRA)))
        t0 = time.time()
        report = finetune.train(finetune.defaults(), agent=laya._agent, lock=laya_lock())
        if report.get("saved"):
            with laya_lock():
                laya.reload_head()
        _path(STATE).write_text(json.dumps({"trained_at": time.time(), "trained_rows": n_rows, "last": report}))
        prev = report.get("previous")
        print(
            f"[reflex] learned from {n_rows} of your requests in {time.time() - t0:.0f}s: intent "
            f"{prev if prev is None else f'{prev:.3f}'} → {report['intent']:.3f}"
            + ("" if report.get("saved") else " (kept the old head)")
        )
        return report
    finally:
        _training = False


async def learn_loop(is_idle) -> None:
    """Background task in the core: retrain when there's enough new data and NEO is idle."""
    idle_since: float | None = None
    while True:
        await asyncio.sleep(60)
        try:
            if not is_idle():
                idle_since = None
                continue
            idle_since = idle_since or time.time()
            if time.time() - idle_since >= _IDLE_S and due():
                await asyncio.to_thread(retrain)
                idle_since = None
        except Exception as e:  # noqa: BLE001 — learning must never take NEO down
            print(f"[reflex] retrain failed: {str(e)[:120]}")
