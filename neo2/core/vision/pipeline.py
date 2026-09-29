"""One camera question, one frame: :func:`look`.

Apple Vision (labels, words, hands) and D-FINE (object boxes) run side by side in threads, for at
most :data:`LOCAL_WAIT_S` — a detector still loading is left to finish in the background and the
answer goes on without its boxes. What they found is handed to ``on_local`` (the live card draws the
boxes while the answer is on its way), then, as short hints with the one still, to the cloud brains
that see (neo2.core.vision.answer: Gemini, then Cerebras; the on-device labels alone when those are
off or down). "What am I holding?": the object whose box meets a hand's.

:func:`preview` is the live card's per-frame boxes (D-FINE only; nothing leaves the Mac).
"""

from __future__ import annotations

import asyncio
import inspect
import re
import time
from collections.abc import Callable
from typing import Any

from PIL import Image

from neo2.core import effects, logger
from neo2.core.vision import answer as _answer
from neo2.core.vision import apple, detect

log = logger(__name__)

LOCAL_WAIT_S = 1.2  # the on-device layers' share of the ≤3 s answer
_REACH = 0.5  # a hand's box grown by this much of its size on each side: what it's holding is in there


_SURE = 0.8  # D-FINE on its own is trusted above this


def _words(s: str) -> set[str]:
    return {w for w in re.split(r"[^a-z]+", s.lower()) if len(w) > 2}


def trusted(dets: list[detect.Det], labels: list[tuple[str, float]] | None) -> list[detect.Det]:
    """D-FINE's boxes worth drawing and passing on: never the person in front of the camera, and
    either very sure or named by Apple Vision too. Its 80 everyday classes put a wrong name on what
    they don't know (a guitar as "scissors"), and a wrong box on the card misleads."""
    seen: set[str] = set().union(*(_words(n) for n, _ in labels)) if labels else set()
    return [d for d in dets if d["label"] != "person" and (d["score"] >= _SURE or _words(d["label"]) & seen)]


def ui_boxes(dets: list[detect.Det]) -> list[dict[str, Any]]:
    """D-FINE's boxes as the camera card draws them: {x, y, w, h, label, score}, 0..1 of the frame."""
    return [{**d["box"], "label": d["label"], "score": d["score"]} for d in dets]


def holding(hands: list[apple.Hand], dets: list[detect.Det]) -> str | None:
    """The object by a hand (not a person): the box that overlaps the hand's reach most, None when
    no object is within reach."""
    best, most = None, 0.0
    for h in hands:
        x, y, w, hh = h.box
        rx0, ry0 = x - w * _REACH, y - hh * _REACH
        rx1, ry1 = x + w * (1 + _REACH), y + hh * (1 + _REACH)
        for d in dets:
            if d["label"] == "person":
                continue
            b = d["box"]
            ox = max(0.0, min(rx1, b["x"] + b["w"]) - max(rx0, b["x"]))
            oy = max(0.0, min(ry1, b["y"] + b["h"]) - max(ry0, b["y"]))
            area = max(1e-6, b["w"] * b["h"])
            share = ox * oy / area * d["score"]  # how much of the object is in reach, by confidence
            if share > most:
                best, most = d["label"], share
    return best if most >= 0.1 else None


def _apple(img: Image.Image) -> tuple[apple.Seen | None, str, float]:
    t0 = time.perf_counter()
    try:
        return apple.analyse(img), "", _ms(t0)
    except (Exception, effects.EffectRefused) as e:  # noqa: BLE001 — Unavailable, a Vision error, effects off
        return None, _why("Apple Vision", e), _ms(t0)


def _detect(img: Image.Image) -> tuple[list[detect.Det] | None, str, float]:
    t0 = time.perf_counter()
    try:
        return detect.detect(img), "", _ms(t0)
    except (Exception, effects.EffectRefused) as e:  # noqa: BLE001
        return None, _why("the object detector", e), _ms(t0)


def _why(who: str, e: BaseException) -> str:
    from neo2.core.vision import Unavailable

    return str(e) if isinstance(e, Unavailable) else f"{who} failed ({type(e).__name__})"


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


async def _tell(cb: Callable[[dict[str, Any]], Any] | None, found: dict[str, Any]) -> None:
    if cb is None:
        return
    try:
        r = cb(found)
        if inspect.isawaitable(r):
            await r
    except Exception as e:  # noqa: BLE001 — the card's trouble never stops the answer
        log.info(f"[vision] on_local failed: {type(e).__name__}")


async def look(
    question: str,
    img: Image.Image,
    *,
    on_local: Callable[[dict[str, Any]], Any] | None = None,
    local_wait_s: float = LOCAL_WAIT_S,
) -> dict[str, Any]:
    """Answer ``question`` about the frame ``img``: {answer, by, cloud, labels, boxes, text, hands,
    holding, notes, ms: {apple, detect, answer, total}}. ``boxes`` are the card's (:func:`ui_boxes`);
    ``notes``: why a layer was left out (not installed, not downloaded), for the user."""
    t0 = time.perf_counter()
    _answer.prewarm()  # (Cerebras's connection, in case Gemini can't answer: nothing when warm)
    ta = asyncio.ensure_future(asyncio.to_thread(_apple, img))
    td = asyncio.ensure_future(asyncio.to_thread(_detect, img))
    await asyncio.wait({ta, td}, timeout=max(0.0, local_wait_s))
    notes: list[str] = []
    seen, why_a, ms_a = ta.result() if ta.done() else (None, "", None)
    dets, why_d, ms_d = td.result() if td.done() else (None, "", None)
    if not ta.done():
        notes.append("Apple Vision was too slow for this answer")
    if not td.done():
        notes.append("the object detector is still loading")
    notes += [w for w in (why_a, why_d) if w]
    dets = trusted(dets or [], seen.labels if seen else None)
    hands_found = seen.hands if seen else []
    found: dict[str, Any] = {
        "labels": [[n, c] for n, c in (seen.labels if seen else [])],
        "boxes": ui_boxes(dets),
        "text": list(seen.text) if seen else [],
        "hands": apple.hands_summary(hands_found),
        "holding": holding(hands_found, dets),
    }
    await _tell(on_local, found)
    t1 = time.perf_counter()
    hints = {**found, "objects": [d["label"] for d in dets]}
    reply = await _answer.ask(img, question, hints)
    ms = {"apple": ms_a, "detect": ms_d, "answer": _ms(t1), "total": _ms(t0)}
    return {**found, "answer": reply.text, "by": reply.by, "cloud": reply.cloud, "notes": notes, "ms": ms}


def preview(img: Image.Image) -> list[dict[str, Any]]:
    """The live card's boxes for one preview frame (D-FINE, on this Mac; [] when it can't run).
    Blocking: call it in a thread."""
    try:
        return ui_boxes(trusted(detect.detect(img), None))
    except (Exception, effects.EffectRefused) as e:  # noqa: BLE001 — no boxes is a fine preview
        log.debug(f"[vision] preview boxes: {type(e).__name__}")
        return []


__all__ = ["LOCAL_WAIT_S", "holding", "look", "preview", "trusted", "ui_boxes"]
