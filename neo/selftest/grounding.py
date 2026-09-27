"""Scoring a local grounder on this Mac, by evidence (P1, opt-in, user-run).

A local grounder (a vision model the user installed) points at things on screen: "the Bravo
button" → a pixel. Which model and bit width to use is chosen by how well it does here, on the
controls test window's own targets, whose frames the window reports exactly:

    python -m neo.selftest controls --save-shot        # keeps the window's picture and its targets
    python -m neo.selftest grounding --grounder pkg.module:ground [--shot PATH]

``ground(png: bytes, instruction: str) -> (x, y) | None`` gives a pixel of the picture. Each target
is asked for by its description; a hit is a point inside the target's frame (the tilted board's
square: inside that square). Reports the hit rate and p50/p90 seconds per call. Nothing is sent
anywhere by this module — whatever the grounder does is the grounder's.
"""

from __future__ import annotations

import importlib
import io
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from neo.selftest import geometry as G
from neo.selftest import layout as L

# What each target is asked for as (never its coordinates).
ASKS: dict[str, str] = {
    "button_a": f"the {L.BUTTON_A} button",
    "button_b": f"the {L.BUTTON_B} button",
    "hindi": "the pink button with a Hindi label",
    "field": f"the text field that says '{L.FIELD_HINT}'",
    "double": "the box that says 'Double-click me'",
    "marker": "the magenta square",
    "tiny": "the tiny cyan square",
    "tile": "the blue 'Drag me' tile",
    "well": "the 'Drop here' box",
}
SQUARES = ("e2", "a8", "h1")  # asked for on the tilted board: "square e2 on the tilted board"

Grounder = Callable[[bytes, str], "tuple[float, float] | None"]


def load(spec: str) -> Grounder:
    """``pkg.module:function`` → the function."""
    mod, _, fn = spec.partition(":")
    if not mod or not fn:
        raise ValueError("--grounder must be module:function")
    return getattr(importlib.import_module(mod), fn)


def latest_shot(folder: Path) -> Path | None:
    shots = sorted(folder.glob("window-*.png"))
    return shots[-1] if shots else None


def _inside_polygon(poly: list[tuple[float, float]], x: float, y: float) -> bool:
    inside = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def evaluate(ground: Grounder, png: bytes, meta: dict[str, Any]) -> dict[str, Any]:
    """Ask for every target in one saved picture of the window; score the answers."""
    from PIL import Image

    rx, ry, _rw, _rh = meta["region"]
    scale = float(meta.get("scale") or 1.0)  # screen points per image pixel
    Image.open(io.BytesIO(png)).verify()
    rows = []
    for name, ask in ASKS.items():
        frame = meta["targets"].get(name)
        if frame:
            rows.append((name, ask, lambda x, y, f=frame: f[0] <= x < f[0] + f[2] and f[1] <= y < f[1] + f[3]))
    if quad := meta.get("tilt_quad"):
        q = [tuple(p) for p in quad]
        for sq in SQUARES:
            poly = G.cell_polygon(q, *G.cell_of(sq))
            rows.append((f"tilt_{sq}", f"square {sq} on the tilted board", lambda x, y, p=poly: _inside_polygon(p, x, y)))
    out, times = [], []
    for name, ask, hit in rows:
        t = time.perf_counter()
        try:
            got = ground(png, ask)
            err = ""
        except Exception as e:  # noqa: BLE001 - a grounder that fails on one ask is scored on it
            got, err = None, type(e).__name__
        times.append(time.perf_counter() - t)
        ok = False
        if got is not None:
            x, y = rx + float(got[0]) * scale, ry + float(got[1]) * scale
            ok = bool(hit(x, y))
        out.append({"target": name, "asked": ask, "hit": ok, "answered": got is not None, "error": err})
    xs = sorted(times)
    return {
        "targets": len(out),
        "hit_rate": round(sum(r["hit"] for r in out) / len(out), 3) if out else None,
        "p50_s": round(statistics.median(xs), 3) if xs else None,
        "p90_s": round(xs[min(len(xs) - 1, int(0.9 * (len(xs) - 1) + 0.5))], 3) if xs else None,
        "rows": out,
    }


def run(grounder: str, shot: Path | None = None) -> dict[str, Any]:
    from neo.config import settings

    shot = shot or latest_shot(settings().data_dir / "selftest")
    if shot is None or not shot.exists():
        return {"error": "no saved window picture: run `python -m neo.selftest controls --save-shot` first"}
    meta = json.loads(shot.with_suffix(".json").read_text(encoding="utf-8"))
    return {"shot": str(shot), "grounder": grounder, **evaluate(load(grounder), shot.read_bytes(), meta)}
