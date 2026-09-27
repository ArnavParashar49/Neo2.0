"""Boards NEO finds in the pixels: a model names squares ("e2 to e4"), NEO works out the points.

A game board, a sudoku or any grid of cells drawn on a canvas has nothing in it for
Accessibility, and a model that guesses pixel positions for every square misses. So NEO finds the
grid itself, from one picture it already took, and maps every square to a point on the screen:

- **flat grids** (``find_grid``): between two neighbouring squares of a board the colour flips,
  so every internal grid line is a straight edge that runs the whole length of the board (pieces
  sit inside squares and never break it); a grid drawn with lines has the same lines. Their
  spacing gives the squares, and where they start and end gives the board's edges (an invisible
  outer border doesn't matter). A 1–3 px gap that resizing leaves where four squares meet is
  bridged.
- **boards seen at an angle** (a 3D board, tilted and textured): the corners where four squares
  meet (X-corners) are saddle points of the brightness. NEO finds them, grows a lattice from them
  and fits a perspective mapping (homography) board → screen (``find_tilted``, NEO's own), or it
  starts from the 4 corners a model that can see gave once (``from_corners``) and refines the
  mapping on the X-corners near where they should be. Either way the result is checked: the
  squares' colours must alternate, the rows and columns just outside must not (else the board is
  shifted by a square), and a board mapped from a model's corners is marked as such.

Square names: row and column always (``r1c1`` = top-left as seen); chess names (a1–h8) only on an
8×8 board whose orientation is known — from where the light and dark pieces stand, or said by a
model that can see. After a move NEO compares the board with how it looked before
(``changed_cells``) and says what is where now (``occupancy``).

The rules the tools keep (the wiring lives in the pointer tools):

- A brain that can't see never gets or sends coordinates: square names only, and only on a board
  NEO found itself in a fresh picture (``points_for``); a board mapped from a seeing model's
  corners is refused to it.
- Nothing here names an app: a board is found from the pixels and the caller's evidence (a large
  area of a window with no accessible elements), whatever draws it.
- numpy only: no capture, no events, no threads, nothing process-wide — library mode may import it.

Coordinates: image pixels are continuous (pixel k spans [k, k+1)); a picture maps to screen points
as screen = origin + pixel × points-per-pixel, like a screenshot's Shot. A ``Board`` keeps its
mapping in screen points, so any later picture of the same place (another scale, a region) can be
compared with it.
"""

from __future__ import annotations

import itertools
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---- pictures ---------------------------------------------------------------------------------


@dataclass
class Picture:
    """An RGB picture (h × w × 3, uint8) and how its pixels map to the screen."""

    rgb: np.ndarray
    x0: float = 0.0
    y0: float = 0.0
    ppp: float = 1.0  # screen points per pixel
    _gray: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def of(cls, image: Any, x0: float = 0.0, y0: float = 0.0, ppp: float = 1.0) -> Picture:
        """From a PIL image or an array (RGB, RGBA or grey)."""
        a = np.asarray(image)
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=2)
        elif a.shape[2] == 4:
            a = a[..., :3]
        return cls(np.ascontiguousarray(a, dtype=np.uint8), float(x0), float(y0), float(ppp))

    @classmethod
    def from_shot(cls, image: Any, shot: Any) -> Picture:
        """A screenshot's picture with its Shot (anything with x0, y0 and ppp)."""
        return cls.of(image, shot.x0, shot.y0, shot.ppp)

    @property
    def gray(self) -> np.ndarray:
        if self._gray is None:
            self._gray = self.rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        return self._gray

    @property
    def to_screen(self) -> np.ndarray:
        """Image pixels → screen points (3 × 3)."""
        return np.array([[self.ppp, 0, self.x0], [0, self.ppp, self.y0], [0, 0, 1.0]])


# ---- geometry ---------------------------------------------------------------------------------


def _apply(H: np.ndarray, pts) -> np.ndarray:
    p = np.asarray(pts, float).reshape(-1, 2)
    q = np.c_[p, np.ones(len(p))] @ H.T
    return q[:, :2] / q[:, 2:3]


def _norm(p: np.ndarray) -> np.ndarray:
    """Hartley normalisation: centred, mean distance √2."""
    c = p.mean(0)
    s = math.sqrt(2) / max(1e-9, float(np.mean(np.linalg.norm(p - c, axis=1))))
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])


def homography(src, dst) -> np.ndarray | None:
    """The 3 × 3 mapping taking ``src`` points to ``dst`` (least squares over ≥ 4 pairs)."""
    s, d = np.asarray(src, float), np.asarray(dst, float)
    if len(s) < 4 or len(s) != len(d):
        return None
    Ts, Td = _norm(s), _norm(d)
    sn, dn = _apply(Ts, s), _apply(Td, d)
    A = np.zeros((2 * len(s), 9))
    x, y, u, v = sn[:, 0], sn[:, 1], dn[:, 0], dn[:, 1]
    A[0::2, 0:3] = np.c_[x, y, np.ones_like(x)]
    A[0::2, 6:9] = -np.c_[u * x, u * y, u]
    A[1::2, 3:6] = np.c_[x, y, np.ones_like(x)]
    A[1::2, 6:9] = -np.c_[v * x, v * y, v]
    try:
        Vt = np.linalg.svd(A)[2]
    except np.linalg.LinAlgError:
        return None
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    if abs(H[2, 2]) < 1e-12 or not np.all(np.isfinite(H)):
        return None
    return H / H[2, 2]


def _sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear samples of ``img`` (h × w or h × w × c) at continuous pixel positions."""
    h, w = img.shape[:2]
    fx = np.clip(xs - 0.5, 0, w - 1.001)
    fy = np.clip(ys - 0.5, 0, h - 1.001)
    x0, y0 = fx.astype(np.int32), fy.astype(np.int32)
    ax, ay = fx - x0, fy - y0
    if img.ndim == 3:
        ax, ay = ax[..., None], ay[..., None]
    a = img[y0, x0].astype(np.float32)
    b = img[y0, x0 + 1].astype(np.float32)
    c = img[y0 + 1, x0].astype(np.float32)
    d = img[y0 + 1, x0 + 1].astype(np.float32)
    return (a * (1 - ax) + b * ax) * (1 - ay) + (c * (1 - ax) + d * ax) * ay


_MAX_CELLS = 24  # the most squares on a side NEO looks for
_MIN_SQUARE = 12  # pixels: a square smaller than this is no square anyone plays on

# ---- flat grids -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    """A flat grid found in a picture, in that picture's pixels."""

    x0: float
    y0: float
    cw: float  # cell size, px
    ch: float
    cols: int
    rows: int
    lines: bool = False  # drawn with lines (a sudoku), not alternating squares

    def centre(self, col: int, row: int) -> tuple[float, float]:
        return self.x0 + (col + 0.5) * self.cw, self.y0 + (row + 0.5) * self.ch

    @property
    def H(self) -> np.ndarray:
        """Board units → image pixels."""
        return np.array([[self.cw, 0, self.x0], [0, self.ch, self.y0], [0, 0, 1.0]])


def _runs(edge: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per column of ``edge`` (runs go down axis 0): the length of its longest run of True and the
    index of that run's last element."""
    h, w = edge.shape
    e = np.zeros((w, h + 2), np.int8)
    e[:, 1:-1] = edge.T
    d = np.diff(e, axis=1)
    sc, sp = np.nonzero(d == 1)
    _, ep = np.nonzero(d == -1)
    best = np.zeros(w, np.int64)
    end = np.zeros(w, np.int64)
    if len(sp):
        length = ep - sp
        order = np.lexsort((length, sc))
        cs = sc[order]
        last = np.r_[cs[1:] != cs[:-1], True]
        best[cs[last]] = length[order][last]
        end[cs[last]] = ep[order][last] - 1
    return best, end


def _close(edge: np.ndarray, gap: int = 3) -> np.ndarray:
    """Bridge gaps of up to ``gap`` pixels along axis 0 (where four squares meet, a resampled
    picture blends the corner and the flip vanishes for a pixel or two); the ends stay put."""
    h = edge.shape[0]
    p = np.zeros((h + 2 * gap, edge.shape[1]), bool)
    p[gap : gap + h] = edge
    dil = p[gap : gap + h].copy()
    for d in range(1, gap + 1):
        dil |= p[gap - d : gap - d + h]
        dil |= p[gap + d : gap + d + h]
    q = np.ones((h + 2 * gap, edge.shape[1]), bool)
    q[gap : gap + h] = dil
    out = dil.copy()
    for d in range(1, gap + 1):
        out &= q[gap - d : gap - d + h]
        out &= q[gap + d : gap + d + h]
    return out


def _candidates(edge: np.ndarray, min_run: int) -> list[tuple[float, float, float]]:
    """(position, run start, run end) of each straight line at least ``min_run`` long (lines in
    adjacent columns merged: the strongest stands for them)."""
    cols = np.nonzero(edge.sum(axis=0) >= min_run // 2)[0]  # (only columns with edge enough for a line)
    length = np.zeros(edge.shape[1], np.int64)
    end = np.zeros(edge.shape[1], np.int64)
    if len(cols):
        length[cols], end[cols] = _runs(_close(edge[:, cols]))
    idx = np.nonzero(length >= min_run)[0]
    out: list[tuple[float, float, float]] = []
    group: list[int] = []
    for i in [*idx.tolist(), None]:
        if group and (i is None or i - group[-1] > 2):
            j = max(group, key=lambda k: length[k])
            out.append((float(j), float(end[j] - length[j] + 1), float(end[j] + 1)))
            group = []
        if i is not None:
            group.append(int(i))
    return out


def _progression(lines: list[tuple[float, float, float]], min_lines: int):
    """The longest run of equally spaced lines that share one extent, where that extent is a whole
    number of spacings (a board's lines, not a column of buttons): (first, step, count, start, end)."""
    best = None
    n = len(lines)
    for i in range(n):
        s0, e0 = lines[i][1], lines[i][2]
        for j in range(i + 1, n):
            step = lines[j][0] - lines[i][0]
            if step < 8:
                continue
            slack = max(4.0, step * 0.15)
            if abs(lines[j][1] - s0) > slack or abs(lines[j][2] - e0) > slack:
                continue
            tol = max(2.5, step * 0.05)
            picked = [lines[i][0], lines[j][0]]
            while True:
                want = lines[i][0] + len(picked) * step
                nxt = [ln for ln in lines if abs(ln[0] - want) <= tol and abs(ln[1] - s0) <= slack and abs(ln[2] - e0) <= slack]
                if not nxt:
                    break
                picked.append(min(nxt, key=lambda ln: abs(ln[0] - want))[0])
            k = len(picked)
            if k < min_lines or (best is not None and k <= best[2]):
                continue
            fit = np.polyfit(np.arange(k), np.array(picked), 1)
            step_fit = float(fit[0])
            cells = (e0 - s0) / step_fit
            if abs(cells - round(cells)) > 0.2 or round(cells) < k - 1:
                continue
            best = (float(fit[1]), step_fit, k, s0, e0)
    return best


def _axis(first: float, step: float, k: int, lo: float, hi: float) -> tuple[float, float, int] | None:
    """(edge, cell size, cells) along one axis: the extent ``lo``–``hi`` (where the other set of
    lines starts and ends) cut into cells of about ``step``, the ``k`` lines from ``first`` landing
    on the cells' edges. Lines at both ends (a visible border) set the edges themselves."""
    cells = round((hi - lo) / step)
    if cells < 1 or abs((hi - lo) - cells * step) > step * 0.2:
        return None
    size = (hi - lo) / cells
    pos = first + 1  # (a flip between pixels i and i+1 is an edge at i+1)
    m = (pos - lo) / size
    if abs(m - round(m)) > 0.15 or round(m) < 0 or round(m) + k - 1 > cells:
        return None  # the lines aren't the extent's cell edges: something else
    if round(m) == 0 and round(m) + k - 1 == cells:
        return pos, step, cells  # its border is drawn: the outer lines are the edges
    return lo, size, cells


def find_grid(gray: np.ndarray, *, min_cells: int = 3, flip: float = 18.0) -> Grid | None:
    """A flat grid of cells in a grey picture, or None: alternating squares (a chess or checkers
    board) or a grid drawn with lines (sudoku, go); in the picture's own pixels."""
    g = np.asarray(gray, np.float32)
    if g.ndim != 2 or min(g.shape) < 40:
        return None
    h, w = g.shape
    ex = np.abs(np.diff(g, axis=1)) > flip  # a flip between x and x+1
    min_run = max(40, int(min(h, w) * 0.1))
    v = _progression(_candidates(ex, min_run), 2)
    if v is None:
        return None
    vfirst, vstep, vn, top, bottom = v
    a, b = max(0, int(top) - 3), min(h - 1, int(bottom) + 3)  # (with the board's own top and bottom edges)
    ey = np.abs(np.diff(g[a : b + 1], axis=0)) > flip
    hz = _progression([(p + a, s, e) for p, s, e in _candidates(ey.T, min_run)], 2)
    if hz is None:
        return None
    hfirst, hstep, hn, left, right = hz
    cx, cy = _axis(vfirst, vstep, vn, left, right), _axis(hfirst, hstep, hn, top, bottom)
    if cx is None or cy is None:
        return None
    (x0, cw, cols), (y0, ch, rows) = cx, cy
    if not (min_cells <= cols <= _MAX_CELLS and min_cells <= rows <= _MAX_CELLS) or min(cw, ch) < _MIN_SQUARE or not (0.5 < cw / ch < 2.0):
        return None  # (a transparency checkerboard's tiny squares, a table's long rows: no board)
    return Grid(float(x0), float(y0), float(cw), float(ch), int(cols), int(rows), lines=_drawn_with_lines(g, x0, y0, cw, ch, cols, rows))


def _drawn_with_lines(g: np.ndarray, x0: float, y0: float, cw: float, ch: float, cols: int, rows: int) -> bool:
    """Lines on one ground (a sudoku) rather than alternating squares: neighbouring cells' middles
    are about as bright as each other."""
    ys = y0 + (np.arange(rows) + 0.5) * ch
    xs = x0 + (np.arange(cols) + 0.5) * cw
    # each cell's typical brightness: the median of a few points around its middle, off any piece
    ox, oy = np.meshgrid([-0.38, -0.3, 0.3, 0.38], [-0.38, -0.3, 0.3, 0.38])
    px, py = np.broadcast_arrays(xs[None, :, None] + ox.ravel() * cw, ys[:, None, None] + oy.ravel() * ch)
    vals = np.median(_sample(g, px, py).reshape(rows, cols, -1), axis=2)
    par = (np.arange(rows)[:, None] + np.arange(cols)[None, :]) % 2 == 0
    return abs(float(np.median(vals[par])) - float(np.median(vals[~par]))) < 12.0


# ---- X-corners: where four squares meet ---------------------------------------------------------


def _box(a: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)² box, edges repeated."""
    if r <= 0:
        return a
    k = 2 * r + 1
    h, w = a.shape
    p = np.pad(a.astype(np.float32, copy=False), r, mode="edge")
    v = p[0:h].copy()
    for d in range(1, k):
        v += p[d : d + h]
    out = v[:, 0:w].copy()
    for d in range(1, k):
        out += v[:, d : d + w]
    return out * np.float32(1.0 / (k * k))


def _smooth(a: np.ndarray, r: int) -> np.ndarray:
    """Three box passes: close to a Gaussian (σ ≈ √((2r+1)² − 1)/2)."""
    for _ in range(3):
        a = _box(a, r)
    return a


def _saddle(g: np.ndarray, r: int) -> np.ndarray:
    """How strongly each pixel is a saddle of the smoothed brightness (≥ 0): the corner of four
    alternating squares is one, whatever its angle or perspective; an edge, a blob or wood grain
    along one direction is not."""
    s = _smooth(g.astype(np.float32), r)
    p = np.pad(s, 1, mode="edge")
    ixx = p[1:-1, 2:] - 2 * s + p[1:-1, :-2]
    iyy = p[2:, 1:-1] - 2 * s + p[:-2, 1:-1]
    ixy = (p[2:, 2:] - p[2:, :-2] - p[:-2, 2:] + p[:-2, :-2]) * 0.25
    return np.maximum(ixy * ixy - ixx * iyy, 0.0)


def _maxfilter(a: np.ndarray, r: int) -> np.ndarray:
    p = np.pad(a, r, mode="constant", constant_values=-1)
    h, w = a.shape
    m = p[:, r : r + w].copy()
    for d in range(-r, r + 1):
        np.maximum(m, p[:, r + d : r + d + w], out=m)
    out = m[r : r + h].copy()
    for d in range(-r, r + 1):
        np.maximum(out, m[r + d : r + d + h], out=out)
    return out


def _peaks(resp: np.ndarray, rad: int, rel: float = 0.04, most: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima of ``resp`` (points at sub-pixel precision, strengths), strongest first."""
    top = float(resp.max()) if resp.size else 0.0
    if top <= 0:
        return np.zeros((0, 2)), np.zeros(0)
    m = (resp >= _maxfilter(resp, rad)) & (resp > top * rel)
    ys, xs = np.nonzero(m)
    if len(xs) > most:
        keep = np.argsort(-resp[ys, xs])[:most]
        ys, xs = ys[keep], xs[keep]
    pts = np.stack([xs, ys], 1).astype(np.float64)
    return _subpixel(resp, pts), resp[ys, xs]


def _subpixel(R: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """A peak's position to a fraction of a pixel (a parabola through it and its neighbours), as
    continuous coordinates (the pixel's middle is +0.5)."""
    h, w = R.shape
    out = pts.copy() + 0.5
    for n, (x, y) in enumerate(pts.astype(int)):
        if 0 < x < w - 1:
            a, b, c = R[y, x - 1], R[y, x], R[y, x + 1]
            d = a - 2 * b + c
            if d < 0:
                out[n, 0] += float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))
        if 0 < y < h - 1:
            a, b, c = R[y - 1, x], R[y, x], R[y + 1, x]
            d = a - 2 * b + c
            if d < 0:
                out[n, 1] += float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))
    return out


# ---- a lattice of X-corners -----------------------------------------------------------------------


def _spacing(H: np.ndarray, i: float, j: float) -> float:
    p = _apply(H, [(i, j), (i + 1, j), (i, j + 1)])
    return float(min(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[0])))


_QUAD = np.array([(a * 0.18 + da, b * 0.18 + db) for a, b in ((1, 1), (-1, -1), (1, -1), (-1, 1)) for da in (-0.05, 0, 0.05) for db in (-0.05, 0, 0.05)])


def _xcheck(g: np.ndarray, H: np.ndarray, keys: np.ndarray, sign: float = 0.0) -> np.ndarray:
    """How much each lattice point (i, j) looks like a corner of four alternating squares in this
    mapping: the two squares on one diagonal alike, the other two alike, the pairs different — signed
    by the point's parity, so a whole board agrees (> 0 with ``sign`` = the board's own sign)."""
    keys = np.asarray(keys, float).reshape(-1, 2)
    pts = (keys[:, None, :] + _QUAD[None, :, :]).reshape(-1, 2)
    xy = _apply(H, pts)
    v = _sample(g, xy[:, 0], xy[:, 1]).reshape(len(keys), 4, 9).mean(axis=2)
    a, b = (v[:, 0] + v[:, 1]) / 2, (v[:, 2] + v[:, 3]) / 2
    spread = np.abs(v[:, 0] - v[:, 1]) + np.abs(v[:, 2] - v[:, 3])
    par = np.where((keys[:, 0] + keys[:, 1]) % 2 == 0, 1.0, -1.0)
    c = (a - b) * par
    ok = np.abs(a - b) > np.maximum(12.0, 1.5 * spread)
    if sign:
        c = c * sign
    return np.where(ok, c, 0.0)


def _grow(g: np.ndarray, P: np.ndarray, lat: dict[tuple[int, int], int], sign: float, tol: float = 0.22) -> dict[tuple[int, int], int]:
    """Grow a lattice (lattice point → index into ``P``) one ring of neighbours at a time: fit the
    mapping to what is found, predict each neighbour, take the candidate there when it is a corner
    of this board's squares."""
    used = set(lat.values())
    for _ in range(2 * _MAX_CELLS):
        keys = list(lat)
        H = homography(keys, P[[lat[k] for k in keys]])
        if H is None:
            break
        have = set(keys)
        F = np.array(sorted({(i + di, j + dj) for i, j in keys for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))} - have), float)
        F = F[np.abs(F).max(axis=1) <= _MAX_CELLS] if len(F) else F
        if not len(F):
            break
        Q = _apply(H, F)
        sp = np.minimum(np.linalg.norm(_apply(H, F + (1, 0)) - Q, axis=1), np.linalg.norm(_apply(H, F + (0, 1)) - Q, axis=1))
        D = np.linalg.norm(Q[:, None, :] - P[None, :, :], axis=2)
        k = np.argmin(D, axis=1)
        near = D[np.arange(len(F)), k] <= tol * sp
        if not near.any():
            break
        F, k = F[near], k[near]
        good = _xcheck(g, H, F, sign) > 0
        added = 0
        for (i, j), idx in zip(F[good].astype(int).tolist(), k[good].tolist(), strict=True):
            if idx not in used:
                lat[(i, j)] = idx
                used.add(idx)
                added += 1
        if not added:
            break
    return lat


def _clean(P: np.ndarray, lat: dict[tuple[int, int], int]) -> tuple[dict[tuple[int, int], int], np.ndarray | None, float]:
    """Drop lattice points far from the fitted mapping; (lattice, mapping, rms error / spacing)."""
    for _ in range(3):
        keys = list(lat)
        if len(keys) < 4:
            return lat, None, 1.0
        src = np.array(keys, float)
        dst = P[[lat[k] for k in keys]]
        H = homography(src, dst)
        if H is None:
            return lat, None, 1.0
        res = np.linalg.norm(_apply(H, src) - dst, axis=1)
        sp = _spacing(H, *src.mean(0))
        bad = res > max(1.5, 3.0 * float(np.median(res)), 0.12 * sp)
        if not bad.any():
            return lat, H, float(np.sqrt(np.mean(res**2)) / max(sp, 1e-9))
        lat = {k: v for k, v, b in zip(keys, [lat[k] for k in keys], bad, strict=True) if not b}
    return lat, None, 1.0


def _seek_lattice(g: np.ndarray, P: np.ndarray, strength: np.ndarray, seeds: int = 24) -> tuple[dict[tuple[int, int], int], np.ndarray | None]:
    """NEO's own lattice: from the strongest candidates, a unit cell (a point, two neighbours in
    different directions and the fourth corner where they say) whose corner looks like four
    alternating squares, grown as far as such corners go."""
    n = len(P)
    if n < 9:
        return {}, None
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    best: dict[tuple[int, int], int] = {}
    best_H = None
    done: set[int] = set()
    for s in np.argsort(-strength)[:seeds]:
        if int(s) in done:
            continue
        cells = []
        for a, b in itertools.combinations(np.argsort(D[s])[1:9], 2):
            u, v = P[a] - P[s], P[b] - P[s]
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu < 6 or nv < 6 or not (0.33 < nu / nv < 3.0):
                continue
            if abs(u[0] * v[1] - u[1] * v[0]) / (nu * nv) < 0.55:  # (≥ about 33° apart)
                continue
            d4 = np.linalg.norm(P - (P[s] + u + v), axis=1)
            c = int(np.argmin(d4))
            if d4[c] <= 0.25 * min(nu, nv) and c not in (s, a, b):
                cells.append((int(a), int(b), c))
        if not cells:
            continue
        # the corner of four alternating squares, judged on all the candidate cells at once (as if flat
        # there: a unit cell is small enough)
        U = np.array([P[a] - P[s] for a, _, _ in cells])
        V = np.array([P[b] - P[s] for _, b, _ in cells])
        pts = P[s] + _QUAD[None, :, :1] * U[:, None, :] + _QUAD[None, :, 1:] * V[:, None, :]
        q = _sample(g, pts[..., 0].ravel(), pts[..., 1].ravel()).reshape(len(cells), 4, 9).mean(axis=2)
        contrast = np.abs(q[:, 0] + q[:, 1] - q[:, 2] - q[:, 3]) / 2
        spread = np.abs(q[:, 0] - q[:, 1]) + np.abs(q[:, 2] - q[:, 3])
        for n in np.nonzero(contrast > np.maximum(12.0, 1.5 * spread))[0]:
            a, b, c = cells[n]
            cell = {(0, 0): int(s), (1, 0): a, (0, 1): b, (1, 1): c}
            H = homography(list(cell), P[list(cell.values())])
            if H is None:
                continue
            x = _xcheck(g, H, list(cell))
            if not (np.all(x > 0) or np.all(x < 0)):
                continue
            lat = _grow(g, P, cell, float(np.sign(x[0])))
            lat, H, err = _clean(P, lat)
            done.update(lat.values())
            if H is not None and err < 0.08 and len(lat) > len(best):
                best, best_H = lat, H
            break  # one good unit cell per seed: its lattice is the one there
        if len(best) >= 30:
            break
    return best, best_H


# ---- squares: their colours alternate ------------------------------------------------------------

# Where a square's own colour shows: near its edges and corners (a piece stands in its middle and,
# seen at an angle, rises over the squares behind it).
_RIM = np.array(
    [(x + dx, y + dy) for x, y in ((0.13, 0.13), (0.5, 0.12), (0.87, 0.13), (0.88, 0.5), (0.87, 0.87), (0.5, 0.88), (0.13, 0.87), (0.12, 0.5))
     for dx in (-0.04, 0, 0.04) for dy in (-0.04, 0, 0.04)]
)  # fmt: skip


def _rim_colours(rgb: np.ndarray, H: np.ndarray, squares: np.ndarray) -> np.ndarray:
    """RGB samples round each square's rim: (squares, samples, 3)."""
    pts = (squares[:, None, :].astype(float) + _RIM[None, :, :]).reshape(-1, 2)
    xy = _apply(H, pts)
    return _sample(rgb, xy[:, 0], xy[:, 1]).reshape(len(squares), len(_RIM), 3)


def _inside(shape: tuple[int, ...], H: np.ndarray, squares: np.ndarray) -> np.ndarray:
    """Which squares lie wholly in the picture."""
    h, w = shape[:2]
    cs = np.array([(0, 0), (1, 0), (0, 1), (1, 1)], float)
    xy = _apply(H, (squares[:, None, :] + cs[None]).reshape(-1, 2)).reshape(len(squares), 4, 2)
    return np.all((xy[..., 0] >= 0) & (xy[..., 0] <= w) & (xy[..., 1] >= 0) & (xy[..., 1] <= h), axis=1)


@dataclass
class Tones:
    even: np.ndarray  # the colour of squares whose col + row is even
    odd: np.ndarray

    @property
    def gap(self) -> float:
        return float(np.linalg.norm(self.even - self.odd))


def _tones(rgb: np.ndarray, H: np.ndarray, squares: np.ndarray) -> Tones:
    cols = _rim_colours(rgb, H, squares)
    even = (squares[:, 0] + squares[:, 1]) % 2 == 0
    return Tones(np.median(cols[even].reshape(-1, 3), axis=0), np.median(cols[~even].reshape(-1, 3), axis=0))


def _agree(rgb: np.ndarray, H: np.ndarray, squares: np.ndarray, t: Tones) -> np.ndarray:
    """Per square: does its rim show its own tone (the one its parity says), not the other?"""
    cols = _rim_colours(rgb, H, squares)
    even = ((squares[:, 0] + squares[:, 1]) % 2 == 0)[:, None]
    own = np.where(even[..., None], t.even, t.odd)
    other = np.where(even[..., None], t.odd, t.even)
    d_own = np.linalg.norm(cols - own, axis=2)
    d_other = np.linalg.norm(cols - other, axis=2)
    near = (d_own < d_other) & (d_own < max(28.0, 0.55 * t.gap))
    return near.mean(axis=1) >= 0.34


def _block(c0: int, r0: int, c1: int, r1: int) -> np.ndarray:
    """Squares (col, row) with c0 ≤ col ≤ c1 and r0 ≤ row ≤ r1."""
    cc, rr = np.meshgrid(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1))
    return np.stack([cc.ravel(), rr.ravel()], 1)


def _extent(rgb: np.ndarray, H: np.ndarray, c0: int, r0: int, c1: int, r1: int) -> tuple[tuple[int, int, int, int], Tones, float] | None:
    """How far the alternating squares go, from a block known to be on the board: a row or column
    more on any side while its squares alternate like the board's; ((c0, r0, c1, r1), tones, share
    of all the board's squares that agree)."""
    sq = _block(c0, r0, c1, r1)
    seen = _inside(rgb.shape, H, sq)
    t = _tones(rgb, H, sq[seen] if seen.any() else sq)
    if t.gap < 30:
        return None  # no two colours: not a board of squares
    for _ in range(2 * _MAX_CELLS):
        grew = False
        for side in ("left", "right", "top", "bottom"):
            if c1 - c0 + 1 >= _MAX_CELLS and side in ("left", "right"):
                continue
            if r1 - r0 + 1 >= _MAX_CELLS and side in ("top", "bottom"):
                continue
            strip = {
                "left": _block(c0 - 1, r0, c0 - 1, r1),
                "right": _block(c1 + 1, r0, c1 + 1, r1),
                "top": _block(c0, r0 - 1, c1, r0 - 1),
                "bottom": _block(c0, r1 + 1, c1, r1 + 1),
            }[side]
            seen = _inside(rgb.shape, H, strip)
            if seen.sum() < max(2, 0.6 * len(strip)):
                continue
            if _agree(rgb, H, strip[seen], t).mean() >= 0.7:
                c0, r0, c1, r1 = (c0 - (side == "left"), r0 - (side == "top"), c1 + (side == "right"), r1 + (side == "bottom"))
                grew = True
        if not grew:
            break
    sq = _block(c0, r0, c1, r1)
    seen = _inside(rgb.shape, H, sq)
    if seen.mean() < 0.9:
        return None  # (a board mostly off the picture: no use)
    return (c0, r0, c1, r1), t, float(_agree(rgb, H, sq[seen], t).mean())


def _upright(H: np.ndarray, at) -> tuple[np.ndarray, np.ndarray]:
    """A lattice→image mapping turned so its first axis goes right and its second down, as seen:
    (the new mapping, M taking new lattice coordinates to the old ones)."""
    at = (float(at[0]), float(at[1]))
    p = _apply(H, [at, (at[0] + 1, at[1]), (at[0], at[1] + 1)])
    du, dv = p[1] - p[0], p[2] - p[0]
    A = np.eye(3)
    if abs(du[0]) / (np.linalg.norm(du) + 1e-9) < abs(dv[0]) / (np.linalg.norm(dv) + 1e-9):
        A = np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 1]])  # swap: new (c, r) = old (j, i)
        du, dv = dv, du
    M = A @ np.diag([1.0 if du[0] >= 0 else -1.0, 1.0 if dv[1] >= 0 else -1.0, 1.0])
    return H @ M, M


# ---- boards -------------------------------------------------------------------------------------

_FILES = "abcdefghijklmnopqrstuvwxyz"
SIDES = ("bottom", "top", "left", "right")


@dataclass
class Board:
    """A board NEO knows: ``H`` maps board units (col, row; (0, 0) the top-left corner of the
    playing area as seen, rows going down the picture) to screen points."""

    H: np.ndarray
    rows: int
    cols: int
    source: str  # "neo": NEO found the grid in the pixels itself; "vision": from a seeing model's corners
    flat: bool = True  # seen straight on (no perspective): pieces lie within their squares
    lines: bool = False  # a grid drawn with lines, not alternating squares
    orientation: str = ""  # where the light pieces' home rows are ("bottom", …; "" unknown)
    oriented_by: str = ""  # "pieces" (NEO judged it) or "model" (a seeing model said so)
    shot: str = ""
    at: float = 0.0  # time.monotonic() of the picture it was found or last checked in
    window: tuple[float, float, float, float] | None = None  # the window's frame then (screen points)
    tones: Tones | None = None
    agree: float = 1.0  # share of squares whose colour alternates as a board's should
    state: np.ndarray | None = field(default=None, repr=False)  # how every square looked last time
    rim: np.ndarray | None = field(default=None, repr=False)  # …and its rim (a move leaves rims alone)
    occupancy: list[str] | None = None  # per row, top first: '.' empty, 'L' light piece, 'D' dark piece

    # -- geometry --
    def centre(self, col: int, row: int) -> tuple[float, float]:
        x, y = _apply(self.H, [(col + 0.5, row + 0.5)])[0]
        return float(x), float(y)

    def corners(self) -> np.ndarray:
        return _apply(self.H, [(0, 0), (self.cols, 0), (self.cols, self.rows), (0, self.rows)])

    def bbox(self) -> tuple[float, float, float, float]:
        """(x, y, w, h) in screen points: the playing area's bounding box."""
        c = self.corners()
        x0, y0 = c.min(0)
        x1, y1 = c.max(0)
        return float(x0), float(y0), float(x1 - x0), float(y1 - y0)

    def cell_at(self, x: float, y: float) -> tuple[int, int] | None:
        u, v = _apply(np.linalg.inv(self.H), [(x, y)])[0]
        c, r = math.floor(u), math.floor(v)
        return (c, r) if 0 <= c < self.cols and 0 <= r < self.rows else None

    def square_size(self) -> float:
        """The smallest square's shorter side, in points."""
        return min(_spacing(self.H, c, r) for c in (0, self.cols - 1) for r in (0, self.rows - 1))

    # -- names --
    @property
    def chess(self) -> bool:
        """Names like a1–h8 apply: 8×8 alternating squares whose orientation is known."""
        return self.rows == 8 and self.cols == 8 and not self.lines and self.orientation in SIDES

    def name(self, col: int, row: int) -> str:
        if not self.chess:
            return f"r{row + 1}c{col + 1}"
        o = self.orientation
        f, r = {"bottom": (col, 8 - row), "top": (7 - col, row + 1), "left": (row, col + 1), "right": (7 - row, 8 - col)}[o]
        return f"{_FILES[f]}{r}"

    def parse(self, name: str) -> tuple[int, int]:
        """(col, row) of a square's name, or ValueError saying which names this board takes."""
        s = re.sub(r"\s+", "", str(name or "")).lower()
        m = re.fullmatch(r"r(?:ow)?(\d{1,2})c(?:ol)?(\d{1,2})", s)
        if m:
            row, col = int(m.group(1)) - 1, int(m.group(2)) - 1
            if 0 <= row < self.rows and 0 <= col < self.cols:
                return col, row
            raise ValueError(f"{name!r} is off the board, which has {self.rows} rows and {self.cols} columns (r1c1 … r{self.rows}c{self.cols}).")
        m = re.fullmatch(r"([a-z])(\d{1,2})", s)
        if m:
            if not self.chess:
                why = "its orientation isn't known" if self.rows == 8 == self.cols and not self.lines else f"it is {self.rows}×{self.cols}, not a chess board"
                raise ValueError(
                    f"this board's squares have no chess names ({why}): name them by row and column from the top-left "
                    f"as seen, r1c1 … r{self.rows}c{self.cols}."
                )
            f, r = _FILES.index(m.group(1)), int(m.group(2))
            if not (0 <= f < 8 and 1 <= r <= 8):
                raise ValueError(f"{name!r} is off the board (a1 … h8).")
            o = self.orientation
            col, row = {"bottom": (f, 8 - r), "top": (7 - f, r - 1), "left": (r - 1, f), "right": (8 - r, 7 - f)}[o]
            return col, row
        raise ValueError(f"{str(name)[:20]!r} isn't a square's name here: use {self.name(0, 0)} … {self.name(self.cols - 1, self.rows - 1)}.")

    def names(self) -> str:
        return f"{self.name(0, self.rows - 1)}–{self.name(self.cols - 1, 0)}" if self.chess else f"r1c1–r{self.rows}c{self.cols}"


def _board(pic: Picture, H_img: np.ndarray, c0: int, r0: int, c1: int, r1: int, **kw) -> Board:
    """A Board from a lattice→image mapping and the block of squares that is the board."""
    shift = np.array([[1.0, 0, c0], [0, 1.0, r0], [0, 0, 1]])
    t = kw.get("tones")
    if t is not None and (c0 + r0) % 2:  # (the squares' parity counted from the board's own corner)
        kw["tones"] = Tones(t.odd, t.even)
    return Board(pic.to_screen @ H_img @ shift, rows=r1 - r0 + 1, cols=c1 - c0 + 1, **kw)


def _perspective(H: np.ndarray, cols: int, rows: int) -> float:
    """How far from flat a board looks: the ratio of its largest to smallest square side."""
    sides = [_spacing(H, c, r) for c in (0, cols - 1) for r in (0, rows - 1)]
    return max(sides) / max(1e-9, min(sides))


def _work(pic: Picture, area: tuple[float, float, float, float] | None, f: int):
    """The grey picture to look for corners in: ``area`` (image pixels x, y, w, h) or all of it,
    averaged down ``f`` times. (grey, factor, crop x, crop y)."""
    g = pic.gray
    h, w = g.shape
    ox = oy = 0
    if area is not None:
        x, y, aw, ah = area
        ox, oy = max(0, int(math.floor(x))), max(0, int(math.floor(y)))
        g = g[oy : min(h, int(math.ceil(y + ah))), ox : min(w, int(math.ceil(x + aw)))]
    if f > 1:
        hh, ww = g.shape[0] // f * f, g.shape[1] // f * f
        g = g[:hh, :ww].reshape(hh // f, f, ww // f, f).mean(axis=(1, 3))
    return np.ascontiguousarray(g, np.float32), f, ox, oy


_WORK_SQUARE = 24  # pixels: X-corners are looked for where a square is about this big


def _scales(pic: Picture, area, square: float | None) -> list[int]:
    """How far to average the picture down before looking for X-corners: to about _WORK_SQUARE
    pixels a square when the square's size is known; else the picture at most ~1000 px long, then
    coarser and finer (big squares' wood grain would drown their corners; small ones would blur)."""
    if square:
        return [max(1, round(square / _WORK_SQUARE))]
    long = max(area[2], area[3]) if area is not None else max(pic.rgb.shape[:2])
    f0 = max(1, math.ceil(long / 1000))
    return list(dict.fromkeys([f0, 2 * f0, max(1, f0 // 2)]))


def _lift(f: int, ox: int, oy: int) -> np.ndarray:
    """Working-picture pixels → image pixels."""
    return np.array([[f, 0, ox], [0, f, oy], [0, 0, 1.0]])


def _settle(pic: Picture, H_img: np.ndarray, keys: np.ndarray, **kw) -> tuple[Board | None, str]:
    """From a lattice of X-corners (``keys``, mapped to image pixels by ``H_img``) to a checked
    board: turned upright, its extent found where its squares stop alternating."""
    Hu, M = _upright(H_img, keys.mean(0))
    k = np.rint(_apply(np.linalg.inv(M), keys)).astype(int)
    c0, r0 = k.min(0) - 1
    c1, r1 = k.max(0)
    got = _extent(pic.rgb, Hu, int(c0), int(r0), int(c1), int(r1))
    if got is None:
        return None, "the squares there don't alternate in two colours"
    (c0, r0, c1, r1), tones, agree = got
    if agree < 0.75:
        return None, f"only {agree:.0%} of the squares alternate as a board's should"
    if min(c1 - c0, r1 - r0) + 1 < 3:
        return None, "too few squares"
    if _spacing(Hu, c0, r0) < _MIN_SQUARE * 0.6:
        return None, "the squares are too small to be a board's"
    b = _board(pic, Hu, c0, r0, c1, r1, tones=tones, agree=agree, **kw)
    b.flat = _perspective(b.H, b.cols, b.rows) < 1.03
    return b, ""


def find_tilted(pic: Picture, area: tuple[float, float, float, float] | None = None, *, square: float | None = None) -> Board | None:
    """A board of alternating squares in ``pic`` (or its ``area``, image pixels), seen at any angle
    — NEO's own: from the X-corners it finds, no model involved. ``square``: about how big a square
    is (pixels), when known. None when there is none."""
    for f in _scales(pic, area, square):
        g, f, ox, oy = _work(pic, area, f)
        if min(g.shape) < 40:
            continue
        P, strength = _peaks(_saddle(g, 2), 3)
        lat, H = _seek_lattice(g, P, strength)
        if H is None or len(lat) < 12:
            continue
        keys = np.array(list(lat), float)
        if np.any(keys.max(0) - keys.min(0) < 2):
            continue
        b, _ = _settle(pic, _lift(f, ox, oy) @ H, keys, source="neo")
        if b is not None:
            return b
    return None


def _order(corners) -> np.ndarray:
    """Four points as top-left, top-right, bottom-right, bottom-left (as seen)."""
    p = np.asarray(corners, float).reshape(4, 2)
    c = p.mean(0)
    p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]  # clockwise on screen (y down)
    first = int(np.argmin(p[:, 0] + p[:, 1]))
    return np.roll(p, -first, axis=0)


def _seeded(g: np.ndarray, H: np.ndarray, n_c: int, n_r: int) -> tuple[dict[tuple[int, int], int], np.ndarray | None, np.ndarray]:
    """X-corners for a board whose mapping is roughly known (``H``: lattice → working pixels): each
    predicted inner corner matched to the saddle nearest it, kept when it is a corner of this
    board's squares, the rest grown from them as NEO grows its own lattices."""
    P, _ = _peaks(_saddle(g, 2), 3)
    if len(P) < 8:
        return {}, None, P
    lat: dict[tuple[int, int], int] = {}
    for reach in (0.4, 0.25):
        keys = np.array([(i, j) for i in range(1, n_c) for j in range(1, n_r)], float)
        Q = _apply(H, keys)
        sp = np.minimum(np.linalg.norm(_apply(H, keys + (1, 0)) - Q, axis=1), np.linalg.norm(_apply(H, keys + (0, 1)) - Q, axis=1))
        D = np.linalg.norm(Q[:, None, :] - P[None, :, :], axis=2)
        k = np.argmin(D, axis=1)
        near = D[np.arange(len(keys)), k] <= reach * sp
        x = _xcheck(g, H, keys[near])
        sign = 1.0 if (x > 0).sum() >= (x < 0).sum() else -1.0
        lat = {}
        for (i, j), idx, xv in zip(keys[near].astype(int).tolist(), k[near].tolist(), x.tolist(), strict=True):
            if xv * sign > 0 and idx not in lat.values():
                lat[(i, j)] = idx
        lat, H1, err = _clean(P, lat)
        if H1 is None or len(lat) < 8 or err > 0.1:
            return {}, None, P
        lat = _grow(g, P, lat, sign)
        lat, H1, err = _clean(P, lat)
        if H1 is None or err > 0.1:
            return {}, None, P
        H = H1
    return lat, H, P


def from_corners(pic: Picture, corners, *, rows: int | None = None, cols: int | None = None) -> tuple[Board | None, str]:
    """A board from the 4 corners of its playing area as a model that can see gave them (image
    pixels of ``pic``, any order), refined and checked by NEO: first NEO looks for the board's
    X-corners on its own around there; else the mapping the corners imply is moved onto the
    X-corners near where it says they are. Either way the squares must alternate, and stop
    alternating at the edges. The board is marked as mapped from a model's corners.
    (board, "") or (None, why not)."""
    try:
        quad = _order(corners)
    except (ValueError, TypeError):
        return None, "four corners are needed, each [x, y]"
    if not np.all(np.isfinite(quad)):
        return None, "the corners must be numbers"
    n_c, n_r = int(cols or 8), int(rows or 8)
    if not (2 <= n_c <= _MAX_CELLS and 2 <= n_r <= _MAX_CELLS):
        return None, f"a board has 2 to {_MAX_CELLS} squares on a side"
    d1, d2 = quad[2] - quad[0], quad[3] - quad[1]
    area = abs(float(d1[0] * d2[1] - d1[1] * d2[0])) / 2
    if area < 40 * 40:
        return None, "those corners enclose almost nothing"
    H0 = homography([(0, 0), (n_c, 0), (n_c, n_r), (0, n_r)], quad)
    if H0 is None:
        return None, "those corners don't make a board"
    square = math.sqrt(area) / max(n_c, n_r)
    pad = 1.2 * square
    x0, y0 = quad.min(0) - pad
    x1, y1 = quad.max(0) + pad
    region = (float(x0), float(y0), float(x1 - x0), float(y1 - y0))

    def fits(b: Board, reach: float) -> bool:
        got = _apply(np.linalg.inv(pic.to_screen), b.corners())
        return all(np.min(np.linalg.norm(got - q, axis=1)) <= reach * square for q in quad)

    own = find_tilted(pic, region, square=square)  # (NEO's own grid, where the corners say: kept when they agree)
    if own is not None and fits(own, 1.0) and ((rows is None and cols is None) or (own.rows, own.cols) == (n_r, n_c)):
        own.source = "vision"
        return own, ""
    g, f, ox, oy = _work(pic, region, _scales(pic, region, square)[0])
    L = _lift(f, ox, oy)
    lat, H, P = _seeded(g, np.linalg.inv(L) @ H0, n_c, n_r)
    if H is None:
        return None, "NEO couldn't find the squares' corners near those corners — check them"
    b, why = _settle(pic, L @ H, np.array(list(lat), float), source="vision")
    if b is None:
        return None, why
    if (rows or cols or b.rows != b.cols) and (b.rows != n_r or b.cols != n_c):
        return None, f"the squares there make a {b.rows}×{b.cols} board, not {n_r}×{n_c}"
    if not fits(b, 0.75):
        return None, "the board NEO found there doesn't have those corners — check them"
    return b, ""


# ---- pieces --------------------------------------------------------------------------------------

_S = 16  # a flat square is looked at as S × S samples
_T = np.linspace(-0.05, 0.3, 6)  # seen at an angle: from the middle toward the viewer…
_U = np.linspace(-0.3, 0.3, 7)  # …and across; |u| ≥ 0.25 is the edge of a piece's foot
_EDGE = np.abs(np.repeat(_U[None, :], len(_T), 0).ravel()) >= 0.25


def _front(b: Board) -> np.ndarray:
    """The way toward the viewer in board units: where the picture goes down. A piece seen at an
    angle rises the other way, over the squares behind it."""
    c = (b.cols / 2, b.rows / 2)
    p = _apply(b.H, [c, (c[0] + 0.01, c[1]), (c[0], c[1] + 0.01)])
    J = np.column_stack([p[1] - p[0], p[2] - p[0]]) / 0.01  # d(screen)/d(board)
    try:
        d = np.linalg.solve(J, np.array([0.0, 1.0]))
    except np.linalg.LinAlgError:
        d = np.array([0.0, 1.0])
    return d / (np.linalg.norm(d) + 1e-9)


def _layout(b: Board) -> tuple[np.ndarray, np.ndarray]:
    """Where each square is looked at, in board units from its top-left corner: (points judged for
    a piece, points that show the square's own colour)."""
    if b.flat:
        t = 0.1 + 0.8 * (np.arange(_S) + 0.5) / _S
        xx, yy = np.meshgrid(t, t)
        body = np.stack([xx.ravel(), yy.ravel()], 1)
        i = np.arange(_S)
        edge = np.minimum.outer(np.minimum(i, _S - 1 - i), np.minimum(i, _S - 1 - i)) < 2
        return body, body[edge.ravel()]
    f = _front(b)
    s = np.array([-f[1], f[0]])
    body = np.array([(0.5, 0.5) + f * t + s * u for t in _T for u in _U])
    corners = np.array([(0.13, 0.13), (0.87, 0.13), (0.13, 0.87), (0.87, 0.87)])
    near = corners[np.argsort(-((corners - 0.5) @ f))[:2]]  # the two corners nearest the viewer
    ground = np.array([p + (dx, dy) for p in near for dx in (-0.03, 0, 0.03) for dy in (-0.03, 0, 0.03)])
    return body, ground


def _look(pic: Picture, b: Board) -> tuple[np.ndarray, np.ndarray]:
    """Every square's samples in this picture: (piece points, ground points), each (rows, cols, n, 3)."""
    body, ground = _layout(b)
    Hi = np.linalg.inv(pic.to_screen) @ b.H
    sq = _block(0, 0, b.cols - 1, b.rows - 1).astype(float)
    out = []
    for pts in (body, ground):
        xy = _apply(Hi, (sq[:, None, :] + pts[None, :, :]).reshape(-1, 2))
        out.append(_sample(pic.rgb, xy[:, 0], xy[:, 1]).reshape(b.rows, b.cols, len(pts), 3))
    return out[0], out[1]


_LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def _pieceness(pic: Picture, b: Board) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(which samples show something other than the square, the samples, the square's colour)."""
    body, ground = _look(pic, b)
    if b.flat:
        bg = np.median(ground, axis=2, keepdims=True)
        return np.abs(body - bg).sum(axis=3) > 60, body, bg
    n = ground.shape[2] // 2
    g1, g2 = np.median(ground[:, :, :n], axis=2), np.median(ground[:, :, n:], axis=2)
    par = (np.arange(b.rows)[:, None] + np.arange(b.cols)[None, :]) % 2 == 0
    tone = np.where(par[..., None], b.tones.even, b.tones.odd) if b.tones is not None else (g1 + g2) / 2
    same = np.linalg.norm(g1 - g2, axis=2, keepdims=True) < 30
    bg = np.where(same, (g1 + g2) / 2, tone)[:, :, None, :]
    piece = np.abs(body - bg).sum(axis=3) > 75
    # a shadow is the square's own colour, darker: not a piece
    k = (body * bg).sum(axis=3) / np.maximum((bg * bg).sum(axis=3), 1e-6)
    resid = np.linalg.norm(body - k[..., None] * bg, axis=3)
    piece &= ~((k > 0.45) & (k < 0.95) & (resid < 12))
    return piece, body, bg


def occupancy(pic: Picture, b: Board) -> list[str]:
    """Per row, top first: '.' an empty square, 'L' a light piece on it, 'D' a dark one — judged by
    how much of the square's middle differs from its own colour, and whether that part is lighter
    or darker. Seen at an angle, the front of a piece's foot is what is judged, and it must be as
    wide as a foot: what rises from the square in front (a narrower body or head) isn't a piece."""
    piece, body, bg = _pieceness(pic, b)
    # a light piece shows parts clearly lighter than its square (a dark one hardly ever does: its
    # outline, its shading and its anti-aliased edges are all darker)
    lighter = (body @ _LUMA) > (bg @ _LUMA) + 15
    light_share = np.where(piece, lighter, False).sum(axis=2) / np.maximum(piece.sum(axis=2), 1)
    if b.flat:
        there = piece.mean(axis=2) >= 0.06
    else:
        there = (piece[:, :, ~_EDGE].mean(axis=2) >= 0.4) & (piece[:, :, _EDGE].mean(axis=2) >= 0.3)
    return ["".join(("L" if light_share[r, c] >= 0.25 else "D") if there[r, c] else "." for c in range(b.cols)) for r in range(b.rows)]


def orientation(occ: list[str]) -> str:
    """Where the light pieces' home rows are, judged from where the pieces stand: "bottom", "top",
    "left", "right", or "" when the pieces don't say (too few, or mixed)."""
    light = np.array([(c, r) for r, row in enumerate(occ) for c, ch in enumerate(row) if ch == "L"], float)
    dark = np.array([(c, r) for r, row in enumerate(occ) for c, ch in enumerate(row) if ch == "D"], float)
    if len(light) < 4 or len(dark) < 4:
        return ""
    dc, dr = light.mean(0) - dark.mean(0)
    if abs(dr) >= 2 and abs(dr) >= 2 * abs(dc):
        return "bottom" if dr > 0 else "top"
    if abs(dc) >= 2 and abs(dc) >= 2 * abs(dr):
        return "right" if dc > 0 else "left"
    return ""


def changed_cells(b: Board, before: np.ndarray, after: np.ndarray) -> list[tuple[int, int]]:
    """Squares (col, row) whose look changed between two states (``state``): what a move did. A
    square changed when enough of its samples did (noise, JPEG and a blinking highlight don't).
    Seen at an angle, a change only in the middle of a square whose neighbour toward the viewer
    changed too is that neighbour's piece rising over it, not a change of its own."""
    hit = np.abs(before - after).mean(axis=3) > 25  # (rows, cols, samples)
    moved = hit.mean(axis=2) > (0.05 if b.flat else 0.15)
    if not b.flat:
        wide = hit[:, :, _EDGE].mean(axis=2) > 0.15
        f = _front(b)
        step = (int(round(f[0])), int(round(f[1])))
        keep = moved.copy()
        for r, c in zip(*np.nonzero(moved & ~wide), strict=True):
            fc, fr = c + step[0], r + step[1]
            if 0 <= fr < b.rows and 0 <= fc < b.cols and moved[fr, fc]:
                keep[r, c] = False
        moved = keep
    rows, cols = np.nonzero(moved)
    return [(int(c), int(r)) for r, c in zip(rows, cols, strict=True)]


def state(pic: Picture, b: Board) -> np.ndarray:
    """How every square looks now (its piece points), to compare with a later picture."""
    return _look(pic, b)[0]


def _rims(pic: Picture, b: Board) -> np.ndarray:
    Hi = np.linalg.inv(pic.to_screen) @ b.H
    return _rim_colours(pic.rgb, Hi, _block(0, 0, b.cols - 1, b.rows - 1))


def _covered(pic: Picture, b: Board) -> bool:
    """Is the board no longer where it was — covered (a sheet, a menu), moved or gone? A move changes
    squares' middles, and a highlight or a piece's shadow a few squares' rims; something over the
    board changes many rims at once, and a board that moved no longer alternates where it was."""
    sq = _block(0, 0, b.cols - 1, b.rows - 1)
    Hi = np.linalg.inv(pic.to_screen) @ b.H
    seen = _inside(pic.rgb.shape, Hi, sq)
    if seen.mean() < 0.9:
        return True
    if b.rim is not None:
        rim = _rims(pic, b)
        changed = (np.abs(rim - b.rim).mean(axis=2) > 12).mean(axis=1) > 0.3
        if changed.sum() > max(6, 0.12 * len(sq)):
            return True
    return bool(not b.lines and b.tones is not None and _agree(pic.rgb, Hi, sq[seen], b.tones).mean() < b.agree - 0.12)


# ---- what NEO knows: the boards it found ---------------------------------------------------------

FRESH_S = 60.0  # a board last seen longer ago than this is looked at again before a square is acted on
_KEEP = 8
_boards: dict[str, Board] = {}  # by screenshot id
_latest: Board | None = None
_pictures: dict[str, Picture] = {}  # screenshots NEO saw a board-like area in but no grid: for map_board
_KEEP_PICTURES = 2


def forget() -> None:
    """Drop every board and picture (tests; a new conversation)."""
    global _latest
    _boards.clear()
    _pictures.clear()
    _latest = None


def latest() -> Board | None:
    return _latest


def for_shot(shot: str) -> Board | None:
    return _boards.get(str(shot or "").strip())


def _same_place(a: Board, b: Board) -> bool:
    """The same board (same size, corners within a quarter of a square): one seen again."""
    if (a.rows, a.cols) != (b.rows, b.cols):
        return False
    tol = 0.25 * min(a.square_size(), b.square_size())
    return bool(np.all(np.linalg.norm(a.corners() - b.corners(), axis=1) <= tol))


def _remember(b: Board) -> Board:
    global _latest
    if b.shot:
        _boards[b.shot] = b
        while len(_boards) > _KEEP:
            _boards.pop(next(iter(_boards)))
    _latest = b
    return b


def _read(pic: Picture, b: Board, now: float, previous: Board | None) -> list[str]:
    """Fill in what the pieces say (occupancy, orientation, how it looks) and what changed since
    ``previous``, the same board seen before; the changed squares' names."""
    since: list[str] = []
    if not b.lines:
        b.occupancy = occupancy(pic, b)
        o = orientation(b.occupancy)
        if o:
            b.orientation, b.oriented_by = o, "pieces"
        elif previous is not None and previous.orientation:
            b.orientation, b.oriented_by = previous.orientation, previous.oriented_by
    b.state, b.rim = state(pic, b), _rims(pic, b)
    b.at = now
    if previous is not None and previous.state is not None and previous.state.shape == b.state.shape:
        since = [b.name(c, r) for c, r in changed_cells(b, previous.state, b.state)]
    return since


@dataclass
class Survey:
    """What a look at a picture found: a board (and what changed on it since the last look), or
    only a hint that there may be one (a large area without accessible elements, no grid)."""

    board: Board | None = None
    since: list[str] = field(default_factory=list)
    hint: str = ""  # NEO's words for a board-like area it couldn't map
    covered: bool = False


def survey(
    pic: Picture,
    *,
    shot: str = "",
    area: tuple[float, float, float, float] | None = None,
    canvas: bool = False,
    sheet: bool = False,
    window: tuple[float, float, float, float] | None = None,
    now: float | None = None,
) -> Survey:
    """Look for a board in a screenshot, on evidence only. ``area``: the part of the picture to
    look in (screen points), e.g. a window's content without its toolbar. ``canvas``: the caller's
    evidence that ``area`` is a large region with no accessible elements (a view drawn in one
    piece) — only then are grids drawn with lines and boards seen at an angle looked for; a flat
    board of alternating squares is looked for in any case (finding one is its own evidence).
    ``sheet``: a sheet or dialog is open over the window (Accessibility says so): nothing is read
    under it. ``window``: the window's frame now, to notice later that it moved."""
    now = time.monotonic() if now is None else now
    if sheet:
        return Survey(covered=True)
    region = None
    if area is not None:
        inv = np.linalg.inv(pic.to_screen)
        (ax0, ay0), (ax1, ay1) = _apply(inv, [(area[0], area[1]), (area[0] + area[2], area[1] + area[3])])
        region = (float(ax0), float(ay0), float(ax1 - ax0), float(ay1 - ay0))
    b = _flat(pic, region, canvas)
    if b is None and canvas:
        b = find_tilted(pic, region)
    if b is None:
        if canvas and shot:
            _pictures[shot] = pic
            while len(_pictures) > _KEEP_PICTURES:
                _pictures.pop(next(iter(_pictures)))
            return Survey(hint=shot)
        return Survey()
    b.shot, b.window = shot, window
    previous = _latest if _latest is not None and _same_place(_latest, b) else None
    since = _read(pic, b, now, previous)
    _remember(b)
    return Survey(board=b, since=since)


def _flat(pic: Picture, region, canvas: bool) -> Board | None:
    g = pic.gray
    ox = oy = 0
    if region is not None:
        x, y, w, h = region
        ox, oy = max(0, int(x)), max(0, int(y))
        g = g[oy : max(oy, int(y + h)), ox : max(ox, int(x + w))]
    grid = find_grid(g) if g.size else None
    if grid is None or (grid.lines and not canvas):
        return None  # (a table or a calendar drawn with lines is no board: only on a canvas)
    H_img = _lift(1, ox, oy) @ grid.H
    b = Board(pic.to_screen @ H_img, grid.rows, grid.cols, "neo", flat=True, lines=grid.lines)
    if not grid.lines:
        sq = _block(0, 0, grid.cols - 1, grid.rows - 1)
        t = _tones(pic.rgb, H_img, sq)
        if t.gap < 30:
            return None
        b.tones, b.agree = t, float(_agree(pic.rgb, H_img, sq, t).mean())
        if b.agree < 0.75:
            return None
    return b


# ---- evidence: where a board may be ---------------------------------------------------------------


# Roles whose content Accessibility reads out even when they hold no elements: never a drawn view.
_READ_ROLES = frozenset({"AXTextArea", "AXTextField", "AXStaticText", "AXWebArea", "AXTable", "AXList", "AXOutline", "AXBrowser", "AXCell", "AXRow"})


def canvas_area(
    window: tuple[float, float, float, float] | None, elements, *, least: float = 0.2, grid: int = 32
) -> tuple[float, float, float, float] | None:
    """The part of a window drawn in one piece (screen points), when it is at least ``least`` of
    the window: the evidence a board may be there. ``elements``: every element's frame from the
    window's Accessibility tree, as (x, y, w, h) or (x, y, w, h, role). A large element holding no
    others (and not text, a list or a page) is such a view; else, when the view isn't exposed at
    all, the largest part of the window no element covers. An element that holds others is a
    container, not content."""
    if window is None or window[2] < 100 or window[3] < 100:
        return None
    wx, wy, ww, wh = (float(v) for v in window)
    items = [e for e in elements if e and len(e) >= 4 and e[2] > 0 and e[3] > 0]
    F = np.array([e[:4] for e in items], float).reshape(-1, 4)
    roles = [str(e[4]) if len(e) > 4 else "" for e in items]
    occ = np.zeros((grid, grid), bool)
    if len(F):
        cx, cy = F[:, 0] + F[:, 2] / 2, F[:, 1] + F[:, 3] / 2
        holds = (cx[None, :] > F[:, None, 0]) & (cx[None, :] < F[:, None, 0] + F[:, None, 2])
        holds &= (cy[None, :] > F[:, None, 1]) & (cy[None, :] < F[:, None, 1] + F[:, None, 3])
        np.fill_diagonal(holds, False)
        leaf = ~holds.any(axis=1)
        # (clipped to the window: a view scrolled partly out of it counts for what shows)
        x0, y0 = np.maximum(F[:, 0], wx), np.maximum(F[:, 1], wy)
        x1, y1 = np.minimum(F[:, 0] + F[:, 2], wx + ww), np.minimum(F[:, 1] + F[:, 3], wy + wh)
        shown = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
        big = shown >= least * ww * wh
        views = [i for i in np.nonzero(leaf & big)[0] if roles[i] not in _READ_ROLES]
        if views:
            i = max(views, key=lambda k: shown[k])
            return float(x0[i]), float(y0[i]), float(x1[i] - x0[i]), float(y1[i] - y0[i])
        if any(roles[i] in _READ_ROLES for i in np.nonzero(leaf & big)[0]):
            return None  # a page, a text or a list fills it: read that, not pixels
        for x, y, w, h in F[leaf & ~big]:
            c0, c1 = int(np.floor((x - wx) / ww * grid)), int(np.ceil((x + w - wx) / ww * grid))
            r0, r1 = int(np.floor((y - wy) / wh * grid)), int(np.ceil((y + h - wy) / wh * grid))
            occ[max(0, r0) : max(0, min(grid, r1)), max(0, c0) : max(0, min(grid, c1))] = True
    best, rect = 0, None
    heights = np.zeros(grid, int)
    for r in range(grid):  # the largest empty rectangle: a histogram of empty runs, row by row
        heights = np.where(occ[r], 0, heights + 1)
        stack: list[int] = []
        for c in range(grid + 1):
            hc = heights[c] if c < grid else 0
            while stack and heights[stack[-1]] >= hc:
                top = stack.pop()
                left = stack[-1] + 1 if stack else 0
                area = int(heights[top]) * (c - left)
                if area > best:
                    best, rect = area, (left, r - int(heights[top]) + 1, c - left, int(heights[top]))
            stack.append(c)
    if rect is None or best < least * grid * grid:
        return None
    c, r, w, h = rect
    return (wx + c * ww / grid, wy + r * wh / grid, w * ww / grid, h * wh / grid)


# ---- words for the model -----------------------------------------------------------------------


def _rows_text(b: Board) -> str:
    """The pieces as compact rows, top row first: runs of empty squares as digits."""
    out = []
    for row in b.occupancy or []:
        s, run = "", 0
        for ch in row:
            if ch == ".":
                run += 1
                continue
            s += (str(run) if run else "") + ch
            run = 0
        out.append(s + (str(run) if run else ""))
    return "/".join(out)


_SIDE_WORDS = {"bottom": "at the bottom", "top": "at the top", "left": "on the left", "right": "on the right"}


def note(b: Board, since: list[str] | None = None, *, blind: bool = False) -> str:
    """NEO's own words about a board it knows (no coordinates, nothing read off the screen): its
    size, how its squares are named, how to act on them, the pieces, what changed."""
    kind = "a grid" if b.lines else "a board"
    how = "NEO found" if b.source == "neo" else "NEO mapped (from the corners given, checked by NEO)"
    seen = "" if b.flat else ", seen at an angle"
    text = f"{how} {kind} of {b.rows}×{b.cols} squares{seen}. "
    if b.chess:
        who = "judged from the pieces" if b.oriented_by == "pieces" else "as the model said"
        text += f"Squares are named {b.names()}: White (the light pieces) {_SIDE_WORDS[b.orientation]}, {who}. "
    else:
        text += f"Squares are named by row and column from the top-left as seen: r1c1 … r{b.rows}c{b.cols}"
        if b.rows == b.cols == 8 and not b.lines:
            text += " (which side is White's isn't clear from the pieces"
            text += ")" if blind else '; a model that can see may say it: map_board orientation="bottom", …)'
        text += ". "
    if blind and b.source != "neo":
        text += "A model that can't see can't act on this board's squares (its corners came from a model); take a new screenshot."
        return text
    c, r = min(4, b.cols - 1), max(0, b.rows - 2)
    text += f'To act on a square, name it — click cell="{b.name(c, r)}" · drag from="{b.name(c, r)}" to="{b.name(c, max(0, r - 2))}"'
    text += " — and NEO works out the points."
    if b.occupancy is not None:
        text += f" Pieces now, top row first (L light, D dark, digits: empty squares): {_rows_text(b)}"
        text += "." if b.flat else " (read from the front of each piece's foot: from a low angle, a tall piece can hide the square behind it)."
    if since:
        text += f" Changed since the last look: {', '.join(since[:12])}."
    return text


def covered_note() -> str:
    return "A sheet or dialog is open over this window: a board under it can't be read or played — answer the sheet first, then look again."


def hint_note(shot: str, *, blind: bool = False) -> str:
    """For a large area without accessible elements where NEO found no grid."""
    if blind:
        return (
            "This window has a large area with no accessible elements, and NEO couldn't find a board's squares in it "
            "by itself; mapping it needs a model that can see."
        )
    return (
        "This window has a large area with no accessible elements, and NEO couldn't find a board's squares in it by "
        f'itself. If it is a board, give its playing area\'s 4 corners once: map_board shot="{shot}" '
        "corners=[[x,y],[x,y],[x,y],[x,y]] (pixels of this screenshot; rows and cols if not 8×8) — NEO checks them "
        "and then maps every square."
    )


def look(
    pic: Picture,
    *,
    shot: str,
    blind: bool = False,
    sheet: bool = False,
    area: tuple[float, float, float, float] | None = None,
    canvas: bool = False,
    window: tuple[float, float, float, float] | None = None,
    now: float | None = None,
) -> str:
    """The screenshot note's board part in one call (NEO's own words; "" when there is nothing to
    say): a board found (or seen again, with what changed), one under a sheet, or a board-like area
    NEO couldn't map."""
    s = survey(pic, shot=shot, area=area, canvas=canvas, sheet=sheet, window=window, now=now)
    if s.covered:
        return covered_note() if canvas or (_latest is not None and window is not None and _latest.window == window) else ""
    if s.board is not None:
        return note(s.board, s.since, blind=blind)
    return hint_note(shot, blind=blind) if s.hint else ""


def region(b: Board, margin: float = 0.03) -> tuple[int, int, int, int]:
    """The screen rectangle (points: x, y, w, h) to capture to look at the board again."""
    x, y, w, h = b.bbox()
    mx, my = w * margin, h * margin
    return int(math.floor(x - mx)), int(math.floor(y - my)), int(math.ceil(w + 2 * mx)), int(math.ceil(h + 2 * my))


# ---- acting on squares -------------------------------------------------------------------------


class BoardError(Exception):
    """Nothing was done; the message says why and what to do instead (no coordinates in it)."""


def blind_refusal() -> str:
    return (
        "Error: you can't see the screen, and this board's squares were mapped from a seeing model's corners, "
        "not by NEO from the pixels, so I won't act on them for you. Take a new screenshot: if NEO finds the "
        "board itself, its squares can be named."
    )


def _pick(a: dict) -> Board | None:
    shot = str(a.get("shot") or "").strip()
    if shot:
        return for_shot(shot) or (None if shot in _pictures else _latest)
    return _latest


def points_for(
    a: dict,
    keys: tuple[str, ...],
    *,
    blind: bool = False,
    now: float | None = None,
    window: tuple[float, float, float, float] | None = None,
) -> tuple[list[tuple[float, float]], Board, list[str]]:
    """The screen points for squares named in ``a[k]`` for each of ``keys`` ("cell", or "from" and
    "to"), on the board of ``a["shot"]`` or the latest one: (points, board, the names as the board
    writes them). A brain that can't see gets squares only on a board NEO found itself in a fresh
    picture. Raises BoardError (no coordinates in its words)."""
    now = time.monotonic() if now is None else now
    b = _pick(a)
    if b is None:
        raise BoardError("Error: NEO doesn't know a board here yet — take a screenshot of it first (NEO finds its squares).")
    if blind and b.source != "neo":
        raise BoardError(blind_refusal())
    if now - b.at > FRESH_S:
        raise BoardError(f"Error: NEO last saw that board {now - b.at:.0f}s ago — take a new screenshot first (it may have moved).")
    if window is not None and b.window is not None and max(abs(p - q) for p, q in zip(window, b.window, strict=True)) > 2:
        raise BoardError("Error: the board's window moved since NEO found the board — take a new screenshot first.")
    pts, names = [], []
    for k in keys:
        try:
            col, row = b.parse(a.get(k))
        except ValueError as e:
            raise BoardError(f"Error: {k}: {e}") from None
        pts.append(b.centre(col, row))
        names.append(b.name(col, row))
    return pts, b, names


@dataclass
class Move:
    """What a click or a drag on a board did, in NEO's words (square names only)."""

    said: str
    changed: bool | None  # None: couldn't tell (the board isn't where it was)
    cells: list[str] = field(default_factory=list)


def after_move(b: Board, pic: Picture, frm: str = "", to: str = "", *, now: float | None = None) -> Move:
    """Compare the board in ``pic`` (a picture taken after the input) with how it looked before:
    which squares changed, and what is on the ones acted on now. ``frm``/``to``: the squares'
    names (a click: ``to`` only). The board's look is updated, and it counts as freshly seen."""
    now = time.monotonic() if now is None else now
    if b.state is None:
        return Move("", None)
    if _covered(pic, b):
        return Move("The board isn't where it was (covered, moved or gone) — take a screenshot before the next move.", None)
    now_state = state(pic, b)
    cells = [b.name(c, r) for c, r in changed_cells(b, b.state, now_state)]
    b.state, b.rim, b.at = now_state, _rims(pic, b), now
    if not b.lines:
        b.occupancy = occupancy(pic, b)
    if not cells:
        return Move("Nothing changed on the board — the move may not have been taken (not allowed, not your turn, or it missed).", False)

    def what(name: str) -> str:
        if b.occupancy is None:
            return ""
        col, row = b.parse(name)
        ch = b.occupancy[row][col]
        return "empty" if ch == "." else ("a light piece" if ch == "L" else "a dark piece")

    bits = []
    if frm:
        w = what(frm)
        bits.append(f"{frm} is now empty" if w == "empty" else (f"{frm} still has {w}" if w else f"{frm} changed" if frm in cells else f"{frm} didn't change"))
    if to:
        w = what(to)
        bits.append(f"{to} has {w}" if w and w != "empty" else (f"{to} is empty" if w else f"{to} changed" if to in cells else f"{to} didn't change"))
    others = [c for c in cells if c not in (frm, to)]
    if others:
        bits.append("also changed: " + ", ".join(others[:12]))
    return Move("On the board: " + "; ".join(bits) + ".", True, cells)


# ---- a seeing model's corners ------------------------------------------------------------------


def map_board(a: dict, *, blind: bool = False, picture: Picture | None = None, now: float | None = None) -> tuple[str, bool]:
    """The map_board tool's work: a model that can see gives a board's 4 corners once (pixels of a
    screenshot, or Gemini's 0–1000 [y, x] points), NEO refines and checks them and maps every square;
    or it says which side White is on. (text, ok). ``picture``: that screenshot's picture, when the
    caller keeps it (else the one NEO kept when it saw a board-like area there)."""
    now = time.monotonic() if now is None else now
    if blind:
        return (
            "Error: you can't see the screen — no screenshot ever reached you — so you can't give a board's corners "
            "or say which side is White's. Tell the user the model that can see the screen is busy right now.",
            False,
        )
    shot = str(a.get("shot") or "").strip()
    side = str(a.get("orientation") or "").strip().lower()
    if side and side not in SIDES:
        return f"Error: orientation is where White's pieces start: one of {', '.join(SIDES)}.", False
    if not a.get("corners") and not a.get("corners_norm"):
        b = for_shot(shot) if shot else _latest
        if b is None or not side:
            return "Error: give corners=[[x,y],…] (the playing area's 4 corners, pixels of the screenshot) with its shot id.", False
        if b.lines or (b.rows, b.cols) != (8, 8):
            return f"Error: that is a {b.rows}×{b.cols} {'grid' if b.lines else 'board'}: its squares have no chess names.", False
        b.orientation, b.oriented_by = side, "model"
        return note(b), True
    pic = picture if picture is not None else _pictures.get(shot)
    if pic is None:
        return f"Error: NEO no longer has screenshot {shot[:16]!r} — take a new one, then give the corners in it.", False
    h, w = pic.rgb.shape[:2]
    try:
        if a.get("corners"):
            pts = np.array(a["corners"], float).reshape(4, 2)
        else:
            yx = np.array(a["corners_norm"], float).reshape(4, 2)
            pts = np.stack([yx[:, 1] / 1000 * w, yx[:, 0] / 1000 * h], 1)
    except (TypeError, ValueError):
        return "Error: corners must be 4 points, each [x, y] (or corners_norm: 4 × [y, x] on a 0–1000 scale).", False
    if not np.all(np.isfinite(pts)) or np.any(pts < -1) or np.any(pts[:, 0] > w + 1) or np.any(pts[:, 1] > h + 1):
        return f"Error: the corners must lie in the screenshot, which is {w}×{h} px.", False
    try:
        rows = int(a["rows"]) if a.get("rows") not in (None, "") else None
        cols = int(a["cols"]) if a.get("cols") not in (None, "") else None
    except (TypeError, ValueError):
        return "Error: rows and cols must be whole numbers.", False
    b, why = from_corners(pic, pts, rows=rows, cols=cols)
    if b is None:
        return f"Error: {why}.", False
    b.shot = shot
    previous = _latest if _latest is not None and _same_place(_latest, b) else None
    since = _read(pic, b, now, previous)
    if side and (b.rows, b.cols) == (8, 8) and not b.lines:
        b.orientation, b.oriented_by = side, "model"
    _remember(b)
    return note(b, since), True
