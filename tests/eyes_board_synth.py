"""Synthetic screens with boards for the board helper's tests and benchmarks (nothing here touches
the Mac: no capture, no input, no Accessibility).

- ``flat_screen``: a macOS-like screen with a flat 2D board (glyph pieces), drawn at 2 px per point
  and reduced to points the way NEO's screenshots are.
- ``tilted_screen``: a textured wooden board seen in perspective (a camera above and in front of
  it, like a 3D chess app's default view), with upright shaded 3D pieces standing on the squares,
  their shadows, a wooden frame around the playing area and a window behind it.

Both return ``(image, truth)``: ``truth["H"]`` maps board units (col, row; (0, 0) = the top-left
corner of the playing area as seen, rows going down the picture) to image pixels (continuous
coordinates: pixel k spans [k, k+1)).
"""

from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W_PT, H_PT = 1512, 982
_SF = "/System/Library/Fonts/SFNS.ttf"
_UNI = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"

START = ("rnbqkbnr", "pppppppp", "........", "........", "........", "........", "PPPPPPPP", "RNBQKBNR")
_GLYPH = {"k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟"}


def _font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except Exception:  # noqa: BLE001 - a machine without the font: PIL's own
        return None


def truth_occupancy(position) -> list[str]:
    """'.' empty, 'L' a light (upper-case) piece, 'D' a dark one — per row, top row first."""
    return ["".join("." if ch == "." else ("D" if ch.islower() else "L") for ch in row) for row in position]


def moved(position, frm: str, to: str, *, white_at_bottom: bool = True):
    """The position after moving the piece on ``frm`` to ``to`` (chess names; no rules)."""
    grid = [list(r) for r in position]

    def rc(sq):
        f, r = "abcdefgh".index(sq[0]), int(sq[1])
        return (8 - r, f) if white_at_bottom else (r - 1, 7 - f)

    (r1, c1), (r2, c2) = rc(frm), rc(to)
    grid[r2][c2], grid[r1][c1] = grid[r1][c1], "."
    return tuple("".join(r) for r in grid)


# ---- a flat board ----------------------------------------------------------------------------


def flat_screen(
    *,
    board_at=(420, 150),
    sq_pt: int = 76,
    n: int = 8,
    position=START,
    light=(238, 238, 210),
    dark=(118, 150, 86),
    sheet: bool = False,
    highlight=(),
    seed: int = 1,
    scale: int = 2,
):
    """A 1512×982 pt screen (returned at 1 px per point) with a flat board and a sidebar."""
    rnd = random.Random(seed)
    Wp, Hp = W_PT * scale, H_PT * scale
    img = Image.new("RGB", (Wp, Hp), (236, 236, 238))
    d = ImageDraw.Draw(img)
    f = _font(_SF, 13 * scale)
    d.rectangle([0, 0, Wp, 24 * scale], fill=(246, 246, 246))
    x = 40 * scale
    for word in ("Game", "Edit", "View", "Window", "Help"):
        d.text((x, 4 * scale), word, fill=(20, 20, 20), font=f)
        x += (len(word) * 9 + 22) * scale
    wx, wy, ww, wh = 360, 90, 900, 800
    d.rounded_rectangle([wx * scale, wy * scale, (wx + ww) * scale, (wy + wh) * scale], 12 * scale, fill=(250, 250, 250), outline=(200, 200, 200))
    bx, by = board_at
    pf = _font(_UNI, int(sq_pt * 0.8) * scale)
    for r in range(n):
        for c in range(n):
            x0, y0 = (bx + c * sq_pt) * scale, (by + r * sq_pt) * scale
            fill = light if (r + c) % 2 == 0 else dark
            if (c, r) in highlight:
                fill = (246, 246, 105)
            d.rectangle([x0, y0, x0 + sq_pt * scale - 1, y0 + sq_pt * scale - 1], fill=fill)
            p = position[r][c] if r < len(position) and c < len(position[r]) else "."
            if p == ".":
                continue
            ink, edge = ((10, 10, 10), (0, 0, 0)) if p.islower() else ((255, 255, 255), (0, 0, 0))
            if pf is not None:
                d.text((x0 + sq_pt * scale * 0.1, y0 + sq_pt * scale * 0.02), _GLYPH[p.lower()], fill=ink, font=pf, stroke_width=scale, stroke_fill=edge)
            else:
                m = sq_pt * scale * 0.2
                d.ellipse([x0 + m, y0 + m, x0 + sq_pt * scale - m, y0 + sq_pt * scale - m], fill=ink, outline=edge, width=scale)
    for i, lab in enumerate(("New Game", "Undo Move", "Hint", "Settings")):
        y = (160 + i * 44) * scale
        d.rounded_rectangle([1100 * scale, y, 1230 * scale, y + 28 * scale], 6 * scale, fill=(255, 255, 255), outline=(180, 180, 180))
        d.text((1112 * scale, y + 5 * scale), lab, fill=(0, 0, 0), font=f)
    for i in range(18):
        d.text((20 * scale, (60 + i * 26) * scale), f"Item {i + 1}: {rnd.choice(['Recent', 'Saved', 'Online'])} game", fill=(40, 40, 40), font=f)
    if sheet:
        sx, sy, sw, sh = 560, 90, 500, 170
        d.rectangle([sx * scale, sy * scale, (sx + sw) * scale, (sy + sh) * scale], fill=(245, 245, 245), outline=(150, 150, 150))
        d.text(((sx + 20) * scale, (sy + 20) * scale), "Start a New Game?", fill=(0, 0, 0), font=f)
    out = img.reduce(scale) if scale > 1 else img
    H = np.array([[sq_pt, 0, bx], [0, sq_pt, by], [0, 0, 1]], float)
    return out, {"H": H, "rows": n, "cols": n, "sq": sq_pt, "board": (bx, by, n * sq_pt, n * sq_pt), "position": position}


def thin_grid_screen(*, at=(500, 200), cell: int = 50, n: int = 9, line: int = 2, border: int = 4):
    """A sudoku-like grid: dark lines on white, a thicker outer border, digits in some cells."""
    img = Image.new("RGB", (W_PT, H_PT), (240, 240, 242))
    d = ImageDraw.Draw(img)
    x0, y0 = at
    size = n * cell
    d.rectangle([x0, y0, x0 + size, y0 + size], fill=(255, 255, 255))
    for i in range(n + 1):
        w = border if i in (0, n) else line
        p = i * cell
        d.rectangle([x0 + p - w // 2, y0 - border // 2, x0 + p - w // 2 + w - 1, y0 + size + border // 2], fill=(30, 30, 30))
        d.rectangle([x0 - border // 2, y0 + p - w // 2, x0 + size + border // 2, y0 + p - w // 2 + w - 1], fill=(30, 30, 30))
    f = _font(_SF, int(cell * 0.6))
    rnd = random.Random(3)
    for r in range(n):
        for c in range(n):
            if rnd.random() < 0.35:
                d.text((x0 + c * cell + cell * 0.3, y0 + r * cell + cell * 0.12), str(rnd.randint(1, 9)), fill=(20, 20, 20), font=f)
    H = np.array([[cell, 0, x0], [0, cell, y0], [0, 0, 1]], float)
    return img, {"H": H, "rows": n, "cols": n}


# ---- a board in perspective ----------------------------------------------------------------------

_HEIGHT = {"p": 0.8, "r": 0.95, "n": 1.05, "b": 1.2, "q": 1.4, "k": 1.55}


def _profile(kind: str) -> list[tuple[float, float]]:
    """(height, radius) of an upright piece, in squares: a base, a body narrowing up, a head."""
    h = _HEIGHT[kind]
    head = 0.17 if kind == "p" else 0.2
    return [
        (0.0, 0.34),
        (0.1, 0.34),
        (0.16, 0.25),
        (h * 0.55, 0.13),
        (h * 0.62, 0.2 if kind != "p" else 0.16),
        (h * 0.68, 0.12),
        (h - head * 1.6, head),
        (h - head * 0.6, head),
        (h, 0.06),
    ]


class Camera:
    """A pinhole camera looking at the middle of an n×n board lying in the plane z = 0, from
    ``elev`` degrees above it, turned ``yaw`` degrees about the board's middle. World units are
    squares; x to the right, y away from the camera, z up. Image pixels at ``ss`` × points."""

    def __init__(self, *, elev: float, yaw: float, dist: float, focal: float, centre: tuple[float, float], n: int = 8, ss: int = 2):
        e, a = math.radians(elev), math.radians(yaw)
        self.n = n
        self.cam = np.array([dist * math.cos(e) * math.sin(a), -dist * math.cos(e) * math.cos(a), dist * math.sin(e)])
        fwd = -self.cam / np.linalg.norm(self.cam)
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        self.R = np.stack([right, -up, fwd])  # camera axes: x right, y down, z forward
        self.K = np.array([[focal * ss, 0, centre[0] * ss], [0, focal * ss, centre[1] * ss], [0, 0, 1]])
        self.ss = ss

    def project(self, pts: np.ndarray) -> np.ndarray:
        """World points (…, 3) → image pixels (…, 2) at the supersampled scale."""
        pc = (pts - self.cam) @ self.R.T
        p = pc @ self.K.T
        return p[..., :2] / p[..., 2:3]

    def depth(self, pts: np.ndarray) -> np.ndarray:
        return ((pts - self.cam) @ self.R.T)[..., 2]

    def board_H(self) -> np.ndarray:
        """Board units (col, row: row 0 the far side) → supersampled image pixels, for z = 0."""
        half = self.n / 2
        A = np.array([[1.0, 0, -half], [0, -1.0, half], [0, 0, 1.0]])  # (col,row,1) → (x,y,1) world
        t = -self.R @ self.cam
        M = self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], t])
        return M @ A


def _wood(u: np.ndarray, v: np.ndarray, rng: np.random.Generator, along_u: np.ndarray, phase: np.ndarray, amp: float) -> np.ndarray:
    """A grain pattern (−1…1): wavy stripes along one axis of each square."""
    s = np.where(along_u, v, u)
    t = np.where(along_u, u, v)
    wav = np.sin(t * 2.3 + phase) * 0.35 + np.sin(t * 7.1 + phase * 2) * 0.08
    g = np.sin((s * 9.0 + wav + phase) * 2 * math.pi)
    fine = np.sin((s * 37.0 + wav * 3 + phase * 3) * 2 * math.pi)
    return amp * (0.7 * g + 0.3 * fine)


def tilted_screen(
    *,
    elev: float = 55.0,
    yaw: float = 0.0,
    n: int = 8,
    position=START,
    focal: float = 1150.0,
    dist: float = 17.0,
    centre=(756, 520),
    light=(222, 190, 142),
    dark=(122, 78, 44),
    frame=(78, 50, 30),
    background=(52, 54, 60),
    grain: float = 14.0,
    lighting: float = 0.14,
    shadows: bool = True,
    noise: float = 3.0,
    sheet: bool = False,
    seed: int = 7,
    ss: int = 2,
):
    """A 1512×982 pt screen (1 px per point) showing a textured board in perspective with 3D
    pieces (upper-case light, lower-case dark; ``position`` row 0 is the far side)."""
    rng = np.random.default_rng(seed)
    cam = Camera(elev=elev, yaw=yaw, dist=dist, focal=focal, centre=centre, n=n, ss=ss)
    Hs = cam.board_H()
    Wp, Hp = W_PT * ss, H_PT * ss
    # background: a window's dark gradient (coordinates broadcast, not full grids: a render stays small)
    yy = np.arange(Hp, dtype=np.float32)[:, None]
    bg = np.array(background, np.float32)
    img = np.empty((Hp, Wp, 3), np.float32)
    img[:] = bg
    img *= (1.0 - 0.18 * (yy / Hp))[..., None]
    # the board and its frame, per pixel (inverse mapping), inside the frame's bounding box
    rim = 0.7
    corners = np.array([[-rim, -rim], [n + rim, -rim], [n + rim, n + rim], [-rim, n + rim]])
    pc = np.c_[corners, np.ones(4)] @ Hs.T
    pc = pc[:, :2] / pc[:, 2:]
    x0, y0 = np.floor(pc.min(0)).astype(int)
    x1, y1 = np.ceil(pc.max(0)).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(Wp, x1), min(Hp, y1)
    Hi = np.linalg.inv(Hs)
    X, Y = np.meshgrid(np.arange(x0, x1, dtype=np.float32) + 0.5, np.arange(y0, y1, dtype=np.float32) + 0.5)
    q = np.stack([X, Y, np.ones_like(X)], -1) @ Hi.T
    u, v = q[..., 0] / q[..., 2], q[..., 1] / q[..., 2]
    inside = (u >= 0) & (u < n) & (v >= 0) & (v < n)
    in_frame = (u >= -rim) & (u < n + rim) & (v >= -rim) & (v < n + rim) & ~inside
    col, row = np.floor(u).astype(int), np.floor(v).astype(int)
    par = (np.clip(col, 0, n - 1) + np.clip(row, 0, n - 1)) % 2
    sq_rng = rng.uniform(0, 2 * math.pi, (n, n))
    along = ((np.arange(n)[:, None] + np.arange(n)[None, :]) % 2 == 0)
    phase = sq_rng[np.clip(row, 0, n - 1), np.clip(col, 0, n - 1)]
    au = along[np.clip(row, 0, n - 1), np.clip(col, 0, n - 1)]
    g = _wood(u, v, rng, au, phase, grain)
    base = np.where(par[..., None] == 0, np.array(light, np.float32), np.array(dark, np.float32))
    shade = 1.0 + lighting * (0.6 - (u / n) * 0.7 - (v / n) * 0.5)  # light from the far left
    board_px = base * shade[..., None] + g[..., None] * np.array([1.0, 0.8, 0.6])
    fr = np.array(frame, np.float32) * shade[..., None] + _wood(u, v, rng, np.ones_like(au), np.zeros_like(phase), grain * 0.8)[..., None]
    region = img[y0:y1, x0:x1]
    region[inside] = board_px[inside]
    region[in_frame] = fr[in_frame]
    arr = img
    # pieces, far to near; each with a soft shadow on the board first
    pieces = []
    for r in range(n):
        for c in range(n):
            p = position[r][c] if r < len(position) and c < len(position[r]) else "."
            if p != ".":
                pieces.append((r, c, p))
    half = n / 2
    ang = np.linspace(0, 2 * math.pi, 28, endpoint=False)

    def paint(poly: np.ndarray, outline: bool = False):
        """(mask, outline mask, y-slice, x-slice) of a polygon, cut to its bounding box."""
        bx0, by0 = np.floor(poly.min(0)).astype(int) - 2
        bx1, by1 = np.ceil(poly.max(0)).astype(int) + 2
        bx0, by0, bx1, by1 = max(0, bx0), max(0, by0), min(Wp, bx1), min(Hp, by1)
        if bx1 <= bx0 or by1 <= by0:
            return None
        mask = Image.new("L", (bx1 - bx0, by1 - by0), 0)
        md = ImageDraw.Draw(mask)
        pts = [(float(x - bx0), float(y - by0)) for x, y in poly]
        md.polygon(pts, fill=255)
        if outline:
            md.line(pts + pts[:1], fill=128, width=ss)
        m = np.asarray(mask)
        return m, slice(by0, by1), slice(bx0, bx1)

    if shadows:
        for r, c, _p in pieces:
            wx, wy = c + 0.5 - half, half - (r + 0.5)
            ring = np.stack([wx + 0.18 + 0.36 * np.cos(ang), wy + 0.1 + 0.3 * np.sin(ang), np.zeros_like(ang)], -1)
            got = paint(cam.project(ring))
            if got:
                m, ys, xs = got
                arr[ys, xs][m > 0] *= 0.72
    order = sorted(pieces, key=lambda t: -float(cam.depth(np.array([t[1] + 0.5 - half, half - (t[0] + 0.5), 0.0]))))
    for r, c, p in order:
        wx, wy = c + 0.5 - half, half - (r + 0.5)
        pts = []
        for z, rad in _profile(p.lower()):
            ring = np.stack([wx + rad * np.cos(ang), wy + rad * np.sin(ang), np.full_like(ang, z)], -1)
            pts.append(cam.project(ring))
        hull = _hull(np.concatenate(pts))
        got = paint(hull, outline=True)
        if not got:
            continue
        m, ys, xs = got
        colour = np.array((236, 230, 214) if p.isupper() else (46, 38, 34), np.float32)
        h, w = m.shape
        gx = np.linspace(0, 1, w)[None, :]
        gy = np.linspace(0, 1, h)[:, None]
        sh = 0.72 + 0.4 * (1 - np.abs(gx - 0.35) * 1.6).clip(0, 1) - 0.12 * gy
        body = colour[None, None, :] * sh[..., None]
        patch = arr[ys, xs]
        patch[m > 0] = body[m > 0]
        patch[m == 128] *= 0.45
    if sheet:
        sx, sy = 560 * ss, 90 * ss
        arr[sy : sy + 300 * ss, sx : sx + 500 * ss] = (236, 236, 236)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    small = out.reduce(ss) if ss > 1 else out
    if noise:
        a = np.asarray(small).astype(np.float32) + rng.normal(0, noise, (small.height, small.width, 3))
        small = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    H1 = np.diag([1.0 / ss, 1.0 / ss, 1.0]) @ Hs
    return small, {"H": H1 / H1[2, 2], "rows": n, "cols": n, "position": position, "camera": cam}


def _hull(pts: np.ndarray) -> np.ndarray:
    """Convex hull (monotone chain) of 2D points."""
    p = sorted(map(tuple, np.round(pts, 3)))
    if len(p) < 3:
        return np.array(p)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper: list = []
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1])


def board_to_image(H: np.ndarray, pts) -> np.ndarray:
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:]
